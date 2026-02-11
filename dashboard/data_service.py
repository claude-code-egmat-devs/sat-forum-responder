"""
Data service layer for SAT Forum Dashboard.
Provides read-only access to the webhooks SQLite database.
"""

import sqlite3
import os
import time as _time
import json
import requests
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = os.getenv("SAT_DB_PATH", "/opt/sat-forum-responder/db/webhooks.db")

# ── Mixpanel configuration ──
_MIXPANEL_ENABLED = True
_MIXPANEL_CONFIG = {
    "project_id": "2151823",
    "username": "MetricsTracking.3632e6.mp-service-account",
    "secret": "xSHkTT56anfOIAJ7jWg0A4GP6fImk03W",
}
_vote_cache = {"data": None, "ts": 0}  # single cache for all vote events
_VOTE_CACHE_TTL = 300  # 5 minutes

# Usernames to exclude from dashboard (test accounts)
_TEST_USERS = ("sujeev.testing",)


def _dedup_join(alias="w"):
    """Return INNER JOIN clause to keep only the latest row per correlation_id."""
    return f"INNER JOIN (SELECT MAX(id) as id FROM webhooks GROUP BY correlation_id) _dedup ON {alias}.id = _dedup.id"



@contextmanager
def get_db():
    """Get a read-only SQLite connection."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _has_metadata_table(conn):
    """Check if webhook_metadata table exists (safe for older DBs)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='webhook_metadata'"
    ).fetchone()
    return row[0] > 0


# Cache for extra columns detection (per-process, no need to re-check)
_extra_columns_cache = None

def _get_extra_columns(conn):
    """Detect which optional columns exist in the webhooks table.
    Returns a set of column names from {quality_score, sub_classification, hil_reason}."""
    global _extra_columns_cache
    if _extra_columns_cache is not None:
        return _extra_columns_cache
    cols = conn.execute("PRAGMA table_info(webhooks)").fetchall()
    col_names = {c[1] for c in cols}
    optional = {"quality_score", "sub_classification", "hil_reason"}
    _extra_columns_cache = optional & col_names
    return _extra_columns_cache


def _date_filter(start_date=None, end_date=None):
    """Build WHERE clause and params for date filtering."""
    clauses = []
    params = {}
    if start_date:
        clauses.append("received_at >= :start_date")
        params["start_date"] = start_date
    if end_date:
        clauses.append("received_at < :end_date")
        params["end_date"] = end_date
    where = " AND ".join(clauses)
    return f"WHERE {where}" if where else "", params


def _test_user_exclusion(conn, prefix="w"):
    """Return AND clauses to exclude test users and dry-run queries.
    Requires metadata table + JOIN for user filtering; dry-run filter always applies."""
    clauses = []
    params = {}

    # Exclude dry-run queries (always, no JOIN needed)
    clauses.append("forum_post_status IS NULL OR forum_post_status != 'skipped_dry_run'")

    # Exclude test users (only if metadata table exists)
    if _has_metadata_table(conn):
        for i, user in enumerate(_TEST_USERS):
            key = f"_excl_user_{i}"
            clauses.append(f"(m.posted_by IS NULL OR m.posted_by NOT LIKE :{key})")
            params[key] = f"%{user}%"

    return " AND ".join(f"({c})" for c in clauses), params


def _where_clause(date_where, extra_clause):
    """Combine date WHERE clause and extra exclusion clause into valid SQL.
    Handles the case where date_where is empty (needs WHERE instead of AND)."""
    if date_where and extra_clause:
        return f"{date_where} AND {extra_clause}"
    elif date_where:
        return date_where
    elif extra_clause:
        return f"WHERE {extra_clause}"
    return ""


def get_summary_stats(start_date=None, end_date=None):
    """Return top-level summary metrics."""
    date_where, params = _date_filter(start_date, end_date)

    with get_db() as conn:
        has_meta = _has_metadata_table(conn)
        excl_clause, excl_params = _test_user_exclusion(conn)
        params.update(excl_params)

        if has_meta:
            meta_join = "LEFT JOIN webhook_metadata m ON webhooks.correlation_id = m.correlation_id"
        else:
            meta_join = ""
        # Prefix date filter columns
        dw = date_where.replace("received_at", "webhooks.received_at") if date_where else ""
        combined_where = _where_clause(dw, excl_clause)

        sql = f"""
            SELECT
                COUNT(*) as total_queries,
                SUM(CASE WHEN status IN ('hil_exception', 'hil_concept') THEN 1 ELSE 0 END) as hil_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                SUM(CASE WHEN forum_post_status = 'posted' THEN 1 ELSE 0 END) as auto_posted,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                ROUND(AVG(CASE WHEN processing_time_ms > 0 THEN processing_time_ms END)) as avg_processing_ms,
                SUM(CASE WHEN images_transcribed > 0 THEN 1 ELSE 0 END) as with_images
            FROM webhooks
            {_dedup_join("webhooks")}
            {meta_join}
            {combined_where}
        """
        row = conn.execute(sql, params).fetchone()
        total = row["total_queries"] or 0
        hil = row["hil_count"] or 0
        errors = row["error_count"] or 0
        auto_posted = row["auto_posted"] or 0
        return {
            "total_queries": total,
            "hil_count": hil,
            "hil_pct": round(hil / total * 100, 1) if total > 0 else 0,
            "error_count": errors,
            "error_pct": round(errors / total * 100, 1) if total > 0 else 0,
            "auto_posted": auto_posted,
            "completed": row["completed"] or 0,
            "avg_processing_ms": row["avg_processing_ms"] or 0,
            "with_images": row["with_images"] or 0,
        }


def get_classification_breakdown(start_date=None, end_date=None):
    """Return query counts grouped by classification."""
    date_where, params = _date_filter(start_date, end_date)

    with get_db() as conn:
        has_meta = _has_metadata_table(conn)
        excl_clause, excl_params = _test_user_exclusion(conn)
        params.update(excl_params)

        if has_meta:
            meta_join = "LEFT JOIN webhook_metadata m ON webhooks.correlation_id = m.correlation_id"
        else:
            meta_join = ""
        dw = date_where.replace("received_at", "webhooks.received_at") if date_where else ""
        combined_where = _where_clause(dw, excl_clause)

        sql = f"""
            SELECT
                COALESCE(classification, 'Unknown') as classification,
                COUNT(*) as count
            FROM webhooks
            {_dedup_join("webhooks")}
            {meta_join}
            {combined_where}
            GROUP BY classification
            ORDER BY count DESC
        """
        rows = conn.execute(sql, params).fetchall()
        return [{"classification": r["classification"], "count": r["count"]} for r in rows]


def get_status_breakdown(start_date=None, end_date=None):
    """Return query counts grouped by processing status."""
    date_where, params = _date_filter(start_date, end_date)

    with get_db() as conn:
        has_meta = _has_metadata_table(conn)
        excl_clause, excl_params = _test_user_exclusion(conn)
        params.update(excl_params)

        if has_meta:
            meta_join = "LEFT JOIN webhook_metadata m ON webhooks.correlation_id = m.correlation_id"
        else:
            meta_join = ""
        dw = date_where.replace("received_at", "webhooks.received_at") if date_where else ""
        combined_where = _where_clause(dw, excl_clause)

        sql = f"""
            SELECT
                COALESCE(status, 'unknown') as status,
                COUNT(*) as count
            FROM webhooks
            {_dedup_join("webhooks")}
            {meta_join}
            {combined_where}
            GROUP BY status
            ORDER BY count DESC
        """
        rows = conn.execute(sql, params).fetchall()
        return [{"status": r["status"], "count": r["count"]} for r in rows]


def get_forum_post_status(start_date=None, end_date=None):
    """Return query counts grouped by forum post status."""
    date_where, params = _date_filter(start_date, end_date)

    with get_db() as conn:
        has_meta = _has_metadata_table(conn)
        excl_clause, excl_params = _test_user_exclusion(conn)
        params.update(excl_params)

        if has_meta:
            meta_join = "LEFT JOIN webhook_metadata m ON webhooks.correlation_id = m.correlation_id"
        else:
            meta_join = ""
        dw = date_where.replace("received_at", "webhooks.received_at") if date_where else ""
        combined_where = _where_clause(dw, excl_clause)

        sql = f"""
            SELECT
                COALESCE(forum_post_status, 'none') as forum_post_status,
                COUNT(*) as count
            FROM webhooks
            {_dedup_join("webhooks")}
            {meta_join}
            {combined_where}
            GROUP BY forum_post_status
            ORDER BY count DESC
        """
        rows = conn.execute(sql, params).fetchall()
        return [{"forum_post_status": r["forum_post_status"], "count": r["count"]} for r in rows]


def get_daily_volume(start_date=None, end_date=None, limit=30):
    """Return daily query counts for the bar chart."""
    date_where, params = _date_filter(start_date, end_date)

    with get_db() as conn:
        has_meta = _has_metadata_table(conn)
        excl_clause, excl_params = _test_user_exclusion(conn)
        params.update(excl_params)

        if has_meta:
            meta_join = "LEFT JOIN webhook_metadata m ON webhooks.correlation_id = m.correlation_id"
        else:
            meta_join = ""
        dw = date_where.replace("received_at", "webhooks.received_at") if date_where else ""
        combined_where = _where_clause(dw, excl_clause)

        sql = f"""
            SELECT
                DATE(webhooks.received_at) as date,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status IN ('hil_exception', 'hil_concept') THEN 1 ELSE 0 END) as hil,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM webhooks
            {_dedup_join("webhooks")}
            {meta_join}
            {combined_where}
            GROUP BY DATE(webhooks.received_at)
            ORDER BY date DESC
            LIMIT :limit
        """
        params["limit"] = limit
        rows = conn.execute(sql, params).fetchall()
        result = [
            {
                "date": r["date"],
                "total": r["total"],
                "completed": r["completed"],
                "hil": r["hil"],
                "errors": r["errors"],
            }
            for r in rows
        ]
        result.reverse()  # chronological order
        return result


def get_recent_queries(start_date=None, end_date=None, limit=20):
    """Return the most recent queries for the table view.
    LEFT JOINs webhook_metadata if the table exists, otherwise returns nulls."""
    date_where, params = _date_filter(start_date, end_date)

    with get_db() as conn:
        has_meta = _has_metadata_table(conn)

        if has_meta:
            meta_cols = ", m.discussion_id, m.entity_name, m.entity_id, m.platform_name, m.posted_by, m.forum_post_subject"
            meta_join = "LEFT JOIN webhook_metadata m ON w.correlation_id = m.correlation_id"
        else:
            meta_cols = ", NULL as discussion_id, NULL as entity_name, NULL as entity_id, NULL as platform_name, NULL as posted_by, NULL as forum_post_subject"
            meta_join = ""

        # Prefix webhooks columns with w. when joining
        date_where_prefixed = date_where.replace("received_at", "w.received_at") if date_where else ""

        excl_clause, excl_params = _test_user_exclusion(conn)
        params.update(excl_params)
        combined_where = _where_clause(date_where_prefixed, excl_clause)

        sql = f"""
            SELECT
                w.id,
                w.correlation_id,
                w.received_at,
                w.processed_at,
                w.status,
                w.classification,
                w.forum_post_status,
                w.processing_time_ms,
                w.images_transcribed,
                w.error_message
                {meta_cols}
            FROM webhooks w
            {_dedup_join()}
            {meta_join}
            {combined_where}
            ORDER BY w.received_at DESC
            LIMIT :limit
        """
        params["limit"] = limit
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": r["id"],
                "correlation_id": r["correlation_id"],
                "received_at": r["received_at"],
                "processed_at": r["processed_at"],
                "status": r["status"],
                "classification": r["classification"],
                "forum_post_status": r["forum_post_status"],
                "processing_time_ms": r["processing_time_ms"],
                "images_transcribed": r["images_transcribed"],
                "error_message": r["error_message"],
                "discussion_id": r["discussion_id"],
                "entity_name": r["entity_name"],
                "entity_id": r["entity_id"],
                "platform_name": r["platform_name"],
                "posted_by": r["posted_by"],
                "forum_post_subject": r["forum_post_subject"],
            }
            for r in rows
        ]



def _build_analysis(detail):
    """Generate a human-readable analysis paragraph for why a query was not posted."""
    import json as _json

    status = detail.get("status") or ""
    fps = detail.get("forum_post_status") or ""
    err = detail.get("error_message") or ""
    quality_score = detail.get("quality_score")
    sub_classification = detail.get("sub_classification")
    hil_reason = detail.get("hil_reason")

    parts = []

    # ── Main explanation ──
    if status == "error":
        parts.append(f"Processing failed with an error before the forum posting stage. Error: {err}" if err else "Processing failed with an error before the forum posting stage.")

    elif status in ("hil_exception", "hil_concept"):
        base = "This query was escalated to Human-in-the-Loop (HIL) review."
        if hil_reason:
            base += f" Reason: {hil_reason}"
        else:
            base += " The system determined that automated processing was not appropriate for this query."
        parts.append(base)

    elif fps == "skipped_quality_hil":
        if quality_score is not None:
            parts.append(f"Quality score: {quality_score}/100 (threshold: 85). The response was routed to HIL because quality validation failed.")
        else:
            parts.append("The generated response did not meet the quality threshold (score below 85). The query has been escalated to HIL for manual response.")

    elif fps == "skipped_validation":
        parts.append("The query classification was not eligible for automated forum posting. Only specific classification types are allowed to be auto-posted.")

    elif fps == "skipped_dry_run":
        parts.append("Dry-run mode was active when this query was processed. The response was generated but not posted to the forum.")

    elif fps == "skipped_hil":
        base = "The HIL (Human-in-the-Loop) flag was set during processing, so the response was not auto-posted to the forum."
        if hil_reason:
            base += f" Reason: {hil_reason}"
        parts.append(base)

    elif fps == "failed":
        forum_err = detail.get("forum_post_error") or ""
        parts.append(f"The forum API call failed when attempting to post the response. Error: {forum_err}" if forum_err else "The forum API call failed when attempting to post the response.")

    elif fps == "skipped":
        parts.append("This query was skipped and not posted to the forum.")

    else:
        parts.append("This query was not posted to the forum.")

    # ── Quality score line (if not already mentioned) ──
    if quality_score is not None and fps != "skipped_quality_hil":
        parts.append(f"Quality score: {quality_score}/100.")

    # ── Sub-classification details ──
    if sub_classification:
        try:
            sc = _json.loads(sub_classification) if isinstance(sub_classification, str) else sub_classification
            classification = detail.get("classification") or ""

            if classification == "Pointing_Out_Corrections" and sc.get("validation_classification"):
                vc = sc["validation_classification"]
                line = f"Student correction evaluated as {vc}."
                if sc.get("error_type"):
                    line += f" Error type: {sc['error_type']}."
                if sc.get("confidence_level"):
                    line += f" Confidence: {sc['confidence_level']}."
                parts.append(line)

            elif classification == "Variation_of_Question":
                items = []
                if sc.get("interaction_type"):
                    items.append(f"Interaction: {sc['interaction_type']}")
                if sc.get("exception_type"):
                    items.append(f"Exception: {sc['exception_type']}")
                if sc.get("followup_type"):
                    items.append(f"Follow-up: {sc['followup_type']}")
                if items:
                    parts.append("Sub-classification: " + ", ".join(items) + ".")

            elif classification == "Alternate_Approach":
                items = []
                if sc.get("approach_status"):
                    items.append(f"Approach: {sc['approach_status']}")
                if sc.get("mistake_type"):
                    items.append(f"Mistake: {sc['mistake_type']}")
                if sc.get("understanding_status"):
                    items.append(f"Understanding: {sc['understanding_status']}")
                if items:
                    parts.append("Sub-classification: " + ", ".join(items) + ".")

            elif classification == "Genuine_Doubt" and sc.get("exception_flag"):
                parts.append(f"Exception flag: {sc['exception_flag']}.")

        except Exception:
            pass

    return " ".join(parts)


def _build_workflow(detail):
    """Generate a step-by-step workflow timeline for a query."""
    steps = []
    status = detail.get("status") or ""
    fps = detail.get("forum_post_status") or ""
    classification = detail.get("classification")
    images = detail.get("images_transcribed") or 0

    # Step 1: Webhook Received (always completed if row exists)
    steps.append({
        "step": 1,
        "label": "Webhook Received",
        "status": "completed",
        "detail": detail.get("received_at") or ""
    })

    # Step 2: Image Transcription (if applicable)
    if images > 0:
        steps.append({
            "step": 2,
            "label": "Image Transcription",
            "status": "completed",
            "detail": f"{images} image(s) transcribed"
        })

    # Step 3: Classification
    if status == "error" and not classification:
        steps.append({
            "step": 3,
            "label": "Classification",
            "status": "failed",
            "detail": detail.get("error_message") or "Classification failed"
        })
        return steps

    if classification:
        steps.append({
            "step": 3,
            "label": "Classification",
            "status": "completed",
            "detail": classification
        })
    elif status != "error":
        steps.append({
            "step": 3,
            "label": "Classification",
            "status": "completed",
            "detail": "Completed"
        })

    # Step 4: Response Generation
    if status == "error":
        steps.append({
            "step": 4,
            "label": "Response Generation",
            "status": "failed",
            "detail": detail.get("error_message") or "Processing error"
        })
        return steps

    if status in ("hil_exception", "hil_concept"):
        steps.append({
            "step": 4,
            "label": "Response Generation",
            "status": "hil",
            "detail": "Escalated to HIL"
        })
        return steps

    steps.append({
        "step": 4,
        "label": "Response Generation",
        "status": "completed",
        "detail": "Response generated"
    })

    # Step 5: Quality Check
    quality_score = detail.get("quality_score")
    if fps == "skipped_quality_hil":
        q_detail = f"Score: {quality_score}/100 (threshold: 85)" if quality_score is not None else "Score below threshold (85)"
        steps.append({
            "step": 5,
            "label": "Quality Check",
            "status": "failed",
            "detail": q_detail
        })
        return steps

    if fps in ("posted", "skipped_dry_run"):
        q_detail = f"Score: {quality_score}/100 — Passed" if quality_score is not None else "Passed"
        steps.append({
            "step": 5,
            "label": "Quality Check",
            "status": "completed",
            "detail": q_detail
        })
    elif fps in ("skipped_validation", "skipped_hil", "skipped"):
        steps.append({
            "step": 5,
            "label": "Quality Check",
            "status": "skipped",
            "detail": "Skipped (validation/HIL)"
        })
    else:
        steps.append({
            "step": 5,
            "label": "Quality Check",
            "status": "completed",
            "detail": "Completed"
        })

    # Step 6: Forum Posting
    if fps == "posted":
        steps.append({
            "step": 6,
            "label": "Forum Posting",
            "status": "completed",
            "detail": "Posted successfully"
        })
    elif fps == "failed":
        steps.append({
            "step": 6,
            "label": "Forum Posting",
            "status": "failed",
            "detail": detail.get("forum_post_error") or "Posting failed"
        })
    elif fps == "skipped_dry_run":
        steps.append({
            "step": 6,
            "label": "Forum Posting",
            "status": "skipped",
            "detail": "Dry-run mode active"
        })
    elif fps == "skipped_validation":
        steps.append({
            "step": 6,
            "label": "Forum Posting",
            "status": "skipped",
            "detail": "Classification not eligible"
        })
    elif fps == "skipped_hil":
        steps.append({
            "step": 6,
            "label": "Forum Posting",
            "status": "skipped",
            "detail": "HIL flag set"
        })
    else:
        steps.append({
            "step": 6,
            "label": "Forum Posting",
            "status": "skipped",
            "detail": fps or "Not posted"
        })

    return steps



def get_filtered_queries(filter_type, start_date=None, end_date=None, limit=100):
    """Return queries filtered by type (hil, error, completed) with metadata.
    Used for the drill-down list when clicking summary cards."""
    date_where, params = _date_filter(start_date, end_date)

    # Map filter type to SQL conditions
    type_conditions = {
        "hil": "w.status IN ('hil_exception', 'hil_concept')",
        "error": "w.status = 'error'",
        "completed": "w.status = 'completed'",
    }
    condition = type_conditions.get(filter_type)
    if not condition:
        return []

    with get_db() as conn:
        has_meta = _has_metadata_table(conn)
        excl_clause, excl_params = _test_user_exclusion(conn)
        params.update(excl_params)

        if has_meta:
            meta_cols = ", m.discussion_id, m.entity_name, m.entity_id, m.platform_name, m.posted_by, m.forum_post_subject"
            meta_join = "LEFT JOIN webhook_metadata m ON w.correlation_id = m.correlation_id"
        else:
            meta_cols = ", NULL as discussion_id, NULL as entity_name, NULL as entity_id, NULL as platform_name, NULL as posted_by, NULL as forum_post_subject"
            meta_join = ""

        date_where_prefixed = date_where.replace("received_at", "w.received_at") if date_where else ""

        # Build WHERE: combine date filter + type condition + exclusion
        if date_where_prefixed:
            where_clause = f"{date_where_prefixed} AND {condition}"
        else:
            where_clause = f"WHERE {condition}"
        if excl_clause:
            where_clause += f" AND {excl_clause}"

        extra_cols = _get_extra_columns(conn)
        extra_select = ""
        if "quality_score" in extra_cols:
            extra_select += ",\n                w.quality_score"
        if "sub_classification" in extra_cols:
            extra_select += ",\n                w.sub_classification"
        if "hil_reason" in extra_cols:
            extra_select += ",\n                w.hil_reason"

        sql = f"""
            SELECT
                w.id,
                w.correlation_id,
                w.received_at,
                w.processed_at,
                w.status,
                w.classification,
                w.forum_post_status,
                w.processing_time_ms,
                w.images_transcribed,
                w.error_message
                {extra_select}
                {meta_cols}
            FROM webhooks w
            {_dedup_join()}
            {meta_join}
            {where_clause}
            ORDER BY w.received_at DESC
            LIMIT :limit
        """
        params["limit"] = limit
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": r["id"],
                "correlation_id": r["correlation_id"],
                "received_at": r["received_at"],
                "processed_at": r["processed_at"],
                "status": r["status"],
                "classification": r["classification"],
                "forum_post_status": r["forum_post_status"],
                "processing_time_ms": r["processing_time_ms"],
                "images_transcribed": r["images_transcribed"],
                "error_message": r["error_message"],
                "quality_score": r["quality_score"] if "quality_score" in extra_cols else None,
                "sub_classification": r["sub_classification"] if "sub_classification" in extra_cols else None,
                "hil_reason": r["hil_reason"] if "hil_reason" in extra_cols else None,
                "discussion_id": r["discussion_id"],
                "entity_name": r["entity_name"],
                "entity_id": r["entity_id"],
                "platform_name": r["platform_name"],
                "posted_by": r["posted_by"],
                "forum_post_subject": r["forum_post_subject"],
            }
            for r in rows
        ]


def get_query_detail(correlation_id):
    """Return full detail for a single query by correlation_id.
    Joins webhooks + webhook_metadata. Returns None if not found."""
    with get_db() as conn:
        has_meta = _has_metadata_table(conn)

        if has_meta:
            meta_cols = ", m.discussion_id, m.entity_name, m.entity_id, m.platform_name, m.posted_by, m.forum_post_subject"
            meta_join = "LEFT JOIN webhook_metadata m ON w.correlation_id = m.correlation_id"
        else:
            meta_cols = ", NULL as discussion_id, NULL as entity_name, NULL as entity_id, NULL as platform_name, NULL as posted_by, NULL as forum_post_subject"
            meta_join = ""

        extra_cols = _get_extra_columns(conn)
        extra_select = ""
        if "quality_score" in extra_cols:
            extra_select += ",\n                w.quality_score"
        if "sub_classification" in extra_cols:
            extra_select += ",\n                w.sub_classification"
        if "hil_reason" in extra_cols:
            extra_select += ",\n                w.hil_reason"

        sql = f"""
            SELECT
                w.id,
                w.correlation_id,
                w.received_at,
                w.processed_at,
                w.status,
                w.classification,
                w.forum_post_status,
                w.processing_time_ms,
                w.images_transcribed,
                w.error_message,
                w.forum_post_error
                {extra_select}
                {meta_cols}
            FROM webhooks w
            {_dedup_join()}
            {meta_join}
            WHERE w.correlation_id = ?
        """
        row = conn.execute(sql, (correlation_id,)).fetchone()
        if not row:
            return None

        detail = {
            "id": row["id"],
            "correlation_id": row["correlation_id"],
            "received_at": row["received_at"],
            "processed_at": row["processed_at"],
            "status": row["status"],
            "classification": row["classification"],
            "forum_post_status": row["forum_post_status"],
            "processing_time_ms": row["processing_time_ms"],
            "images_transcribed": row["images_transcribed"],
            "error_message": row["error_message"],
            "forum_post_error": row["forum_post_error"],
            "quality_score": row["quality_score"] if "quality_score" in extra_cols else None,
            "sub_classification": row["sub_classification"] if "sub_classification" in extra_cols else None,
            "hil_reason": row["hil_reason"] if "hil_reason" in extra_cols else None,
            "discussion_id": row["discussion_id"],
            "entity_name": row["entity_name"],
            "entity_id": row["entity_id"],
            "platform_name": row["platform_name"],
            "posted_by": row["posted_by"],
            "forum_post_subject": row["forum_post_subject"],
        }

        detail["analysis"] = _build_analysis(detail)
        detail["workflow"] = _build_workflow(detail)

        return detail




def get_query_logs(correlation_id):
    """Fetch and parse journalctl logs for a correlation ID.
    Returns structured events, markdown summary, and Mermaid flowchart."""
    import subprocess
    import re

    # Fetch logs from journalctl
    try:
        result = subprocess.run(
            ["journalctl", "-u", "sat-forum-responder", "--no-pager", "--output=short-precise"],
            capture_output=True, text=True, timeout=15
        )
        all_lines = result.stdout.splitlines()
    except Exception as e:
        return {"error": f"Failed to read logs: {e}", "events": [], "markdown": "", "mermaid": ""}

    # Find lines matching correlation ID, PLUS context lines that follow
    # (Classification:, Validation:, COST:, etc. are logged without the correlation ID)
    matched_indices = set()
    for i, line in enumerate(all_lines):
        if correlation_id in line:
            matched_indices.add(i)
            _context_keywords = [
                'Classification:', 'Primary Intent:', 'Key Indicators:',
                'Reasoning:', 'Decision Path:', 'Exception Flag:',
                'Validation Classification:', 'Validation Explanation:',
                'TOKENS:', 'COST:', 'JSON parse error',
                'Claude API call successful', 'Pointing_Out_Corrections',
                'Variation_of_Question', 'Alternate_Approach', 'Genuine_Doubt',
            ]
            # Look FORWARD up to 8 lines for context
            for j in range(1, 9):
                if i + j < len(all_lines):
                    next_line = all_lines[i + j]
                    if re.search(r'\[\w{8}-\w{4}-\w{4}-\w{4}-\w{12}\]', next_line) and correlation_id not in next_line:
                        break
                    if any(kw in next_line for kw in _context_keywords):
                        matched_indices.add(i + j)
            # Look BACKWARD up to 5 lines for context (COST, JSON errors logged before tool response)
            for j in range(1, 6):
                if i - j >= 0:
                    prev_line = all_lines[i - j]
                    if correlation_id in prev_line:
                        break  # stop at previous correlation ID line
                    if any(kw in prev_line for kw in _context_keywords):
                        matched_indices.add(i - j)

    if not matched_indices:
        return {"error": "No logs found for this correlation ID", "events": [], "markdown": "", "mermaid": ""}

    lines = [all_lines[i] for i in sorted(matched_indices)]

    # Parse log lines into structured events
    events = []
    for line in lines:
        # Extract timestamp and message
        # Format: "Feb 08 13:21:24 srv... python[...]: 2026-02-08 13:21:24,708 - module - LEVEL - message"
        ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)', line)
        level_match = re.search(r' - (INFO|WARNING|ERROR|DEBUG) - ', line)
        ts = ts_match.group(1) if ts_match else ""
        level = level_match.group(1) if level_match else "INFO"

        # Extract the message after the level
        msg = ""
        if level_match:
            msg = line[level_match.end():]
        else:
            # Fallback: take everything after correlation ID
            idx = line.find(correlation_id)
            if idx >= 0:
                msg = line[idx + len(correlation_id) + 2:]

        # Clean up: remove correlation ID prefix like [xxxx-xxx]
        msg = re.sub(r'^\[[\w-]+\]\s*', '', msg.strip())

        events.append({"timestamp": ts, "level": level, "message": msg})

    # ── Detect key milestones from log messages ──
    milestones = []
    student_text = ""
    a1_result = ""
    og_result = ""
    og_reasoning = ""
    a2_classification = ""
    a2_reasoning = ""
    a2_decision_path = ""
    tool_name = ""
    tool_result_lines = []
    json_error = ""
    quality_score = None
    quality_reason = ""
    airtable_saved = False
    forum_status = ""
    teams_sent = False
    final_status = ""
    final_forum = ""
    final_time = ""
    validation_classification = ""
    images_count = 0
    api_costs = []

    for ev in events:
        m = ev["message"]
        lv = ev["level"]

        if "Received correlation ID" in m:
            milestones.append(("received", ev["timestamp"], "Webhook Received"))
        elif "Successfully fetched forum data" in m:
            milestones.append(("fetched", ev["timestamp"], "Forum data fetched from API"))
        elif "forumPostText:" in m:
            student_text = m.split("forumPostText:", 1)[-1].strip().strip("'\"")[:200]
        elif "base64EncodedImages count:" in m:
            try:
                images_count = int(m.split("count:")[-1].strip())
            except:
                pass
        elif "a1_triage FULL RESPONSE" in m:
            milestones.append(("a1", ev["timestamp"], "A1 Triage completed"))
        elif "A1 classified as" in m:
            a1_result = m.split("A1 classified as")[-1].split("-")[0].strip()
        elif "OG Detection result:" in m:
            og_result = m.split("result:")[-1].strip()
            milestones.append(("og", ev["timestamp"], f"OG Detection: {og_result}"))
        elif "OG Detection reasoning:" in m:
            og_reasoning = m.split("reasoning:")[-1].strip()[:300]
        elif "Not an OG question" in m:
            pass  # already captured
        elif "OG flow resulted in HIL" in m:
            milestones.append(("og_hil", ev["timestamp"], "OG question detected — routed to HIL"))
        elif "a2_deep_sm FULL RESPONSE" in m or "a2_deep" in m.lower() and "FULL RESPONSE" in m:
            milestones.append(("a2", ev["timestamp"], "A2 Deep Classification completed"))
        elif "a2_deep_sm Classification:" in m:
            a2_classification = m.split("Classification:")[-1].strip()
        elif m.startswith("Classification:") and not a2_classification:
            a2_classification = m.split("Classification:")[-1].strip()
        elif m.startswith("Reasoning:"):
            a2_reasoning = m.split("Reasoning:")[-1].strip()[:400]
        elif m.startswith("Decision Path:"):
            a2_decision_path = m.split("Decision Path:")[-1].strip()[:300]
        elif m.startswith("Primary Intent:"):
            pass  # captured in a2 section
        elif re.match(r'Running tool_\d', m):
            tool_name = re.search(r'tool_\d+', m).group()
            milestones.append(("tool_start", ev["timestamp"], f"Running {tool_name}"))
        elif "FULL RESPONSE" in m and "tool_" in m:
            tn = re.search(r'tool_\d+', m)
            if tn:
                tool_name = tn.group()
            milestones.append(("tool_done", ev["timestamp"], f"{tool_name} completed"))
        elif "Validation Classification:" in m:
            validation_classification = m.split("Validation Classification:")[-1].strip()
        elif "Exception Flag:" in m:
            tool_result_lines.append(m)
        elif "JSON parse error" in m:
            json_error = m.strip()
            milestones.append(("json_error", ev["timestamp"], f"JSON Parse Error: {json_error}"))
        elif "Quality score" in m and "/100" in m:
            qs_match = re.search(r'(\d+)/100', m)
            if qs_match:
                quality_score = int(qs_match.group(1))
            if "below threshold" in m:
                quality_reason = "Below threshold (85+) — routing to HIL"
                milestones.append(("quality_fail", ev["timestamp"], f"Quality: {quality_score}/100 — FAILED"))
            else:
                quality_reason = "Passed"
                milestones.append(("quality_pass", ev["timestamp"], f"Quality: {quality_score}/100 — Passed"))
        elif "Saved to Airtable" in m:
            airtable_saved = True
            table = m.split("(")[-1].split(")")[0] if "(" in m else "unknown"
            milestones.append(("airtable", ev["timestamp"], f"Saved to Airtable ({table})"))
        elif "Skipping forum post" in m:
            forum_status = "skipped"
            milestones.append(("forum_skip", ev["timestamp"], m[:120]))
        elif "Posted to forum" in m or "Forum post successful" in m:
            forum_status = "posted"
            milestones.append(("forum_post", ev["timestamp"], "Posted to forum"))
        elif "Teams notification sent" in m:
            teams_sent = True
            milestones.append(("teams", ev["timestamp"], "Teams notification sent"))
        elif "Completed processing for" in m:
            # Parse: "Completed processing for xxx in 12345ms - Status: completed, Forum: posted, Images: 0"
            status_match = re.search(r'Status:\s*([\w_]+)', m)
            forum_match = re.search(r'Forum:\s*([\w_]+)', m)
            time_match = re.search(r'in (\d+)ms', m)
            if status_match:
                final_status = status_match.group(1)
            if forum_match:
                final_forum = forum_match.group(1)
            if time_match:
                final_time = time_match.group(1) + "ms"
            milestones.append(("done", ev["timestamp"], f"Completed — {final_status} / {final_forum} / {final_time}"))
        elif "Processing failed" in m or "Error processing" in m:
            milestones.append(("error", ev["timestamp"], m[:200]))
        elif "HIL required:" in m:
            milestones.append(("hil", ev["timestamp"], m[:200]))
        elif "COST:" in m:
            cost_match = re.search(r'total=\$(\d+\.\d+)', m)
            if cost_match:
                api_costs.append(float(cost_match.group(1)))
        elif "Pointing_Out_Corrections with" in m:
            # Extract validation result: "Pointing_Out_Corrections with INVALID classification - will post"
            if "INVALID" in m:
                validation_classification = "INVALID"
            elif "VALID" in m:
                validation_classification = "VALID"
            milestones.append(("validation", ev["timestamp"], m.strip()[:150]))

    # ── Build Markdown summary ──
    md_parts = []
    md_parts.append(f"# Processing Log: {correlation_id[:12]}...")
    md_parts.append("")

    if student_text:
        md_parts.append("## Student Post")
        md_parts.append(f"> {student_text}...")
        md_parts.append("")

    md_parts.append("## Processing Pipeline")
    md_parts.append("")

    if a1_result:
        md_parts.append(f"### 1. A1 Triage")
        md_parts.append(f"- **Result**: {a1_result}")
        md_parts.append("")

    if og_result:
        md_parts.append(f"### 2. OG Detection")
        md_parts.append(f"- **Result**: {og_result}")
        if og_reasoning:
            md_parts.append(f"- **Reasoning**: {og_reasoning}")
        md_parts.append("")

    if a2_classification:
        md_parts.append(f"### 3. A2 Deep Classification")
        md_parts.append(f"- **Classification**: {a2_classification}")
        if a2_reasoning:
            md_parts.append(f"- **Reasoning**: {a2_reasoning}")
        if a2_decision_path:
            md_parts.append(f"- **Decision Path**: {a2_decision_path}")
        md_parts.append("")

    if tool_name:
        md_parts.append(f"### 4. Tool Execution: {tool_name}")
        if validation_classification:
            md_parts.append(f"- **Validation**: {validation_classification}")
        if json_error:
            md_parts.append(f"- **ERROR**: {json_error}")
        md_parts.append("")

    if quality_score is not None:
        md_parts.append(f"### 5. Quality Check")
        md_parts.append(f"- **Score**: {quality_score}/100 (threshold: 85)")
        md_parts.append(f"- **Result**: {quality_reason}")
        md_parts.append("")

    if airtable_saved:
        md_parts.append(f"### 6. Airtable")
        md_parts.append(f"- Saved successfully")
        md_parts.append("")

    md_parts.append(f"### Final Result")
    md_parts.append(f"- **Status**: {final_status}")
    md_parts.append(f"- **Forum**: {final_forum}")
    md_parts.append(f"- **Processing Time**: {final_time}")
    if api_costs:
        md_parts.append(f"- **API Cost**: ${sum(api_costs):.4f}")
    md_parts.append("")

    if json_error:
        md_parts.append("## Issues Detected")
        md_parts.append(f"- {json_error}")
        md_parts.append("")

    markdown = "\n".join(md_parts)

    # ── Build Mermaid flowchart ──
    mermaid_lines = ["graph TD"]

    # Define nodes
    mermaid_lines.append('    A["Webhook Received"] --> B["A1 Triage"]')

    if a1_result:
        mermaid_lines.append(f'    B -->|"{a1_result}"| C["OG Detection"]')
    else:
        mermaid_lines.append('    B --> C["OG Detection"]')

    if og_result and "is_og=True" in og_result:
        mermaid_lines.append('    C -->|"OG Question"| HIL_OG["HIL: OG Question"]')
        mermaid_lines.append('    style HIL_OG fill:#ea580c,color:#fff')
    else:
        mermaid_lines.append('    C -->|"Not OG"| D["A2 Classification"]')

        if a2_classification:
            safe_class = a2_classification.replace('"', "#quot;")
            tool_label = tool_name.replace("_", " ").title() if tool_name else "Tool"
            mermaid_lines.append(f'    D -->|"{safe_class}"| E["{tool_label}"]')
        else:
            mermaid_lines.append('    D --> E["Tool Execution"]')

        if json_error:
            mermaid_lines.append('    E -->|"JSON Error"| E_ERR["Parse Error"]')
            mermaid_lines.append('    E_ERR --> F["Quality Check"]')
            mermaid_lines.append('    style E_ERR fill:#dc2626,color:#fff')
        else:
            mermaid_lines.append('    E --> F["Quality Check"]')

        if quality_score is not None:
            if quality_score >= 85:
                mermaid_lines.append(f'    F -->|"{quality_score}/100 ✓"| G["Forum Post"]')
                if final_forum == "posted":
                    mermaid_lines.append('    G --> H["Posted ✓"]')
                    mermaid_lines.append('    style H fill:#059669,color:#fff')
                elif "skipped_validation" in (final_forum or ""):
                    mermaid_lines.append('    G --> H["Skipped: Validation"]')
                    mermaid_lines.append('    style H fill:#d97706,color:#fff')
                else:
                    safe_forum = (final_forum or "skipped").replace('"', "'")
                    mermaid_lines.append(f'    G --> H["{safe_forum}"]')
                    mermaid_lines.append('    style H fill:#d97706,color:#fff')
                mermaid_lines.append('    style F fill:#059669,color:#fff')
            else:
                mermaid_lines.append(f'    F -->|"{quality_score}/100 ✗"| HIL_Q["HIL: Quality Failed"]')
                mermaid_lines.append('    style F fill:#dc2626,color:#fff')
                mermaid_lines.append('    style HIL_Q fill:#ea580c,color:#fff')
        else:
            # No quality score — check forum status
            if final_forum == "posted":
                mermaid_lines.append('    F --> G["Posted ✓"]')
                mermaid_lines.append('    style G fill:#059669,color:#fff')
            elif "skipped" in (final_forum or ""):
                safe_forum = (final_forum or "skipped").replace('"', "'")
                mermaid_lines.append(f'    F --> G["{safe_forum}"]')
                mermaid_lines.append('    style G fill:#d97706,color:#fff')
            else:
                mermaid_lines.append('    F --> G["Done"]')

    # Style completed nodes
    mermaid_lines.append('    style A fill:#0891b2,color:#fff')
    mermaid_lines.append('    style B fill:#0891b2,color:#fff')
    mermaid_lines.append('    style C fill:#0891b2,color:#fff')
    mermaid_lines.append('    style D fill:#0891b2,color:#fff')
    if tool_name:
        mermaid_lines.append('    style E fill:#0891b2,color:#fff')

    mermaid = "\n".join(mermaid_lines)

    return {
        "correlation_id": correlation_id,
        "events": [{"timestamp": ev["timestamp"], "level": ev["level"], "message": ev["message"][:300]} for ev in events],
        "milestones": milestones,
        "markdown": markdown,
        "mermaid": mermaid,
        "summary": {
            "student_text": student_text,
            "a1_result": a1_result,
            "og_result": og_result,
            "og_reasoning": og_reasoning,
            "a2_classification": a2_classification,
            "a2_reasoning": a2_reasoning,
            "tool_name": tool_name,
            "validation_classification": validation_classification,
            "json_error": json_error,
            "quality_score": quality_score,
            "quality_reason": quality_reason,
            "final_status": final_status,
            "final_forum": final_forum,
            "final_time": final_time,
            "api_cost": round(sum(api_costs), 4) if api_costs else None,
            "images": images_count,
        },
    }



def get_all_vote_events():
    """Fetch all vote events filtered to Neuron expert forum posts.
    Uses Forum_Post_Expert event (FP_Platform=NEURON) to get valid post IDs,
    then filters vote events and deduplicates by (email, post_id, event, action).
    Returns list of {ts, type, action} dicts for client-side date filtering.
    Uses 5-min server-side cache."""
    if not _MIXPANEL_ENABLED:
        return {"available": False, "events": [], "error": "Mixpanel not configured"}

    if _vote_cache["data"] is not None and (_time.time() - _vote_cache["ts"]) < _VOTE_CACHE_TTL:
        return _vote_cache["data"]

    try:
        # Use PST (Mixpanel project timezone) to avoid to_date being in the future
        from datetime import timezone as _tz
        _pst = _tz(timedelta(hours=-8))
        today = datetime.now(_pst)
        today_str = today.strftime("%Y-%m-%d")
        api_from = (today - timedelta(days=15)).strftime("%Y-%m-%d")

        # Step 1: Get expert post IDs from Forum_Post_Expert events (NEURON only)
        expert_resp = requests.get(
            "https://data.mixpanel.com/api/2.0/export",
            params={
                "project_id": _MIXPANEL_CONFIG["project_id"],
                "from_date": api_from,
                "to_date": today_str,
                "event": json.dumps(["Forum_Post_Expert"]),
                "where": 'properties["FP_Platform"] == "SAT"',
            },
            auth=(_MIXPANEL_CONFIG["username"], _MIXPANEL_CONFIG["secret"]),
            timeout=60,
        )
        expert_resp.raise_for_status()

        expert_post_ids = set()
        raw_expert = expert_resp.text.strip()
        if raw_expert:
            for line in raw_expert.split("\n"):
                if not line.strip():
                    continue
                evt = json.loads(line)
                post_id = evt.get("properties", {}).get("FP_Forum_Post_id")
                if post_id is not None:
                    expert_post_ids.add(str(post_id))

        # Step 2: Fetch raw vote events from export API (last 15 days)
        resp = requests.get(
            "https://data.mixpanel.com/api/2.0/export",
            params={
                "project_id": _MIXPANEL_CONFIG["project_id"],
                "from_date": api_from,
                "to_date": today_str,
                "event": json.dumps(["Upvoted_Forum_Post", "Downvoted_Forum_Post"]),
            },
            auth=(_MIXPANEL_CONFIG["username"], _MIXPANEL_CONFIG["secret"]),
            timeout=90,
        )
        resp.raise_for_status()

        all_events = []
        seen_votes = set()
        raw_text = resp.text.strip()
        if raw_text:
            for line in raw_text.split("\n"):
                if not line.strip():
                    continue
                event = json.loads(line)
                props = event.get("properties", {})
                post_id = str(props.get("FP_Forum_Post_id", ""))

                # Only count votes on posts created by Neuron responder
                if post_id not in expert_post_ids:
                    continue

                event_name = event.get("event", "")
                action = props.get("FP_Vote_Action", "")
                ev_type = "up" if event_name == "Upvoted_Forum_Post" else "down"
                ev_ts = props.get("time", 0)

                # Deduplicate: export API returns multiple rows per vote
                dedup_key = (post_id, ev_ts, ev_type, action)
                if dedup_key in seen_votes:
                    continue
                seen_votes.add(dedup_key)

                all_events.append({
                    "ts": ev_ts,
                    "type": ev_type,
                    "action": action,
                })

        result = {"available": True, "events": all_events, "expert_posts": len(expert_post_ids)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"available": False, "events": [], "error": str(e)}

    _vote_cache["data"] = result
    _vote_cache["ts"] = _time.time()
    return result



def get_all_raw_queries():
    """Return ALL webhook rows with metadata, excluding test users and dry-runs.
    No date filtering, no Mixpanel. Used for client-side aggregation."""
    with get_db() as conn:
        has_meta = _has_metadata_table(conn)
        excl_clause, excl_params = _test_user_exclusion(conn)

        if has_meta:
            meta_cols = ", m.discussion_id, m.entity_name, m.entity_id, m.platform_name, m.posted_by, m.forum_post_subject"
            meta_join = "LEFT JOIN webhook_metadata m ON w.correlation_id = m.correlation_id"
        else:
            meta_cols = ", NULL as discussion_id, NULL as entity_name, NULL as entity_id, NULL as platform_name, NULL as posted_by, NULL as forum_post_subject"
            meta_join = ""

        where = f"WHERE {excl_clause}" if excl_clause else ""

        extra_cols = _get_extra_columns(conn)
        extra_select = ""
        if "quality_score" in extra_cols:
            extra_select += ",\n                w.quality_score"
        if "sub_classification" in extra_cols:
            extra_select += ",\n                w.sub_classification"
        if "hil_reason" in extra_cols:
            extra_select += ",\n                w.hil_reason"

        sql = f"""
            SELECT
                w.id,
                w.correlation_id,
                w.received_at,
                w.processed_at,
                w.status,
                w.classification,
                w.forum_post_status,
                w.processing_time_ms,
                w.images_transcribed,
                w.error_message
                {extra_select}
                {meta_cols}
            FROM webhooks w
            {_dedup_join()}
            {meta_join}
            {where}
            ORDER BY w.received_at DESC
        """
        rows = conn.execute(sql, excl_params).fetchall()
        return [
            {
                "id": r["id"],
                "correlation_id": r["correlation_id"],
                "received_at": r["received_at"],
                "processed_at": r["processed_at"],
                "status": r["status"],
                "classification": r["classification"],
                "forum_post_status": r["forum_post_status"],
                "processing_time_ms": r["processing_time_ms"],
                "images_transcribed": r["images_transcribed"],
                "error_message": r["error_message"],
                "quality_score": r["quality_score"] if "quality_score" in extra_cols else None,
                "sub_classification": r["sub_classification"] if "sub_classification" in extra_cols else None,
                "hil_reason": r["hil_reason"] if "hil_reason" in extra_cols else None,
                "discussion_id": r["discussion_id"],
                "entity_name": r["entity_name"],
                "entity_id": r["entity_id"],
                "platform_name": r["platform_name"],
                "posted_by": r["posted_by"],
                "forum_post_subject": r["forum_post_subject"],
            }
            for r in rows
        ]


def get_all_dashboard_data(start_date=None, end_date=None):
    """Return all dashboard data in a single call."""
    return {
        "summary": get_summary_stats(start_date, end_date),
        "classification": get_classification_breakdown(start_date, end_date),
        "status": get_status_breakdown(start_date, end_date),
        "forum_post_status": get_forum_post_status(start_date, end_date),
        "daily_volume": get_daily_volume(start_date, end_date),
        "recent_queries": get_recent_queries(start_date, end_date),
        "mixpanel": get_all_vote_events(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
