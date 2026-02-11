"""
SAT Forum Dashboard - Flask Application
Serves the dashboard UI and provides JSON API endpoints.
"""

import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

load_dotenv()

from data_service import get_all_dashboard_data, get_recent_queries, get_query_detail, get_filtered_queries, get_query_logs, get_all_raw_queries, get_all_vote_events

app = Flask(__name__)

# Prefix for running behind nginx at /sat-forum-dashboard/
PREFIX = os.getenv("APP_PREFIX", "/sat-forum-dashboard")


@app.route("/")
def dashboard():
    """Serve the main dashboard page."""
    return render_template("dashboard.html", prefix=PREFIX)


@app.route("/api/stats")
def api_stats():
    """Return all dashboard metrics as JSON."""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    data = get_all_dashboard_data(start_date, end_date)
    return jsonify(data)


@app.route("/api/queries")
def api_queries():
    """Return recent queries as JSON."""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    limit = request.args.get("limit", 20, type=int)
    rows = get_recent_queries(start_date, end_date, limit=limit)
    return jsonify(rows)



@app.route("/api/queries/filtered")
def api_queries_filtered():
    """Return queries filtered by type (hil, error, completed)."""
    filter_type = request.args.get("type", "")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    limit = request.args.get("limit", 100, type=int)
    if filter_type not in ("hil", "error", "completed"):
        return jsonify({"error": "Invalid filter type. Use: hil, error, completed"}), 400
    rows = get_filtered_queries(filter_type, start_date, end_date, limit=limit)
    return jsonify(rows)


@app.route("/api/query/<correlation_id>")
def api_query_detail(correlation_id):
    """Return full detail for a single query."""
    detail = get_query_detail(correlation_id)
    if detail is None:
        return jsonify({"error": "Query not found"}), 404
    return jsonify(detail)


@app.route("/api/query/<correlation_id>/logs")
def api_query_logs(correlation_id):
    """Return parsed processing logs for a single query."""
    data = get_query_logs(correlation_id)
    return jsonify(data)



@app.route("/api/raw-data")
def api_raw_data():
    """Return all webhook rows for client-side aggregation. Fast, no Mixpanel."""
    rows = get_all_raw_queries()
    return jsonify({"queries": rows})


@app.route("/api/mixpanel")
def api_mixpanel():
    """Return all vote events for client-side date filtering (5-min server cache)."""
    data = get_all_vote_events()
    return jsonify(data)


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5012))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
