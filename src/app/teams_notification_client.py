"""
Teams Notification Client for SAT Forum Responder
Sends notifications to Microsoft Teams via Power Automate webhook
"""

import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TeamsNotificationClient:
    """Client for sending notifications to Microsoft Teams via Power Automate"""

    def __init__(self, webhook_url: str, chat_id: str):
        """Initialize Teams Notification client"""
        self.webhook_url = webhook_url
        self.chat_id = chat_id
        self.headers = {"Content-Type": "application/json"}
        logger.info(f"Teams Notification Client initialized - Chat ID: {chat_id[:30]}...")

    def send_notification(
        self,
        message_body: str,
        email: str = "system@prismlearning.com"
    ) -> Dict[str, Any]:
        """Send a notification to Teams"""
        payload = {
            "chat_id": self.chat_id,
            "email": email,
            "message_body": message_body
        }

        try:
            logger.info("Sending Teams notification...")

            response = requests.post(
                self.webhook_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code in [200, 201, 202]:
                logger.info("Teams notification sent successfully")
                return {"success": True, "status_code": response.status_code}
            else:
                logger.warning(
                    f"Teams notification failed - Status: {response.status_code}, "
                    f"Response: {response.text[:200] if response.text else 'No response'}"
                )
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text[:200] if response.text else "Unknown error"
                }

        except requests.exceptions.Timeout:
            logger.error("Teams notification timeout")
            return {"success": False, "error": "Timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Teams notification error: {e}")
            return {"success": False, "error": str(e)}

    def send_processing_notification(
        self,
        correlation_id: str,
        status: str,
        forum_post_status: Optional[str],
        posted_by_email: Optional[str],
        classification: Optional[str] = None,
        error_message: Optional[str] = None,
        html_cleaned: bool = False,
        images_transcribed: int = 0
    ) -> Dict[str, Any]:
        """Send a forum processing notification to Teams"""
        status_emoji = self._get_status_emoji(status, forum_post_status)
        status_text = self._get_status_text(status, forum_post_status)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        html_parts = [
            f"<p><b>{status_emoji} SAT Forum Post Processed</b></p>",
            f"<p><b>Status:</b> {status_text}<br>",
            f"<b>Correlation ID:</b> <code>{correlation_id}</code><br>",
            f"<b>Posted By:</b> {posted_by_email or 'N/A'}<br>",
        ]

        if classification:
            html_parts.append(f"<b>Classification:</b> {classification}<br>")

        if forum_post_status:
            forum_status_text = self._get_forum_post_status_text(forum_post_status)
            html_parts.append(f"<b>Forum Reply:</b> {forum_status_text}<br>")

        if images_transcribed > 0:
            html_parts.append(f"<b>Images Transcribed:</b> {images_transcribed}<br>")

        if html_cleaned:
            html_parts.append("<b>HTML Fixed:</b> Yes (parsing errors were auto-corrected)<br>")

        if error_message:
            html_parts.append(f"<b>Error:</b> {error_message[:200]}<br>")

        html_parts.append("</p>")
        html_parts.append(f"<p><i>Processed at {timestamp}</i></p>")

        message_body = "".join(html_parts)

        return self.send_notification(
            message_body=message_body,
            email=posted_by_email or "system@prismlearning.com"
        )

    def send_deflection_alert(
        self,
        correlation_id: str,
        nsm_category: str,
        forum_post_text: Optional[str] = None,
        platform: Optional[str] = None,
        posted_by_email: Optional[str] = None,
        entity_name: Optional[str] = None,
        entity_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a deflection alert when an NSM query redirects student to support@e-gmat.com"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Truncate long question text
        question_display = (forum_post_text or "N/A")[:500]
        if forum_post_text and len(forum_post_text) > 500:
            question_display += "..."

        html_parts = [
            "<p><b>🔶 SAT NSM Deflection Alert — Expert Follow-up Needed</b></p>",
            "<p>Student redirected to support@e-gmat.com — a strategy expert should follow up.</p>",
            "<p>",
            f"<b>NSM Category:</b> {nsm_category}<br>",
            f"<b>Correlation ID:</b> {correlation_id}<br>",
            f"<b>Platform:</b> {platform or 'N/A'}<br>",
            f"<b>Entity:</b> {entity_name or 'N/A'} (ID: {entity_id or 'N/A'})<br>",
            f"<b>Posted By:</b> {posted_by_email or 'N/A'}<br>",
            f"<b>Student Question:</b> {question_display}",
            "</p>",
            f"<p><i>Deflection detected at {timestamp}</i></p>"
        ]

        message_body = "".join(html_parts)

        return self.send_notification(
            message_body=message_body,
            email=posted_by_email or "system@prismlearning.com"
        )

    def _get_status_emoji(self, status: str, forum_post_status: Optional[str]) -> str:
        if status == "completed" and forum_post_status == "posted":
            return "✅"
        elif status == "completed" and forum_post_status == "posted_hil":
            return "🟠"
        elif status == "completed":
            return "🟢"
        elif status == "hil_exception":
            return "🟡"
        elif status == "url_detected":
            return "🔗"
        elif status == "error":
            return "❌"
        else:
            return "ℹ️"

    def _get_status_text(self, status: str, forum_post_status: Optional[str]) -> str:
        status_map = {
            "completed": "Completed Successfully",
            "hil_exception": "Human-in-Loop Required (HIL)",
            "url_detected": "URL Detected - Skipped",
            "error": "Processing Error",
            "pending": "Pending"
        }
        return status_map.get(status, status)

    def _get_forum_post_status_text(self, forum_post_status: str) -> str:
        status_map = {
            "posted": "✅ Posted to Forum",
            "posted_hil": "🟠 Posted (HIL Acknowledgment — needs human follow-up)",
            "failed": "❌ Failed to Post",
            "skipped_hil": "⏭️ Skipped (HIL)",
            "skipped_url": "⏭️ Skipped (URL)",
            "skipped_validation": "⏭️ Skipped (Validation)",
            "skipped": "⏭️ Skipped"
        }
        return status_map.get(forum_post_status, forum_post_status)
