"""
Email Notification Client for SAT Forum Responder
Sends email notifications via Power Automate webhook
"""

import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EmailNotificationClient:
    """Client for sending email notifications via Power Automate"""

    def __init__(self, webhook_url: str, default_recipient: str = "support@e-gmat.com"):
        self.webhook_url = webhook_url
        self.default_recipient = default_recipient
        self.headers = {"Content-Type": "application/json"}
        logger.info("Email Notification Client initialized")

    def send_email(
        self,
        subject: str,
        body: str,
        recipient_email: Optional[str] = None,
        is_html: bool = True
    ) -> Dict[str, Any]:
        """Send an email via Power Automate webhook"""
        payload = {
            "recipient_email": recipient_email or self.default_recipient,
            "subject": subject,
            "body": body,
            "is_html": is_html
        }

        try:
            logger.info(f"Sending email notification to {payload['recipient_email']}...")

            response = requests.post(
                self.webhook_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code in [200, 201, 202]:
                logger.info("Email notification sent successfully")
                return {"success": True, "status_code": response.status_code}
            else:
                logger.warning(
                    f"Email notification failed - Status: {response.status_code}, "
                    f"Response: {response.text[:200] if response.text else 'No response'}"
                )
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text[:200] if response.text else "Unknown error"
                }

        except requests.exceptions.Timeout:
            logger.error("Email notification timeout")
            return {"success": False, "error": "Timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Email notification error: {e}")
            return {"success": False, "error": str(e)}

    def send_deflection_email(
        self,
        correlation_id: str,
        nsm_category: str,
        forum_post_text: Optional[str] = None,
        platform: Optional[str] = None,
        posted_by_email: Optional[str] = None,
        entity_name: Optional[str] = None,
        entity_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a deflection alert email when an NSM query redirects student to support"""
        subject = f"[SAT Forum Deflection] {nsm_category} — Expert Follow-up Needed"

        question_display = (forum_post_text or "N/A")[:1000]
        if forum_post_text and len(forum_post_text) > 1000:
            question_display += "..."

        body = (
            "<h3>SAT NSM Deflection Alert</h3>"
            "<p>A student's SAT forum query was classified as an NSM deflection category. "
            "The student has been redirected to <b>support@e-gmat.com</b> — "
            "a strategy expert should follow up.</p>"
            "<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
            f"<tr><td><b>NSM Category</b></td><td>{nsm_category}</td></tr>"
            f"<tr><td><b>Correlation ID</b></td><td>{correlation_id}</td></tr>"
            f"<tr><td><b>Platform</b></td><td>{platform or 'N/A'}</td></tr>"
            f"<tr><td><b>Entity Name</b></td><td>{entity_name or 'N/A'}</td></tr>"
            f"<tr><td><b>Entity ID</b></td><td>{entity_id or 'N/A'}</td></tr>"
            f"<tr><td><b>Posted By</b></td><td>{posted_by_email or 'N/A'}</td></tr>"
            f"<tr><td><b>Student Question</b></td><td>{question_display}</td></tr>"
            "</table>"
        )

        return self.send_email(subject=subject, body=body)
