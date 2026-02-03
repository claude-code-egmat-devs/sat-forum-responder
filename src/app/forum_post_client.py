"""
Forum Post Client for SAT Forum Responder
Posts generated responses to the PrismLearning SAT forum via API
"""

import requests
import logging
import re
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)


def clean_html(html_content: str) -> str:
    """Clean and sanitize HTML content to fix common parsing errors"""
    if not html_content:
        return ""

    cleaned = html_content

    # Fix common character encoding issues
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '-',
        'â€"': '-',
        'â€¢': '•',
        'Â': '',
        '\x00': '',
        '\r\n': '\n',
        '\r': '\n',
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    # Remove control characters except newlines and tabs
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)

    # Fix unclosed self-closing tags
    self_closing_tags = ['br', 'hr', 'img', 'input', 'meta', 'link']
    for tag in self_closing_tags:
        cleaned = re.sub(rf'<{tag}(\s[^>]*)?>(?!/)', rf'<{tag}\1/>', cleaned, flags=re.IGNORECASE)

    # Find and close unclosed tags
    tag_pattern = re.compile(r'<(/?)(\w+)([^>]*)>')
    void_elements = {'br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'param', 'source', 'track', 'wbr'}

    def get_unclosed_tags(html_str):
        stack = []
        for match in tag_pattern.finditer(html_str):
            is_closing = match.group(1) == '/'
            tag_name = match.group(2).lower()
            if tag_name in void_elements:
                continue
            if is_closing:
                if stack and stack[-1] == tag_name:
                    stack.pop()
            else:
                if not match.group(3).rstrip().endswith('/'):
                    stack.append(tag_name)
        return stack

    unclosed = get_unclosed_tags(cleaned)
    for tag in reversed(unclosed):
        cleaned += f'</{tag}>'

    # Fix malformed HTML entities
    cleaned = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', cleaned)

    # Remove script/style tags for safety
    cleaned = re.sub(r'<script[^>]*>.*?</script>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)

    return cleaned.strip()


class ForumPostClient:
    """Client for posting responses to the PrismLearning SAT forum API"""

    def __init__(self, url: str, api_key: str):
        """Initialize Forum Post client"""
        self.url = url
        self.api_key = api_key
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        logger.info(f"Forum Post Client initialized - URL: {url}")

    def post_response(
        self,
        correlation_id: str,
        parent_id: int,
        post_text: str,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """Post a response to the forum"""
        current_post_text = post_text
        html_cleaned = False

        for attempt in range(retry_count):
            payload = {
                "correlationId": correlation_id,
                "parentId": parent_id,
                "postSubject": "Re:",
                "queryState": "REPLY_READY",
                "postText": current_post_text
            }

            try:
                logger.info(f"Posting response to forum - Correlation ID: {correlation_id}, Parent ID: {parent_id} (attempt {attempt + 1})")

                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code in [200, 201, 202]:
                    logger.info(f"Successfully posted response to forum - Correlation ID: {correlation_id}")
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        "response": response.json() if response.text else {},
                        "html_cleaned": html_cleaned
                    }
                else:
                    error_text = response.text[:500] if response.text else 'No response body'
                    logger.warning(f"Forum post failed - Status: {response.status_code}, Response: {error_text}")

                    is_parsing_error = self._is_parsing_error(response.status_code, error_text)

                    if is_parsing_error and not html_cleaned:
                        logger.info("Detected parsing error, attempting to clean HTML and retry...")
                        current_post_text = clean_html(post_text)
                        html_cleaned = True
                        continue

                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        if not html_cleaned:
                            logger.info("Client error, attempting to clean HTML and retry once more...")
                            current_post_text = clean_html(post_text)
                            html_cleaned = True
                            continue

                        return {
                            "success": False,
                            "status_code": response.status_code,
                            "error": error_text
                        }

            except requests.exceptions.Timeout:
                logger.error(f"Forum post timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Forum post connection error (attempt {attempt + 1}): {e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Forum post request error (attempt {attempt + 1}): {e}")

            if attempt < retry_count - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        return {
            "success": False,
            "error": f"Failed after {retry_count} attempts"
        }

    def _is_parsing_error(self, status_code: int, error_text: str) -> bool:
        """Check if the error is related to HTML/JSON parsing"""
        if not error_text:
            return False

        error_lower = error_text.lower()
        parsing_indicators = [
            'parse', 'parsing', 'invalid', 'malformed', 'syntax',
            'unexpected', 'encoding', 'character', 'json', 'html',
            'xml', 'entity', 'unterminated', 'unclosed', 'illegal', 'bad request'
        ]

        return any(indicator in error_lower for indicator in parsing_indicators)

    def post_forum_response(self, forum_data: Dict[str, Any], html_response: str) -> Dict[str, Any]:
        """Post a response using forum data and generated HTML response"""
        correlation_id = forum_data.get("correlationId") or forum_data.get("Forum_Corr_ID")
        parent_id = forum_data.get("parentId") or forum_data.get("id") or forum_data.get("forumId")

        if not correlation_id:
            logger.error("Missing correlation ID for forum post")
            return {"success": False, "error": "Missing correlation ID"}

        if not parent_id:
            logger.error("Missing parent ID for forum post")
            return {"success": False, "error": "Missing parent ID"}

        try:
            parent_id = int(parent_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid parent ID format: {parent_id}")
            return {"success": False, "error": f"Invalid parent ID format: {parent_id}"}

        return self.post_response(
            correlation_id=correlation_id,
            parent_id=parent_id,
            post_text=html_response
        )
