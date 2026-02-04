"""
Airtable client for SAT Forum Responder
Handles all Airtable operations for SAT Forum Posts and Agent System Outputs tables
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class AirtableClient:
    """Client for interacting with Airtable API"""

    def __init__(self, api_key: str, base_id: str, table_name: str):
        """Initialize Airtable client"""
        self.api_key = api_key
        self.base_id = base_id
        self.table_name = table_name
        self.base_url = f"https://api.airtable.com/v0/{base_id}/{table_name}"
        # Agent System Outputs table URL
        self.agent_outputs_url = f"https://api.airtable.com/v0/{base_id}/Agent%20System%20Outputs"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _create_record(self, url: str, fields: Dict[str, Any], retry_count: int = 3) -> Optional[str]:
        """Create a new record in Airtable at specified URL"""
        payload = {"fields": fields}

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    record_id = data.get("id")
                    logger.info(f"Created Airtable record: {record_id}")
                    return record_id
                else:
                    logger.warning(f"Airtable create failed: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Airtable request error (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)

        return None

    def create_record(self, fields: Dict[str, Any], retry_count: int = 3) -> Optional[str]:
        """Create a new record in the main table"""
        return self._create_record(self.base_url, fields, retry_count)

    def _update_record(self, url: str, record_id: str, fields: Dict[str, Any], retry_count: int = 3) -> bool:
        """Update an existing record in Airtable"""
        full_url = f"{url}/{record_id}"
        payload = {"fields": fields}

        for attempt in range(retry_count):
            try:
                response = requests.patch(
                    full_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    logger.info(f"Updated Airtable record: {record_id}")
                    return True
                else:
                    logger.warning(f"Airtable update failed: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Airtable request error (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)

        return False

    def update_record(self, record_id: str, fields: Dict[str, Any], retry_count: int = 3) -> bool:
        """Update an existing record in the main table"""
        return self._update_record(self.base_url, record_id, fields, retry_count)

    def _find_by_correlation_id(self, url: str, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Find a record by correlation_id in specified table"""
        try:
            formula = f"{{Forum_Corr_ID}}=\047{correlation_id}\047"
            params = {"filterByFormula": formula}

            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])
                if records:
                    return records[0]
                return None
            else:
                logger.error(f"Airtable search failed: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Airtable search error: {e}")
            return None

    def find_by_correlation_id(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Find a record by correlation_id in main table"""
        return self._find_by_correlation_id(self.base_url, correlation_id)

    def upsert_forum_response(self, data: Dict[str, Any]) -> bool:
        """Create or update a forum response record in SAT Forum Posts table"""
        correlation_id = data.get("correlation_id")

        existing_record = self.find_by_correlation_id(correlation_id)

        # SAT Forum Posts field mappings (without agent outputs)
        fields = {
            "Forum_Corr_ID": data.get("correlation_id"),
            "postedBy": data.get("posted_by"),
            "forumPostSubject": data.get("forum_post_subject"),
            "ForumPostText": data.get("forum_post_text"),
            "parentPostQuery": data.get("parent_post_query"),
            "parentPostResponse": data.get("parent_post_response"),
            "isImageBase64Encoded": str(data.get("image_base64_encoded", "")),
            "type": data.get("post_type"),
            "environment": data.get("environment"),
            "classification": data.get("classification"),
            "response": data.get("expert_reply_html"),
            "url_check": data.get("url_check", "false")
        }

        # Remove None values
        fields = {k: v for k, v in fields.items() if v is not None}

        if existing_record:
            record_id = existing_record["id"]
            return self.update_record(record_id, fields)
        else:
            record_id = self.create_record(fields)
            return record_id is not None

    def upsert_agent_outputs(self, data: Dict[str, Any]) -> bool:
        """Create or update agent system outputs in Agent System Outputs table"""
        correlation_id = data.get("correlation_id")

        existing_record = self._find_by_correlation_id(self.agent_outputs_url, correlation_id)

        # Agent System Outputs field mappings
        fields = {
            "Forum_Corr_ID": data.get("correlation_id"),
            "urls_list": data.get("urls_list", ""),
            "a1_triage_output": data.get("a1_triage_output", ""),
            "a2_classification_output": data.get("a2_classification_output", ""),
            "tool_response_output": data.get("tool_response_output", "")
        }

        # Remove None values
        fields = {k: v for k, v in fields.items() if v is not None}

        if existing_record:
            record_id = existing_record["id"]
            return self._update_record(self.agent_outputs_url, record_id, fields)
        else:
            record_id = self._create_record(self.agent_outputs_url, fields)
            return record_id is not None
