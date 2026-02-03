"""
Claude API Client for SAT Forum Responder
Handles all interactions with Claude API including text and vision capabilities
Uses Claude Opus 4.5 model
"""

import anthropic
import json
import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Client for interacting with Claude API"""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-5-20251101",
        max_tokens: int = 20000,
        thinking_budget: int = 6000
    ):
        """
        Initialize Claude client

        Args:
            api_key: Anthropic API key
            model: Model name (default: Claude Opus 4.5)
            max_tokens: Maximum output tokens
            thinking_budget: Extended thinking budget tokens
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        logger.info(f"Claude client initialized with model: {model}")

    def call_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        retry_count: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Call Claude with system and user prompts

        Args:
            system_prompt: System instructions
            user_prompt: User message
            retry_count: Number of retries on failure

        Returns:
            Dictionary with response data or None
        """
        for attempt in range(retry_count):
            try:
                start_time = time.time()

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    },
                    temperature=1.0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )

                execution_time = int((time.time() - start_time) * 1000)

                # Extract response content
                text_response = ""
                thinking_content = ""

                for block in response.content:
                    if block.type == "thinking":
                        thinking_content = block.thinking
                    elif block.type == "text":
                        text_response += block.text

                logger.info(f"Claude API call successful ({execution_time}ms)")

                return {
                    "response": text_response,
                    "thinking": thinking_content,
                    "execution_time_ms": execution_time,
                    "model": self.model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }

            except anthropic.APIError as e:
                logger.error(f"Claude API error (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error calling Claude (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))

        return None

    def call_agent_with_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data: str,
        media_type: str = "image/png",
        retry_count: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Call Claude with vision capability for image analysis

        Args:
            system_prompt: System instructions
            user_prompt: User message
            image_data: Base64 encoded image data (without data URI prefix)
            media_type: MIME type of the image
            retry_count: Number of retries on failure

        Returns:
            Dictionary with response data or None
        """
        for attempt in range(retry_count):
            try:
                start_time = time.time()

                # Prepare content with image
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    },
                    temperature=1.0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": content}
                    ]
                )

                execution_time = int((time.time() - start_time) * 1000)

                # Extract response
                text_response = ""
                thinking_content = ""

                for block in response.content:
                    if block.type == "thinking":
                        thinking_content = block.thinking
                    elif block.type == "text":
                        text_response += block.text

                logger.info(f"Claude vision API call successful ({execution_time}ms)")

                return {
                    "response": text_response,
                    "thinking": thinking_content,
                    "execution_time_ms": execution_time,
                    "model": self.model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }

            except anthropic.APIError as e:
                logger.error(f"Claude vision API error (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.error(f"Unexpected error calling Claude vision (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))

        return None

    def call_agent_with_multiple_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[Dict[str, str]],
        retry_count: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Call Claude with multiple images

        Args:
            system_prompt: System instructions
            user_prompt: User message
            images: List of dicts with 'data' (base64) and 'media_type' keys
            retry_count: Number of retries on failure

        Returns:
            Dictionary with response data or None
        """
        for attempt in range(retry_count):
            try:
                start_time = time.time()

                # Build content with all images
                content = []
                for img in images:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.get("media_type", "image/png"),
                            "data": img["data"]
                        }
                    })

                content.append({
                    "type": "text",
                    "text": user_prompt
                })

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    },
                    temperature=1.0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": content}
                    ]
                )

                execution_time = int((time.time() - start_time) * 1000)

                text_response = ""
                thinking_content = ""

                for block in response.content:
                    if block.type == "thinking":
                        thinking_content = block.thinking
                    elif block.type == "text":
                        text_response += block.text

                logger.info(f"Claude multi-image API call successful ({execution_time}ms, {len(images)} images)")

                return {
                    "response": text_response,
                    "thinking": thinking_content,
                    "execution_time_ms": execution_time,
                    "model": self.model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }

            except anthropic.APIError as e:
                logger.error(f"Claude multi-image API error (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.error(f"Unexpected error calling Claude multi-image (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))

        return None

    def parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from Claude's response

        Args:
            response_text: Text response from Claude

        Returns:
            Parsed JSON dictionary or None
        """
        try:
            # Try to find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in response")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Problematic JSON: {response_text[:500]}...")

            # Try extracting from code blocks
            try:
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    if json_end != -1:
                        json_str = response_text[json_start:json_end].strip()
                        return json.loads(json_str)

                if "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    if json_end != -1:
                        json_str = response_text[json_start:json_end].strip()
                        if json_str.startswith('{'):
                            return json.loads(json_str)

            except Exception as retry_error:
                logger.error(f"JSON retry parse also failed: {retry_error}")

            return None
