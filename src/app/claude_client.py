"""
Claude API Client for SAT Forum Responder
Handles all interactions with Claude API including text and vision capabilities
Uses Claude Opus 4.5 model
"""

import anthropic
import json
import logging
import sys
import time
from typing import Dict, Any, Optional, List

# Add AI monitoring library to path
sys.path.insert(0, "/home/.ai_monitoring/lib")
try:
    from ai_usage_logger import AIUsageLogger
    AI_LOGGING_ENABLED = True
except ImportError:
    AI_LOGGING_ENABLED = False

logger = logging.getLogger(__name__)

# Claude Opus 4.5 Pricing (per million tokens)
OPUS_INPUT_PRICE_PER_MILLION = 15.0   # $15 per million input tokens
OPUS_OUTPUT_PRICE_PER_MILLION = 75.0  # $75 per million output tokens


def calculate_cost(input_tokens: int, output_tokens: int) -> Dict[str, float]:
    """
    Calculate the cost of an API call based on token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens (includes thinking tokens)

    Returns:
        Dictionary with cost breakdown
    """
    input_cost = (input_tokens / 1_000_000) * OPUS_INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OPUS_OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6)
    }


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

        # Initialize AI usage logger
        if AI_LOGGING_ENABLED:
            self.usage_logger = AIUsageLogger(app_name="SAT_Forum_Responder")
            logger.info("AI usage logging enabled")
        else:
            self.usage_logger = None
            logger.warning("AI usage logging not available")

        logger.info(f"Claude client initialized with model: {model}")

    def _log_usage(self, purpose: str, input_tokens: int, output_tokens: int,
                   execution_time: int, success: bool = True, error: str = None):
        """Log AI usage to centralized monitoring system"""
        if self.usage_logger:
            try:
                self.usage_logger.log_call(
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    purpose=purpose,
                    latency_ms=execution_time,
                    success=success,
                    error=error,
                    thinking_enabled=True
                )
            except Exception as e:
                logger.warning(f"Failed to log AI usage: {e}")

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

                # Use streaming for long requests (required for high max_tokens)
                text_response = ""
                thinking_content = ""
                input_tokens = 0
                output_tokens = 0

                with self.client.messages.stream(
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
                ) as stream:
                    response = stream.get_final_message()

                execution_time = int((time.time() - start_time) * 1000)

                # Extract response content
                for block in response.content:
                    if block.type == "thinking":
                        thinking_content = block.thinking
                    elif block.type == "text":
                        text_response += block.text

                # Get token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                # Calculate cost
                cost = calculate_cost(input_tokens, output_tokens)

                logger.info(f"Claude API call successful ({execution_time}ms)")
                logger.info(f"  TOKENS: input={input_tokens}, output={output_tokens}")
                logger.info(f"  COST: input=${cost['input_cost']:.6f}, output=${cost['output_cost']:.6f}, total=${cost['total_cost']:.6f}")

                # Log to centralized monitoring
                self._log_usage("text_query", input_tokens, output_tokens, execution_time)

                return {
                    "response": text_response,
                    "thinking": thinking_content,
                    "execution_time_ms": execution_time,
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost
                }

            except anthropic.APIError as e:
                logger.error(f"Claude API error (attempt {attempt + 1}): {e}")
                if attempt >= 1:
                    self._log_usage("text_query", 0, 0, 0, success=False, error=str(e))
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

                # Use streaming for long requests
                with self.client.messages.stream(
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
                ) as stream:
                    response = stream.get_final_message()

                execution_time = int((time.time() - start_time) * 1000)

                # Extract response
                text_response = ""
                thinking_content = ""

                for block in response.content:
                    if block.type == "thinking":
                        thinking_content = block.thinking
                    elif block.type == "text":
                        text_response += block.text

                # Calculate cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = calculate_cost(input_tokens, output_tokens)

                logger.info(f"Claude vision API call successful ({execution_time}ms)")
                logger.info(f"  TOKENS: input={input_tokens}, output={output_tokens}")
                logger.info(f"  COST: input=${cost['input_cost']:.6f}, output=${cost['output_cost']:.6f}, total=${cost['total_cost']:.6f}")

                # Log to centralized monitoring
                self._log_usage("vision_single", input_tokens, output_tokens, execution_time)

                return {
                    "response": text_response,
                    "thinking": thinking_content,
                    "execution_time_ms": execution_time,
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost
                }

            except anthropic.APIError as e:
                logger.error(f"Claude vision API error (attempt {attempt + 1}): {e}")
                if attempt >= 1:
                    self._log_usage("vision_single", 0, 0, 0, success=False, error=str(e))
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

                # Use streaming for long requests
                with self.client.messages.stream(
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
                ) as stream:
                    response = stream.get_final_message()

                execution_time = int((time.time() - start_time) * 1000)

                text_response = ""
                thinking_content = ""

                for block in response.content:
                    if block.type == "thinking":
                        thinking_content = block.thinking
                    elif block.type == "text":
                        text_response += block.text

                # Calculate cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = calculate_cost(input_tokens, output_tokens)

                logger.info(f"Claude multi-image API call successful ({execution_time}ms, {len(images)} images)")
                logger.info(f"  TOKENS: input={input_tokens}, output={output_tokens}")
                logger.info(f"  COST: input=${cost['input_cost']:.6f}, output=${cost['output_cost']:.6f}, total=${cost['total_cost']:.6f}")

                # Log to centralized monitoring
                self._log_usage("vision_multi", input_tokens, output_tokens, execution_time)

                return {
                    "response": text_response,
                    "thinking": thinking_content,
                    "execution_time_ms": execution_time,
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost
                }

            except anthropic.APIError as e:
                logger.error(f"Claude multi-image API error (attempt {attempt + 1}): {e}")
                if attempt >= 1:
                    self._log_usage("vision_multi", 0, 0, 0, success=False, error=str(e))
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.error(f"Unexpected error calling Claude multi-image (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))

        return None

    def call_agent_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_name: str,
        tool_description: str,
        tool_schema: Dict[str, Any],
        retry_count: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Two-call structured output with thinking:
          Call 1: Thinking enabled (no tools) - Claude reasons deeply about the task.
          Call 2: Forced tool_choice (no thinking) - extracts structured JSON from Call 1's reasoning.
        This avoids the API restriction that disallows thinking with forced tool_choice.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            tool_name: Name of the tool (e.g. 'classify_doubt')
            tool_description: Description of what the tool does
            tool_schema: JSON Schema for the tool's input_schema
            retry_count: Number of retries on failure

        Returns:
            Dictionary with response data including structured 'parsed' field, or None
        """
        for attempt in range(retry_count):
            try:
                start_time = time.time()

                # -- Call 1: Thinking + reasoning (no tools) --
                logger.info("Structured call — Step 1: reasoning with thinking enabled")
                with self.client.messages.stream(
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
                ) as stream:
                    step1_response = stream.get_final_message()

                # Extract thinking and text from Call 1
                thinking_content = ""
                reasoning_text = ""
                for block in step1_response.content:
                    if block.type == "thinking":
                        thinking_content = block.thinking
                    elif block.type == "text":
                        reasoning_text += block.text

                step1_input = step1_response.usage.input_tokens
                step1_output = step1_response.usage.output_tokens
                step1_cost = calculate_cost(step1_input, step1_output)
                logger.info(f"  Step 1 done — tokens: {step1_input}+{step1_output}, cost: ${step1_cost['total_cost']:.6f}")

                # -- Call 2: Forced tool extraction (no thinking) --
                logger.info("Structured call — Step 2: extracting structured JSON")
                tool_def = {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": tool_schema,
                }

                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system="Extract the classification from the assistant's analysis into the structured tool. Do not change any values — just map them to the schema fields.",
                    tools=[tool_def],
                    tool_choice={"type": "tool", "name": tool_name},
                    messages=[
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": reasoning_text},
                        {"role": "user", "content": "Now output your classification using the tool."}
                    ]
                ) as stream:
                    step2_response = stream.get_final_message()

                execution_time = int((time.time() - start_time) * 1000)

                # Extract structured tool input from Call 2
                parsed = None
                for block in step2_response.content:
                    if block.type == "tool_use":
                        parsed = block.input

                step2_input = step2_response.usage.input_tokens
                step2_output = step2_response.usage.output_tokens
                step2_cost = calculate_cost(step2_input, step2_output)

                total_input = step1_input + step2_input
                total_output = step1_output + step2_output
                total_cost = calculate_cost(total_input, total_output)

                logger.info(f"  Step 2 done — tokens: {step2_input}+{step2_output}, cost: ${step2_cost['total_cost']:.6f}")
                logger.info(f"Claude structured call successful ({execution_time}ms)")
                logger.info(f"  TOTAL TOKENS: input={total_input}, output={total_output}")
                logger.info(f"  TOTAL COST: ${total_cost['total_cost']:.6f}")

                self._log_usage("structured_query", total_input, total_output, execution_time)

                return {
                    "response": json.dumps(parsed) if parsed else "",
                    "parsed": parsed,
                    "thinking": thinking_content,
                    "execution_time_ms": execution_time,
                    "model": self.model,
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "cost": total_cost
                }

            except anthropic.APIError as e:
                logger.error(f"Claude structured API error (attempt {attempt + 1}): {e}")
                if attempt >= 1:
                    self._log_usage("structured_query", 0, 0, 0, success=False, error=str(e))
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.error(f"Unexpected error in structured call (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(3 * (attempt + 1))

        return None

    def repair_json_with_haiku(self, broken_text: str) -> Optional[Dict[str, Any]]:
        """
        Last-resort fallback: send broken JSON to Haiku to repair it.
        Uses a separate lightweight client call (no thinking, no streaming).

        Args:
            broken_text: The raw text that failed JSON parsing

        Returns:
            Parsed JSON dict or None
        """
        try:
            logger.info("Attempting JSON repair with Haiku...")

            # Truncate to avoid excessive token usage
            text_to_repair = broken_text[:8000]

            repair_response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": (
                        "The following text was supposed to be valid JSON but has formatting errors. "
                        "Extract and return ONLY the corrected JSON object. Fix any missing commas, "
                        "unescaped quotes, trailing commas, or other syntax issues. "
                        "Return ONLY the raw JSON with no markdown, no explanation, no code blocks.\n\n"
                        f"{text_to_repair}"
                    )
                }]
            )

            repaired_text = repair_response.content[0].text.strip()

            # Calculate cost for repair call
            repair_input = repair_response.usage.input_tokens
            repair_output = repair_response.usage.output_tokens
            # Haiku pricing: $0.80/M input, $4/M output
            repair_cost = (repair_input / 1_000_000) * 0.80 + (repair_output / 1_000_000) * 4.0
            logger.info(f"  Haiku repair: {repair_input}+{repair_output} tokens, ${repair_cost:.4f}")

            self._log_usage("json_repair_haiku", repair_input, repair_output, 0)

            # Try to parse the repaired JSON
            start_idx = repaired_text.find('{')
            end_idx = repaired_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                result = json.loads(repaired_text[start_idx:end_idx])
                logger.info("Haiku JSON repair successful")
                return result

            logger.error("Haiku repair did not return valid JSON")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Haiku repair output still invalid JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Haiku repair failed: {e}")
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

            # Last resort: try Haiku repair
            repaired = self.repair_json_with_haiku(response_text)
            if repaired:
                return repaired

            return None
