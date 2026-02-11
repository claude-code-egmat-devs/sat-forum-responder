"""
SAT Forum Responder - Forum Processor
Processes forum posts through the multi-agent system with enhanced image transcription
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .app.config import config
from .app.claude_client import ClaudeClient
from .app.airtable_client import AirtableClient
from .app.url_detector import url_detector
from .app.forum_post_client import ForumPostClient
from .app.teams_notification_client import TeamsNotificationClient
from .app.content_processor import ForumContentProcessor
from .app.html_validator import html_validator, ValidationResult

logger = logging.getLogger(__name__)


# ── Tool-use schemas for structured output (guarantees valid JSON) ──
A1_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["SM_Doubt", "SAT_Strategy_Doubt", "Unrelated_to_SAT", "Gratitude"],
            "description": "The triage classification of the student doubt"
        },
        "justification": {
            "type": "object",
            "properties": {
                "primary_intent": {"type": "string", "description": "What the student is primarily trying to achieve"},
                "key_indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific phrases or keywords from student doubt"
                },
                "classification_reasoning": {"type": "string", "description": "Why this doubt fits the chosen category"},
                "decision_process": {"type": "string", "description": "Which classification steps led to this conclusion"}
            },
            "required": ["primary_intent", "key_indicators", "classification_reasoning", "decision_process"]
        }
    },
    "required": ["classification", "justification"]
}

A2_DEEP_SM_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["Genuine_Doubt", "Pointing_Out_Corrections", "Alternate_Approach", "Variation_of_Question"],
            "description": "The deep SM classification of the student doubt"
        },
        "justification": {
            "type": "object",
            "properties": {
                "primary_intent": {"type": "string", "description": "What the student's main goal appears to be"},
                "key_indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Quoted phrases from the student's doubt"
                },
                "classification_reasoning": {"type": "string", "description": "Why this doubt fits the chosen category"},
                "supporting_evidence": {"type": "string", "description": "Quoted relevant portions connected to category characteristics"},
                "decision_path": {"type": "string", "description": "Which decision tree steps led to this classification"}
            },
            "required": ["primary_intent", "key_indicators", "classification_reasoning", "decision_path"]
        }
    },
    "required": ["classification", "justification"]
}

_CLASSIFIER_SCHEMAS = {
    "a1_triage": {
        "tool_name": "classify_triage",
        "tool_description": "Classify a student forum doubt into one of four triage categories: SM_Doubt, SAT_Strategy_Doubt, Unrelated_to_SAT, or Gratitude.",
        "schema": A1_TRIAGE_SCHEMA,
    },
    "a2_deep_sm": {
        "tool_name": "classify_deep_sm",
        "tool_description": "Perform deep classification of a subject-matter doubt into: Genuine_Doubt, Pointing_Out_Corrections, Alternate_Approach, or Variation_of_Question.",
        "schema": A2_DEEP_SM_SCHEMA,
    },
}


class ForumProcessor:
    """Processes SAT forum posts through the multi-agent system with image transcription"""

    def __init__(self):
        """Initialize the forum processor with all clients"""
        # Load API configurations
        anthropic_config = config.get_anthropic_config()
        airtable_config = config.get_airtable_config()

        # Initialize Claude client with Opus 4.5
        self.claude_client = ClaudeClient(
            api_key=anthropic_config.get('api_key', ''),
            model=anthropic_config.get('model', 'claude-opus-4-5-20251101'),
            max_tokens=anthropic_config.get('max_tokens', 20000),
            thinking_budget=anthropic_config.get('thinking_budget', 6000)
        )

        # Initialize Content Processor (for image transcription)
        self.content_processor = ForumContentProcessor(self.claude_client)

        # Initialize Airtable client
        self.airtable_client = AirtableClient(
            api_key=airtable_config.get('api_key', ''),
            base_id=airtable_config.get('base_id', ''),
            table_name=airtable_config.get('table_name', ''),
            agent_outputs_table=airtable_config.get('agent_outputs_table', '')
        )

        # Initialize Forum Post client
        forum_post_config = config.get_forum_post_api_config()
        self.forum_post_client = ForumPostClient(
            url=forum_post_config.get('url', ''),
            api_key=forum_post_config.get('api_key', '')
        ) if forum_post_config.get('url') else None

        # Initialize Teams Notification client
        teams_config = config.get_teams_notification_config()
        self.teams_client = TeamsNotificationClient(
            webhook_url=teams_config.get('webhook_url', ''),
            chat_id=teams_config.get('chat_id', '')
        ) if teams_config.get('webhook_url') else None

        # Load prompts
        self.prompts = self._load_prompts()

        logger.info("SAT Forum Processor initialized")

    def _load_prompts(self) -> Dict[str, str]:
        """Load all prompt files"""
        prompts = {}
        prompt_dir = config.PROMPTS_DIR

        # SAT-specific prompt file mappings
        prompt_files = {
            "a1_triage": "SAT - Classification - A1 Triage Classifier.txt",
            "a2_deep_sm": "SAT - Classification - A2 Deep SM Classifier.txt",
            "tool_3": "SAT - Genuine Doubt.txt",
            "tool_4": "SAT - Point Out Corrections.txt",
            "tool_4_validator": "SAT - Point Out Corrections - Validator.txt",
            "tool_4_responder": "SAT - Point Out Corrections - Responder.txt",
            "tool_5": "SAT - Variation of Question.txt",
            "tool_6": "SAT - AlternateVsSimilar.txt",
            "tool_7": "SAT - Response Formatter.txt",
            "html_formatter": "html formatter Prompt.txt"
        }

        for key, filename in prompt_files.items():
            filepath = prompt_dir / filename
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    prompts[key] = f.read()
                logger.info(f"Loaded prompt: {key}")
            else:
                logger.warning(f"Prompt file not found: {filename}")

        return prompts

    def _prepare_user_prompt(self, forum_data: Dict[str, Any]) -> str:
        """Prepare user prompt from forum data (with transcribed images)"""
        question_data_raw = forum_data.get("questionDataVO", {})

        # Handle both dict and list formats
        if isinstance(question_data_raw, list):
            question_data = question_data_raw[0] if question_data_raw else {}
            all_questions = question_data_raw
        else:
            question_data = question_data_raw
            all_questions = [question_data] if question_data else []

        passage_data = forum_data.get("passageDataVO")

        # Get forum post text (SAT uses ForumPostText with capital F sometimes)
        forum_post_text = forum_data.get("forumPostText", "") or forum_data.get("ForumPostText", "")

        # Build main prompt using SAT-specific XML tags
        prompt = f"""
<SAT_Question>
{question_data.get("questionText", "")}

{question_data.get("questionStem", "")}
</SAT_Question>

<SAT_Solution>
{question_data.get("generalFeedback", "")}
</SAT_Solution>

<Student_Doubt>
Subject: {forum_data.get("forumPostSubject", "")}

{forum_post_text}
</Student_Doubt>
"""

        # Add passage if present
        if passage_data:
            if isinstance(passage_data, dict):
                passage_text = passage_data.get("PassageTabListString", "") or passage_data.get("passageText", "")
            else:
                passage_text = str(passage_data)
            if passage_text:
                prompt = f"<Passage>{passage_text}</Passage>\n\n" + prompt

        # Note about multiple questions
        if len(all_questions) > 1:
            prompt = f"<Note>This passage has {len(all_questions)} questions total</Note>\n\n" + prompt

        # Add transcriptions from base64EncodedImages array
        if forum_data.get("_base64_transcriptions"):
            transcriptions = forum_data["_base64_transcriptions"]
            trans_text = "\n".join([
                f"[Image {t['index']+1} Transcription: {t['transcription']}]"
                for t in transcriptions
            ])
            prompt += f"\n\n<Attached_Images>\n{trans_text}\n</Attached_Images>"

        # Add previous exchange if present
        if forum_data.get("parentPostQuery"):
            prompt += f"""
<previous_exchange>
<original_doubt>
{forum_data.get("parentPostQuery")}
</original_doubt>

<expert_response>
{forum_data.get("parentPostResponse")}
</expert_response>
</previous_exchange>
"""

        return prompt

    def _run_classifier(self, prompt_key: str, forum_data: Dict[str, Any],
                        correlation_id: str, sequence: int) -> Optional[Dict[str, Any]]:
        """Run a classifier agent using structured output (tool_use) for guaranteed valid JSON."""
        logger.info(f"Running {prompt_key}...")

        user_prompt = self._prepare_user_prompt(forum_data)

        # Use structured output if schema is available for this classifier
        schema_config = _CLASSIFIER_SCHEMAS.get(prompt_key)
        if schema_config:
            result = self.claude_client.call_agent_structured(
                system_prompt=self.prompts[prompt_key],
                user_prompt=user_prompt,
                tool_name=schema_config["tool_name"],
                tool_description=schema_config["tool_description"],
                tool_schema=schema_config["schema"],
            )
            if result:
                parsed = result.get("parsed")
            else:
                parsed = None
        else:
            # Fallback to text-based parsing for unknown classifiers
            result = self.claude_client.call_agent(
                system_prompt=self.prompts[prompt_key],
                user_prompt=user_prompt
            )
            if result:
                parsed = self.claude_client.parse_json_response(result["response"])
            else:
                parsed = None

        if result and parsed:
                classification = parsed.get('classification')
                logger.info(f"{prompt_key} Classification: {classification}")

                # Log full response details
                logger.info(f"[{correlation_id}] {prompt_key} FULL RESPONSE:")
                logger.info(f"  Classification: {classification}")
                if parsed.get('justification'):
                    justification = parsed.get('justification', {})
                    logger.info(f"  Primary Intent: {justification.get('primary_intent', 'N/A')}")
                    logger.info(f"  Key Indicators: {justification.get('key_indicators', [])}")
                    logger.info(f"  Reasoning: {justification.get('classification_reasoning', 'N/A')}")
                    logger.info(f"  Decision Path: {justification.get('decision_path', 'N/A')}")
                if parsed.get('confidence'):
                    logger.info(f"  Confidence: {parsed.get('confidence')}")

                tool_output = {
                    "correlation_id": correlation_id,
                    "tool_name": prompt_key.upper(),
                    "tool_sequence": sequence,
                    "tool_output": parsed,
                    "execution_status": "success",
                    "execution_time_ms": result.get("execution_time_ms"),
                    "classification_result": classification,
                    "exception_flag": False
                }

                return {"raw": result, "parsed": parsed, "tool_output": tool_output}

        return None

    def _run_specialized_tool(self, tool_key: str, forum_data: Dict[str, Any],
                              correlation_id: str, sequence: int) -> Optional[Dict[str, Any]]:
        """Run specialized response tool"""
        logger.info(f"Running {tool_key}...")

        user_prompt = self._prepare_user_prompt(forum_data)
        result = self.claude_client.call_agent(
            system_prompt=self.prompts[tool_key],
            user_prompt=user_prompt
        )

        if result:
            parsed = self.claude_client.parse_json_response(result["response"])

            # Determine exception flag
            exception_flag = False
            if parsed:
                exception_flag = (
                    parsed.get("Exception_Flag") == "Yes" or
                    parsed.get("exception_flag") == True or
                    parsed.get("exception_flag") == "Yes"
                )

            # Log full response details
            logger.info(f"[{correlation_id}] {tool_key} FULL RESPONSE:")
            if parsed:
                logger.info(f"  Exception Flag: {exception_flag}")
                if exception_flag:
                    logger.warning(f"  *** HIL EXCEPTION TRIGGERED ***")
                    logger.warning(f"  Exception Reason: {parsed.get('Exception_Reason', parsed.get('exception_reason', 'N/A'))}")
                    logger.warning(f"  Exception Details: {parsed.get('exception_details', 'N/A')}")

                # Log validation result if present (for Pointing_Out_Corrections)
                if parsed.get('validation_result'):
                    val_result = parsed['validation_result']
                    logger.info(f"  Validation Classification: {val_result.get('classification', 'N/A')}")
                    logger.info(f"  Validation Explanation: {val_result.get('explanation', 'N/A')}")

                # Log metadata if present
                if parsed.get('metadata'):
                    metadata = parsed['metadata']
                    logger.info(f"  Metadata: {json.dumps(metadata, indent=4)}")
                    if metadata.get('hil_escalation'):
                        logger.warning(f"  *** HIL ESCALATION FROM METADATA ***")
                        logger.warning(f"  HIL Reason: {metadata.get('hil_reason', 'N/A')}")

                # Log response summary
                response_text = parsed.get('response', parsed.get('Response', ''))
                if isinstance(response_text, dict):
                    logger.info(f"  Response Keys: {list(response_text.keys())}")
                elif response_text:
                    logger.info(f"  Response Preview: {str(response_text)[:500]}...")

                # Log any analysis or reasoning
                if parsed.get('analysis'):
                    logger.info(f"  Analysis: {str(parsed.get('analysis'))[:500]}...")
                if parsed.get('reasoning'):
                    logger.info(f"  Reasoning: {str(parsed.get('reasoning'))[:500]}...")

            tool_output = {
                "correlation_id": correlation_id,
                "tool_name": tool_key.upper(),
                "tool_sequence": sequence,
                "tool_output": parsed,
                "execution_status": "success",
                "execution_time_ms": result.get("execution_time_ms"),
                "classification_result": None,
                "exception_flag": exception_flag
            }

            return {"raw": result, "parsed": parsed, "tool_output": tool_output}

        return None

    def _run_tool_4_two_step(self, forum_data: Dict[str, Any],
                              correlation_id: str, sequence: int) -> Optional[Dict[str, Any]]:
        """
        Two-call processing for Pointing_Out_Corrections:
          Call 1: Validation (Agent 1) -> validation_result, analysis, response_guidance
          Call 2: Response generation (Agent 2) -> response_html, metadata
        """
        user_prompt = self._prepare_user_prompt(forum_data)

        # --- Call 1: Validator ---
        logger.info(f"[{correlation_id}] Running tool_4 validator (Agent 1)...")
        result1 = self.claude_client.call_agent(
            system_prompt=self.prompts["tool_4_validator"],
            user_prompt=user_prompt
        )

        if not result1:
            logger.error(f"[{correlation_id}] tool_4 validator call failed")
            return None

        parsed1 = self.claude_client.parse_json_response(result1["response"])
        if not parsed1:
            logger.error(f"[{correlation_id}] tool_4 validator JSON parse failed")
            return None

        # Log Agent 1 output
        val_result = parsed1.get("validation_result", {})
        logger.info(f"[{correlation_id}] tool_4 Validator result:")
        logger.info(f"  Classification: {val_result.get('classification', 'N/A')}")
        logger.info(f"  Confidence: {val_result.get('confidence_level', 'N/A')}")
        logger.info(f"  Error Type: {val_result.get('error_type', 'N/A')}")
        logger.info(f"  HIL Flag: {val_result.get('HIL_flag', 'N/A')}")

        # Check HIL_flag - if VALID, skip Agent 2 (will go to HIL anyway)
        hil_flag = val_result.get("HIL_flag")
        is_hil = hil_flag is True or str(hil_flag).lower() == "true"

        if is_hil:
            logger.info(f"[{correlation_id}] VALID error detected (HIL_flag=true) — proceeding to Agent 2 for acknowledgment response")

        # --- Call 2: Responder ---
        logger.info(f"[{correlation_id}] Running tool_4 responder (Agent 2)...")
        responder_prompt = (
            f"<agent1_output>\n{json.dumps(parsed1, indent=2)}\n</agent1_output>\n\n"
            f"{user_prompt}"
        )

        result2 = self.claude_client.call_agent(
            system_prompt=self.prompts["tool_4_responder"],
            user_prompt=responder_prompt
        )

        if result2:
            parsed2 = self.claude_client.parse_json_response(result2["response"])
            if parsed2:
                # Merge Agent 1 + Agent 2 outputs
                merged = {**parsed1, **parsed2}
                logger.info(f"[{correlation_id}] tool_4 Responder result:")
                logger.info(f"  response_html present: {bool(merged.get('response_html'))}")
                logger.info(f"  metadata: {json.dumps(merged.get('metadata', {}), indent=2)}")

                # Compute total execution time
                total_time = (result1.get("execution_time_ms", 0) or 0) + (result2.get("execution_time_ms", 0) or 0)

                tool_output = {
                    "correlation_id": correlation_id,
                    "tool_name": "TOOL_4_TWO_STEP",
                    "tool_sequence": sequence,
                    "tool_output": merged,
                    "execution_status": "success",
                    "execution_time_ms": total_time,
                    "classification_result": val_result.get("classification"),
                    "exception_flag": False
                }
                return {"raw": result2, "parsed": merged, "tool_output": tool_output}

        # Fall back to Agent 1 only if Agent 2 fails
        logger.warning(f"[{correlation_id}] Agent 2 (Responder) failed — falling back to Agent 1 only")
        tool_output = {
            "correlation_id": correlation_id,
            "tool_name": "TOOL_4_VALIDATOR",
            "tool_sequence": sequence,
            "tool_output": parsed1,
            "execution_status": "success",
            "execution_time_ms": result1.get("execution_time_ms"),
            "classification_result": val_result.get("classification"),
            "exception_flag": False
        }
        return {"raw": result1, "parsed": parsed1, "tool_output": tool_output}

    def _format_to_html(self, expert_reply: str, correlation_id: str, sequence: int) -> Optional[Dict[str, Any]]:
        """Format expert reply to clean HTML"""
        logger.info("Formatting expert reply to HTML...")

        try:
            result = self.claude_client.call_agent(
                system_prompt=self.prompts["html_formatter"],
                user_prompt=expert_reply
            )

            if result:
                parsed = self.claude_client.parse_json_response(result["response"])
                if parsed and "formatted_html" in parsed:
                    logger.info("HTML formatting completed")

                    tool_output = {
                        "correlation_id": correlation_id,
                        "tool_name": "HTML_FORMATTER",
                        "tool_sequence": sequence,
                        "tool_output": parsed,
                        "execution_status": "success",
                        "execution_time_ms": result.get("execution_time_ms"),
                        "classification_result": None,
                        "exception_flag": False
                    }

                    return {
                        "formatted_html": parsed["formatted_html"],
                        "tool_output": tool_output
                    }

            return None

        except Exception as e:
            logger.error(f"Error formatting HTML: {e}")
            return None

    def process_forum_post(self, forum_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single forum post through the agent system"""
        correlation_id = forum_data.get("correlationId") or forum_data.get("Forum_Corr_ID")
        logger.info(f"Processing SAT forum post: {correlation_id}")

        results = {
            "correlation_id": correlation_id,
            "forum_data": forum_data,
            "image_processing_stats": None,
            "a1_result": None,
            "a2_result": None,
            "tool_result": None,
            "final_response": None,
            "final_response_html": None,
            "hil_flag": False,
            "processing_status": "pending",
            "tool_outputs": [],
            "url_check": False,
            "urls_list": []
        }

        try:
            # Step 0a: Check for URLs
            has_urls, detected_urls = url_detector.check_forum_data(forum_data)
            results["url_check"] = has_urls
            results["urls_list"] = detected_urls

            if has_urls:
                logger.warning(f"URL(s) detected in forum post - skipping processing. URLs: {detected_urls}")
                results["processing_status"] = "url_detected"
                return results

            # Step 0b: Process all images (URLs and base64) in all fields
            logger.info("Processing images in forum data...")
            processed_forum_data = self.content_processor.process_forum_data(forum_data)
            results["image_processing_stats"] = self.content_processor.get_processing_stats()

            # Use processed data for rest of pipeline
            forum_data = processed_forum_data

            # Step 1: A1 Triage Classifier
            a1_result = self._run_classifier("a1_triage", forum_data, correlation_id, sequence=1)
            if not a1_result:
                results["processing_status"] = "error"
                return results

            results["a1_result"] = a1_result
            if "tool_output" in a1_result:
                results["tool_outputs"].append(a1_result["tool_output"])

            a1_classification = a1_result["parsed"].get("classification")

            # Check if non-SM doubt (SAT uses SM_Doubt, SAT_Strategy_Doubt, Unrelated_to_SAT, Gratitude)
            if a1_classification != "SM_Doubt":
                logger.info(f"Non-SM doubt detected: {a1_classification} - Triggering HIL")
                results["hil_flag"] = True
                results["processing_status"] = "hil_exception"
                return results

            # Step 2: A2 Deep SM Classifier
            a2_result = self._run_classifier("a2_deep_sm", forum_data, correlation_id, sequence=2)
            if not a2_result:
                results["processing_status"] = "error"
                return results

            results["a2_result"] = a2_result
            if "tool_output" in a2_result:
                results["tool_outputs"].append(a2_result["tool_output"])

            a2_classification = a2_result["parsed"].get("classification")

            # Route to specialized tool
            tool_mapping = {
                "Genuine_Doubt": "tool_3",
                "Pointing_Out_Corrections": "tool_4",
                "Variation_of_Question": "tool_5",
                "Alternate_Approach": "tool_6"
            }

            tool_key = tool_mapping.get(a2_classification)
            if not tool_key:
                logger.error(f"Unknown classification: {a2_classification}")
                results["processing_status"] = "error"
                return results

            # Step 3: Run specialized tool (tool_4 uses two-step pattern)
            if tool_key == "tool_4":
                tool_result = self._run_tool_4_two_step(forum_data, correlation_id, sequence=3)
            else:
                tool_result = self._run_specialized_tool(tool_key, forum_data, correlation_id, sequence=3)
            if not tool_result:
                results["processing_status"] = "error"
                return results

            results["tool_result"] = tool_result
            if "tool_output" in tool_result:
                results["tool_outputs"].append(tool_result["tool_output"])

            # Extract final response
            parsed_response = tool_result["parsed"]

            if isinstance(parsed_response.get("response"), dict):
                response_obj = parsed_response["response"]
                response_parts = []
                # Try structured keys (all possible subkeys across response types)
                for key in ["greeting", "main_response", "worked_solution", "corrections_needed",
                             "conceptual_gaps", "next_steps", "comparison_to_official",
                             "clarifications_needed", "closing"]:
                    if key in response_obj and response_obj[key]:
                        response_parts.append(response_obj[key])
                # Fallback: try "content" key (SAT format)
                if not response_parts and response_obj.get("content"):
                    response_parts.append(response_obj["content"])
                # Last resort: join all string values from the dict
                if not response_parts:
                    for v in response_obj.values():
                        if isinstance(v, str) and len(v) > 20:
                            response_parts.append(v)
                results["final_response"] = "\n\n".join(response_parts)
                if not results["final_response"]:
                    logger.warning(f"[{correlation_id}] Could not extract response from dict keys: {list(response_obj.keys())}")
            elif parsed_response.get("response_html"):
                results["final_response"] = parsed_response.get("response_html")
                results["final_response_html"] = parsed_response.get("response_html")
                logger.info("Response already in HTML format from tool")
            else:
                results["final_response"] = parsed_response.get("Response", parsed_response.get("response", ""))

            # Step 4: Format response to HTML (skip if already HTML)
            if results["final_response"] and not results.get("final_response_html"):
                html_result = self._format_to_html(results["final_response"], correlation_id, sequence=4)
                if html_result:
                    results["final_response_html"] = html_result["formatted_html"]
                    if "tool_output" in html_result:
                        results["tool_outputs"].append(html_result["tool_output"])
                else:
                    logger.warning("HTML formatting failed, using plain text")
                    results["final_response_html"] = results["final_response"]
            elif not results["final_response"]:
                results["final_response_html"] = ""

            # Check for HIL flags
            hil_flag = (
                parsed_response.get("Exception_Flag") == "Yes" or
                parsed_response.get("exception_flag") == True or
                parsed_response.get("exception_flag") == "Yes" or
                (parsed_response.get("metadata", {}).get("hil_escalation") == True) or
                (parsed_response.get("metadata", {}).get("hil_escalation") == "true")
            )
            if hil_flag:
                results["hil_flag"] = True
                results["processing_status"] = "hil_exception"
                logger.warning(f"[{correlation_id}] ========== HIL EXCEPTION SUMMARY ==========")
                logger.warning(f"[{correlation_id}] A1 Classification: {results.get('a1_result', {}).get('parsed', {}).get('classification', 'N/A')}")
                logger.warning(f"[{correlation_id}] A2 Classification: {a2_classification}")
                logger.warning(f"[{correlation_id}] Exception_Flag: {parsed_response.get('Exception_Flag', parsed_response.get('exception_flag', 'N/A'))}")
                logger.warning(f"[{correlation_id}] Exception_Reason: {parsed_response.get('Exception_Reason', parsed_response.get('exception_reason', 'N/A'))}")
                logger.warning(f"[{correlation_id}] Metadata HIL: {parsed_response.get('metadata', {}).get('hil_escalation', 'N/A')}")
                logger.warning(f"[{correlation_id}] Metadata HIL Reason: {parsed_response.get('metadata', {}).get('hil_reason', 'N/A')}")
                logger.warning(f"[{correlation_id}] ============================================")
            else:
                results["processing_status"] = "completed"

            # Override for Pointing_Out_Corrections acknowledgment cases
            # Allow VALID/PARTIALLY_VALID/AMBIGUOUS to post an acknowledgment response
            if (a2_classification == "Pointing_Out_Corrections" and results.get("final_response_html")):
                tool_parsed = results.get("tool_result", {}).get("parsed", {})
                val_class = tool_parsed.get("validation_result", {}).get("classification", "").upper()
                if val_class in ("VALID", "PARTIALLY_VALID", "AMBIGUOUS"):
                    logger.info(f"[{correlation_id}] Pointing_Out_Corrections {val_class} — overriding to post acknowledgment response")
                    results["hil_acknowledgment"] = True
                    results["processing_status"] = "completed"

            logger.info(f"[{correlation_id}] Processing completed: {results['processing_status']}")
            return results

        except Exception as e:
            logger.error(f"Error processing forum post: {e}")
            results["processing_status"] = "error"
            return results

    def _extract_course_name(self, forum_data: Dict[str, Any]) -> Optional[str]:
        """Extract course name from questionDataVO"""
        question_data_vo = forum_data.get("questionDataVO", {})
        if isinstance(question_data_vo, list):
            return question_data_vo[0].get("courseName") if question_data_vo else None
        else:
            return question_data_vo.get("courseName")

    def _should_post_to_forum(self, results: Dict[str, Any]) -> bool:
        """
        Determine if the response should be posted to the forum.

        Posting Rules:
        - Genuine_Doubt: Always post
        - Pointing_Out_Corrections: Only post if validation_result.classification == "INVALID"
        - Variation_of_Question: Always post
        - Alternate_Approach: Always post
        - HIL exceptions: Never post
        - Errors: Never post
        """
        if results.get("processing_status") != "completed":
            return False

        a2_classification = None
        if results.get("a2_result") and results["a2_result"].get("parsed"):
            a2_classification = results["a2_result"]["parsed"].get("classification")

        if not a2_classification:
            logger.warning("No A2 classification found, skipping forum post")
            return False

        # For Pointing_Out_Corrections, check the validation result
        if a2_classification == "Pointing_Out_Corrections":
            tool_result = results.get("tool_result", {})
            parsed = tool_result.get("parsed", {})
            validation_result = parsed.get("validation_result", {})
            validation_classification = validation_result.get("classification", "").upper()

            if validation_classification == "INVALID":
                logger.info(f"Pointing_Out_Corrections with INVALID classification - will post")
                return True
            elif validation_classification in ("VALID", "PARTIALLY_VALID", "AMBIGUOUS"):
                logger.info(f"Pointing_Out_Corrections with {validation_classification} classification - will post acknowledgment")
                return True
            else:
                logger.info(f"Pointing_Out_Corrections with {validation_classification} classification - will NOT post")
                return False

        return True

    def _extract_sub_classification(self, classification, results):
        """
        Extract sub-classification details based on the tool used.
        Returns a JSON string with relevant sub-classification fields.
        """
        if not results.get("tool_result") or not results["tool_result"].get("parsed"):
            return None

        parsed = results["tool_result"]["parsed"]
        sub_class_data = {}

        try:
            if classification == "Pointing_Out_Corrections":
                validation_result = parsed.get("validation_result", {})
                if validation_result:
                    sub_class_data = {
                        "validation_classification": validation_result.get("classification"),
                        "error_type": validation_result.get("error_type"),
                        "confidence_level": validation_result.get("confidence_level"),
                        "HIL_flag": validation_result.get("HIL_flag")
                    }

            elif classification == "Variation_of_Question":
                sub_class_data = {
                    "interaction_type": parsed.get("interaction_type"),
                    "exception_type": parsed.get("exception_type"),
                    "followup_type": parsed.get("followup_type")
                }
                metadata = parsed.get("metadata", {})
                if metadata:
                    sub_class_data["complexity_comparison"] = metadata.get("complexity_comparison")
                    sub_class_data["classification_confidence"] = metadata.get("classification_confidence")

            elif classification == "Alternate_Approach":
                sub_class_data = {
                    "routing_classification": parsed.get("classification"),
                    "approach_status": parsed.get("approach_status"),
                    "understanding_status": parsed.get("understanding_status"),
                    "mistake_type": parsed.get("mistake_type")
                }

            elif classification == "Genuine_Doubt":
                if parsed.get("Exception_Flag") or parsed.get("exception_flag"):
                    sub_class_data = {
                        "exception_flag": parsed.get("Exception_Flag") or parsed.get("exception_flag"),
                        "exception_reason": parsed.get("Exception_Reason") or parsed.get("exception_reason")
                    }

            # Filter out None values and return JSON
            sub_class_data = {k: v for k, v in sub_class_data.items() if v is not None}
            return json.dumps(sub_class_data, ensure_ascii=False) if sub_class_data else None

        except Exception as e:
            logger.error(f"Error extracting sub-classification: {e}")
            return None

    def _ai_quality_check(self, html_content, forum_data, correlation_id):
        """
        Perform AI-powered quality check on the response using Claude.
        Returns dict with ai_score (0-100), ai_feedback, ai_issues, ai_recommendation
        """
        logger.info(f"[{correlation_id}] Running AI quality check...")

        quality_check_prompt = """You are a quality assurance reviewer for student-facing educational content on a SAT preparation forum.

Review the following HTML response that will be shown to a student. Evaluate it on these criteria:

1. **Coherence (25 points)**: Is the response logically structured and easy to follow?
2. **Completeness (25 points)**: Does it fully address a student's doubt about SAT content?
3. **Tone (20 points)**: Is it professional, encouraging, and educational?
4. **Accuracy (20 points)**: Are there any obvious mathematical, logical, or factual errors?
5. **Formatting (10 points)**: Is the HTML well-formatted and readable?

Also check for these issues:
- Any placeholder text or incomplete sections
- Internal system markers that shouldn't be visible
- Inappropriate or confusing content
- Missing greeting or closing

Respond with ONLY a JSON object (no markdown, no explanation):
{
  "score": <0-100>,
  "coherence_score": <0-25>,
  "completeness_score": <0-25>,
  "tone_score": <0-20>,
  "accuracy_score": <0-20>,
  "formatting_score": <0-10>,
  "feedback": "<brief summary of quality assessment>",
  "issues": ["<list of specific issues found, or empty array if none>"],
  "recommendation": "<APPROVE|REVIEW|REJECT>"
}"""

        try:
            student_doubt = forum_data.get("forumPostText") or forum_data.get("ForumPostText") or ""

            user_prompt = f"""<student_doubt>
{student_doubt[:1000]}
</student_doubt>

<response_to_review>
{html_content[:8000]}
</response_to_review>

Review this response and provide your quality assessment."""

            result = self.claude_client.call_agent(
                system_prompt=quality_check_prompt,
                user_prompt=user_prompt
            )

            if result and result.get("response"):
                parsed = self.claude_client.parse_json_response(result["response"])
                if parsed:
                    ai_score = parsed.get("score", 0)
                    logger.info(
                        f"[{correlation_id}] AI Quality Check: Score={ai_score}/100, "
                        f"Recommendation={parsed.get('recommendation', 'N/A')}"
                    )

                    if parsed.get("issues"):
                        logger.info(f"[{correlation_id}] AI Quality Issues: {parsed['issues']}")

                    return {
                        "ai_score": ai_score,
                        "ai_feedback": parsed.get("feedback", ""),
                        "ai_issues": parsed.get("issues", []),
                        "ai_recommendation": parsed.get("recommendation", "REVIEW"),
                        "ai_breakdown": {
                            "coherence": parsed.get("coherence_score", 0),
                            "completeness": parsed.get("completeness_score", 0),
                            "tone": parsed.get("tone_score", 0),
                            "accuracy": parsed.get("accuracy_score", 0),
                            "formatting": parsed.get("formatting_score", 0)
                        }
                    }

            logger.warning(f"[{correlation_id}] AI quality check returned no valid response")
            return {"ai_score": 0, "ai_feedback": "AI check failed", "ai_issues": ["AI check failed"], "ai_recommendation": "REVIEW"}

        except Exception as e:
            logger.error(f"[{correlation_id}] AI quality check error: {e}")
            return {"ai_score": 0, "ai_feedback": f"Error: {e}", "ai_issues": [str(e)], "ai_recommendation": "REVIEW"}

    def _validate_and_score_html(self, html_content, forum_data, correlation_id):
        """
        Comprehensive HTML validation with quality scoring.
        Returns dict with is_valid, cleaned_html, quality_score, should_post, etc.
        """
        # Step 1: HTML structure validation
        html_result = html_validator.validate(html_content, correlation_id)

        # Step 2: AI-powered quality check
        ai_result = self._ai_quality_check(html_result.cleaned_html, forum_data, correlation_id)

        # Step 3: Calculate combined score (50% HTML validation, 50% AI score)
        combined_score = int((html_result.quality_score * 0.5) + (ai_result.get("ai_score", 0) * 0.5))

        # Step 4: Determine if should post (threshold: 85+)
        should_post = (
            html_result.is_valid and
            combined_score >= 85 and
            ai_result.get("ai_recommendation") != "REJECT"
        )

        # Log comprehensive validation summary
        logger.info(
            f"[{correlation_id}] === VALIDATION SUMMARY ===\n"
            f"  HTML Score: {html_result.quality_score}/100\n"
            f"  AI Score: {ai_result.get('ai_score', 0)}/100\n"
            f"  Combined Score: {combined_score}/100\n"
            f"  AI Recommendation: {ai_result.get('ai_recommendation', 'N/A')}\n"
            f"  Should Post: {should_post} (threshold: 85+)\n"
            f"  HTML Errors: {len(html_result.errors)}\n"
            f"  HTML Warnings: {len(html_result.warnings)}\n"
            f"  Auto-fixes Applied: {len(html_result.auto_fixes)}"
        )

        return {
            "is_valid": html_result.is_valid,
            "cleaned_html": html_result.cleaned_html,
            "quality_score": combined_score,
            "html_score": html_result.quality_score,
            "ai_score": ai_result.get("ai_score", 0),
            "should_post": should_post,
            "html_validation": {
                "structure_score": html_result.structure_score,
                "security_score": html_result.security_score,
                "content_score": html_result.content_score,
                "formatting_score": html_result.formatting_score,
                "errors": html_result.errors,
                "warnings": html_result.warnings,
                "auto_fixes": html_result.auto_fixes
            },
            "ai_quality": ai_result
        }

    def save_results(self, results: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        """Save results to Airtable and post to forum"""
        save_status = {
            "airtable_saved": False,
            "forum_post_status": None,
            "forum_post_error": None,
            "teams_notified": False
        }

        try:
            forum_data = results["forum_data"]
            images_transcribed = results.get("image_processing_stats", {}).get("total_images", 0)

            if results.get("processing_status") == "error":
                logger.warning("Processing failed, skipping save to Airtable")
                if self.teams_client:
                    try:
                        teams_result = self.teams_client.send_processing_notification(
                            correlation_id=results["correlation_id"],
                            status="error",
                            forum_post_status=None,
                            posted_by_email=forum_data.get("postedBy") or forum_data.get("parentPostedBy"),
                            classification=None,
                            error_message="Processing failed",
                            images_transcribed=images_transcribed
                        )
                        if teams_result.get("success"):
                            save_status["teams_notified"] = True
                    except Exception as te:
                        logger.error(f"Error sending Teams notification: {te}")
                return save_status

            final_response = results.get("final_response", "")
            if isinstance(final_response, dict):
                final_response = json.dumps(final_response, ensure_ascii=False)

            final_response_html = results.get("final_response_html", final_response)
            if isinstance(final_response_html, dict):
                final_response_html = json.dumps(final_response_html, ensure_ascii=False)

            # Validate and score HTML before posting
            validation_result = None
            quality_score = 0
            validation_details = None

            if final_response_html:
                validation_result = self._validate_and_score_html(
                    final_response_html,
                    forum_data,
                    results["correlation_id"]
                )
                # Use cleaned HTML
                final_response_html = validation_result.get("cleaned_html", final_response_html)
                quality_score = validation_result.get("quality_score", 0)
                validation_details = json.dumps({
                    "html_score": validation_result.get("html_score"),
                    "ai_score": validation_result.get("ai_score"),
                    "combined_score": quality_score,
                    "should_post": validation_result.get("should_post"),
                    "html_validation": validation_result.get("html_validation"),
                    "ai_quality": validation_result.get("ai_quality")
                }, ensure_ascii=False)

            classification = None
            if results.get("a2_result") and results["a2_result"].get("parsed"):
                classification = results["a2_result"]["parsed"].get("classification")

            # Extract sub-classification
            sub_classification = self._extract_sub_classification(classification, results)

            classification_justification = {}
            if results.get("a2_result") and results["a2_result"].get("parsed"):
                classification_justification = results["a2_result"]["parsed"].copy()
            if results.get("image_processing_stats"):
                classification_justification["image_processing"] = results["image_processing_stats"]

            course_name = self._extract_course_name(forum_data)

            url_check = results.get("url_check", False)
            urls_list = results.get("urls_list", [])
            urls_list_json = json.dumps(urls_list) if urls_list else ""

            airtable_data = {
                "correlation_id": results["correlation_id"],
                "posted_by": forum_data.get("postedBy"),
                "parent_posted_by": forum_data.get("parentPostedBy"),
                "forum_post_subject": forum_data.get("forumPostSubject"),
                "forum_post_text": forum_data.get("forumPostText") or forum_data.get("ForumPostText"),
                "image_base64_encoded": str(forum_data.get("isImageBase64Encoded", False)),
                "parent_post_query": forum_data.get("parentPostQuery"),
                "parent_post_response": forum_data.get("parentPostResponse"),
                "entity_id": str(forum_data.get("entityId", "")) if forum_data.get("entityId") else None,
                "entity_name": forum_data.get("entityName"),
                "platform_name": forum_data.get("platformName"),
                "course_name": course_name,
                "forum_id": str(forum_data.get("forumId", "")) if forum_data.get("forumId") else None,
                "parent_id": str(forum_data.get("parentId") or forum_data.get("id") or ""),
                "post_type": forum_data.get("type"),
                "environment": forum_data.get("environment"),
                "classification": classification_justification.get("classification"),
                "sub_classification": sub_classification,
                "classification_justification": json.dumps(classification_justification),
                "expert_reply": final_response,
                "expert_reply_html": final_response_html,
                "url_check": str(url_check).lower(),
                "urls_list": urls_list_json,
                "images_transcribed": images_transcribed,
                "quality_score": quality_score,
                "validation_details": validation_details,
                "request_received_at": forum_data.get("requestReceivedAt", ""),
            }

            success = self.airtable_client.upsert_forum_response(airtable_data)
            if success:
                logger.info(f"Saved to Airtable (SAT Forum Posts): {results['correlation_id']}")
                save_status["airtable_saved"] = True
            else:
                logger.error("Failed to save to Airtable (SAT Forum Posts)")

            # Extract agent outputs as JSON strings
            a1_output = ""
            if results.get("a1_result") and results["a1_result"].get("parsed"):
                a1_output = json.dumps(results["a1_result"]["parsed"], ensure_ascii=False)

            a2_output = ""
            if results.get("a2_result") and results["a2_result"].get("parsed"):
                a2_output = json.dumps(results["a2_result"]["parsed"], ensure_ascii=False)

            tool_output = ""
            if results.get("tool_result") and results["tool_result"].get("parsed"):
                tool_output = json.dumps(results["tool_result"]["parsed"], ensure_ascii=False)

            # Save to Agent System Outputs table
            agent_outputs_data = {
                "correlation_id": results["correlation_id"],
                "urls_list": urls_list_json,
                "a1_triage_output": a1_output,
                "a2_classification_output": a2_output,
                "tool_response_output": tool_output
            }
            agent_success = self.airtable_client.upsert_agent_outputs(agent_outputs_data)
            if agent_success:
                logger.info(f"Saved to Airtable (Agent System Outputs): {results['correlation_id']}")
                save_status["agent_outputs_saved"] = True
            else:
                logger.error("Failed to save to Airtable (Agent System Outputs)")

            # Determine if we should post to forum (dual gate: classification + quality)
            should_post_classification = self._should_post_to_forum(results)

            # Check quality validation threshold (85+)
            quality_validated = validation_result.get("should_post", False) if validation_result else False

            # Combined posting decision: both gates must pass
            should_post = should_post_classification and quality_validated

            html_was_cleaned = False
            if dry_run:
                logger.info(
                    f"[DRY-RUN MODE] Skipping forum post for: {results['correlation_id']} | "
                    f"Quality Score: {quality_score}/100 | "
                    f"Would Post (classification): {should_post_classification} | "
                    f"Would Post (quality): {quality_validated} | "
                    f"Final Would Post: {should_post}"
                )
                save_status["forum_post_status"] = "skipped_dry_run"
            elif not quality_validated and should_post_classification:
                # Quality threshold not met - route to HIL
                logger.warning(
                    f"[{results['correlation_id']}] Quality score {quality_score}/100 below threshold (85+) - routing to HIL"
                )
                save_status["forum_post_status"] = "skipped_quality_hil"
                results["hil_flag"] = True
                results["processing_status"] = "hil_exception"
            elif (should_post and
                final_response_html and
                self.forum_post_client):

                logger.info(f"Posting response to forum for: {results['correlation_id']} (Quality: {quality_score}/100)")
                forum_result = self.forum_post_client.post_forum_response(
                    forum_data=forum_data,
                    html_response=final_response_html
                )

                if forum_result.get("success"):
                    logger.info(f"Successfully posted to forum: {results['correlation_id']}")
                    save_status["forum_post_status"] = "posted"
                    html_was_cleaned = forum_result.get("html_cleaned", False)
                    # Convert to posted_hil if this was an acknowledgment post
                    if results.get("hil_acknowledgment"):
                        save_status["forum_post_status"] = "posted_hil"
                        results["hil_flag"] = True
                else:
                    logger.error(f"Failed to post to forum: {forum_result.get('error')}")
                    save_status["forum_post_status"] = "failed"
                    save_status["forum_post_error"] = forum_result.get("error", "Unknown error")
            elif results.get("processing_status") == "hil_exception":
                save_status["forum_post_status"] = "skipped_hil"
            elif not should_post_classification and results.get("processing_status") == "completed":
                save_status["forum_post_status"] = "skipped_validation"
                logger.info(f"Skipping forum post - classification does not require posting: {results['correlation_id']}")
            else:
                save_status["forum_post_status"] = "skipped"

            # Send Teams notification
            if self.teams_client:
                try:
                    classification = None
                    if results.get("a2_result") and results["a2_result"].get("parsed"):
                        classification = results["a2_result"]["parsed"].get("classification")

                    error_msg = save_status.get("forum_post_error")

                    teams_result = self.teams_client.send_processing_notification(
                        correlation_id=results["correlation_id"],
                        status=results.get("processing_status", "unknown"),
                        forum_post_status=save_status.get("forum_post_status"),
                        posted_by_email=forum_data.get("postedBy") or forum_data.get("parentPostedBy"),
                        classification=classification,
                        error_message=error_msg,
                        html_cleaned=html_was_cleaned,
                        images_transcribed=images_transcribed
                    )

                    if teams_result.get("success"):
                        save_status["teams_notified"] = True
                        logger.info(f"Teams notification sent for: {results['correlation_id']}")
                    else:
                        logger.warning(f"Teams notification failed: {teams_result.get('error')}")
                except Exception as te:
                    logger.error(f"Error sending Teams notification: {te}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")

        save_status["quality_score"] = quality_score
        save_status["sub_classification"] = sub_classification
        if save_status.get("forum_post_status") == "skipped_quality_hil":
            save_status["hil_reason"] = "quality_below_threshold"
        elif save_status.get("forum_post_status") == "posted_hil":
            tool_parsed = results.get("tool_result", {}).get("parsed", {})
            val_class = tool_parsed.get("validation_result", {}).get("classification", "unknown").lower()
            save_status["hil_reason"] = f"hil_acknowledgment_{val_class}"
        else:
            save_status["hil_reason"] = None
        return save_status
