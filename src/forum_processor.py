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

logger = logging.getLogger(__name__)


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
            table_name=airtable_config.get('table_name', '')
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
            "tool_5": "SAT - Variation of Question.txt",
            "tool_6": "SAT - AlternateVsSimilar.txt",
            "tool_7": "SAT - Response Formatter.txt"
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
        """Run a classifier agent"""
        logger.info(f"Running {prompt_key}...")

        user_prompt = self._prepare_user_prompt(forum_data)
        result = self.claude_client.call_agent(
            system_prompt=self.prompts[prompt_key],
            user_prompt=user_prompt
        )

        if result:
            parsed = self.claude_client.parse_json_response(result["response"])
            if parsed:
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

                # Log response summary
                response_text = parsed.get('response', parsed.get('Response', ''))
                if isinstance(response_text, dict):
                    logger.info(f"  Response Keys: {list(response_text.keys())}")
                elif response_text:
                    logger.info(f"  Response Preview: {str(response_text)[:500]}...")

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

    def _format_to_html(self, expert_reply: str, correlation_id: str, sequence: int) -> Optional[Dict[str, Any]]:
        """Format expert reply to clean HTML using Tool 7"""
        logger.info("Formatting expert reply to HTML...")

        try:
            # Use the response formatter prompt (tool_7)
            result = self.claude_client.call_agent(
                system_prompt=self.prompts["tool_7"],
                user_prompt=expert_reply
            )

            if result:
                parsed = self.claude_client.parse_json_response(result["response"])
                if parsed and "response_html" in parsed:
                    logger.info("HTML formatting completed")

                    tool_output = {
                        "correlation_id": correlation_id,
                        "tool_name": "RESPONSE_FORMATTER",
                        "tool_sequence": sequence,
                        "tool_output": parsed,
                        "execution_status": "success",
                        "execution_time_ms": result.get("execution_time_ms"),
                        "classification_result": None,
                        "exception_flag": False
                    }

                    return {
                        "formatted_html": parsed["response_html"],
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

            # Step 3: Run specialized tool
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
                # SAT format: response.content contains the full response
                if "content" in response_obj and response_obj["content"]:
                    results["final_response"] = response_obj["content"]
                else:
                    # GMAT format: response has multiple parts
                    response_parts = []
                    for key in ["greeting", "main_response", "worked_solution", "comparison_to_official", "closing"]:
                        if key in response_obj and response_obj[key]:
                            response_parts.append(response_obj[key])
                    results["final_response"] = "\n\n".join(response_parts)
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
                logger.warning(f"[{correlation_id}] ============================================")
            else:
                results["processing_status"] = "completed"

            logger.info(f"[{correlation_id}] Processing completed: {results['processing_status']}")
            return results

        except Exception as e:
            logger.error(f"Error processing forum post: {e}")
            results["processing_status"] = "error"
            return results

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
            else:
                logger.info(f"Pointing_Out_Corrections with {validation_classification} classification - will NOT post")
                return False

        return True

    def save_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
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

            classification = None
            if results.get("a2_result") and results["a2_result"].get("parsed"):
                classification = results["a2_result"]["parsed"].get("classification")

            url_check = results.get("url_check", False)
            urls_list = results.get("urls_list", [])
            urls_list_json = json.dumps(urls_list) if urls_list else ""

            # SAT-specific Airtable data
            airtable_data = {
                "correlation_id": results["correlation_id"],
                "posted_by": forum_data.get("postedBy"),
                "forum_post_subject": forum_data.get("forumPostSubject"),
                "forum_post_text": forum_data.get("forumPostText") or forum_data.get("ForumPostText"),
                "image_base64_encoded": str(forum_data.get("isImageBase64Encoded", False)),
                "parent_post_query": forum_data.get("parentPostQuery"),
                "parent_post_response": forum_data.get("parentPostResponse"),
                "post_type": forum_data.get("type"),
                "environment": forum_data.get("environment"),
                "classification": classification,
                "expert_reply_html": final_response_html,
                "url_check": str(url_check).lower(),
                "urls_list": urls_list_json
            }

            success = self.airtable_client.upsert_forum_response(airtable_data)
            if success:
                logger.info(f"Saved to Airtable: {results['correlation_id']}")
                save_status["airtable_saved"] = True
            else:
                logger.error("Failed to save to Airtable")

            # Determine if we should post to forum
            should_post = self._should_post_to_forum(results)

            html_was_cleaned = False
            if (should_post and
                results.get("final_response_html") and
                self.forum_post_client):

                logger.info(f"Posting response to forum for: {results['correlation_id']}")
                forum_result = self.forum_post_client.post_forum_response(
                    forum_data=forum_data,
                    html_response=results["final_response_html"]
                )

                if forum_result.get("success"):
                    logger.info(f"Successfully posted to forum: {results['correlation_id']}")
                    save_status["forum_post_status"] = "posted"
                    html_was_cleaned = forum_result.get("html_cleaned", False)
                else:
                    logger.error(f"Failed to post to forum: {forum_result.get('error')}")
                    save_status["forum_post_status"] = "failed"
                    save_status["forum_post_error"] = forum_result.get("error", "Unknown error")
            elif results.get("processing_status") == "hil_exception":
                save_status["forum_post_status"] = "skipped_hil"
            elif not should_post and results.get("processing_status") == "completed":
                save_status["forum_post_status"] = "skipped_validation"
                logger.info(f"Skipping forum post - validation result does not require posting: {results['correlation_id']}")
            else:
                save_status["forum_post_status"] = "skipped"

            # Send Teams notification
            if self.teams_client:
                try:
                    teams_result = self.teams_client.send_processing_notification(
                        correlation_id=results["correlation_id"],
                        status=results.get("processing_status", "unknown"),
                        forum_post_status=save_status.get("forum_post_status"),
                        posted_by_email=forum_data.get("postedBy") or forum_data.get("parentPostedBy"),
                        classification=classification,
                        error_message=save_status.get("forum_post_error"),
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

        return save_status
