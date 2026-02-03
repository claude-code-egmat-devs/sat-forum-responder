"""
Content Processor Module for SAT Forum Responder
Processes forum data to detect and replace images with transcriptions across all relevant fields
"""

import logging
from typing import Dict, Any, List, Tuple

from .image_transcriber import ImageTranscriber, ContentImageProcessor

logger = logging.getLogger(__name__)


class ForumContentProcessor:
    """Processes forum data to transcribe all embedded images"""

    # Fields to scan for images in questionDataVO
    QUESTION_DATA_FIELDS = [
        "questionText",
        "generalFeedback",
        "questionStem",
        "questionImageTranscript",
        "feedbackImageTranscript",
        "feedbackVideoTranscript"
    ]

    PASSAGE_DATA_FIELDS = [
        "PassageTabListString",
        "passageText"
    ]

    TOP_LEVEL_FIELDS = [
        "forumPostText",
        "ForumPostText",  # SAT uses capital F
        "forumPostSubject",
        "parentPostQuery",
        "parentPostResponse"
    ]

    def __init__(self, claude_client):
        """
        Initialize ForumContentProcessor

        Args:
            claude_client: Instance of ClaudeClient with vision capability
        """
        self.transcriber = ImageTranscriber(claude_client)
        self.image_processor = ContentImageProcessor(self.transcriber)
        self.processing_stats = {
            "total_images": 0,
            "fields_processed": []
        }

    def process_forum_data(self, forum_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process forum data and transcribe all images in relevant fields

        Args:
            forum_data: Raw forum data dictionary

        Returns:
            Processed forum data with images replaced by transcriptions
        """
        # Reset stats
        self.processing_stats = {
            "total_images": 0,
            "fields_processed": []
        }

        # Create a copy to avoid modifying original
        processed_data = forum_data.copy()

        # Process top-level fields
        for field in self.TOP_LEVEL_FIELDS:
            if field in processed_data and processed_data[field]:
                processed_data[field], count = self._process_field(
                    processed_data[field], field
                )
                if count > 0:
                    self.processing_stats["fields_processed"].append(field)
                    self.processing_stats["total_images"] += count

        # Process questionDataVO
        processed_data = self._process_question_data(processed_data)

        # Process passageDataVO
        processed_data = self._process_passage_data(processed_data)

        # Process base64EncodedImages array
        processed_data = self._process_base64_images_array(processed_data)

        # Log summary
        if self.processing_stats["total_images"] > 0:
            logger.info(
                f"Image processing complete: {self.processing_stats['total_images']} images "
                f"transcribed from fields: {self.processing_stats['fields_processed']}"
            )
        else:
            logger.info("No images found to transcribe in forum data")

        return processed_data

    def _process_field(self, content: str, field_name: str) -> Tuple[str, int]:
        """Process a single field for images"""
        if not content or not isinstance(content, str):
            return content, 0

        return self.image_processor.process_content(content, field_name)

    def _process_question_data(self, forum_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process questionDataVO fields"""
        question_data_raw = forum_data.get("questionDataVO")
        if not question_data_raw:
            return forum_data

        # Handle both dict and list formats
        if isinstance(question_data_raw, list):
            question_data_list = question_data_raw
        else:
            question_data_list = [question_data_raw]

        for i, question_data in enumerate(question_data_list):
            if not isinstance(question_data, dict):
                continue

            for field in self.QUESTION_DATA_FIELDS:
                if field in question_data and question_data[field]:
                    field_path = f"questionDataVO[{i}].{field}" if len(question_data_list) > 1 else f"questionDataVO.{field}"
                    question_data[field], count = self._process_field(
                        question_data[field], field_path
                    )
                    if count > 0:
                        self.processing_stats["fields_processed"].append(field_path)
                        self.processing_stats["total_images"] += count

            # Process answer choices if present
            if "answerChoicesMap" in question_data:
                for j, choice in enumerate(question_data.get("answerChoicesMap", [])):
                    if isinstance(choice, dict):
                        for choice_field in ["answerContent", "answerFeedback"]:
                            if choice_field in choice and choice[choice_field]:
                                field_path = f"answerChoicesMap[{j}].{choice_field}"
                                choice[choice_field], count = self._process_field(
                                    choice[choice_field], field_path
                                )
                                if count > 0:
                                    self.processing_stats["fields_processed"].append(field_path)
                                    self.processing_stats["total_images"] += count

        # Update forum_data
        if isinstance(question_data_raw, list):
            forum_data["questionDataVO"] = question_data_list
        else:
            forum_data["questionDataVO"] = question_data_list[0] if question_data_list else {}

        return forum_data

    def _process_passage_data(self, forum_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process passageDataVO fields"""
        passage_data = forum_data.get("passageDataVO")
        if not passage_data:
            return forum_data

        if isinstance(passage_data, dict):
            for field in self.PASSAGE_DATA_FIELDS:
                if field in passage_data and passage_data[field]:
                    field_path = f"passageDataVO.{field}"
                    passage_data[field], count = self._process_field(
                        passage_data[field], field_path
                    )
                    if count > 0:
                        self.processing_stats["fields_processed"].append(field_path)
                        self.processing_stats["total_images"] += count

            forum_data["passageDataVO"] = passage_data

        elif isinstance(passage_data, str):
            # If passageDataVO is a string, process it directly
            forum_data["passageDataVO"], count = self._process_field(
                passage_data, "passageDataVO"
            )
            if count > 0:
                self.processing_stats["fields_processed"].append("passageDataVO")
                self.processing_stats["total_images"] += count

        return forum_data

    def _process_base64_images_array(self, forum_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the base64EncodedImages array
        These are standalone images attached to the forum post
        """
        base64_images = forum_data.get("base64EncodedImages", [])
        if not base64_images or not isinstance(base64_images, list):
            return forum_data

        transcriptions = []

        for i, image_item in enumerate(base64_images):
            if isinstance(image_item, dict):
                base64_data = image_item.get("encodedImage", "")
                extension = image_item.get("extension", "png")
            elif isinstance(image_item, str):
                base64_data = image_item
                extension = "png"
            else:
                continue

            if not base64_data:
                continue

            logger.info(f"Transcribing base64EncodedImages[{i}]")
            transcription = self.transcriber.transcribe_from_base64(base64_data, extension)

            if transcription:
                transcriptions.append({
                    "index": i,
                    "transcription": transcription,
                    "extension": extension
                })
                self.processing_stats["total_images"] += 1
                self.processing_stats["fields_processed"].append(f"base64EncodedImages[{i}]")

        # Store transcriptions for use in prompts
        if transcriptions:
            forum_data["_base64_transcriptions"] = transcriptions

        return forum_data

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics from the last processing run"""
        return self.processing_stats.copy()

    def clear_cache(self):
        """Clear the image transcription cache"""
        self.image_processor.clear_cache()
