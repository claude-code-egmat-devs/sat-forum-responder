"""
Image Transcriber Module for SAT Forum Responder
Handles transcription of images from both public URLs and base64 encoded data
"""

import re
import base64
import logging
import requests
from typing import Optional, Tuple
from io import BytesIO

logger = logging.getLogger(__name__)


class ImageTranscriber:
    """Transcribes images using Claude Vision API"""

    def __init__(self, claude_client):
        """
        Initialize ImageTranscriber

        Args:
            claude_client: Instance of ClaudeClient with vision capability
        """
        self.claude_client = claude_client
        self.system_prompt = """You are an expert at transcribing mathematical and educational content from images.
Your task is to accurately transcribe ALL content visible in the image including:
- Mathematical expressions and formulas (use LaTeX notation where appropriate)
- Text and paragraphs
- Tables and their contents
- Diagrams (describe them precisely)
- Graphs and charts (describe axes, data points, trends)
- Any labels, annotations, or captions

Be precise, detailed, and maintain the logical structure of the content."""

        self.user_prompt = "Please transcribe all content from this image accurately and completely."

    def transcribe_from_base64(self, base64_data: str, extension: str = "png") -> Optional[str]:
        """
        Transcribe image from base64 encoded data

        Args:
            base64_data: Base64 encoded image string (without data URI prefix)
            extension: Image extension (png, jpg, jpeg, gif, webp)

        Returns:
            Transcribed text or None on failure
        """
        try:
            # Clean base64 data - remove data URI prefix if present
            if "base64," in base64_data:
                base64_data = base64_data.split("base64,")[1]

            # Remove whitespace
            base64_data = base64_data.strip()

            # Map extension to media type
            media_type_map = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp"
            }
            media_type = media_type_map.get(extension.lower(), "image/png")

            logger.info(f"Transcribing base64 image (type: {media_type})")

            result = self.claude_client.call_agent_with_vision(
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                image_data=base64_data,
                media_type=media_type
            )

            if result and result.get("response"):
                logger.info(f"Base64 image transcription successful ({result.get('execution_time_ms')}ms)")
                return result["response"]

            logger.warning("Base64 image transcription returned empty response")
            return None

        except Exception as e:
            logger.error(f"Error transcribing base64 image: {e}")
            return None

    def transcribe_from_url(self, image_url: str) -> Optional[str]:
        """
        Download image from URL and transcribe it

        Args:
            image_url: Public URL of the image

        Returns:
            Transcribed text or None on failure
        """
        try:
            logger.info(f"Downloading image from URL: {image_url[:100]}...")

            # Download image
            response = requests.get(image_url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ImageTranscriber/1.0)"
            })
            response.raise_for_status()

            # Get content type
            content_type = response.headers.get("Content-Type", "image/png")

            # Map content type to extension
            type_to_ext = {
                "image/png": "png",
                "image/jpeg": "jpg",
                "image/jpg": "jpg",
                "image/gif": "gif",
                "image/webp": "webp"
            }

            # Extract base type (ignore charset etc)
            base_type = content_type.split(";")[0].strip().lower()
            extension = type_to_ext.get(base_type, "png")

            # Convert to base64
            image_base64 = base64.b64encode(response.content).decode("utf-8")

            logger.info(f"Image downloaded ({len(response.content)} bytes), transcribing...")

            # Transcribe using base64 method
            return self.transcribe_from_base64(image_base64, extension)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image from URL: {e}")
            return None
        except Exception as e:
            logger.error(f"Error transcribing image from URL: {e}")
            return None

    def extract_extension_from_url(self, url: str) -> str:
        """Extract image extension from URL"""
        url_lower = url.lower()
        for ext in ["png", "jpg", "jpeg", "gif", "webp", "svg"]:
            if f".{ext}" in url_lower:
                return ext
        return "png"  # Default


class ContentImageProcessor:
    """Processes content to detect and transcribe embedded images"""

    # Regex patterns for image detection
    BASE64_PATTERN = re.compile(
        r'(data:image/(png|jpeg|jpg|gif|webp);base64,([A-Za-z0-9+/=]+))',
        re.IGNORECASE
    )

    # Pattern for base64 in src attribute
    IMG_BASE64_PATTERN = re.compile(
        r'<img[^>]*src=["\']data:image/(png|jpeg|jpg|gif|webp);base64,([A-Za-z0-9+/=]+)["\'][^>]*>',
        re.IGNORECASE | re.DOTALL
    )

    # Pattern for image URLs in img tags
    IMG_URL_PATTERN = re.compile(
        r'<img[^>]*src=["\']((https?://[^"\']+\.(png|jpg|jpeg|gif|webp)))["\'][^>]*>',
        re.IGNORECASE | re.DOTALL
    )

    # Pattern for standalone image URLs (not in img tags)
    STANDALONE_URL_PATTERN = re.compile(
        r'(?<!["\'])(https?://[^\s<>"\']+\.(png|jpg|jpeg|gif|webp))(?!["\'])',
        re.IGNORECASE
    )

    def __init__(self, image_transcriber: ImageTranscriber):
        """
        Initialize ContentImageProcessor

        Args:
            image_transcriber: Instance of ImageTranscriber
        """
        self.transcriber = image_transcriber
        self.transcription_cache = {}  # Cache to avoid re-transcribing same images

    def process_content(self, content: str, field_name: str = "content") -> Tuple[str, int]:
        """
        Process content to detect and replace images with transcriptions

        Args:
            content: HTML or text content that may contain images
            field_name: Name of the field being processed (for logging)

        Returns:
            Tuple of (processed_content, num_images_transcribed)
        """
        if not content:
            return content, 0

        num_transcribed = 0
        processed = content

        # Process base64 images in img tags
        processed, count = self._process_base64_images(processed, field_name)
        num_transcribed += count

        # Process URL images in img tags
        processed, count = self._process_url_images(processed, field_name)
        num_transcribed += count

        # Process standalone data URIs
        processed, count = self._process_standalone_base64(processed, field_name)
        num_transcribed += count

        return processed, num_transcribed

    def _process_base64_images(self, content: str, field_name: str) -> Tuple[str, int]:
        """Process and replace base64 images in img tags"""
        count = 0

        for match in self.IMG_BASE64_PATTERN.finditer(content):
            full_match = match.group(0)
            extension = match.group(1)
            base64_data = match.group(2)

            # Create cache key
            cache_key = hash(base64_data[:100] + base64_data[-100:])

            if cache_key in self.transcription_cache:
                transcription = self.transcription_cache[cache_key]
                logger.info(f"Using cached transcription for base64 image in {field_name}")
            else:
                logger.info(f"Transcribing base64 image in {field_name}")
                transcription = self.transcriber.transcribe_from_base64(base64_data, extension)
                if transcription:
                    self.transcription_cache[cache_key] = transcription

            if transcription:
                replacement = f'[Image Transcription: {transcription}]'
                content = content.replace(full_match, replacement)
                count += 1
            else:
                logger.warning(f"Failed to transcribe base64 image in {field_name}")

        return content, count

    def _process_url_images(self, content: str, field_name: str) -> Tuple[str, int]:
        """Process and replace URL images in img tags"""
        count = 0

        for match in self.IMG_URL_PATTERN.finditer(content):
            full_match = match.group(0)
            image_url = match.group(1)

            # Cache key from URL
            cache_key = hash(image_url)

            if cache_key in self.transcription_cache:
                transcription = self.transcription_cache[cache_key]
                logger.info(f"Using cached transcription for URL image in {field_name}")
            else:
                logger.info(f"Transcribing URL image in {field_name}: {image_url[:80]}...")
                transcription = self.transcriber.transcribe_from_url(image_url)
                if transcription:
                    self.transcription_cache[cache_key] = transcription

            if transcription:
                replacement = f'[Image Transcription: {transcription}]'
                content = content.replace(full_match, replacement)
                count += 1
            else:
                logger.warning(f"Failed to transcribe URL image in {field_name}")

        return content, count

    def _process_standalone_base64(self, content: str, field_name: str) -> Tuple[str, int]:
        """Process standalone base64 data URIs (not in img tags)"""
        count = 0

        for match in self.BASE64_PATTERN.finditer(content):
            full_match = match.group(0)
            extension = match.group(2)
            base64_data = match.group(3)

            # Skip if already processed (part of img tag)
            if f'src="{full_match}"' in content or f"src='{full_match}'" in content:
                continue

            cache_key = hash(base64_data[:100] + base64_data[-100:])

            if cache_key in self.transcription_cache:
                transcription = self.transcription_cache[cache_key]
            else:
                logger.info(f"Transcribing standalone base64 in {field_name}")
                transcription = self.transcriber.transcribe_from_base64(base64_data, extension)
                if transcription:
                    self.transcription_cache[cache_key] = transcription

            if transcription:
                replacement = f'[Image Transcription: {transcription}]'
                content = content.replace(full_match, replacement)
                count += 1

        return content, count

    def clear_cache(self):
        """Clear the transcription cache"""
        self.transcription_cache.clear()
