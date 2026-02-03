"""
URL Detection utility for SAT Forum Responder
Detects URLs in forum post text to skip processing for posts containing links
Excludes image URLs which are handled by the image transcriber
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class URLDetector:
    """Detects URLs in text content, excluding image URLs"""

    # Image extensions to exclude from URL detection (these are transcribed instead)
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp', '.tiff'}

    URL_PATTERNS = [
        # Standard HTTP/HTTPS URLs
        r'https?://[^\s<>"\']+',
        # URLs starting with www
        r'www\.[^\s<>"\']+',
        # Google services
        r'drive\.google\.com/[^\s<>"\']+',
        r'docs\.google\.com/[^\s<>"\']+',
        r'sheets\.google\.com/[^\s<>"\']+',
        r'slides\.google\.com/[^\s<>"\']+',
        r'forms\.google\.com/[^\s<>"\']+',
        # URL shorteners
        r'bit\.ly/[^\s<>"\']+',
        r'tinyurl\.com/[^\s<>"\']+',
        r'goo\.gl/[^\s<>"\']+',
        r't\.co/[^\s<>"\']+',
        r'ow\.ly/[^\s<>"\']+',
        r'is\.gd/[^\s<>"\']+',
        r'buff\.ly/[^\s<>"\']+',
        r'rb\.gy/[^\s<>"\']+',
        r'cutt\.ly/[^\s<>"\']+',
        r'shorturl\.at/[^\s<>"\']+',
        # Cloud storage
        r'dropbox\.com/[^\s<>"\']+',
        r'onedrive\.live\.com/[^\s<>"\']+',
        r'1drv\.ms/[^\s<>"\']+',
        r'box\.com/[^\s<>"\']+',
        r'icloud\.com/[^\s<>"\']+',
        # Image hosting
        r'imgur\.com/[^\s<>"\']+',
        r'i\.imgur\.com/[^\s<>"\']+',
        r'imgbb\.com/[^\s<>"\']+',
        r'postimg\.cc/[^\s<>"\']+',
        r'flickr\.com/[^\s<>"\']+',
        r'prnt\.sc/[^\s<>"\']+',
        r'gyazo\.com/[^\s<>"\']+',
        # Video hosting
        r'youtube\.com/[^\s<>"\']+',
        r'youtu\.be/[^\s<>"\']+',
        r'vimeo\.com/[^\s<>"\']+',
        r'dailymotion\.com/[^\s<>"\']+',
        # Document sharing
        r'scribd\.com/[^\s<>"\']+',
        r'slideshare\.net/[^\s<>"\']+',
        r'academia\.edu/[^\s<>"\']+',
        r'researchgate\.net/[^\s<>"\']+',
        # Paste sites
        r'pastebin\.com/[^\s<>"\']+',
        r'hastebin\.com/[^\s<>"\']+',
    ]

    def __init__(self):
        """Initialize URL detector with compiled regex patterns"""
        combined_pattern = '|'.join(f'({pattern})' for pattern in self.URL_PATTERNS)
        self.url_regex = re.compile(combined_pattern, re.IGNORECASE)
        logger.info("URL Detector initialized")

    def _is_image_url(self, url: str) -> bool:
        """Check if URL points to an image file"""
        url_lower = url.lower()
        # Remove query parameters for extension check
        base_url = url_lower.split('?')[0]
        return any(base_url.endswith(ext) for ext in self.IMAGE_EXTENSIONS)

    def detect_urls(self, text: str, exclude_images: bool = True) -> List[str]:
        """
        Detect all URLs in the given text

        Args:
            text: Text to search for URLs
            exclude_images: If True, exclude image URLs (they are transcribed instead)
        """
        if not text:
            return []

        matches = self.url_regex.findall(text)

        urls = []
        for match in matches:
            if isinstance(match, tuple):
                for url in match:
                    if url and url.strip():
                        urls.append(url.strip())
            elif match and match.strip():
                urls.append(match.strip())

        # Deduplicate while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url.lower() not in seen:
                # Exclude image URLs if requested
                if exclude_images and self._is_image_url(url):
                    logger.debug(f"Excluding image URL from detection: {url[:50]}...")
                    continue
                seen.add(url.lower())
                unique_urls.append(url)

        return unique_urls

    def check_forum_data(self, forum_data: dict) -> Tuple[bool, List[str]]:
        """Check forum data for URLs in forumPostText and parentPostQuery"""
        all_urls = []

        # Check forumPostText (SAT uses ForumPostText with capital F)
        forum_post_text = forum_data.get("forumPostText", "") or forum_data.get("ForumPostText", "")
        if forum_post_text:
            urls = self.detect_urls(forum_post_text)
            if urls:
                logger.info(f"URLs detected in forumPostText: {urls}")
                all_urls.extend(urls)

        # Check parentPostQuery
        parent_post_query = forum_data.get("parentPostQuery", "")
        if parent_post_query:
            urls = self.detect_urls(parent_post_query)
            if urls:
                logger.info(f"URLs detected in parentPostQuery: {urls}")
                all_urls.extend(urls)

        # Deduplicate
        seen = set()
        unique_urls = []
        for url in all_urls:
            if url.lower() not in seen:
                seen.add(url.lower())
                unique_urls.append(url)

        has_urls = len(unique_urls) > 0

        if has_urls:
            logger.warning(f"URL check FAILED - {len(unique_urls)} URL(s) detected: {unique_urls}")
        else:
            logger.info("URL check PASSED - No URLs detected")

        return has_urls, unique_urls


# Singleton instance
url_detector = URLDetector()
