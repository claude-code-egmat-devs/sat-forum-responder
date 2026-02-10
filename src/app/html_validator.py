"""
HTML Validator for Forum Responder
Validates and scores HTML content before posting to forum
"""

import re
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of HTML validation with quality scoring"""
    is_valid: bool = True
    quality_score: int = 100
    original_html: str = ""
    cleaned_html: str = ""

    # Detailed breakdown
    structure_score: int = 100
    security_score: int = 100
    content_score: int = 100
    formatting_score: int = 100

    # Issues found
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    auto_fixes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class HTMLValidator:
    """Comprehensive HTML validator for student-facing content"""

    # Tags that are self-closing
    VOID_ELEMENTS = {
        'br', 'hr', 'img', 'input', 'meta', 'link', 'area',
        'base', 'col', 'embed', 'param', 'source', 'track', 'wbr'
    }

    # Dangerous tags to remove
    DANGEROUS_TAGS = {'script', 'style', 'iframe', 'object', 'embed', 'form'}

    # Event handler attributes to remove
    EVENT_HANDLERS = {
        'onclick', 'ondblclick', 'onmousedown', 'onmouseup', 'onmouseover',
        'onmousemove', 'onmouseout', 'onkeydown', 'onkeypress', 'onkeyup',
        'onload', 'onerror', 'onsubmit', 'onreset', 'onfocus', 'onblur',
        'onchange', 'onscroll', 'onresize'
    }

    # Placeholder patterns that indicate incomplete content
    PLACEHOLDER_PATTERNS = [
        r'\[your\s+response\s+here\]',
        r'\[insert\s+.*?\]',
        r'\[todo\]',
        r'\[placeholder\]',
        r'<your\s+response>',
        r'lorem\s+ipsum',
    ]

    # Internal markers that should not appear in student-facing content
    INTERNAL_MARKERS = [
        r'"?exception_flag"?\s*[:=]',
        r'"?hil_flag"?\s*[:=]',
        r'"?classification"?\s*[:=]\s*"',
        r'"?validation_result"?\s*[:=]',
        r'<thinking>',
        r'</thinking>',
        r'<scratchpad>',
        r'</scratchpad>',
        r'<internal>',
        r'<system>',
        r'AGENT\s*\d+:',
        r'Step\s*\d+[A-Z]?\.\d+:',
    ]

    # Encoding fixes
    ENCODING_FIXES = {
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2013': '-',
        '\u2014': '--',
        '\u2022': '*',
        '\u2026': '...',
        '\x00': '',
        '\r\n': '\n',
        '\r': '\n',
    }

    def __init__(self):
        """Initialize the HTML validator"""
        self.tag_pattern = re.compile(r'<(/?)(\w+)([^>]*)>')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )

    def validate(self, html_content: str, correlation_id: str = "") -> ValidationResult:
        """
        Validate HTML content and return a scored result.
        """
        result = ValidationResult(original_html=html_content)

        if not html_content:
            result.is_valid = False
            result.quality_score = 0
            result.errors.append("Empty content")
            logger.error(f"[{correlation_id}] HTML Validation FAILED: Empty content")
            return result

        cleaned = html_content

        # Layer 1: Pre-validation
        cleaned, pre_score, pre_issues = self._pre_validation(cleaned)
        result.content_score = pre_score
        result.errors.extend([e for e, is_error in pre_issues if is_error])
        result.warnings.extend([e for e, is_error in pre_issues if not is_error])

        # Layer 2: Content sanitization
        cleaned, sanitize_fixes = self._sanitize_content(cleaned)
        result.auto_fixes.extend(sanitize_fixes)

        # Layer 3: HTML structure validation
        cleaned, struct_score, struct_fixes = self._validate_structure(cleaned)
        result.structure_score = struct_score
        result.auto_fixes.extend(struct_fixes)

        # Layer 4: Security sanitization
        cleaned, sec_score, sec_fixes = self._security_sanitization(cleaned)
        result.security_score = sec_score
        result.auto_fixes.extend(sec_fixes)

        # Layer 5: Content quality checks
        content_score, content_issues = self._content_quality_check(cleaned)
        result.content_score = min(result.content_score, content_score)
        result.errors.extend([e for e, is_error in content_issues if is_error])
        result.warnings.extend([e for e, is_error in content_issues if not is_error])

        # Layer 6: Formatting quality
        cleaned, format_score, format_fixes = self._formatting_quality(cleaned)
        result.formatting_score = format_score
        result.auto_fixes.extend(format_fixes)

        # Calculate overall quality score (weighted average)
        result.quality_score = self._calculate_overall_score(
            structure=result.structure_score,
            security=result.security_score,
            content=result.content_score,
            formatting=result.formatting_score
        )

        # Determine if valid (no blocking errors and score >= 50)
        result.is_valid = len(result.errors) == 0 and result.quality_score >= 50
        result.cleaned_html = cleaned

        # Log the validation result
        self._log_validation(correlation_id, result)

        return result

    def _pre_validation(self, html: str) -> Tuple[str, int, List[Tuple[str, bool]]]:
        """Pre-validation checks for basic content requirements"""
        issues = []
        score = 100

        if len(html.strip()) < 50:
            issues.append(("Content too short (< 50 chars)", True))
            score -= 50

        if len(html) > 50000:
            html = html[:50000]
            issues.append(("Content truncated (> 50,000 chars)", False))
            score -= 10

        return html, score, issues

    def _sanitize_content(self, html: str) -> Tuple[str, List[str]]:
        """Sanitize content: encoding fixes, emoji removal, control chars"""
        fixes = []
        cleaned = html

        # Fix encoding issues
        for old, new in self.ENCODING_FIXES.items():
            if old in cleaned:
                cleaned = cleaned.replace(old, new)
                fixes.append(f"Fixed encoding: {repr(old)}")

        # Remove emojis
        if self.emoji_pattern.search(cleaned):
            cleaned = self.emoji_pattern.sub('', cleaned)
            fixes.append("Removed emojis")

        # Remove control characters (except newlines, tabs)
        original = cleaned
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
        if cleaned != original:
            fixes.append("Removed control characters")

        return cleaned, fixes

    def _validate_structure(self, html: str) -> Tuple[str, int, List[str]]:
        """Validate and fix HTML structure"""
        fixes = []
        score = 100
        cleaned = html

        # Fix self-closing tags
        for tag in self.VOID_ELEMENTS:
            pattern = rf'<{tag}(\s[^>]*)?>(?!/)'
            if re.search(pattern, cleaned, re.IGNORECASE):
                cleaned = re.sub(pattern, rf'<{tag}\1/>', cleaned, flags=re.IGNORECASE)
                fixes.append(f"Fixed self-closing <{tag}> tags")

        # Find and close unclosed tags
        unclosed = self._get_unclosed_tags(cleaned)
        if unclosed:
            for tag in reversed(unclosed):
                cleaned += f'</{tag}>'
            fixes.append(f"Closed unclosed tags: {', '.join(unclosed)}")
            score -= len(unclosed) * 5

        # Remove orphaned closing tags
        orphaned = self._get_orphaned_closing_tags(cleaned)
        if orphaned:
            for tag in orphaned:
                cleaned = re.sub(rf'</{tag}>', '', cleaned, count=1)
            fixes.append(f"Removed orphaned closing tags: {', '.join(orphaned)}")
            score -= len(orphaned) * 3

        # Fix malformed HTML entities
        original = cleaned
        cleaned = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);)', '&amp;', cleaned)
        if cleaned != original:
            fixes.append("Fixed malformed HTML entities")
            score -= 5

        return cleaned, max(score, 0), fixes

    def _get_unclosed_tags(self, html: str) -> List[str]:
        """Get list of tags that are opened but not closed"""
        stack = []
        for match in self.tag_pattern.finditer(html):
            is_closing = match.group(1) == '/'
            tag_name = match.group(2).lower()

            if tag_name in self.VOID_ELEMENTS:
                continue
            if tag_name in self.DANGEROUS_TAGS:
                continue

            if is_closing:
                if stack and stack[-1] == tag_name:
                    stack.pop()
            else:
                if not match.group(3).rstrip().endswith('/'):
                    stack.append(tag_name)

        return stack

    def _get_orphaned_closing_tags(self, html: str) -> List[str]:
        """Get list of closing tags without matching opening tags"""
        orphaned = []
        stack = []

        for match in self.tag_pattern.finditer(html):
            is_closing = match.group(1) == '/'
            tag_name = match.group(2).lower()

            if tag_name in self.VOID_ELEMENTS:
                continue
            if tag_name in self.DANGEROUS_TAGS:
                continue

            if is_closing:
                if stack and stack[-1] == tag_name:
                    stack.pop()
                else:
                    orphaned.append(tag_name)
            else:
                if not match.group(3).rstrip().endswith('/'):
                    stack.append(tag_name)

        return orphaned

    def _security_sanitization(self, html: str) -> Tuple[str, int, List[str]]:
        """Remove security risks from HTML"""
        fixes = []
        score = 100
        cleaned = html

        # Remove dangerous tags
        for tag in self.DANGEROUS_TAGS:
            pattern = rf'<{tag}[^>]*>.*?</{tag}>'
            if re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL):
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
                fixes.append(f"Removed <{tag}> tags")
                score -= 20

            pattern = rf'<{tag}[^>]*/?\s*>'
            if re.search(pattern, cleaned, re.IGNORECASE):
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
                fixes.append(f"Removed unclosed <{tag}> tag")
                score -= 10

        # Remove event handlers
        for handler in self.EVENT_HANDLERS:
            pattern = rf'\s{handler}\s*=\s*["\'][^"\']*["\']'
            if re.search(pattern, cleaned, re.IGNORECASE):
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
                fixes.append(f"Removed {handler} event handler")
                score -= 15

        # Remove javascript: URLs
        if re.search(r'javascript:', cleaned, re.IGNORECASE):
            cleaned = re.sub(r'(href|src)\s*=\s*["\']javascript:[^"\']*["\']', '', cleaned, flags=re.IGNORECASE)
            fixes.append("Removed javascript: URLs")
            score -= 20

        # Remove data: URLs (except safe image types)
        pattern = r'(href|src)\s*=\s*["\']data:(?!image/(png|jpeg|gif|webp))[^"\']*["\']'
        if re.search(pattern, cleaned, re.IGNORECASE):
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            fixes.append("Removed suspicious data: URLs")
            score -= 15

        return cleaned, max(score, 0), fixes

    def _content_quality_check(self, html: str) -> Tuple[int, List[Tuple[str, bool]]]:
        """Check content quality for student-facing appropriateness"""
        issues = []
        score = 100
        html_lower = html.lower()

        # Check for placeholder text
        for pattern in self.PLACEHOLDER_PATTERNS:
            if re.search(pattern, html_lower):
                issues.append((f"Placeholder text detected: {pattern}", True))
                score -= 30

        # Check for internal markers
        for pattern in self.INTERNAL_MARKERS:
            if re.search(pattern, html, re.IGNORECASE):
                issues.append((f"Internal marker detected: {pattern}", True))
                score -= 25

        # Check for JSON artifacts
        json_patterns = [
            r'^\s*\{',
            r'"[a-z_]+"\s*:\s*["\[\{]',
            r'\}\s*$',
        ]
        json_count = sum(1 for p in json_patterns if re.search(p, html))
        if json_count >= 2:
            issues.append(("Possible JSON artifact in response", False))
            score -= 15

        # Check for proper greeting
        greeting_patterns = [
            r'^<[^>]*>\s*(hello|hi|hey|thank|good|great|welcome)',
            r'^\s*(hello|hi|hey|thank|good|great|welcome)',
        ]
        has_greeting = any(re.search(p, html_lower) for p in greeting_patterns)
        if not has_greeting:
            issues.append(("Response may lack proper greeting", False))
            score -= 5

        # Check for proper closing
        closing_patterns = [
            r'happy\s+learning',
            r'upvote',
            r'follow-up',
            r'feel\s+free',
            r'hope\s+this\s+helps',
            r'best\s+regards',
        ]
        has_closing = any(re.search(p, html_lower) for p in closing_patterns)
        if not has_closing:
            issues.append(("Response may lack proper closing", False))
            score -= 5

        # COMPLETENESS CHECK
        plain_text = re.sub(r'<[^>]+>', ' ', html)
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()
        word_count = len(plain_text.split())

        if word_count < 50:
            issues.append((f"Response too short: only {word_count} words (minimum 50)", True))
            score -= 40
        elif word_count < 100:
            issues.append((f"Response may be incomplete: only {word_count} words", False))
            score -= 15

        paragraph_count = len(re.findall(r'</p>', html, re.IGNORECASE))
        if paragraph_count < 2:
            issues.append(("Response has only 1 paragraph - may be incomplete", False))
            score -= 10

        closing_only_patterns = [
            r'^<p>\s*(if\s+this\s+addresses|hope\s+this\s+helps|happy\s+learning|feel\s+free)',
        ]
        if any(re.search(p, html_lower) for p in closing_only_patterns) and paragraph_count <= 1:
            issues.append(("Response appears to be ONLY a closing with no content", True))
            score -= 50

        return max(score, 0), issues

    def _formatting_quality(self, html: str) -> Tuple[str, int, List[str]]:
        """Check and fix formatting quality"""
        fixes = []
        score = 100
        cleaned = html

        original = cleaned
        cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
        if cleaned != original:
            fixes.append("Reduced excessive newlines")
            score -= 5

        original = cleaned
        cleaned = re.sub(r' {3,}', '  ', cleaned)
        if cleaned != original:
            fixes.append("Reduced excessive spaces")
            score -= 3

        original = cleaned
        cleaned = cleaned.strip()
        if cleaned != original:
            fixes.append("Trimmed whitespace")

        lines = cleaned.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 500)
        if long_lines > 0:
            score -= long_lines * 2
            fixes.append(f"Warning: {long_lines} very long lines detected")

        return cleaned, max(score, 0), fixes

    def _calculate_overall_score(self, structure: int, security: int,
                                  content: int, formatting: int) -> int:
        """Calculate weighted overall quality score"""
        weights = {
            'security': 0.30,
            'content': 0.35,
            'structure': 0.20,
            'formatting': 0.15
        }

        weighted_score = (
            security * weights['security'] +
            content * weights['content'] +
            structure * weights['structure'] +
            formatting * weights['formatting']
        )

        return int(round(weighted_score))

    def _log_validation(self, correlation_id: str, result: ValidationResult):
        """Log validation results for server-side tracking"""
        logger.info(
            f"[{correlation_id}] HTML Validation: "
            f"Score={result.quality_score}/100 "
            f"(Structure={result.structure_score}, Security={result.security_score}, "
            f"Content={result.content_score}, Formatting={result.formatting_score}) "
            f"Valid={result.is_valid}"
        )

        if result.errors:
            logger.warning(f"[{correlation_id}] Validation ERRORS: {result.errors}")
        if result.warnings:
            logger.info(f"[{correlation_id}] Validation warnings: {result.warnings}")
        if result.auto_fixes:
            logger.info(f"[{correlation_id}] Auto-fixes applied: {result.auto_fixes}")


# Singleton instance
html_validator = HTMLValidator()
