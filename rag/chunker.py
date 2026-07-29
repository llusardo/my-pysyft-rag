"""Document chunker: splits raw markdown text into chunks for embedding."""

import logging
import re

logger = logging.getLogger(__name__)

# Matches a fenced code block (```...```), any language tag or none,
# including blank lines inside it. DOTALL so "." spans newlines.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _split_into_units(text: str) -> list[str]:
    """Split text into paragraph/code-block units, in order.

    A fenced code block is kept as a single unit (never split on the blank
    lines that commonly appear inside multi-line code examples). Everything
    outside code fences is split on blank lines ("\\n\\n"), same as before.

    Args:
        text: Raw text to split.

    Returns:
        Ordered list of non-empty units — either a whole code block or a
        stripped paragraph.
    """
    units: list[str] = []
    last_end = 0

    for match in _CODE_FENCE_RE.finditer(text):
        before = text[last_end:match.start()]
        units.extend(p.strip() for p in before.split("\n\n") if p.strip())
        units.append(match.group().strip())
        last_end = match.end()

    tail = text[last_end:]
    units.extend(p.strip() for p in tail.split("\n\n") if p.strip())

    return units


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks of roughly chunk_size characters.

    Splits on paragraph boundaries (blank lines) first, then greedily
    groups consecutive units together until adding the next one would
    exceed chunk_size. This avoids cutting sentences mid-way whenever a
    paragraph break is available nearby. Fenced code blocks (```...```)
    are treated as atomic units — a blank line inside a code block never
    causes a split there, so code examples never get torn across chunks.

    Args:
        text: Raw text to split (e.g. contents of a markdown file).
        chunk_size: Target maximum size of each chunk, in characters.

    Returns:
        List of text chunks. Empty list if text is empty.

    Raises:
        TypeError: If text is None.
    """
    if text is None:
        raise TypeError("text must be a string, not None")
    if text == "":
        return []

    # Paragraphs and fenced code blocks, in document order. A code block
    # counts as one unit regardless of blank lines inside it; the
    # oversized-unit fallback below applies to both kinds.
    units = _split_into_units(text)

    chunks: list[str] = []
    current = ""

    for unit in units:
        # Unit itself is bigger than chunk_size and there's nothing to
        # group it with usefully — emit it alone rather than losing content
        # (this also covers oversized code blocks, which must stay intact).
        if len(unit) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit)
            continue

        # Would this unit push the current chunk over the limit?
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            current = unit
        else:
            current = candidate

    if current:
        chunks.append(current)

    logger.debug("chunk_text produced %d chunks from %d chars", len(chunks), len(text))
    return chunks
