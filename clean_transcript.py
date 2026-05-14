"""
clean_transcript.py — post-processing for raw voice transcripts.

Usage:
    echo "transcript text" | .venv/bin/python clean_transcript.py
    .venv/bin/python clean_transcript.py --test
    .venv/bin/python clean_transcript.py < file.txt

Importable:
    from clean_transcript import clean_transcript
    result = clean_transcript("raw text here")
"""

import re
import sys


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_1 = (
    "So funny enough, I was talking about the parakeet model and using it for reviewing with voice. "
    "So that's pretty cool, um I mention anger at Dina um I think this was right after the goddess's party "
    "and the no PDA thing and then yeah feeling just very let down by her and giving up um okay and so now "
    "I'm starting to write out my text message and I'm really trying to explain like why it is that I feel "
    "this way and um yeah I try a few different times damn I probably spend it wouldn't surprise me if I wow "
    "okay so with spaces and all that um this is almost a hundred lines that I've written basically the entire "
    "days um um no it was about Dina that's really frustrating daily review says that I worked a lot on this "
    "day um I was pretty pissed though feeling pulled back and forth on um how to engage with this wait a second "
    "this is Monday oh this was Sunday this was Sunday I think the next day is when I get frustrated with Thor "
    "this is me losing um motivation to continue being in conjunction with her"
)

SAMPLE_2 = (
    "I mention here that I am grateful for having Moxie. For being able to keep going and push hard. "
    "I apply to sleep awake oh man cool thing here the level of understanding that I was getting when "
    "developing Mistbot reminded me of how I'd have to learn super technical songs and each section required "
    "its own like intricate understanding of what was happening yeah I felt I felt an analogy to how that was "
    "no no how those two crossed over go and learn to play hard songs go and build big stuff wow I want to "
    "train some models I want to make robots this is really cool gives me this is Opus gives me a good line "
    "here you're a systems thinker you already know that a system without a stop condition runs until something "
    "breaks you are the something oh my god this is the day where Thor and I have to check in he asked me at "
    "the end if I felt supported and I lied I felt let down I mostly feel I was hearing a lot of Matt you could "
    "have done better interesting that's it for that day"
)


# ---------------------------------------------------------------------------
# Step 1: Filler removal
# ---------------------------------------------------------------------------

# These are removed outright — pure noise with no discourse value.
# "yeah", "okay", "oh" are NOT in this list; they carry meaning and are
# handled selectively in the sentence-boundary step.
_FILLER_PATTERNS = [
    r"\bum\s+um\b",   # doubled um first
    r"\buh\s+um\b",
    r"\bum\s+uh\b",
    r"\byeah\s+um\b",  # yeah+um combo → keep neither
    r"\bum\b",
    r"\buh\b",
    r"\bhmm\b",
    r"\bhm\b",
]

_filler_re = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)


def _remove_fillers(text: str) -> str:
    text = _filler_re.sub(" ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 2: Repeated-word deduplication
# ---------------------------------------------------------------------------

def _dedup_repeated_words(text: str) -> str:
    """
    Collapse immediately repeated words or short phrases.
      "I felt I felt" -> "I felt"
      "no no" -> "no"
    """
    # two-word phrases first
    text = re.sub(r"\b(\w+\s+\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    # single words
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Step 3: Inline punctuation fixes (before sentence splitting)
#
# These transforms clean up specific awkward constructions in-place without
# introducing sentence breaks, so later patterns see cleaner input.
# ---------------------------------------------------------------------------

_INLINE_FIXES = [
    # "oh this was Sunday" → strip the dangling "oh" that's really just a
    # filler before a date-realisation ("oh wait, this was Sunday")
    (re.compile(r"\boh\s+(?=this\s+(?:is|was)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b)", re.I), ""),

    # "okay so" as a filler transition → just "so"
    (re.compile(r"\bokay\s+so\b", re.I), "so"),

    # "and then yeah" → "and then" (yeah is redundant after a sequential marker)
    (re.compile(r"\band\s+then\s+yeah\b", re.I), "and then"),

    # "wow okay so" → just collapse to nothing (double-filler transition)
    (re.compile(r"\bwow\s+okay\s+so\b", re.I), "wow,"),

    # "if I wow" → "if I —" (trailing wow before a topic change)
    (re.compile(r"\bif\s+I\s+wow\b", re.I), "if I —"),

    # "oh man" before "cool thing here" → keep as inline aside, add comma
    (re.compile(r"\boh\s+man\s+(?=cool\s+thing\s+here\b)", re.I), "oh man — "),

    # general "oh man" / "oh my god" inline — add em-dash framing
    (re.compile(r",?\s*\boh\s+my\s+god\b,?\s*", re.I), " — oh my god — "),
    (re.compile(r",?\s*\boh\s+man\b,?\s*(?!—)", re.I), " — oh man — "),
]


def _apply_inline_fixes(text: str) -> str:
    for pattern, replacement in _INLINE_FIXES:
        text = pattern.sub(replacement, text)
    # clean up any double spaces introduced
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 4: Sentence boundary detection
#
# We insert a "||" sentinel before phrases that signal a new sentence.
# Patterns are applied in order; more specific ones first.
#
# Rules:
#  - A pattern must require real content BEFORE the marker (lookbehind \w
#    or comma) so we don't double-break at the start of a segment.
#  - "wow" and "damn" break BEFORE them and keep what follows.
#  - Sequential markers ("and then") break before them.
#  - "yeah" and "okay" only break when followed by a subject or specific words.
# ---------------------------------------------------------------------------

_BREAK_PATTERNS = [
    # Day/date realisations: "this is Monday", "this was Sunday"
    (re.compile(
        r"(?<=\w)\s+this\s+(is|was)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        re.I),
     r"||This \1 \2"),

    # "wait a second" — meta-comment, full break
    (re.compile(r"(?<=\w)\s+wait\s+a\s+second\b", re.I),
     "||Wait a second —"),

    # Daily review — section header
    (re.compile(r"(?<=\w),?\s+daily\s+review\s+says\b", re.I),
     "||Daily review says"),

    # "cool thing here" — section opener
    (re.compile(r"(?<=\w),?\s+cool\s+thing\s+here\b", re.I),
     "||Cool thing here —"),

    # "wow" — exclamation, break before it
    (re.compile(r"(?<=\w),?\s+wow\b,?\s*", re.I),
     " ||Wow — "),

    # "damn" — break before it
    (re.compile(r"(?<=\w),?\s+damn\b,?\s*", re.I),
     " ||Damn — "),

    # "and then" — sequential narrative marker
    (re.compile(r"(?<=\w),?\s+and\s+then\s+", re.I),
     "||And then "),

    # "so" after comma, starting new clause with pronoun/demonstrative
    (re.compile(r",\s+so\s+(?=(?:that|this|now|I|we|he|she|they)\b)", re.I),
     "||So "),

    # "I mention [here/anger/that]" — diary entry item opener
    (re.compile(r"(?<=\w)\s+I\s+mention(?:ed)?\s+(?=(?:here|anger|that)\b)", re.I),
     "||I mention "),

    # "okay" as reset before "and", "now", "I", "let"
    # (not "okay so" — already collapsed above)
    (re.compile(r"(?<=\w),?\s+okay\s+(?=(?:and|now|I|let)\b)", re.I),
     "||Okay, "),

    # "yeah" starting a new statement — only before subject pronouns
    # Requires something before it (lookbehind \w) so it doesn't fire at
    # start of a segment placed there by a prior break.
    (re.compile(r"(?<=\w),?\s+yeah\s+(?=(?:I|we|he|she|it|that|this)\b)", re.I),
     "||Yeah, "),

    # "but" contrast after comma
    (re.compile(r",\s+but\s+(?=\w)", re.I),
     "||But "),

    # "basically" summary after comma
    (re.compile(r",\s+basically\s+", re.I),
     "||Basically "),
]


def _insert_breaks(text: str) -> str:
    for pattern, replacement in _BREAK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _build_sentences(raw: str) -> list[str]:
    """Split on sentinels and existing terminal punctuation; capitalise each."""
    parts = re.split(r"\|\|", raw)
    sentences = []
    for part in parts:
        # split on existing terminal punctuation
        sub = re.split(r"(?<=[.!?])\s+(?=[A-Z])", part)
        for s in sub:
            s = s.strip().strip(",").strip()
            if not s:
                continue
            # drop very short orphan fragments (single word or dash only)
            if re.fullmatch(r"[\w\s—\-]{1,4}", s):
                continue
            # capitalise first letter
            s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
            # ensure terminal punctuation
            if not re.search(r"[.!?—]$", s):
                s += "."
            sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Step 5: Paragraph grouping
# ---------------------------------------------------------------------------

_PARA_STARTERS = re.compile(
    r"^(So\b|Daily review|Cool thing|Wow|Damn|And then\b|"
    r"This (?:is|was) (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"Wait a second)",
    re.I,
)


def _group_paragraphs(sentences: list[str], max_run: int = 4) -> str:
    if not sentences:
        return ""
    paragraphs: list[list[str]] = [[]]
    run = 0
    for s in sentences:
        is_break = bool(_PARA_STARTERS.match(s))
        if is_break and paragraphs[-1]:
            paragraphs.append([])
            run = 0
        elif run >= max_run:
            paragraphs.append([])
            run = 0
        paragraphs[-1].append(s)
        run += 1
    return "\n\n".join(" ".join(p) for p in paragraphs if p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_transcript(text: str) -> str:
    """
    Full pipeline: filler removal -> dedup -> inline fixes ->
    sentence boundary detection -> paragraph grouping.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = _remove_fillers(text)
    text = _dedup_repeated_words(text)
    text = _apply_inline_fixes(text)
    text = _insert_breaks(text)
    sentences = _build_sentences(text)
    return _group_paragraphs(sentences)


def clean_transcript_regex_only(text: str) -> str:
    """Lighter pass: fillers + dedup + inline fixes only, no sentence splitting."""
    text = re.sub(r"\s+", " ", text).strip()
    text = _remove_fillers(text)
    text = _dedup_repeated_words(text)
    text = _apply_inline_fixes(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_comparison(label: str, original: str) -> None:
    width = 72
    bar = "-" * width
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")
    print("\n[ORIGINAL]")
    print(original)
    print(f"\n{bar}")
    print("[REGEX ONLY — fillers stripped, no sentence splitting]")
    print(clean_transcript_regex_only(original))
    print(f"\n{bar}")
    print("[FULL CLEAN — sentence boundaries + paragraphs]")
    print(clean_transcript(original))
    print()


def _run_tests() -> None:
    print("Running built-in test samples...\n")
    _print_comparison("SAMPLE 1 — journal review with fillers", SAMPLE_1)
    _print_comparison("SAMPLE 2 — cleaner speech, more natural", SAMPLE_2)


def main() -> None:
    args = sys.argv[1:]
    if "--test" in args:
        _run_tests()
        return
    if "--help" in args or "-h" in args:
        print(__doc__)
        return
    if not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        print(
            "Usage: echo 'text' | clean_transcript.py  or  clean_transcript.py --test",
            file=sys.stderr,
        )
        sys.exit(1)
    print(clean_transcript(raw))


if __name__ == "__main__":
    main()
