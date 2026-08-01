"""Span-to-passage matching: the three tiers, and where each match LANDS.

Single source of truth for span matching, shared by
``visualize/stats.py`` (which reports tier rates) and
``eval/heuristic_spans.py`` (which materializes the located text). Keeping one
implementation is what makes the located spans and the EM/norm/approx table
agree by construction -- reimplementing either side would let them drift.

Tiers, mutually exclusive, first hit wins (keep in sync with
visualize/static/index.html):

  * ``em``     -- the span occurs verbatim in the passage;
  * ``norm``   -- it occurs after conservative normalization only (lowercase,
                  straight quotes/dashes, collapsed whitespace), contiguously;
  * ``approx`` -- it aligns somewhere with at most APPROX_RATE (~5%) of its
                  length in character edits (min 2);
  * ``nf``     -- not found.

``locate_span`` additionally returns the (start, end) character offsets IN THE
ORIGINAL PASSAGE, so a norm/approx match can be replaced by the real passage
substring it corresponds to. That substring is verbatim by construction, which
is what the retrieval metrics need: a pseudo-document must never contain text
absent from the document (otherwise a model that paraphrases fluently scores
well on spans that do not exist).

Offsets come from two extra mechanisms beyond plain ``find``:

  * normalization records, for every normalized character, the index of the
    original character it came from (``normalize_with_map``), so a hit in
    normalized space maps back exactly;
  * for ``approx``, the semi-global DP is run once forward to find the
    alignment END, then again on the reversed window/pattern to find the START
    (a reversed semi-global alignment ends where the forward one begins). This
    avoids materializing a full traceback matrix.
"""
import re

import numpy as np

APPROX_RATE = 0.05

NORM_TABLE = str.maketrans({
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "–": "-", "—": "-",   # en/em dash
    " ": " ",                  # nbsp
})
_WS = re.compile(r"[ \t\n\r\f\v]+")


def normalize(text):
    return _WS.sub(" ", text.translate(NORM_TABLE).lower())


def normalize_with_map(text):
    """``(normalized, origin)`` where ``origin[i]`` is the index in ``text`` of
    the character that produced ``normalized[i]``, plus a final sentinel equal
    to ``len(text)`` so ``origin[end]`` is a valid exclusive end offset.

    Mirrors ``normalize`` exactly: translate (1 char -> 1 char, so indices are
    preserved), lowercase (likewise), then collapse each whitespace run to a
    single space attributed to the run's FIRST character. ``str.lower`` can
    change length for a few exotic codepoints (e.g. 'İ'), so it is applied per
    character and any such expansion is attributed to that one source index --
    keeping the map aligned at the cost of a harmless off-by-nothing there.
    """
    translated = text.translate(NORM_TABLE)
    out, origin = [], []
    i, n = 0, len(translated)
    while i < n:
        ch = translated[i]
        if ch in " \t\n\r\f\v":
            j = i
            while j < n and translated[j] in " \t\n\r\f\v":
                j += 1
            out.append(" ")
            origin.append(i)
            i = j
            continue
        low = ch.lower()
        out.append(low)
        origin.extend([i] * len(low))
        i += 1
    origin.append(len(text))
    return "".join(out), origin


def _semiglobal_row(win, pat):
    """Final DP row for aligning all of ``pat`` inside ``win`` with free
    start/end: ``row[j]`` = min edits for an alignment ENDING at ``win[:j]``.
    Vectorized row-wise; the insertion recurrence (a running min) uses the
    prefix-min trick on D[j]-j."""
    n, m = len(win), len(pat)
    if m == 0:
        return np.zeros(n + 1, dtype=np.int32)
    if n == 0:
        return np.full(1, m, dtype=np.int32)
    wc = np.frombuffer(win.encode("utf-32-le"), dtype=np.uint32)
    pc = np.frombuffer(pat.encode("utf-32-le"), dtype=np.uint32)
    idx = np.arange(n + 1, dtype=np.int32)
    prev = np.zeros(n + 1, dtype=np.int32)          # D[0][j] = 0 (free start)
    for i in range(1, m + 1):
        t = np.minimum(prev[:-1] + (wc != pc[i - 1]),   # substitute
                       prev[1:] + 1)                    # skip pattern char
        v = np.concatenate(([np.int32(i)], t)) - idx    # cur[0] = i
        np.minimum.accumulate(v, out=v)                 # insertion: running min
        prev = v + idx
    return prev


def semiglobal_min_err(win, pat):
    """Minimum edits to align ``pat`` fully inside ``win`` (start/end free)."""
    return int(_semiglobal_row(win, pat).min())


#: Anchor starts are floored to this grid so near-duplicate windows collapse to
#: one DP run. The window must therefore be at least this much LONGER than the
#: span, or flooring can push the start back far enough that the span no longer
#: fits inside the window -- which inflates the edit distance and loses the
#: match. (That was a real bug: a 40-char span whose true start was 13 got
#: floored to 0 and scored 9 edits against a budget of 2.)
_ANCHOR_GRID = 16


def _approx_budget(ns):
    """``(max_err, win_len)`` for one normalized span."""
    max_err = max(2, round(len(ns) * APPROX_RATE))
    return max_err, len(ns) + 2 * (max_err + 2) + _ANCHOR_GRID


def _approx_windows(norm_passage, ns):
    """``(window_start, window_text)`` candidates, shared by ``approx_matches``
    and ``_approx_span`` so both judge identical geometry."""
    max_err, win_len = _approx_budget(ns)
    for ws in _anchor_starts(norm_passage, ns, max_err):
        yield ws, norm_passage[ws:ws + win_len]


def _anchor_starts(norm_passage, ns, max_err):
    """Candidate window starts for an approximate match, by pigeonhole: with at
    most ``max_err`` edits, at least one of ``max_err + 1`` span pieces survives
    verbatim, so align only around exact piece hits."""
    pieces = max_err + 1
    piece_len = -(-len(ns) // pieces)
    if piece_len < 4:
        pieces = len(ns) // 4
        piece_len = -(-len(ns) // pieces)
    starts = set()
    for p in range(pieces):
        if len(starts) >= 60:
            break
        off = p * piece_len
        piece = ns[off:off + piece_len]
        frm, hits = 0, 0
        while piece and hits < 50:
            pos = norm_passage.find(piece, frm)
            if pos == -1:
                break
            starts.add(max(0, pos - off - max_err - 2)
                       // _ANCHOR_GRID * _ANCHOR_GRID)
            frm, hits = pos + 1, hits + 1
    return starts


def approx_matches(norm_passage, ns):
    """True when ``ns`` aligns somewhere in ``norm_passage`` within budget."""
    if len(ns) < 8:
        return False
    max_err, _ = _approx_budget(ns)
    return any(semiglobal_min_err(win, ns) <= max_err
               for _ws, win in _approx_windows(norm_passage, ns))


def _approx_span(norm_passage, ns):
    """``(start, end)`` in NORMALIZED coordinates of the best approximate
    alignment of ``ns``, or ``None``. Forward DP locates the alignment end;
    a reversed DP over the prefix locates its start."""
    if len(ns) < 8:
        return None
    max_err, _ = _approx_budget(ns)
    best = None
    for ws, win in _approx_windows(norm_passage, ns):
        row = _semiglobal_row(win, ns)
        err = int(row.min())
        if err > max_err or (best is not None and err >= best[0]):
            continue
        end_in_win = int(row.argmin())              # alignment ends here
        head = win[:end_in_win][::-1]
        back = _semiglobal_row(head, ns[::-1])      # reversed => finds the start
        start_in_win = end_in_win - int(back.argmin())
        best = (err, ws + start_in_win, ws + end_in_win)
    if best is None:
        return None
    return best[1], best[2]


def locate_span(span, passage, norm_passage=None, origin=None):
    """``(tier, start, end)`` for one span against one passage.

    ``start``/``end`` are character offsets into the ORIGINAL ``passage``
    (``end`` exclusive) and are ``None`` only for tier ``nf``. Pass
    ``norm_passage``/``origin`` from ``normalize_with_map(passage)`` to avoid
    renormalizing the passage for every span.
    """
    if not isinstance(span, str) or not span:
        return "nf", None, None
    pos = passage.find(span)
    if pos != -1:
        return "em", pos, pos + len(span)

    if norm_passage is None or origin is None:
        norm_passage, origin = normalize_with_map(passage)
    ns = normalize(span).strip()
    if not ns:
        return "nf", None, None

    pos = norm_passage.find(ns)
    if pos != -1:
        return "norm", origin[pos], origin[pos + len(ns)]

    found = _approx_span(norm_passage, ns)
    if found is not None:
        start, end = found
        if end > start:
            return "approx", origin[start], origin[end]
    return "nf", None, None
