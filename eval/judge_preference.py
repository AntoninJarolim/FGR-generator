"""Pairwise preference judging of span extractions, and its aggregation.

The three existing metric columns are proxies: answer-string containment on two
of four subsets, and two retrieval scores that report what GTE-ModernColBERT
thinks of a spans-only pseudo-document. None of them scores whether a *reader*
would be convinced, and none scores minimality -- whether a system dragged in
text it did not need. This module fills that gap with a blind pairwise judge.

Comprehensiveness is deliberately absent. It cannot be judged from a spans-only
view (you cannot see what both systems missed) and whole-document prompts do not
fit the budget; it belongs to a short-document dataset instead.

Protocol per comparison: one query, both systems' views (eval/judge_view.py),
anonymised as A and B in a randomised order, and four labels --

    plausible_a / plausible_b   yes | no    absolute, per system
    minimality                  a | b | tie comparative
    overall                     a | b | tie comparative

Plausibility is absolute because it is defined that way ("a person reading only
the span would be convinced"); an absolute rate is quotable on its own and is
what identifies samples where *both* systems failed. There is deliberately no
"both bad" verdict: it duplicated plausible_a == plausible_b == "no", it threw
away the "both weak but A is less weak" ordering, and Bradley-Terry folds it into
a tie regardless.

Comparisons that need no judge are resolved without spending a call:

  * both systems empty          -> skipped entirely, excluded from every rate
  * span sets byte-identical    -> recorded as a tie
  * exactly one system empty    -> SENT to the judge. An empty selection can be
    the better answer when the document does not in fact answer the query, so
    this is a real judgement, not a walkover.

Stages, in order. Every stage is resumable and writes its own artifact:

    source eval/eval_env.sh          # views/sampling: numpy only
    $PYTHON eval/judge_preference.py --stage sample
    $PYTHON eval/judge_view.py --systems-file eval/judge_systems.json \
        --items data/eval/judge_items.json
    # the API stages need the `anthropic` SDK and ANTHROPIC_API_KEY (.env_judge),
    # which live in the fgr-generator env, not the pylate one:
    python eval/judge_preference.py --stage smoke     # 1 call, verifies the request shape
    python eval/judge_preference.py --stage submit --limit 110    # pilot
    python eval/judge_preference.py --stage collect
    python eval/judge_preference.py --stage aggregate
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.answer_containment import run_key  # noqa: E402
from eval.heuristic_spans import sidecar_path  # noqa: E402
from eval.judge_view import EMPTY_VIEW, item_key  # noqa: E402

SUBSETS = ["narrativeqa", "qmsum", "summ_screen_fd", "2wikimqa"]

DEFAULT_SYSTEMS = os.path.join(REPO_ROOT, "eval/judge_systems.json")
DEFAULT_ITEMS = os.path.join(REPO_ROOT, "data/eval/judge_items.json")
DEFAULT_VIEW_DIR = os.path.join(REPO_ROOT, "data/eval/judge_views")
DEFAULT_WORK = os.path.join(REPO_ROOT, "data/eval/judge_work")
DEFAULT_JSON_OUT = os.path.join(REPO_ROOT, "data/eval/judge_preference.json")

MODEL = "claude-opus-5"
#: Thinking is ON by default on this model and its tokens bill as output, which
#: would multiply cost several-fold. The `reasoning` field is the explicit chain
#: of thought instead. `disabled` is valid only at effort <= high.
THINKING = {"type": "disabled"}
EFFORT = "low"
MAX_TOKENS = 600
#: temperature/top_p/top_k are REMOVED on this model generation and return a 400.
#: Determinism comes from constrained decoding plus logging every raw response.

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "plausible_a": {"type": "string", "enum": ["yes", "no"]},
        "plausible_b": {"type": "string", "enum": ["yes", "no"]},
        "minimality": {"type": "string", "enum": ["a", "b", "tie"]},
        "overall": {"type": "string", "enum": ["a", "b", "tie"]},
    },
    "required": ["reasoning", "plausible_a", "plausible_b", "minimality", "overall"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are evaluating two systems that extract evidence spans from a document.

Given a query and a document, each system selected a set of spans — contiguous
excerpts copied from the document — meant to justify why that document is
relevant to the query.

Definitions:
- Plausible: a person reading only the selected spans would be convinced the
  document is relevant to the query.
- Minimal: the selection contains only what is needed. Spans should be
  fine-grained phrases or clauses rather than whole paragraphs, and should not
  drag in text that does not help justify relevance.

Each system is shown separately, because the two chose different parts of the
document. Within a system, excerpts appear in document order. Selected text is
marked [[SPAN]]like this[[/SPAN]], surrounded by up to 300 characters of
neighbouring document text for readability. A leading or trailing "…" means the
excerpt is cut from a longer passage.

Rules:
- Only text inside [[SPAN]]…[[/SPAN]] was selected. The surrounding text is shown
  so the excerpt reads naturally — it is not part of the selection and must not
  count as evidence in that system's favour.
- Judge only what is shown. Do not assume anything about parts of the document
  you cannot see, and do not penalise a system for evidence that may exist
  elsewhere in the document.
- A system may select nothing, shown as (no spans selected). When a document does
  not in fact answer the query, selecting nothing is the correct response and is
  better than selecting text that fails to justify relevance. An empty selection
  cannot make a document look relevant, so its plausible verdict is "no" — but it
  may still be the better selection overall.
- Span order carries no meaning. Repeated or overlapping spans are a flaw.
- Minimality is not simply "fewer characters". One span covering a whole page is
  worse than several tight phrases; several redundant phrases are worse than one
  that suffices.
- Use "tie" sparingly, only when the two selections are genuinely equivalent. If
  one is even slightly better, say which. Do not answer "tie" because the choice
  is difficult.
- If neither selection is convincing, still say which is closer to useful.
  Record the failure through plausible_a / plausible_b, not through the comparisons.
- A and B are anonymous; their order is random and carries no meaning.
- Do not include internal or system XML tags in your response.

Reply with JSON:
  reasoning     — at most two sentences comparing the selections.
  plausible_a   — "yes" if a reader of A's spans alone would be convinced the
                  document is relevant to the query, else "no".
  plausible_b   — the same judgement for B.
  minimality    — "a" | "b" | "tie": which selection is better targeted.
  overall       — "a" | "b" | "tie": the better piece of evidence overall.
"""


def user_prompt(query, view_a, view_b):
    return (f"<query>\n{query}\n</query>\n\n"
            f"<system_a>\n{view_a}\n</system_a>\n\n"
            f"<system_b>\n{view_b}\n</system_b>")


# --------------------------- stage: sample ---------------------------

def stage_sample(args):
    """Stratified, deterministic item sample shared by every pair.

    The same items are judged for all pairs (a paired design), so a system
    cannot look good merely by drawing easier documents. Items must exist in
    every non-baseline system's sidecar.
    """
    systems = json.load(open(args.systems))
    per_system = {}
    for s in systems:
        if s["path"].startswith("baseline:"):
            continue
        keys = set()
        with open(sidecar_path(run_key(s["path"])), encoding="utf-8") as f:
            for line in f:
                try:
                    keys.add(item_key(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue
        per_system[s["label"]] = keys
        print(f"  {s['label']:16s} {len(keys)} items in sidecar")

    common = set.intersection(*per_system.values())
    print(f"intersection across {len(per_system)} systems: {len(common)} items")

    by_subset = defaultdict(list)
    for k in common:
        by_subset[k[0]].append(k)
    sample = []
    for sub in SUBSETS:
        ks = sorted(by_subset.get(sub, []),
                    key=lambda k: hashlib.sha256(f"judge|{k}".encode()).hexdigest())
        take = ks[:args.per_subset]
        sample += take
        print(f"  {sub:16s} {len(take)}/{len(ks)} available")
    os.makedirs(os.path.dirname(args.items), exist_ok=True)
    with open(args.items, "w") as f:
        json.dump([list(k) for k in sample], f)
    print(f"wrote {len(sample)} items -> {args.items}")


# --------------------------- comparison building ---------------------------

def load_views(systems, view_dir):
    out = {}
    for s in systems:
        path = os.path.join(view_dir,
                            s["label"].replace("/", "__").replace(":", "--") + ".jsonl")
        if not os.path.exists(path):
            sys.exit(f"no view cache for {s['label']} at {path}\n"
                     f"run: $PYTHON eval/judge_view.py --systems-file ... --items ...")
        d = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                d[item_key(r)] = r
        out[s["label"]] = d
    return out


def build_comparisons(systems, items, views, swap=False, seed=0):
    """-> (to_judge, resolved). Applies the skip rules; randomises A/B."""
    labels = [s["label"] for s in systems]
    to_judge, resolved = [], []
    for x, y in itertools.combinations(labels, 2):
        for k in items:
            va, vb = views[x].get(k), views[y].get(k)
            if va is None or vb is None:
                continue
            ex, ey = va["view"] == EMPTY_VIEW, vb["view"] == EMPTY_VIEW
            if ex and ey:
                resolved.append({"pair": [x, y], "item": list(k),
                                 "outcome": "skipped_both_empty"})
                continue
            if va["view"] == vb["view"]:
                resolved.append({"pair": [x, y], "item": list(k),
                                 "outcome": "auto_tie_identical", "winner": "tie"})
                continue
            # Deterministic per (pair, item) so a swapped re-run is exactly the
            # inverse assignment rather than a fresh coin flip.
            rnd = random.Random(f"{seed}|{x}|{y}|{k}")
            first_is_x = rnd.random() < 0.5
            if swap:
                first_is_x = not first_is_x
            sys_a, sys_b = (x, y) if first_is_x else (y, x)
            to_judge.append({
                "pair": [x, y], "item": list(k),
                "system_a": sys_a, "system_b": sys_b, "swapped": bool(swap),
                "query": views[sys_a][k]["query"],
                "view_a": views[sys_a][k]["view"], "view_b": views[sys_b][k]["view"],
            })
    return to_judge, resolved


def request_params(cmp_):
    p = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        # No cache_control: measured on the pilot, batch requests run in
        # parallel and cannot read a cache entry the others are still writing --
        # only 9% of cacheable tokens were served from cache, so the 1.25x write
        # premium on the other 91% made caching a net ~4% LOSS here.
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user",
                      "content": user_prompt(cmp_["query"], cmp_["view_a"],
                                             cmp_["view_b"])}],
        "thinking": THINKING,
        "output_config": {"effort": EFFORT,
                          "format": {"type": "json_schema", "schema": SCHEMA}},
    }
    return p


def client():
    from anthropic import Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env = os.path.join(REPO_ROOT, ".env_judge")
        if os.path.exists(env):
            for line in open(env):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("no ANTHROPIC_API_KEY (expected in .env_judge)")
    return Anthropic()


# --------------------------- stage: smoke ---------------------------

def stage_smoke(args):
    """One live call. Verifies the request shape and measures real token usage
    before a batch is submitted -- a malformed batch is expensive to discover."""
    systems = json.load(open(args.systems))
    items = [tuple(k) for k in json.load(open(args.items))]
    views = load_views(systems, args.view_dir)
    to_judge, _ = build_comparisons(systems, items, views, seed=args.seed)
    if not to_judge:
        sys.exit("nothing to judge")
    c, cmp_ = client(), to_judge[0]
    print(f"pair={cmp_['pair']} item={cmp_['item']}")
    t0 = time.time()
    r = c.messages.create(**request_params(cmp_))
    text = next(b.text for b in r.content if b.type == "text")
    print(f"\n--- parsed verdict ---\n{json.dumps(json.loads(text), indent=2)}")
    u = r.usage
    print(f"\n--- usage ---\ninput={u.input_tokens} output={u.output_tokens} "
          f"cache_read={getattr(u, 'cache_read_input_tokens', 0)} "
          f"cache_write={getattr(u, 'cache_creation_input_tokens', 0)} "
          f"({time.time() - t0:.1f}s)")
    cost = (u.input_tokens * 5 + u.output_tokens * 25) / 1e6
    print(f"standard ${cost:.5f}/comparison   batch ${cost/2:.5f}/comparison")
    print(f"stop_reason={r.stop_reason}")


# --------------------------- stage: submit / collect ---------------------------

def stage_submit(args):
    systems = json.load(open(args.systems))
    items = [tuple(k) for k in json.load(open(args.items))]
    views = load_views(systems, args.view_dir)
    to_judge, resolved = build_comparisons(systems, items, views,
                                           swap=args.swap, seed=args.seed)
    print(f"comparisons needing a judge: {len(to_judge)}")
    print(f"resolved without a call: {len(resolved)} "
          f"({sum(1 for r in resolved if r['outcome']=='skipped_both_empty')} both-empty, "
          f"{sum(1 for r in resolved if r['outcome']=='auto_tie_identical')} identical)")
    if args.limit:
        rnd = random.Random(f"limit|{args.seed}")
        rnd.shuffle(to_judge)                      # spread the pilot over all pairs
        to_judge = to_judge[:args.limit]
        print(f"--limit -> submitting {len(to_judge)}")

    os.makedirs(args.work, exist_ok=True)
    tag = args.tag
    with open(os.path.join(args.work, f"{tag}.resolved.jsonl"), "w",
              encoding="utf-8") as f:
        for r in resolved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    reqs, mapping = [], {}
    for i, cmp_ in enumerate(to_judge):
        cid = f"c{i:06d}"
        mapping[cid] = {k: cmp_[k] for k in
                        ("pair", "item", "system_a", "system_b", "swapped")}
        reqs.append({"custom_id": cid, "params": request_params(cmp_)})
    with open(os.path.join(args.work, f"{tag}.map.json"), "w") as f:
        json.dump(mapping, f)

    if args.dry_run:
        print("--dry-run: not submitting")
        return
    batch = client().messages.batches.create(requests=reqs)
    with open(os.path.join(args.work, f"{tag}.batch.json"), "w") as f:
        json.dump({"id": batch.id, "n": len(reqs), "swap": args.swap,
                   "created": time.time()}, f)
    print(f"submitted batch {batch.id} ({len(reqs)} requests) status={batch.processing_status}")


def stage_collect(args):
    c = client()
    meta = json.load(open(os.path.join(args.work, f"{args.tag}.batch.json")))
    mapping = json.load(open(os.path.join(args.work, f"{args.tag}.map.json")))
    while True:
        b = c.messages.batches.retrieve(meta["id"])
        if b.processing_status == "ended":
            break
        print(f"  {b.processing_status}: {b.request_counts}", file=sys.stderr)
        time.sleep(30)
    out = os.path.join(args.work, f"{args.tag}.verdicts.jsonl")
    n_ok = n_err = 0
    tok_in = tok_out = tok_cache_read = 0
    with open(out, "w", encoding="utf-8") as f:
        for res in c.messages.batches.results(meta["id"]):
            m = mapping[res.custom_id]
            if res.result.type != "succeeded":
                n_err += 1
                f.write(json.dumps({**m, "error": res.result.type}) + "\n")
                continue
            msg = res.result.message
            try:
                v = json.loads(next(b.text for b in msg.content if b.type == "text"))
            except (StopIteration, json.JSONDecodeError) as e:
                n_err += 1
                f.write(json.dumps({**m, "error": f"parse: {e}"}) + "\n")
                continue
            n_ok += 1
            # input_tokens counts only the UNCACHED prefix; cache creation/read
            # are billed separately (1.25x / 0.10x). Omitting them understates
            # the bill -- on the pilot by ~30%.
            tok_in += (msg.usage.input_tokens
                       + getattr(msg.usage, "cache_creation_input_tokens", 0) or 0)
            tok_cache_read += getattr(msg.usage, "cache_read_input_tokens", 0) or 0
            tok_out += msg.usage.output_tokens
            # Normalise to system identities so a swapped re-run pools with this
            # one instead of being a throwaway diagnostic.
            def sysname(lbl):
                return m["system_a"] if lbl == "a" else m["system_b"]
            f.write(json.dumps({
                **m, "raw": v,
                "plausible": {m["system_a"]: v["plausible_a"],
                              m["system_b"]: v["plausible_b"]},
                "minimality_winner": "tie" if v["minimality"] == "tie" else sysname(v["minimality"]),
                "overall_winner": "tie" if v["overall"] == "tie" else sysname(v["overall"]),
                "usage": {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens},
            }, ensure_ascii=False) + "\n")
    # cache writes already folded into tok_in at 1x; add the 0.25x premium and
    # the 0.10x reads. Batch is 50% off.
    cost = (tok_in * 5 + tok_cache_read * 5 * 0.10 + tok_out * 25) / 1e6 / 2
    print(f"{n_ok} ok, {n_err} failed -> {out}")
    if n_ok:
        print(f"tokens: in={tok_in} cache_read={tok_cache_read} out={tok_out} "
              f"(mean {tok_in//n_ok} in / {tok_out//n_ok} out per comparison)")
        print(f"batch cost ${cost:.4f} total, ${cost/n_ok:.5f} per comparison")


# --------------------------- stage: aggregate ---------------------------

def wilson(k, n, z=1.96):
    """Wilson score interval -- behaves sensibly at small n and near 0/1."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def fit_bt(wins, labels, iters=2000, tol=1e-10):
    """Bradley-Terry by the MM (Zermelo) iteration. wins[(i,j)] = i's win count
    over j, ties already split half to each side."""
    p = {l: 1.0 for l in labels}
    tot = {l: sum(wins.get((l, o), 0) for o in labels if o != l) for l in labels}
    for _ in range(iters):
        new = {}
        for i in labels:
            denom = 0.0
            for j in labels:
                if i == j:
                    continue
                n_ij = wins.get((i, j), 0) + wins.get((j, i), 0)
                if n_ij:
                    denom += n_ij / (p[i] + p[j])
            new[i] = tot[i] / denom if denom > 0 else p[i]
        g = sum(new.values()) / len(new) or 1.0
        new = {k: v / g for k, v in new.items()}
        if max(abs(new[k] - p[k]) for k in p) < tol:
            p = new
            break
        p = new
    return p


def stage_aggregate(args):
    systems = [s["label"] for s in json.load(open(args.systems))]
    verdicts, resolved = [], []
    for fn in sorted(os.listdir(args.work)):
        p = os.path.join(args.work, fn)
        if fn.endswith(".verdicts.jsonl"):
            verdicts += [json.loads(l) for l in open(p, encoding="utf-8")
                         if l.strip() and "error" not in json.loads(l)]
        elif fn.endswith(".resolved.jsonl"):
            resolved += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print(f"{len(verdicts)} judged, {len(resolved)} resolved without a call")
    if not verdicts:
        sys.exit("no verdicts yet")

    # ---- flip rate: same (pair,item) judged in both orders
    by_key = defaultdict(list)
    for v in verdicts:
        by_key[(tuple(v["pair"]), tuple(v["item"]))].append(v)
    both = [g for g in by_key.values() if len({x["swapped"] for x in g}) == 2]
    flips = sum(1 for g in both
                if {x["overall_winner"] for x in g} != {g[0]["overall_winner"]})
    flip_rate = flips / len(both) if both else None

    # ---- pairwise win matrix + BT input (ties split)
    cnt = defaultdict(lambda: defaultdict(int))
    wins_half = defaultdict(float)
    for v in verdicts + [r for r in resolved if r.get("winner") == "tie"]:
        x, y = v["pair"]
        w = v.get("overall_winner", v.get("winner"))
        cnt[(x, y)]["n"] += 1
        if w == "tie":
            cnt[(x, y)]["tie"] += 1
            wins_half[(x, y)] += 0.5
            wins_half[(y, x)] += 0.5
        else:
            cnt[(x, y)][w] += 1
            wins_half[(w, y if w == x else x)] += 1

    print("\n=== pairwise overall win rate (row beats column), decisive only ===")
    hdr = "  " + "".join(f"{s[:13]:>15s}" for s in systems)
    print(f"{'':22s}{hdr}")
    matrix = {}
    for a in systems:
        row = f"  {a[:20]:20s}"
        for b in systems:
            if a == b:
                row += f"{'—':>15s}"
                continue
            key = (a, b) if (a, b) in cnt else (b, a)
            c = cnt.get(key)
            if not c:
                row += f"{'·':>15s}"
                continue
            dec = c["n"] - c["tie"]
            k = c.get(a, 0)
            if dec == 0:
                row += f"{'all tie':>15s}"
                continue
            p, lo, hi = wilson(k, dec)
            matrix[f"{a}|{b}"] = {"win_rate": p, "ci_low": lo, "ci_high": hi,
                                  "decisive": dec, "n": c["n"], "ties": c["tie"]}
            row += f"{p*100:8.1f}±{(hi-lo)/2*100:4.1f}"
        print(row)

    # ---- transitivity over triads
    def beats(a, b):
        m = matrix.get(f"{a}|{b}")
        return None if not m else (m["win_rate"] > 0.5)
    cycles = []
    for tri in itertools.combinations(systems, 3):
        for a, b, c_ in itertools.permutations(tri):
            if beats(a, b) and beats(b, c_) and beats(c_, a):
                cycles.append([a, b, c_])
                break
    print(f"\ntransitivity: {len(cycles)} intransitive triad(s) of "
          f"{len(list(itertools.combinations(systems, 3)))}")
    for c_ in cycles:
        print(f"  cycle: {' > '.join(c_)} > {c_[0]}")

    # ---- absolute plausibility rate (model-free)
    plaus = defaultdict(lambda: [0, 0])
    for v in verdicts:
        for s, yn in v.get("plausible", {}).items():
            plaus[s][1] += 1
            plaus[s][0] += (yn == "yes")
    print("\n=== absolute plausibility rate (model-free) ===")
    for s in systems:
        k, n = plaus.get(s, [0, 0])
        if n:
            p, lo, hi = wilson(k, n)
            print(f"  {s:22s} {p*100:5.1f}%  [{lo*100:5.1f},{hi*100:5.1f}]  n={n}")

    # ---- minimality
    mini = defaultdict(lambda: [0.0, 0])
    for v in verdicts:
        x, y = v["pair"]
        w = v["minimality_winner"]
        for s in (x, y):
            mini[s][1] += 1
        if w == "tie":
            mini[x][0] += 0.5
            mini[y][0] += 0.5
        else:
            mini[w][0] += 1
    print("\n=== minimality win rate (ties split) ===")
    for s in systems:
        k, n = mini.get(s, [0, 0])
        if n:
            print(f"  {s:22s} {100*k/n:5.1f}%  n={n}")

    # ---- BT
    bt = fit_bt(wins_half, systems)
    z = sum(bt.values())
    print("\n=== Bradley-Terry (secondary; matrix above is primary) ===")
    for s in sorted(systems, key=lambda s: -bt[s]):
        print(f"  {s:22s} theta={bt[s]:7.3f}  normalised={bt[s]/z*100:5.1f}%")

    ties_tot = sum(c["tie"] for c in cnt.values())
    n_tot = sum(c["n"] for c in cnt.values())
    both_bad = sum(1 for v in verdicts
                   if set(v.get("plausible", {}).values()) == {"no"})
    print(f"\ntie rate {100*ties_tot/max(1,n_tot):.1f}%   "
          f"both-bad {100*both_bad/max(1,len(verdicts)):.1f}%   "
          f"flip rate {'n/a' if flip_rate is None else f'{100*flip_rate:.1f}% (n={len(both)})'}")

    # ---- artifact for eval/summary_table.py
    runs = {}
    for s in systems:
        k, n = plaus.get(s, [0, 0])
        per = {"bt_winrate": bt[s] / z, "bt_theta": bt[s],
               "plausible_rate": (k / n if n else None), "plausible_n": n,
               "minimality_winrate": (mini[s][0] / mini[s][1] if mini[s][1] else None)}
        runs[s] = {sub: per for sub in SUBSETS}      # pooled fit; see docstring
    data = {"judge_model": MODEL, "effort": EFFORT, "thinking": THINKING["type"],
            "n_judged": len(verdicts), "n_resolved": len(resolved),
            "tie_rate": ties_tot / max(1, n_tot),
            "both_bad_rate": both_bad / max(1, len(verdicts)),
            "flip_rate": flip_rate, "flip_n": len(both),
            "matrix": matrix, "intransitive_triads": cycles, "runs": runs}
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.json_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["sample", "smoke", "submit", "collect", "aggregate"])
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS)
    ap.add_argument("--items", default=DEFAULT_ITEMS)
    ap.add_argument("--view-dir", default=DEFAULT_VIEW_DIR)
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    ap.add_argument("--per-subset", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap comparisons submitted (pilot)")
    ap.add_argument("--swap", action="store_true",
                    help="invert every A/B assignment (position-bias run)")
    ap.add_argument("--tag", default="main", help="names this run's artifacts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"sample": stage_sample, "smoke": stage_smoke, "submit": stage_submit,
     "collect": stage_collect, "aggregate": stage_aggregate}[args.stage](args)


if __name__ == "__main__":
    main()
