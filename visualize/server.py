#!/usr/bin/env python3
"""Local viewer for extracted-relevancy outputs.

Serves ``visualize/static/index.html`` plus a small JSON API over the
``*.jsonl`` files under ``--data-dir`` (default ``data/extracted_relevancy``),
one self-contained record per line.

Pages: ``/`` dashboard, ``/viz`` data viewer, ``/stats`` span mismatch stats.

Files are indexed lazily on first access (line offsets + light per-record
summaries) so GB-scale files never get shipped to the browser; records stream
one at a time. A file whose mtime/size changed (e.g. a run finished and
rewrote it) is re-indexed automatically on the next request. Indexes and
match-rate statistics are cached under ``visualize/.cache/``.

Stdlib only, except the stats computation which uses numpy (see stats.py).
"""
import argparse
import gzip
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def record_key(rec):
    """Join key used to line up the same sample across model output files."""
    return f"{rec.get('subset')}|{rec.get('qid')}|{rec.get('doc_id')}"


def make_summary(idx, rec, spans):
    # "clean_text"/"claim"/"document_id" are the older FDM output schema; a
    # non-string span item is malformed model output found in some old files.
    passage = rec.get("passage") or rec.get("clean_text") or ""
    return {
        "idx": idx,
        "key": record_key(rec),
        "subset": rec.get("subset"),
        "qid": rec.get("qid"),
        "doc_id": rec.get("doc_id") or rec.get("document_id"),
        "query": (rec.get("query") or rec.get("claim") or "")[:200],
        "n_spans": len(spans),
        "n_exact": sum(1 for s in spans if isinstance(s, str) and s and s in passage),
        "was_truncated": rec.get("was_truncated"),
    }


class JsonlIndex:
    """Line-offset index + per-record summaries for one finished JSONL file.

    Building means one full pass over the (possibly multi-GB) file, ~11 s per
    3.6 GB, so the result is persisted to a gzipped sidecar under
    ``visualize/.cache/`` and reloaded (<1 s) on later server runs. The cache
    is keyed on the data file's (mtime, size) and rebuilt when they change.
    """

    def __init__(self, path, cache_path):
        self.path = path
        self.cache_path = cache_path
        self._sig = None
        self.offsets = []
        self.summaries = []
        self.key_to_idx = {}
        self.lock = threading.Lock()
        # Span match-rate stats (see stats.py): computed once, cached beside
        # the index cache, guarded by its own lock so a minutes-long compute
        # never blocks record/list requests for this file.
        self.stats_path = cache_path.removesuffix(".idx.json.gz") + ".stats.json"
        self.stats = None
        self._stats_sig = None
        self.stats_lock = threading.Lock()

    def _signature(self):
        try:
            st = os.stat(self.path)
            return [st.st_mtime, st.st_size]   # list: compares == after JSON round-trip
        except OSError:
            return None

    def ensure(self):
        with self.lock:
            sig = self._signature()
            if sig == self._sig:
                return
            if not self._load_cache(sig):
                self._build()
                self._save_cache(sig)
            self._sig = sig

    def _load_cache(self, sig):
        try:
            with gzip.open(self.cache_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("sig") != sig:
                return False
        except (OSError, json.JSONDecodeError, EOFError):
            return False
        self.offsets, self.summaries = data["offsets"], data["summaries"]
        self.key_to_idx = {}
        for s in self.summaries:
            self.key_to_idx.setdefault(s["key"], s["idx"])
        print(f"[index] {os.path.basename(self.path)}: {len(self.offsets)} records (from cache)")
        return True

    def _save_cache(self, sig):
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            gitignore = os.path.join(CACHE_DIR, ".gitignore")
            if not os.path.exists(gitignore):
                with open(gitignore, "w") as f:
                    f.write("*\n")
            tmp = self.cache_path + ".tmp"
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump({"sig": sig, "offsets": self.offsets, "summaries": self.summaries}, f)
            os.replace(tmp, self.cache_path)
        except OSError as e:
            print(f"[index] warning: could not write cache {self.cache_path}: {e}")

    def _build(self):
        offsets, summaries, key_to_idx = [], [], {}
        with open(self.path, "rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = len(offsets)
                summary = make_summary(idx, rec, rec.get("selected_spans") or [])
                summaries.append(summary)
                key_to_idx.setdefault(summary["key"], idx)
                offsets.append(off)
        self.offsets, self.summaries, self.key_to_idx = offsets, summaries, key_to_idx
        print(f"[index] {os.path.basename(self.path)}: {len(offsets)} records")

    def __len__(self):
        return len(self.offsets)

    def read_record(self, idx):
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            return json.loads(f.readline())

    def ensure_stats(self):
        """Return the match-rate stats rows, computing (one full pass +
        matching, ~1 min per file) or loading the sidecar cache as needed."""
        self.ensure()
        with self.stats_lock:
            sig = self._signature()
            if self.stats is not None and self._stats_sig == sig:
                return self.stats
            from stats import compute_stats, STATS_VERSION
            try:
                with open(self.stats_path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("sig") == sig and data.get("v") == STATS_VERSION:
                    self.stats, self._stats_sig = data["stats"], sig
                    return self.stats
            except (OSError, json.JSONDecodeError):
                pass
            t0 = time.time()
            self.stats = compute_stats(self.path)
            self._stats_sig = sig
            print(f"[stats] {os.path.basename(self.path)}: computed in {time.time() - t0:.1f}s")
            try:
                tmp = self.stats_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"sig": sig, "v": STATS_VERSION, "stats": self.stats}, f)
                os.replace(tmp, self.stats_path)
            except OSError as e:
                print(f"[stats] warning: could not write cache {self.stats_path}: {e}")
            return self.stats


class Registry:
    """Discovers output files under the data dir and caches their indexes."""

    def __init__(self, data_dir, include=""):
        self.data_dir = os.path.abspath(data_dir)
        self.include = include
        self.indexes = {}
        self.lock = threading.Lock()

    def list_sources(self):
        found = []
        for root, _dirs, files in os.walk(self.data_dir):
            for fn in sorted(files):
                if fn.endswith(".jsonl"):
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, self.data_dir)
                    if self.include and self.include not in rel:
                        continue
                    found.append({
                        "name": rel,
                        "size": os.path.getsize(full),
                        "rows": self._rows_if_fresh(rel),
                    })
        return sorted(found, key=lambda d: d["name"])

    def _rows_if_fresh(self, name):
        idx = self.indexes.get(name)
        return len(idx) if idx and idx._signature() == idx._sig else None

    def get(self, name):
        with self.lock:
            index = self.indexes.get(name)
            if index is None:
                index = self.indexes[name] = self._create(name)
        index.ensure()
        return index

    def _create(self, name):
        full = os.path.abspath(os.path.join(self.data_dir, name))
        if (not full.startswith(self.data_dir + os.sep) or not full.endswith(".jsonl")
                or not os.path.isfile(full)):
            raise KeyError(name)
        cache_name = re.sub(r"[^A-Za-z0-9._-]", "_", name) + ".idx.json.gz"
        return JsonlIndex(full, os.path.join(CACHE_DIR, cache_name))


REGISTRY = None    # set in main()
GOLD_PATH = None   # set in main(): path to data/eval/gold_answers.jsonl
GOLD_ANSWERS = None
GOLD_LOCK = threading.Lock()


def load_gold():
    """Lazily load gold_answers.jsonl into a {(subset, qid): record} map.

    The file is small (a few MB) and produced by eval/fetch_gold_answers.py;
    one record per (subset, qid) including misses. Loaded once, then cached.
    """
    global GOLD_ANSWERS
    with GOLD_LOCK:
        if GOLD_ANSWERS is not None:
            return GOLD_ANSWERS
        gold = {}
        if GOLD_PATH and os.path.isfile(GOLD_PATH):
            with open(GOLD_PATH, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    gold[(rec.get("subset"), rec.get("qid"))] = rec
            print(f"[gold] loaded {len(gold)} records from {GOLD_PATH}")
        else:
            print(f"[gold] no gold-answers file at {GOLD_PATH}")
        GOLD_ANSWERS = gold
        return GOLD_ANSWERS


FLAG_TESTS = {
    "truncated": lambda s: bool(s.get("was_truncated")),
    "nonverbatim": lambda s: s["n_exact"] < s["n_spans"],
    "empty": lambda s: s["n_spans"] == 0,
}


def filter_summaries(index, params):
    q = (params.get("q") or "").lower()
    subset = params.get("subset") or ""
    flag_test = FLAG_TESTS.get(params.get("flag") or "")
    out = []
    for s in index.summaries:
        if subset and s["subset"] != subset:
            continue
        if flag_test and not flag_test(s):
            continue
        if q and q not in s["query"].lower():
            continue
        out.append(s)
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    def _send(self, status, body, content_type):
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
        headers = [("Content-Type", content_type), ("Cache-Control", "no-store")]
        if accepts_gzip and len(body) > 2048:
            body = gzip.compress(body, 5)
            headers.append(("Content-Encoding", "gzip"))
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status, message):
        self._json({"error": message}, status=status)

    PAGES = {
        "/": "home.html",          # dashboard
        "/viz": "index.html",      # data viewer
        "/stats": "stats.html",    # span mismatch statistics
    }

    def do_GET(self):
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            route = getattr(self, "route_" + parsed.path.strip("/").replace("api/", "api_"), None)
            if parsed.path in self.PAGES:
                with open(os.path.join(STATIC_DIR, self.PAGES[parsed.path]), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif route:
                route(params)
            else:
                self._error(404, f"unknown path {parsed.path}")
        except KeyError as e:
            self._error(404, f"unknown source or field: {e}")
        except (IndexError, ValueError) as e:
            self._error(400, str(e))
        except BrokenPipeError:
            pass

    # --- API routes (dispatched by path name) ---

    def route_api_files(self, params):
        self._json({"files": REGISTRY.list_sources()})

    def route_api_filemeta(self, params):
        index = REGISTRY.get(params["file"])
        subsets = sorted({s["subset"] for s in index.summaries if s["subset"]})
        counts = {name: sum(1 for s in index.summaries if test(s))
                  for name, test in FLAG_TESTS.items()}
        self._json({"rows": len(index), "subsets": subsets, "counts": counts})

    def route_api_records(self, params):
        index = REGISTRY.get(params["file"])
        filtered = filter_summaries(index, params)
        offset = max(0, int(params.get("offset", 0)))
        limit = min(1000, max(1, int(params.get("limit", 200))))
        self._json({"total": len(filtered), "items": filtered[offset:offset + limit]})

    def route_api_record(self, params):
        index = REGISTRY.get(params["file"])
        idx = int(params["idx"])
        if not 0 <= idx < len(index):
            raise ValueError(f"idx {idx} out of range")
        self._json({"idx": idx, "record": index.read_record(idx)})

    def route_api_stats(self, params):
        index = REGISTRY.get(params["file"])
        self._json({"file": params["file"], "groups": index.ensure_stats()})

    def route_api_gold(self, params):
        rec = load_gold().get((params.get("subset"), params.get("qid")))
        if rec is None or rec.get("join") == "miss":
            self._json({"found": False, "answers": []})
        else:
            self._json({
                "found": True,
                "answers": rec.get("answers") or [],
                "ambiguous": bool(rec.get("ambiguous")),
                "join": rec.get("join"),
                "n_gold_docs": rec.get("n_gold_docs"),
            })

    def route_api_recordByKey(self, params):
        index = REGISTRY.get(params["file"])
        idx = index.key_to_idx.get(params.get("key", ""))
        if idx is None:
            self._json({"idx": None, "record": None})
        else:
            self._json({"idx": idx, "record": index.read_record(idx)})


def main():
    global REGISTRY, GOLD_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data/extracted_relevancy"),
                        help="Directory scanned (recursively) for *.jsonl output files.")
    parser.add_argument("--include", default="long-embed",
                        help="Only list files whose path (relative to --data-dir) contains this "
                             "substring; matches e.g. long-embed-json, long-embed-xml, "
                             "long-embed-xml-constrained. Pass '' to list everything.")
    parser.add_argument("--gold", default=os.path.join(REPO_ROOT, "data/eval/gold_answers.jsonl"),
                        help="Gold QA answers (from eval/fetch_gold_answers.py), joined by (subset, qid).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    REGISTRY = Registry(args.data_dir, include=args.include)
    GOLD_PATH = os.path.abspath(args.gold)

    # Pre-warm every file's index in the background so first interactions are
    # instant. With a valid sidecar cache each file loads in <1 s; otherwise it
    # builds (and caches) here instead of blocking the first browser request.
    def prewarm():
        for f in REGISTRY.list_sources():
            try:
                t0 = time.time()
                REGISTRY.get(f["name"]).ensure_stats()
                print(f"[prewarm] {f['name']} ready in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"[prewarm] {f['name']} failed: {e}")
    threading.Thread(target=prewarm, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FGR viewer: http://{args.host}:{args.port}/  (data dir: {REGISTRY.data_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
