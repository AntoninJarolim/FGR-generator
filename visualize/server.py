#!/usr/bin/env python3
"""Local viewer for extracted-relevancy outputs.

Serves ``visualize/static/index.html`` plus a small JSON API over the
``*.jsonl`` files under ``--data-dir`` (default ``data/extracted_relevancy``),
one self-contained record per line.

Files are indexed lazily on first access (line offsets + light per-record
summaries) so GB-scale files never get shipped to the browser; records stream
one at a time. A file whose mtime/size changed (e.g. a run finished and
rewrote it) is re-indexed automatically on the next request.

Stdlib only -- no third-party dependencies.
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
    passage = rec.get("passage") or ""
    return {
        "idx": idx,
        "key": record_key(rec),
        "subset": rec.get("subset"),
        "qid": rec.get("qid"),
        "doc_id": rec.get("doc_id"),
        "query": (rec.get("query") or "")[:200],
        "n_spans": len(spans),
        "n_exact": sum(1 for s in spans if s and s in passage),
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


class Registry:
    """Discovers output files under the data dir and caches their indexes."""

    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.indexes = {}
        self.lock = threading.Lock()

    def list_sources(self):
        found = []
        for root, _dirs, files in os.walk(self.data_dir):
            for fn in sorted(files):
                if fn.endswith(".jsonl"):
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, self.data_dir)
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


REGISTRY = None  # set in main()

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

    def do_GET(self):
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            route = getattr(self, "route_" + parsed.path.strip("/").replace("api/", "api_"), None)
            if parsed.path in ("/", "/index.html"):
                with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
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

    def route_api_recordByKey(self, params):
        index = REGISTRY.get(params["file"])
        idx = index.key_to_idx.get(params.get("key", ""))
        if idx is None:
            self._json({"idx": None, "record": None})
        else:
            self._json({"idx": idx, "record": index.read_record(idx)})


def main():
    global REGISTRY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data/extracted_relevancy"),
                        help="Directory scanned (recursively) for *.jsonl output files.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    REGISTRY = Registry(args.data_dir)

    # Pre-warm every file's index in the background so first interactions are
    # instant. With a valid sidecar cache each file loads in <1 s; otherwise it
    # builds (and caches) here instead of blocking the first browser request.
    def prewarm():
        for f in REGISTRY.list_sources():
            try:
                t0 = time.time()
                REGISTRY.get(f["name"])
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
