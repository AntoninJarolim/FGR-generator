#!/usr/bin/env python3
"""Local viewer for extracted-relevancy outputs.

Serves ``visualize/static/index.html`` plus a small JSON API over two kinds of
sources:

* finished output files -- ``*.jsonl`` under ``--data-dir``
  (default ``data/extracted_relevancy``), one self-contained record per line;
* live, in-progress runs -- batch ``*_output.jsonl`` files under
  ``--generated-dir`` (default ``data/generated``). Those only hold raw model
  responses keyed by row id, so the server joins them back to the LongEmbed
  input (loaded once, same deterministic order as ``llm_extraction.py``) to
  reconstruct full records while the generation run is still going.

Sources are indexed lazily on first access (line offsets + light per-record
summaries) so GB-scale files never get shipped to the browser; records stream
one at a time. A source whose files changed (new batch finished, file
rewritten) is re-indexed automatically on the next request.
"""
import argparse
import gzip
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
sys.path.insert(0, REPO_ROOT)  # for utils.longembed when joining live runs

LIVE_PREFIX = "live:"


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


# ---------------------------------------------------------------------------
# LongEmbed input, needed to give live-run responses their query/passage back.
# Loaded once on first live-source access; order matches llm_extraction.py.
# ---------------------------------------------------------------------------
_INPUT = {"records": None}
_INPUT_LOCK = threading.Lock()


def longembed_input():
    with _INPUT_LOCK:
        if _INPUT["records"] is None:
            from utils.longembed import load_longembed
            print("[live] loading LongEmbed input records (one-time)...")
            _INPUT["records"] = load_longembed()
        return _INPUT["records"]


class JsonlIndex:
    """Line-offset index + per-record summaries for one finished JSONL file."""

    def __init__(self, path):
        self.path = path
        self._sig = None
        self.offsets = []
        self.summaries = []
        self.key_to_idx = {}
        self.lock = threading.Lock()

    def _signature(self):
        try:
            st = os.stat(self.path)
            return (st.st_mtime, st.st_size)
        except OSError:
            return None

    def ensure(self):
        with self.lock:
            sig = self._signature()
            if sig != self._sig:
                self._build()
                self._sig = sig

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


class LiveRunIndex:
    """Index over an in-progress run directory of batch ``*_output.jsonl``
    files, joined against the LongEmbed input by row id."""

    BATCH_NUM = re.compile(r"batch[_-](\d+)")

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self._sig = None
        self.summaries = []
        self.spans_by_idx = []
        self.row_ids = []
        self.key_to_idx = {}
        self.lock = threading.Lock()

    def _batch_files(self):
        try:
            names = [n for n in os.listdir(self.run_dir) if n.endswith("_output.jsonl")]
        except OSError:
            return []
        names.sort(key=lambda n: int(self.BATCH_NUM.search(n).group(1))
                   if self.BATCH_NUM.search(n) else 0)
        return [os.path.join(self.run_dir, n) for n in names]

    def _signature(self):
        return tuple((p, os.path.getsize(p)) for p in self._batch_files())

    def ensure(self):
        with self.lock:
            sig = self._signature()
            if sig != self._sig:
                self._build()
                self._sig = sig

    @staticmethod
    def _decode_spans(content):
        """Same tolerance as llm_extraction.get_all_responses: a response that
        is not a JSON object with a 'spans' list counts as zero spans."""
        try:
            obj = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return []
        if isinstance(obj, dict) and isinstance(obj.get("spans"), list):
            return [s for s in obj["spans"] if isinstance(s, str)]
        return []

    def _build(self):
        responses = {}
        for path in self._batch_files():
            with open(path, "rb") as f:
                for line in f:
                    try:
                        parsed = json.loads(line)
                        row_id = int(parsed["custom_id"].removeprefix("row_"))
                        content = parsed["response"]["body"]["choices"][0]["message"]["content"]
                    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
                        continue
                    responses[row_id] = content

        records = longembed_input()
        summaries, spans_by_idx, row_ids, key_to_idx = [], [], [], {}
        for row_id in sorted(responses):
            if not 0 <= row_id < len(records):
                continue
            idx = len(summaries)
            spans = self._decode_spans(responses[row_id])
            summary = make_summary(idx, records[row_id], spans)
            summary["row_id"] = row_id
            summaries.append(summary)
            spans_by_idx.append(spans)
            row_ids.append(row_id)
            key_to_idx.setdefault(summary["key"], idx)
        self.summaries, self.spans_by_idx = summaries, spans_by_idx
        self.row_ids, self.key_to_idx = row_ids, key_to_idx
        print(f"[index] live {os.path.basename(self.run_dir)}: "
              f"{len(summaries)} of {len(records)} records generated so far")

    def __len__(self):
        return len(self.summaries)

    def read_record(self, idx):
        rec = dict(longembed_input()[self.row_ids[idx]])
        rec["selected_spans"] = self.spans_by_idx[idx]
        return rec


class Registry:
    """Discovers finished files and live run dirs; caches their indexes."""

    def __init__(self, data_dir, generated_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.generated_dir = os.path.abspath(generated_dir)
        self.indexes = {}
        self.lock = threading.Lock()

    def list_sources(self):
        found = []
        for root, _dirs, files in os.walk(self.data_dir):
            for fn in sorted(files):
                if fn.endswith(".jsonl"):
                    full = os.path.join(root, fn)
                    found.append({
                        "name": os.path.relpath(full, self.data_dir),
                        "size": os.path.getsize(full),
                        "rows": self._rows_if_fresh(os.path.relpath(full, self.data_dir)),
                    })
        if os.path.isdir(self.generated_dir):
            for template in sorted(os.listdir(self.generated_dir)):
                tdir = os.path.join(self.generated_dir, template)
                if not os.path.isdir(tdir):
                    continue
                for run in sorted(os.listdir(tdir)):
                    rdir = os.path.join(tdir, run)
                    if not os.path.isdir(rdir):
                        continue
                    batches = [n for n in os.listdir(rdir) if n.endswith("_output.jsonl")]
                    if not batches:
                        continue
                    name = f"{LIVE_PREFIX}{template}/{run}"
                    found.append({
                        "name": name,
                        "size": sum(os.path.getsize(os.path.join(rdir, n)) for n in batches),
                        "rows": self._rows_if_fresh(name),
                        "live": True,
                    })
        return sorted(found, key=lambda d: (d.get("live", False), d["name"]))

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
        if name.startswith(LIVE_PREFIX):
            rel = name[len(LIVE_PREFIX):]
            full = os.path.abspath(os.path.join(self.generated_dir, rel))
            if not full.startswith(self.generated_dir + os.sep) or not os.path.isdir(full):
                raise KeyError(name)
            return LiveRunIndex(full)
        full = os.path.abspath(os.path.join(self.data_dir, name))
        if (not full.startswith(self.data_dir + os.sep) or not full.endswith(".jsonl")
                or not os.path.isfile(full)):
            raise KeyError(name)
        return JsonlIndex(full)


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
                        help="Directory scanned (recursively) for finished *.jsonl output files.")
    parser.add_argument("--generated-dir", default=os.path.join(REPO_ROOT, "data/generated"),
                        help="Directory of in-progress runs (template/run/batch outputs).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    REGISTRY = Registry(args.data_dir, args.generated_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FGR viewer: http://{args.host}:{args.port}/")
    print(f"  finished outputs: {REGISTRY.data_dir}")
    print(f"  live runs:        {REGISTRY.generated_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
