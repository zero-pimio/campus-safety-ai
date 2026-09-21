"""Exercise the pinned upstream hook functions over local HTTP.

The upstream routing functions are real; Flask plumbing and the downstream
process_alert_hook service are substitutes. This does not test persistence,
authentication, Kafka, image storage, or the platform UI.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

from campus_safety_ai.adapters.easyaiot import EasyAIoTDeliveryError, _post_json

PIN = "49f00960a2907c0066b2c3f6e36c8cb9c9239320"


def verify(upstream: Path) -> dict:
    head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PIN:
        raise ValueError(f"upstream HEAD differs from the audited version: {head}")
    source = upstream / "VIDEO/app/blueprints/alert.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {"api_response", "alert_hook"}]
    if len(nodes) != 2:
        raise ValueError("upstream hook functions changed")
    for node in nodes:
        node.decorator_list = []
    state = {"result": {"status": "success"}, "request": {}}
    namespace = {
        "jsonify": lambda value: value,
        "request": SimpleNamespace(get_json=lambda: state["request"]),
        "process_alert_hook": lambda data: state["result"],
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/video/alert/hook":
                self.send_error(404)
                return
            state["request"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload, status = namespace["alert_hook"]()
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    cases = []
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/video/alert/hook"
        for status in ("success", "skipped", "suppressed", "failed"):
            state["result"] = {"status": status}
            accepted = False
            try:
                _post_json(endpoint, b'{"device_id":"contract-test"}', {}, 2)
                accepted = True
            except EasyAIoTDeliveryError:
                pass
            if accepted != (status == "success"):
                raise AssertionError(f"unexpected acceptance for {status}")
            cases.append({"upstream_status": status, "accepted": accepted})
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
    return {"upstream_commit": head, "cases": cases,
            "scope": "real upstream handler functions over local HTTP; downstream service substituted",
            "full_platform_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.upstream), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")
