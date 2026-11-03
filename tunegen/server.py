"""TuneGen web front end. Python standard library only.

  python server.py            then open  http://127.0.0.1:8000  in your browser
  python server.py --port 9000
Binds to 127.0.0.1 only, so it is reachable from your own computer and nobody else.
"""
import argparse
import dataclasses
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent import TuneGen, load_kb
from weather import OpenMeteoClient, SimHandler, start_sim_server

HERE = os.path.dirname(os.path.abspath(__file__))
KB = load_kb(os.path.join(HERE, "knowledge.json"))


def run_agent(p: dict) -> dict:
    city = str(p.get("city", "")).strip()[:60]
    if not city:
        raise ValueError("city is required")
    try:
        budget = float(p.get("budget"))
    except (TypeError, ValueError):
        raise ValueError("budget must be a number")
    goal = p.get("goal", "balanced")
    events = {}
    try:
        if p.get("warranty_tick"):
            events.setdefault(int(p["warranty_tick"]), []).append(("warranty_active", True))
        if p.get("budget_tick") and p.get("new_budget"):
            events.setdefault(int(p["budget_tick"]), []).append(("budget_inr", float(p["new_budget"])))
    except (TypeError, ValueError):
        raise ValueError("event ticks and new budget must be numbers")
    if any(t < 1 or t > 60 for t in events):
        raise ValueError("event ticks must be between 1 and 60")

    demo, srv = bool(p.get("demo")), None
    try:
        if demo:
            srv, base = start_sim_server()
            SimHandler.fail = bool(p.get("outage"))
            client = OpenMeteoClient(f"{base}/v1/forecast", f"{base}/v1/search", timeout=3, retries=1)
        else:
            client = OpenMeteoClient(timeout=6, retries=1)
        agent = TuneGen(KB, client, city, budget, goal, bool(p.get("warranty")), events, verbose=False)
        rec = agent.run()
    finally:
        if srv:
            srv.shutdown()
            srv.server_close()
    b = agent.beliefs
    return {
        "trace": agent.bdi.trace,
        "recommendation": rec,
        "observation": dataclasses.asdict(b.get("obs")) if b.get("obs") else None,
        "desires": [{"name": d.name, "priority": d.priority, "status": d.status.value, "description": d.description} for d in agent.desires],
        "env_changes": [{"key": k, "old": o, "new": n} for k, o, n, s in b.history if s == "environment"],
        "final_budget": b.get("budget_inr"), "warranty_active": b.get("warranty_active"), "car": b.get("car_name"),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, status, ctype, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "static", "index.html"), "rb") as f:
                return self._send(200, "text/html; charset=utf-8", f.read())
        if self.path == "/api/cities":
            return self._send(200, "application/json", json.dumps(list(KB["cities"])).encode())
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, "text/plain", b"not found")
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 10_000:
                raise ValueError("request too large")
            result = run_agent(json.loads(self.rfile.read(n) or b"{}"))
            self._send(200, "application/json", json.dumps(result, default=str).encode())
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, "application/json", json.dumps({"error": str(e)}).encode())
        except Exception as e:                                   # never crash the server on one bad request
            self._send(500, "application/json", json.dumps({"error": f"internal error: {e}"}).encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError as e:
        sys.exit(f"Could not start on port {a.port}: {e}. Try: python server.py --port 9000")
    print(f"TuneGen is running. Open http://127.0.0.1:{a.port} in your browser. Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
