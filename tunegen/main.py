"""TuneGen CLI.

  python main.py --city Delhi --budget 20000 --goal balanced     # live weather (falls back to typical climate if offline)
  python main.py --demo                                          # offline demo: simulated weather server + 3 mid run surprises
  python main.py --demo --outage                                 # demo where the weather API is down
  python main.py --city Pune --event 6:budget_inr=8000           # inject a belief change at tick 6
"""
import argparse
import os
import sys

from agent import TuneGen, load_kb, render_report
from weather import OpenMeteoClient, SimHandler, start_sim_server

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_event(s):
    tick, rest = s.split(":", 1)
    k, v = rest.split("=", 1)
    val = {"true": True, "false": False}.get(v.lower())
    if val is None:
        try:
            val = float(v) if k == "budget_inr" else v
        except ValueError:
            raise SystemExit(f"bad event value: {s}")
    return int(tick), (k, val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Chennai")
    ap.add_argument("--budget", type=float, default=20000)
    ap.add_argument("--goal", default="balanced", choices=["performance", "mileage", "balanced"])
    ap.add_argument("--warranty", action="store_true", help="car is under warranty")
    ap.add_argument("--demo", action="store_true", help="use the local simulated weather server")
    ap.add_argument("--outage", action="store_true", help="with --demo: simulate the weather API being down")
    ap.add_argument("--event", action="append", default=[], help="TICK:belief=value, e.g. 6:budget_inr=8000")
    a = ap.parse_args()

    events = {}
    for e in a.event:
        t, ev = parse_event(e)
        events.setdefault(t, []).append(ev)
    srv = None
    if a.demo:
        srv, base = start_sim_server()
        SimHandler.fail = a.outage
        client = OpenMeteoClient(f"{base}/v1/forecast", f"{base}/v1/search", timeout=3, retries=1)
        print("*** DEMO MODE: weather comes from a local SIMULATED server, not the real internet ***")
        if not a.event:    # scripted surprises: user reveals warranty, then budget drops
            events = {5: [("warranty_active", True)], 8: [("budget_inr", 8000.0)]}
    else:
        client = OpenMeteoClient()
    try:
        agent = TuneGen(load_kb(os.path.join(HERE, "knowledge.json")), client, a.city, a.budget, a.goal, a.warranty, events)
        agent.run()
        print("\n" + "=" * 78 + "\n" + render_report(agent))
    finally:
        if srv:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    sys.exit(main())
