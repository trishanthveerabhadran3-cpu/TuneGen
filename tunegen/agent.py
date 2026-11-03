"""TuneGen: the integrated agent. Knowledge base (JSON) + live weather API + BDI engine + utility evaluator."""
from __future__ import annotations

import json
import urllib.parse
from typing import Any, Dict, List, Optional

from bdi import BDIAgent, Beliefs, Desire, Step
from utility import best_bundle, reject_reason, score_all
from weather import APIError, Observation, OpenMeteoClient, air_density_ratio, parse_observation

OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b}


def load_kb(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        kb = json.load(f)
    for k in ("vehicle", "cities", "goal_weights", "utility", "wmo_weather_codes", "parts", "rules"):
        if k not in kb:
            raise ValueError(f"knowledge base missing section: {k}")
    return kb


def rec_assumptions(b: Beliefs) -> Dict[str, Any]:
    """Everything a recommendation depends on. If any of it changes, the recommendation is stale."""
    return {k: b.get(k) for k in ("budget_inr", "goal", "warranty_active", "props_sig", "city")}


class TuneGen:
    def __init__(self, kb: Dict[str, Any], client: OpenMeteoClient, city="Bengaluru", budget=20000.0, goal="balanced",
                 warranty_active=False, events: Optional[Dict[int, list]] = None, verbose=True):
        if not isinstance(budget, (int, float)) or budget != budget or budget <= 0:
            raise ValueError(f"budget must be a positive number, got {budget!r}")
        if goal not in kb["goal_weights"]:
            raise ValueError(f"goal must be one of {list(kb['goal_weights'])}")
        self.kb, self.client = kb, client
        self.beliefs = Beliefs({
            "car_id": kb["vehicle"]["id"], "car_name": kb["vehicle"]["name"], "city": city, "budget_inr": float(budget),
            "goal": goal, "warranty_active": warranty_active, "weather_api_up": True,
            "obs": None, "props": None, "props_sig": None, "conditions_for": None,
            "warranty_warning_sent": False, "recommendation": None})
        b = self.beliefs
        self.desires = [
            Desire("warn_warranty_risk", "Warn the user that modifications can void the warranty", 5,
                   lambda b: bool(b.get("warranty_active")), lambda b: bool(b.get("warranty_warning_sent"))),
            Desire("assess_conditions", "Understand current real world conditions for this city", 3,
                   lambda b: True, lambda b: b.get("conditions_for") == b.get("city")),
            Desire("recommend_equipment", "Recommend the best purchasable parts within budget", 2,
                   lambda b: b.get("conditions_for") == b.get("city"),
                   lambda b: (b.get("recommendation") or {}).get("assumptions") == rec_assumptions(b)),
        ]
        plans = {"warn_warranty_risk": self.plan_warn, "assess_conditions": self.plan_assess,
                 "recommend_equipment": self.plan_recommend}
        self.bdi = BDIAgent(self.beliefs, self.desires, plans, events or {}, verbose)

    # ---------------- plan library ----------------
    def plan_warn(self, b):
        return "send_warranty_warning", [Step("warn_user", lambda bb: bb.update("warranty_warning_sent", True))], {}

    def plan_assess(self, b):
        a = {"weather_api_up": b.get("weather_api_up"), "city": b.get("city")}
        if b.get("weather_api_up"):
            return "live_weather_assessment", [Step("fetch_live_weather", self.s_fetch), Step("parse_and_validate", self.s_parse),
                                               Step("apply_knowledge_rules", self.s_rules)], a
        return "offline_climate_assessment", [Step("load_typical_climate_from_kb", self.s_offline),
                                              Step("apply_knowledge_rules", self.s_rules)], a

    def plan_recommend(self, b):
        return "utility_optimised_recommendation", [
            Step("apply_hard_constraints", self.s_filter), Step("score_utility", self.s_score),
            Step("select_optimum_and_bundle", self.s_select), Step("attach_search_links", self.s_publish)], rec_assumptions(b)

    # ---------------- steps: assessment ----------------
    def s_fetch(self, b):
        city = b.get("city")
        try:
            c = self.kb["cities"].get(city)
            lat, lon = (c["lat"], c["lon"]) if c else self.client.geocode(city)
            b.update("raw_weather", self.client.current(lat, lon))
        except APIError as e:
            b.update("weather_api_up", False, "api")
            raise RuntimeError(f"weather API unavailable ({e})")

    def s_parse(self, b):
        try:
            src = "SIMULATED" if "127.0.0.1" in self.client.forecast_url else "LIVE"
            b.update("obs", parse_observation(b.get("raw_weather"), b.get("city"), self.kb["wmo_weather_codes"], src))
        except APIError as e:
            b.update("weather_api_up", False, "api")
            raise RuntimeError(f"bad weather data rejected ({e})")

    def s_offline(self, b):
        t = (self.kb["cities"].get(b.get("city")) or {}).get("typical") or {"temp_c": 25, "humidity_pct": 50, "precip_mm": 0, "pressure_hpa": 1013}
        b.update("obs", Observation(b.get("city"), "typical climate", t["temp_c"], t["humidity_pct"], t["precip_mm"],
                                    t["pressure_hpa"], "unknown", air_density_ratio(t["temp_c"], t["humidity_pct"], t["pressure_hpa"]),
                                    "OFFLINE_DEFAULT"))

    def s_rules(self, b):
        obs, m = b.get("obs"), b.get("obs").metrics()
        props = {"max_aggr": 1.0, "mult": {}, "avoid": [], "advice": [], "fired": []}
        for r in self.kb["rules"]:
            w = r["when"]
            test = lambda c: OPS[c["op"]](m[c["metric"]], c["value"])
            if not (all(map(test, w["all"])) if "all" in w else any(map(test, w["any"]))):
                continue
            t = r["then"]
            props["fired"].append(r["id"])
            props["max_aggr"] = min(props["max_aggr"], t.get("max_aggressiveness", 1.0))
            for tag, f in t.get("part_multipliers", {}).items():
                props["mult"][tag] = round(props["mult"].get(tag, 1.0) * f, 4)
            props["avoid"] += [x for x in t.get("avoid_tags", []) if x not in props["avoid"]]
            props["advice"].append(t["advice"])
        b.update("props", props)
        b.update("props_sig", json.dumps(props, sort_keys=True))
        b.update("conditions_for", b.get("city"))

    # ---------------- steps: recommendation ----------------
    def s_filter(self, b):
        p = b.get("props")
        ctx = {"budget": b.get("budget_inr"), "car_id": b.get("car_id"), "max_aggr": p["max_aggr"],
               "avoid_tags": p["avoid"], "warranty_active": b.get("warranty_active")}
        pool, rejected = [], []
        for part in self.kb["parts"]:
            why = reject_reason(part, ctx)
            (rejected.append((part["name"], why)) if why else pool.append(part))
        b.update("pool", pool)
        b.update("rejected", rejected)

    def s_score(self, b):
        u = self.kb["utility"]
        b.update("scored", score_all(b.get("pool"), b.get("budget_inr"), self.kb["goal_weights"][b.get("goal")],
                                     b.get("props")["mult"], u["risk_lambda"], u["max_delivery_days"]))

    def s_select(self, b):
        scored = b.get("scored")
        b.update("best_single", scored[0] if scored else None)
        b.update("bundle", best_bundle(scored, b.get("budget_inr")) if scored else ([], 0.0))

    def s_publish(self, b):
        tpl = self.kb.get("search_link_template", "https://www.amazon.in/s?k={q}")
        bundle, bu = b.get("bundle")
        items = [{"name": s["part"]["name"], "category": s["part"]["category"], "price_inr": s["part"]["price_inr"],
                  "claimed_gain_hp": s["part"]["hp_gain"], "adj_gain_hp": s["adj_gain"], "utility": round(s["U"], 4),
                  "search_link": tpl.format(q=urllib.parse.quote_plus(f"{s['part']['name']} {b.get('car_name')}"))} for s in bundle]
        single = b.get("best_single")
        obs = b.get("obs")
        b.update("recommendation", {
            "assumptions": rec_assumptions(b), "bundle": items, "bundle_utility": round(bu, 4),
            "total_price_inr": sum(i["price_inr"] for i in items), "total_adj_gain_hp": round(sum(i["adj_gain_hp"] for i in items), 2),
            "best_single": single["part"]["name"] if single else None,
            "rejected": b.get("rejected"), "advice": b.get("props")["advice"], "fired_rules": b.get("props")["fired"],
            "data_source": obs.source, "confidence": "high" if obs.source == "LIVE" else ("medium (simulated feed)" if obs.source == "SIMULATED" else "low (typical climate, no live data)")})

    # ---------------- run + report ----------------
    def run(self, max_ticks=60):
        self.bdi.run(max_ticks)
        return self.beliefs.get("recommendation")


def render_report(agent: TuneGen) -> str:
    b, r = agent.beliefs, agent.beliefs.get("recommendation")
    o = b.get("obs")
    if not r:
        return "No recommendation could be produced."
    L = [f"TUNEGEN REPORT: {b.get('car_name')} | {b.get('city')} | goal: {b.get('goal')} | budget INR {b.get('budget_inr'):,.0f}",
         f"Conditions: {o.temp_c:.1f}C, {o.humidity_pct:.0f}% humidity, {o.precip_mm:.1f}mm rain, {o.pressure_hpa:.0f} hPa, "
         f"air density {o.density_ratio:.3f} of sea level (source: {r['data_source']}, confidence: {r['confidence']})",
         f"Knowledge rules fired: {', '.join(r['fired_rules']) or 'none'}"]
    L += [f"  - {a}" for a in r["advice"]]
    if b.get("warranty_active"):
        L.append("WARNING: the car is under warranty. Parts that may void it were excluded.")
    L.append(f"\nBest single part by utility: {r['best_single']}")
    L.append(f"Best bundle (one per category): total INR {r['total_price_inr']:,.0f}, adjusted gain {r['total_adj_gain_hp']} hp")
    for i in r["bundle"]:
        L.append(f"  + {i['name']} [{i['category']}] INR {i['price_inr']:,.0f}, +{i['adj_gain_hp']} hp, U={i['utility']}\n      search: {i['search_link']}")
    L.append("\nExcluded by hard constraints:")
    L += [f"  x {n}: {why}" for n, why in r["rejected"]]
    L.append("\nNote: part specs are illustrative sample data and links are marketplace SEARCH links, not verified listings.")
    return "\n".join(L)
