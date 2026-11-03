import itertools
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import TuneGen, load_kb
from utility import best_bundle, reject_reason, score_all
from weather import APIError, OpenMeteoClient, SimHandler, parse_observation, start_sim_server

KB = load_kb(os.path.join(os.path.dirname(__file__), "..", "knowledge.json"))
CTX = {"budget": 20000, "car_id": "swift_k12", "max_aggr": 1.0, "avoid_tags": [], "warranty_active": False}


def make(city, budget=20000, goal="balanced", warranty=False, events=None, outage=False):
    srv, base = start_sim_server()
    SimHandler.fail = outage
    cl = OpenMeteoClient(f"{base}/v1/forecast", f"{base}/v1/search", timeout=2, retries=0)
    ag = TuneGen(KB, cl, city, budget, goal, warranty, events, verbose=False)
    ag.run()
    srv.shutdown()
    srv.server_close()
    return ag


class UtilityTests(unittest.TestCase):
    def test_argmax_matches_bruteforce_for_each_goal(self):
        for goal in KB["goal_weights"]:
            pool = [p for p in KB["parts"] if not reject_reason(p, CTX)]
            u = KB["utility"]
            sc = score_all(pool, 20000, KB["goal_weights"][goal], {}, u["risk_lambda"], u["max_delivery_days"])
            self.assertEqual(sc[0]["U"], max(s["U"] for s in sc))
            self.assertEqual(len(pool), 8)

    def test_hard_constraints(self):
        by = {p["id"]: p for p in KB["parts"]}
        self.assertIn("BS6", reject_reason(by["race"], CTX))
        self.assertIn("fit", reject_reason(by["city_intake"], CTX))
        self.assertIn("budget", reject_reason(by["piggy"], {**CTX, "budget": 5000}))
        self.assertIsNotNone(reject_reason({**by["plugs"], "price_inr": float("nan")}, CTX))
        self.assertIsNotNone(reject_reason({**by["plugs"], "reliability": 9}, CTX))

    def test_bundle_is_optimal_and_feasible(self):
        pool = [p for p in KB["parts"] if not reject_reason(p, CTX)]
        sc = score_all(pool, 20000, KB["goal_weights"]["balanced"], {}, 0.15, 14)
        bundle, bu = best_bundle(sc, 20000)
        self.assertLessEqual(sum(s["part"]["price_inr"] for s in bundle), 20000)
        cats = [s["part"]["category"] for s in bundle]
        self.assertEqual(len(cats), len(set(cats)))
        for r in range(1, 5):                         # independent brute-force cross check
            for combo in itertools.combinations(sc, r):
                c = [s["part"]["category"] for s in combo]
                if len(set(c)) == len(c) and sum(s["part"]["price_inr"] for s in combo) <= 20000:
                    self.assertLessEqual(sum(s["U"] for s in combo), bu + 1e-9)

    def test_edge_cases(self):
        self.assertEqual(score_all([], 1000, KB["goal_weights"]["balanced"], {}, 0.15, 14), [])
        with self.assertRaises(ValueError):
            TuneGen(KB, OpenMeteoClient(), budget=-5)
        with self.assertRaises(ValueError):
            TuneGen(KB, OpenMeteoClient(), goal="speed")


class APITests(unittest.TestCase):
    def test_parser_rejects_bad_data(self):
        w = KB["wmo_weather_codes"]
        for raw in ({}, {"current": {"temperature_2m": 30}}, {"current": {"temperature_2m": 900, "relative_humidity_2m": 50, "precipitation": 0, "surface_pressure": 1000}}):
            with self.assertRaises(APIError):
                parse_observation(raw, "x", w, "T")

    def test_outage_raises_api_error(self):
        srv, base = start_sim_server()
        SimHandler.fail = True
        with self.assertRaises(APIError):
            OpenMeteoClient(f"{base}/v1/forecast", timeout=2, retries=0).current(13.08, 80.27)
        srv.shutdown()


class AdaptationTests(unittest.TestCase):
    def names(self, ag):
        return [i["name"] for i in ag.beliefs.get("recommendation")["bundle"]]

    def test_conditions_change_the_plan(self):
        chennai, delhi, shimla = make("Chennai"), make("Delhi"), make("Shimla")
        self.assertNotIn("Cold Air Intake A", self.names(chennai))               # rain removes open intake
        self.assertEqual(delhi.beliefs.get("props")["fired"], ["R_HOT"])
        for n in ("Plug-in Tuner Module", "Piggyback ECU Chip"):                  # heat excludes aggressive ECU
            self.assertNotIn(n, self.names(delhi))
        self.assertEqual(shimla.beliefs.get("props")["fired"], ["R_THIN_AIR"])
        gain = lambda ag: {i["name"]: i["adj_gain_hp"] for i in ag.beliefs.get("recommendation")["bundle"]}
        self.assertNotEqual(gain(delhi), gain(shimla))                            # same parts, different weather adjusted gains
        self.assertNotEqual(delhi.beliefs.get("props_sig"), shimla.beliefs.get("props_sig"))

    def test_midrun_belief_changes_cause_intention_shifts(self):
        ag = make("Chennai", events={5: [("warranty_active", True)], 8: [("budget_inr", 8000.0)]})
        tr = ag.bdi.trace
        adopts = [t["plan"] for t in tr if t.get("event") == "adopt"]
        self.assertIn("send_warranty_warning", adopts)                            # higher priority desire pre-empted
        self.assertEqual(adopts.count("utility_optimised_recommendation"), 3)     # initial + after warranty + after budget
        self.assertTrue(any("pre-empted" in t.get("reason", "") for t in tr))
        self.assertTrue(any("assumptions broken" in t.get("reason", "") for t in tr))
        r = ag.beliefs.get("recommendation")
        self.assertLessEqual(r["total_price_inr"], 8000)
        self.assertTrue(all(i["price_inr"] <= 8000 for i in r["bundle"]))
        self.assertEqual(r["assumptions"]["budget_inr"], 8000.0)
        for p in KB["parts"]:                                                     # nothing that voids warranty survived
            if p["warranty_risk"] > 0.25:
                self.assertNotIn(p["name"], self.names(ag))

    def test_api_outage_falls_back_to_offline_plan(self):
        ag = make("Delhi", outage=True)
        tr = ag.bdi.trace
        self.assertTrue(any(t.get("event") == "step_failed" for t in tr))
        adopts = [t["plan"] for t in tr if t.get("event") == "adopt"]
        self.assertEqual(adopts[:2], ["live_weather_assessment", "offline_climate_assessment"])
        self.assertEqual(ag.beliefs.get("recommendation")["data_source"], "OFFLINE_DEFAULT")
        self.assertIn("low", ag.beliefs.get("recommendation")["confidence"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
