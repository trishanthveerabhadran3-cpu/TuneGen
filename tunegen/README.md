# TuneGen: a goal driven research and recommendation agent for Indian-market car tuning

TuneGen reads the live weather for a city, reasons over a structured knowledge base, and recommends the best
tuning parts for a Maruti Suzuki Swift K12 within a budget. If the world changes mid task (budget drops, warranty
revealed, weather API fails) it drops its plan and builds a new one.

It runs fully locally. **No paid services, no API keys, no packages to install** (Python 3.9+ standard library only).

## Run the web front end
```
python server.py          # then open http://127.0.0.1:8000 in your browser (Ctrl+C to stop)
```

## Run it (command line)
```
python main.py --city Delhi --budget 20000 --goal balanced      # real live weather (needs internet)
python main.py --demo                                           # offline demo with 2 mid-run surprises
python main.py --demo --outage --city Delhi                     # weather API down -> fallback plan
python main.py --city Pune --event 6:budget_inr=8000            # inject your own surprise at tick 6
python -m unittest discover -s tests -v                         # 9 tests
```
If the real API is unreachable, the agent does not crash: it abandons the live plan and uses typical climate from
the knowledge base, and the report says confidence is **low** and the source is **OFFLINE_DEFAULT**.

## Files
| File | Role | Assignment |
|---|---|---|
| `knowledge.json` | Ontology style knowledge base: parts, relations, weather code groups, adaptation rules, goal weights | 4.2 |
| `weather.py` | REST client (Open-Meteo forecast + geocoding), parser, validation, simulated test server | 4.2 |
| `utility.py` | Utility function, hard constraints, argmax and exhaustive bundle search | 3.2 |
| `bdi.py` | Beliefs, Desires, Intentions as separate structures + interpreter loop | 4.1 |
| `agent.py` | Integrates everything: plan library, rules engine, report | all |
| `main.py` | Command line interface | |
| `tests/test_tunegen.py` | Verification of all three assignments | |
| `docs/` | Captured demo runs | |

## Architecture
```
 Open-Meteo REST API --(JSON)--> parse + validate --> Observation --+
                                                                    v
 knowledge.json --rules--> BELIEFS <---- environment events     rules fire -> props (aggressiveness cap,
                              |                                   part multipliers, avoided tags, advice)
                              v
 DESIRES (state machine, priorities) --deliberate--> INTENTION (plan from library, with assumptions)
                              |                              |
                              |   reconsider: pre-empted? assumption broken? step failed? -> replan
                              v                              v
                       utility evaluator: filter -> score -> argmax + best bundle -> report with search links
```

### Utility function (3.2)
`U(i) = Σ w_k·x_k(i) − λ·warranty_risk(i)` with six normalised criteria (weather adjusted performance gain, mileage gain,
cost headroom, reliability, seller trust, delivery speed). Weights depend on the goal (performance / mileage / balanced).
Hard constraints (valid data, fits car, BS6 legal, within budget, not too aggressive for conditions, warranty) remove an
option before scoring. Ties break deterministically. The optimum is verified against brute force in the tests.

### BDI (4.1)
* **Beliefs**: city, budget, goal, warranty status, API health, observation, derived props, recommendation.
* **Desires** (priority): warn about warranty (5) > assess conditions (3) > recommend equipment (2).
  States: INACTIVE, ACTIVE, PURSUING, SUSPENDED, ACHIEVED. An ACHIEVED desire reopens if its result goes stale.
* **Intentions**: ordered step pipelines, each storing the beliefs it assumed. Changed assumption, pre-emption or a failed step triggers replanning.

### Knowledge and live API (4.2)
Weather metrics (temperature, humidity, rain, pressure, computed air density) trigger declarative rules in
`knowledge.json` (heat, rain, thin air, ideal). Rules change the agent's properties instantly and the plan is recomputed.

## Design spec (Assignment 1.1 material)
**Domain**: advisory agent for aftermarket car tuning in India. It researches conditions and parts and recommends, it never touches a car.

| PEAS | Description |
|---|---|
| Performance measure | Utility of the recommended bundle, constraint violations (target: 0), honesty of confidence labels, robustness to failures |
| Environment | Live weather API, parts catalog and seller data, user constraints (budget, goal, warranty), the specific car |
| Actuators | Written report, ranked parts list, advisories, marketplace search links |
| Sensors | REST API responses, user input and mid-run updates, knowledge base |

| Environment property | Classification | Why |
|---|---|---|
| Observability | Partially observable | Only current weather, not the true engine state or true product quality |
| Determinism | Stochastic | Weather and API behaviour are not controlled by the agent |
| Episodic | Sequential | Earlier assessment feeds the recommendation |
| Static | Dynamic | Budget, warranty and weather change mid task |
| Discrete | Mixed | Discrete part choices, continuous weather values |
| Agents | Single agent (self-interested buyer view) | No competing agents modelled |

| | Traditional OOP program | TuneGen agent |
|---|---|---|
| Control | Fixed call sequence | Chooses goals and plans at run time |
| Change | Needs a code change | Replans when beliefs change |
| Failure | Exception, stops | Abandons plan, falls back, reports lower confidence |
| State | Variables | Explicit beliefs, desires, intentions |

## Honest limitations
* Part specs (hp gain, price, ratings) are **illustrative sample data**. Replace with researched real data before real use.
* Links are marketplace **search** links, not verified product pages.
* Weather adjustment factors in the rules are engineering judgment, not measured dyno results.
* The weather feed is structured JSON; the rules are hand written.
* Scripted demo events simulate user messages. In real use the same beliefs would be updated by user input.
