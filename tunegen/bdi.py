"""BDI engine (Assignment 4.1): Beliefs, Desires, Intentions kept as three separate structures.

Interpreter tick: perceive -> deliberate -> reconsider -> plan -> act.
A running Intention is dropped when (a) a higher priority desire pre-empts it or
(b) any belief it assumed when it was planned has changed. A step that raises also abandons it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class Beliefs:
    def __init__(self, initial: Dict[str, Any]):
        self._f = dict(initial)
        self.history: List[Tuple[str, Any, Any, str]] = []

    def get(self, k: str, default: Any = None) -> Any:
        return self._f.get(k, default)

    def update(self, k: str, v: Any, source: str = "agent") -> bool:
        old = self._f.get(k)
        if old == v:
            return False
        self._f[k] = v
        self.history.append((k, old, v, source))
        return True


class DStatus(Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    PURSUING = "PURSUING"
    SUSPENDED = "SUSPENDED"
    ACHIEVED = "ACHIEVED"


@dataclass
class Desire:
    name: str
    description: str
    priority: int
    is_relevant: Callable[[Beliefs], bool]
    is_achieved: Callable[[Beliefs], bool]
    status: DStatus = DStatus.INACTIVE


@dataclass
class Step:
    name: str
    action: Callable[[Beliefs], None]


@dataclass
class Intention:
    desire: str
    plan_name: str
    steps: List[Step]
    assumptions: Dict[str, Any]
    pc: int = 0

    def broken(self, b: Beliefs) -> Dict[str, Tuple[Any, Any]]:
        return {k: (v, b.get(k)) for k, v in self.assumptions.items() if b.get(k) != v}


class BDIAgent:
    def __init__(self, beliefs: Beliefs, desires: List[Desire], plans: Dict[str, Callable], events: Dict[int, list], verbose=True):
        self.b, self.desires, self.plans, self.events = beliefs, desires, plans, events
        self.intention: Optional[Intention] = None
        self.tick = 0
        self.trace: List[Dict[str, Any]] = []
        self.verbose = verbose

    def log(self, kind: str, msg: str, **data):
        self.trace.append({"tick": self.tick, "kind": kind, "msg": msg, **data})
        if self.verbose:
            print(f"[t{self.tick:02d}] {kind:<9} {msg}")

    def _set(self, d: Desire, s: DStatus):
        if d.status != s:
            self.log("DESIRE", f"{d.name}: {d.status.value} -> {s.value}", desire=d.name, new=s.value)
            d.status = s

    def perceive(self):
        for ev in self.events.get(self.tick, []):
            if callable(ev):
                ev(self)
                continue
            k, v = ev
            old = self.b.get(k)
            if self.b.update(k, v, "environment"):
                self.log("EVENT", f"environment update: {k}: {old!r} -> {v!r}", key=k, old=old, new=v)

    def deliberate(self) -> Optional[Desire]:
        for d in self.desires:
            if d.is_achieved(self.b):
                self._set(d, DStatus.ACHIEVED)
            elif d.is_relevant(self.b):
                if d.status in (DStatus.INACTIVE, DStatus.ACHIEVED):
                    self._set(d, DStatus.ACTIVE)
            else:
                self._set(d, DStatus.INACTIVE)
        live = [d for d in self.desires if d.status in (DStatus.ACTIVE, DStatus.PURSUING, DStatus.SUSPENDED)]
        return max(live, key=lambda d: d.priority) if live else None

    def _drop(self, reason: str, kind="abandon"):
        it = self.intention
        self.log("INTENTION", f"ABANDON '{it.plan_name}' at step {it.pc}/{len(it.steps)}: {reason}",
                 event=kind, plan=it.plan_name, desire=it.desire, reason=reason)
        d = next(x for x in self.desires if x.name == it.desire)
        if d.status == DStatus.PURSUING:
            self._set(d, DStatus.SUSPENDED)
        self.intention = None

    def reconsider(self, chosen: Optional[Desire]):
        it = self.intention
        if it is None:
            return
        if chosen is None or chosen.name != it.desire:
            self._drop(f"pre-empted by higher priority desire '{chosen.name if chosen else None}'")
            return
        broken = it.broken(self.b)
        if broken:
            self._drop("assumptions broken: " + ", ".join(f"{k} changed" for k in broken))

    def plan(self, chosen: Desire):
        if self.intention is not None:
            return
        name, steps, assumptions = self.plans[chosen.name](self.b)
        self.intention = Intention(chosen.name, name, steps, assumptions)
        self._set(chosen, DStatus.PURSUING)
        self.log("INTENTION", f"ADOPT '{name}': " + " > ".join(s.name for s in steps), event="adopt", plan=name, desire=chosen.name)

    def act(self):
        it = self.intention
        step = it.steps[it.pc]
        try:
            step.action(self.b)
        except Exception as e:                       # step failure -> abandon, replan next tick
            self._drop(f"step '{step.name}' failed: {e}", kind="step_failed")
            return
        it.pc += 1
        self.log("ACT", f"{it.plan_name} {it.pc}/{len(it.steps)}: {step.name}")
        if it.pc == len(it.steps):
            self.log("INTENTION", f"COMPLETED '{it.plan_name}'", event="complete", plan=it.plan_name, desire=it.desire)
            self.intention = None

    def run(self, max_ticks: int = 60):
        last_event = max(self.events) if self.events else 0
        while self.tick < max_ticks:
            self.tick += 1
            self.perceive()
            chosen = self.deliberate()
            self.reconsider(chosen)
            if chosen is None:
                if self.tick >= last_event:
                    self.log("DONE", "all desires achieved")
                    return
                continue
            self.plan(chosen)
            self.act()
        self.log("STOP", "max ticks reached")
