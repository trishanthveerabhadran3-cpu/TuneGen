"""Live REST API integration (Assignment 4.2): Open-Meteo forecast + geocoding. No API key needed."""
from __future__ import annotations

import json
import math
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LIVE_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FIELDS = "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,weather_code"


class APIError(Exception):
    pass


class OpenMeteoClient:
    def __init__(self, forecast_url=LIVE_FORECAST_URL, geocode_url=LIVE_GEOCODE_URL, timeout=8.0, retries=2):
        self.forecast_url, self.geocode_url, self.timeout, self.retries = forecast_url, geocode_url, timeout, retries

    def _get(self, base: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{base}?{urllib.parse.urlencode(params)}"
        err: Optional[str] = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "TuneGen-Agent/1.0", "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise APIError("response JSON is not an object")
                if data.get("error"):
                    raise APIError(f"API error: {data.get('reason')}")
                return data
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if 400 <= e.code < 500:
                    break
            except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
                err = f"network failure: {getattr(e, 'reason', e)}"
            except json.JSONDecodeError as e:
                err = f"malformed JSON: {e}"
            except APIError as e:
                err = str(e)
            if attempt < self.retries:
                time.sleep(0.3 * 2 ** attempt)
        raise APIError(err or "unknown error")

    def geocode(self, name: str) -> Tuple[float, float]:
        res = (self._get(self.geocode_url, {"name": name, "count": 1, "format": "json"}).get("results") or [])
        if not res:
            raise APIError(f"city '{name}' not found")
        return float(res[0]["latitude"]), float(res[0]["longitude"])

    def current(self, lat: float, lon: float) -> Dict[str, Any]:
        return self._get(self.forecast_url, {"latitude": lat, "longitude": lon, "current": FIELDS, "timezone": "auto"})


@dataclass
class Observation:
    city: str
    time: str
    temp_c: float
    humidity_pct: float
    precip_mm: float
    pressure_hpa: float
    weather_group: str
    density_ratio: float
    source: str          # LIVE | SIMULATED | OFFLINE_DEFAULT

    def metrics(self) -> Dict[str, float]:
        return {"air_temp_c": self.temp_c, "humidity_pct": self.humidity_pct, "precip_mm": self.precip_mm,
                "pressure_hpa": self.pressure_hpa, "density_ratio": self.density_ratio,
                "is_raining": 1.0 if self.weather_group in ("rain", "thunderstorm") else 0.0}


def air_density_ratio(t_c: float, rh: float, p_hpa: float) -> float:
    """Moist air density / ISA sea level (1.225 kg/m3): ideal gas law + Magnus vapour pressure."""
    tk = t_c + 273.15
    e = rh / 100 * 6.112 * math.exp(17.62 * t_c / (243.12 + t_c))
    return (((p_hpa - e) * 100) / (287.058 * tk) + (e * 100) / (461.495 * tk)) / 1.225


def weather_group(code: Optional[int], wmo: Dict[str, list]) -> str:
    for g, codes in wmo.items():
        if code in codes:
            return g
    return "unknown"


def parse_observation(raw: Dict[str, Any], city: str, wmo: Dict[str, list], source: str) -> Observation:
    cur = raw.get("current")
    if not isinstance(cur, dict):
        raise APIError("response has no 'current' block")
    need = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure"]
    missing = [k for k in need if cur.get(k) is None]
    if missing:
        raise APIError(f"response missing fields: {missing}")
    try:
        t, rh, pr, p = (float(cur[k]) for k in need)
    except (TypeError, ValueError):
        raise APIError("non numeric weather values")
    if not (-60 <= t <= 65 and 0 <= rh <= 100 and 0 <= pr < 500 and 300 <= p <= 1100):
        raise APIError(f"values out of physical range: t={t} rh={rh} precip={pr} p={p}")
    code = cur.get("weather_code")
    return Observation(city, str(cur.get("time", "")), t, rh, pr, p,
                       weather_group(int(code) if isinstance(code, (int, float)) else None, wmo),
                       air_density_ratio(t, rh, p), source)


# ---------------- local SIMULATED server (testing without internet only) ----------------
SIM = {12.97: (19.0, 62.0, 0.0, 910.0, 1), 28.61: (41.0, 18.0, 0.0, 990.0, 0),
       13.08: (27.0, 93.0, 6.5, 1006.0, 63), 31.10: (8.0, 55.0, 0.0, 784.0, 2)}


class SimHandler(BaseHTTPRequestHandler):
    fail = False                      # flip to True to simulate an outage

    def log_message(self, *a):
        pass

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if SimHandler.fail:
            return self._send(503, '{"error": true, "reason": "simulated outage"}')
        if u.path == "/v1/search":
            name = q.get("name", [""])[0].lower()
            return self._send(200, json.dumps({"results": [{"latitude": 18.52, "longitude": 73.86}]} if name == "pune" else {}))
        lat = round(float(q.get("latitude", ["0"])[0]), 2)
        t, rh, pr, p, code = SIM.get(lat, (25.0, 50.0, 0.0, 1000.0, 0))
        self._send(200, json.dumps({"current": {"time": "SIMULATED", "temperature_2m": t, "relative_humidity_2m": rh,
                                                "precipitation": pr, "surface_pressure": p, "weather_code": code}}))


def start_sim_server():
    SimHandler.fail = False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SimHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
