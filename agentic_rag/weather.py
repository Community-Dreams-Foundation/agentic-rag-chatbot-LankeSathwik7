from __future__ import annotations

import datetime as dt
import json
import statistics
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class WeatherAnalytics:
    location: str
    start_date: str
    end_date: str
    missing_days: int
    mean_temperature: float
    volatility: float
    anomaly_days: list[str]
    explanation: str

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "missing_days": self.missing_days,
            "mean_temperature": round(self.mean_temperature, 2),
            "volatility": round(self.volatility, 2),
            "anomaly_days": self.anomaly_days,
            "explanation": self.explanation,
        }


def _safe_get_json(url: str, timeout_sec: int = 12) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "agentic-rag-bot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def analyze_open_meteo_timeseries(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> WeatherAnalytics:
    today = dt.date.today()
    end = dt.date.fromisoformat(end_date)
    endpoint = "https://archive-api.open-meteo.com/v1/archive"
    if end > today:
        endpoint = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": str(latitude),
        "longitude": str(longitude),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
        "daily": "temperature_2m_mean",
    }
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    try:
        data = _safe_get_json(url)
    except Exception as exc:
        return WeatherAnalytics(
            location=f"{latitude},{longitude}",
            start_date=start_date,
            end_date=end_date,
            missing_days=0,
            mean_temperature=0.0,
            volatility=0.0,
            anomaly_days=[],
            explanation=f"Open-Meteo request failed: {exc.__class__.__name__}.",
        )

    daily = data.get("daily", {})
    times = daily.get("time", []) or []
    temps = daily.get("temperature_2m_mean", []) or []
    pairs = [(d, t) for d, t in zip(times, temps)]
    valid = [(d, t) for d, t in pairs if isinstance(t, (int, float))]
    missing = len(pairs) - len(valid)

    if not valid:
        return WeatherAnalytics(
            location=f"{latitude},{longitude}",
            start_date=start_date,
            end_date=end_date,
            missing_days=missing,
            mean_temperature=0.0,
            volatility=0.0,
            anomaly_days=[],
            explanation="No valid temperature records were returned by Open-Meteo for this range.",
        )

    temp_values = [t for _, t in valid]
    mean_t = statistics.mean(temp_values)
    vol = statistics.pstdev(temp_values) if len(temp_values) > 1 else 0.0
    anomaly_days: list[str] = []
    if vol > 0:
        for day, temp in valid:
            z = abs((temp - mean_t) / vol)
            if z >= 2.0:
                anomaly_days.append(day)

    explanation = (
        f"Analyzed {len(valid)} daily observations. "
        f"Average temperature was {mean_t:.2f}C with volatility {vol:.2f}. "
        f"Missing days: {missing}. "
        f"Anomaly days (|z|>=2): {', '.join(anomaly_days) if anomaly_days else 'none'}."
    )
    return WeatherAnalytics(
        location=f"{latitude},{longitude}",
        start_date=start_date,
        end_date=end_date,
        missing_days=missing,
        mean_temperature=mean_t,
        volatility=vol,
        anomaly_days=anomaly_days,
        explanation=explanation,
    )
