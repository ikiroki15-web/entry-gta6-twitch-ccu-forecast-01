"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers.

Reads one case (/input/<id>.in) and writes /output/<id>.out with, for each of
hours 2..24 of the release day (UTC), five quantiles of that hour's concurrent
Twitch viewers in the game's category.

Method: an analogue forecaster. The case ships 38 past big launches
(`comparables`), each with its release day's hourly Twitch category averages.
For every comparable with a complete 24-hour release day, take its shape as a
log-ratio curve: ln(viewers_h / viewers_1). The forecast for hour h is the
target's hour-1 print times exp() of that curve's distribution at h.

Not every past launch behaves like a rolling-midnight console launch, which
is what GTA VI is, so analogues are weighted:

- base weight 1.0 for a rolling-midnight launch with no early access
  (its `unlock` text says "midnight local" / "rolling" and it is not tagged
  early-access-contaminated); 0.25 for everything else (simultaneous-global
  unlocks, early-access-contaminated launches, free-to-play launches...).
  A simultaneous unlock has a near-empty hour 1 and a huge spike later, so
  it still carries some shape information, but it must not dominate.
- times a Gaussian kernel on log hour-1 viewers (bandwidth 1.5): launches
  whose first hour drew a similar crowd are more informative about how the
  day unfolds. A launch ten times bigger or smaller still counts, just less.

Per hour the weighted mean and weighted standard deviation of the log-ratio
curve give the five quantiles through the standard normal's z values. The
spread has a floor (0.20 in log units) so a small analogue pool can never
claim false precision, and a cap (1.5) so one wild analogue cannot blow the
interval up. Quantiles are rounded half-up to 2 decimals, like the house
baseline's, and clamped to (0, 1e9].

Fallbacks keep every case valid: if fewer than three usable analogues exist,
fall back to the persistence baseline (every hour at the hour-1 print, spread
0.15 * sqrt(h - 1) in log space). If the case has no hour-1 print, use the
last history point, else 1 viewer.

Standard library only, no network, no randomness: two runs on the same case
write the same bytes. Validated offline by leave-one-out cross-validation
with the shipped program itself, treating each comparable as the target and
scoring against its real release day: on the eleven rolling-midnight,
non-early-access launches it scores a mean pinball loss of 0.269, against
0.363 for the persistence baseline on the same launches (0.432 vs 0.930
over all 37 usable comparables).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Weight of a non-rolling / early-access / otherwise unlike analogue.
OTHER_WEIGHT = 0.25
# Bandwidth of the Gaussian kernel on log hour-1 viewers.
LOG_P1_BANDWIDTH = 1.5
# Floor and cap on the per-hour log spread, so the interval is never absurdly
# narrow (small pool) or absurdly wide (one wild analogue).
SIGMA_FLOOR = 0.20
SIGMA_CAP = 1.5
# Persistence fallback spread, per square-root hour, as in the house baseline.
FALLBACK_SIGMA = 0.15
# A comparable needs a real first hour to define a ratio curve at all.
MIN_P1 = 100
# Minimum usable analogues before the persistence fallback takes over.
MIN_POOL = 3


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as the house baseline rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def clamp_quantiles(values: list[float]) -> list[float | int]:
    """Keep quantiles finite, in (0, 1e9] and non-decreasing, rounded to 2dp."""
    out: list[float | int] = []
    prev = 0.0
    for value in values:
        value = round2(min(max(value, 0.01), 1e9))
        if value < prev:
            value = prev
        out.append(value)
        prev = float(value)
    return out


def is_rolling_midnight(comp: dict) -> bool:
    """True when the launch unlocked at midnight local time, rolling."""
    unlock = str(comp.get("unlock") or "").lower()
    return "midnight local" in unlock or "rolling" in unlock


def is_early_access(comp: dict) -> bool:
    """True when the launch's release day was contaminated by early access."""
    unlock = str(comp.get("unlock") or "").lower()
    return "early-access-contaminated" in unlock


def release_day_curve(comp: dict) -> dict[int, float] | None:
    """Hourly viewers for hours 1..24 of the comparable's release day.

    Returns None unless all 24 hours are present, so a partial curve never
    enters the pool with a silently missing hour.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours: dict[int, float] = {}
    for row in comp.get("hourly") or []:
        try:
            hour = int((parse_time(row["from"]) - day_start).total_seconds() // 3600) + 1
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= hour <= 24:
            hours[hour] = viewers
    if len(hours) < 24:
        return None
    return hours


def analogue_pool(case: dict, p1: float) -> list[tuple[dict[int, float], float]]:
    """(log-ratio curve, weight) for every usable comparable launch."""
    log_p1 = math.log(max(1.0, p1))
    pool: list[tuple[dict[int, float], float]] = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if curve is None:
            continue
        comp_p1 = max(1.0, curve[1])
        if comp_p1 < MIN_P1:
            continue
        base = 1.0 if (is_rolling_midnight(comp) and not is_early_access(comp)) else OTHER_WEIGHT
        gap = (log_p1 - math.log(comp_p1)) / LOG_P1_BANDWIDTH
        weight = base * math.exp(-0.5 * gap * gap)
        log_ratio = {h: math.log(max(1.0, v) / comp_p1) for h, v in curve.items()}
        pool.append((log_ratio, weight))
    return pool


def weighted_mean_sd(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Weighted mean and population standard deviation of (value, weight)."""
    total = sum(w for _, w in pairs)
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    return mean, math.sqrt(var)


def analogue_quantiles(p1: float, pool: list[tuple[dict[int, float], float]], hour: int) -> list[float]:
    """Five quantiles for one hour from the analogues' log-ratio curves."""
    pairs = [(curve[hour], weight) for curve, weight in pool]
    mean, sd = weighted_mean_sd(pairs)
    sd = min(max(sd, SIGMA_FLOOR), SIGMA_CAP)
    return [p1 * math.exp(mean + z * sd) for z in Z]


def persistence_quantiles(p1: float, hour: int) -> list[float]:
    """House-baseline fallback: every hour at the hour-1 print."""
    spread = FALLBACK_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z]


def hour1_print(case: dict) -> float:
    """Hour 1's print; the last history point, else 1, when it is missing."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            return max(1.0, float(row["viewers"]))
    history = case.get("history") or []
    if history:
        return max(1.0, float(history[-1]["viewers"]))
    return 1.0


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    p1 = hour1_print(case)
    pool = analogue_pool(case, p1)
    use_pool = len(pool) >= MIN_POOL
    rows = []
    for row in case["forecast"]:
        hour = int(row["hour"])
        if use_pool:
            raw = analogue_quantiles(p1, pool, hour)
        else:
            raw = persistence_quantiles(p1, hour)
        rows.append({"hour": hour, "q": clamp_quantiles(raw)})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
