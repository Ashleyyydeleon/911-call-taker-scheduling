"""911 call-taker scheduling: demand -> Erlang C -> required staff -> 10-hour shift plan.

The question: given an hourly call forecast, how many call takers does a 911 center need each hour to answer
90% of calls within 15 seconds (the California standard), and where should 10-hour shifts start to meet that?

Method
  1. Demand      hourly calls, planned at the P90 (the level 9 of 10 hours stay under).
  2. Erlang C    the smallest number of call takers N so that P(wait <= 15 s) >= 90% at that hour's volume.
  3. Shrinkage   scheduled staff = N / (1 - shrinkage), so N people are actually on the phones.
  4. Shifts      integer programs choose the 10-hour shift starts (at most 5 distinct start times, never
                 fewer than 3 people in any hour) for a given number of shifts per day.

Data: this repository does NOT include any agency data, forecasts or call records. By default the script runs on a
SYNTHETIC demand curve (a smooth day shape: quietest around 4-5am, busiest around 5pm, about 2.9x apart) so you can
see how the method behaves. Numbers it prints are illustrative, not findings about any real agency.
To use your own demand, pass a CSV with columns: hour (0-23), mean, p90   ->  python scheduling.py --demand my.csv

Usage
  python scheduling.py                 # synthetic demand, prints tables
  python scheduling.py --plot out.png  # also saves a stacked-shift chart (needs matplotlib)
"""
import argparse
import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

TARGET_S, TARGET_PCT = 15, 0.90     # answer 90% of calls within 15 seconds
TALK_S = 116                        # average talk time, seconds (an assumption: replace with local data)
AFTER_CALL_S = 30                   # after-call work, seconds (assumption)
SHRINKAGE = 0.30                    # breaks, meals, training, leave (typical contact-center range 25-35%)
SHIFT_H = 10                        # 4 x 10-hour shifts a week
FTE_HOURS_WEEK = 40
MAX_STARTS = 5                      # keep the roster practical: at most 5 distinct shift start times
MIN_ON_SHIFT = 3                    # never fewer than 3 call takers in any hour

# Estimated minimum 911 call takers by hour, from a published City Auditor audit (Exhibit 19, 2023 schedule).
CURRENT_MIN = np.array([5, 5, 5, 3.1, 3.1, 2.5, 2.5, 2.6, 2.6, 4, 4, 5, 5, 5, 5, 5.2, 5.2, 5, 5, 5, 5, 5, 5, 5])


def hour_label(h):
    return "12am" if h == 0 else "12pm" if h == 12 else f"{h}am" if h < 12 else f"{h - 12}pm"


# --- Demand ---------------------------------------------------------------------------------------

def synthetic_demand():
    """A smooth, made-up day: ~17 calls/hr at the quietest, ~49 at the busiest, P90 20% above the mean."""
    h = np.arange(24)
    shape = 0.5 * (1 + np.cos(2 * np.pi * (h - 17) / 24))   # peaks at 5pm, bottoms out at 5am
    mean = 17 + 32 * shape
    return pd.DataFrame({"hour": h, "mean": mean, "p90": mean * 1.2}).set_index("hour")


def load_demand(path):
    df = pd.read_csv(path).set_index("hour").sort_index()
    if len(df) != 24 or not {"mean", "p90"} <= set(df.columns):
        raise SystemExit("demand CSV needs 24 rows (hour 0-23) and columns: hour, mean, p90")
    return df


# --- Erlang C -------------------------------------------------------------------------------------

def service_level(n, calls_hr, aht):
    """Share of calls answered within TARGET_S with n people on the phones (Erlang C)."""
    a = calls_hr * aht / 3600                      # offered load in Erlangs
    if n <= a:
        return 0.0
    head = sum(a ** k / math.factorial(k) for k in range(n))
    tail = a ** n / math.factorial(n) * n / (n - a)
    p_wait = tail / (head + tail)
    return 1 - p_wait * math.exp(-(n - a) * TARGET_S / aht)


def required(calls_hr, aht, shrink):
    n = 1
    while service_level(n, calls_hr, aht) < TARGET_PCT:
        n += 1
    return n / (1 - shrink)


def sl_scheduled(sched, calls_hr, aht, shrink):
    """Share answered in time when `sched` people are scheduled (fractional staff -> interpolate)."""
    on = sched * (1 - shrink)
    lo = int(math.floor(on))
    frac = on - lo
    s_lo = service_level(lo, calls_hr, aht) if lo >= 1 else 0.0
    return s_lo if frac == 0 else (1 - frac) * s_lo + frac * service_level(lo + 1, calls_hr, aht)


def daily_sl(cov, calls, aht, shrink):
    return float(np.average([sl_scheduled(c, v, aht, shrink) for c, v in zip(cov, calls)], weights=calls))


# --- Shift optimisation ---------------------------------------------------------------------------

def cover_matrix():
    """m[hour, start] = 1 if a shift starting at `start` is on duty during `hour` (wraps past midnight)."""
    m = np.zeros((24, 24))
    for s in range(24):
        for k in range(SHIFT_H):
            m[(s + k) % 24, s] = 1
    return m


def plan_shifts(n_shifts, req, calls, aht, shrink):
    """Place exactly n_shifts 10-hour shifts: an integer program minimising call-weighted shortfall against the
    requirement, then single-shift moves while they raise the share answered in time."""
    A = cover_matrix()
    w = calls / calls.sum()
    Z = np.zeros((24, 24))
    c = np.r_[np.zeros(24), np.zeros(24), w]                                  # variables: x (shifts), y (start used), u (shortfall)
    cover = LinearConstraint(np.c_[A, Z, np.eye(24)], lb=req, ub=np.inf)
    link = LinearConstraint(np.c_[np.eye(24), -30 * np.eye(24), Z], lb=-np.inf, ub=0)
    starts = LinearConstraint(np.r_[np.zeros(24), np.ones(24), np.zeros(24)][None, :], lb=0, ub=MAX_STARTS)
    total = LinearConstraint(np.r_[np.ones(24), np.zeros(48)][None, :], lb=n_shifts, ub=n_shifts)
    floor_ = LinearConstraint(np.c_[A, Z, Z], lb=MIN_ON_SHIFT, ub=np.inf)
    res = milp(c, constraints=[cover, link, starts, total, floor_],
               integrality=np.r_[np.ones(48), np.zeros(24)],
               bounds=Bounds(np.zeros(72), np.r_[np.full(24, np.inf), np.ones(24), np.full(24, np.inf)]))
    x = np.round(res.x[:24]).astype(int)

    def score(v):
        return daily_sl(A @ v, calls, aht, shrink)

    improved = True
    while improved:
        improved = False
        for a in np.flatnonzero(x):
            for b in range(24):
                y = x.copy()
                y[a] -= 1
                y[b] += 1
                if np.count_nonzero(y) <= MAX_STARTS and (A @ y).min() >= MIN_ON_SHIFT and score(y) > score(x) + 1e-9:
                    x, improved = y, True
                    break
            if improved:
                break
    return x


def describe(x):
    return [(hour_label(s), hour_label((s + SHIFT_H) % 24), int(n)) for s, n in enumerate(x) if n > 0]


# --- Run ------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demand", help="CSV with columns hour, mean, p90 (defaults to a synthetic demand curve)")
    ap.add_argument("--plot", help="save a stacked-shift chart to this PNG path (needs matplotlib)")
    args = ap.parse_args()

    demand = load_demand(args.demand) if args.demand else synthetic_demand()
    if not args.demand:
        print("Using SYNTHETIC demand (no real data). Results are illustrative.\n")
    aht = TALK_S + AFTER_CALL_S
    req = np.array([required(v, aht, SHRINKAGE) for v in demand["p90"]])
    calls = demand["mean"].to_numpy()

    table = pd.DataFrame({"calls_per_hr_p90": demand["p90"].round(1), "required": req.round(1),
                          "current_min": CURRENT_MIN, "gap": (req - CURRENT_MIN).round(1)})
    table.index = [hour_label(h) for h in table.index]
    print(table.to_string(), "\n")

    A = cover_matrix()
    today_hours = CURRENT_MIN.sum()
    today = daily_sl(CURRENT_MIN, calls, aht, SHRINKAGE)
    print(f"Today's minimum: {today_hours:.1f} call-taker hours/day -> modeled {today:.0%} answered within {TARGET_S} s\n")
    plans = {}
    for key, n in (("A", 11), ("B", 13), ("C", 16)):
        x = plan_shifts(n, req, calls, aht, SHRINKAGE)
        plans[key] = x
        per_hour = [sl_scheduled(c, v, aht, SHRINKAGE) for c, v in zip(A @ x, calls)]
        fte = (n * SHIFT_H - today_hours) * 7 / FTE_HOURS_WEEK
        print(f"Option {key}: {n} shifts/day, +{fte:.1f} FTE, modeled {daily_sl(A @ x, calls, aht, SHRINKAGE):.0%} "
              f"(worst hour {min(per_hour):.0%})\n    starts (start, end, people): {describe(x)}")

    if args.plot:
        import matplotlib.pyplot as plt
        x = plans["B"]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        bottom = np.zeros(24)
        for h in [s for s in range(24) if x[s] > 0]:
            cov = np.array([x[h] if (hh - h) % 24 < SHIFT_H else 0 for hh in range(24)], dtype=float)
            ax.bar(range(24), cov, bottom=bottom, edgecolor="white", label=f"{x[h]} start {hour_label(h)}")
            bottom += cov
        ax.step(range(24), CURRENT_MIN, where="mid", color="#444", ls="--", label="today's minimum")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([hour_label(h) for h in range(0, 24, 2)])
        ax.set_ylim(0, 9)
        ax.set_ylabel("call takers on shift")
        ax.set_title("Option B (synthetic demand): shifts stacked by start time")
        ax.legend(frameon=False, ncol=2)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print("saved", args.plot)


if __name__ == "__main__":
    main()
