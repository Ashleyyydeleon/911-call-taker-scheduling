# 911 call-taker scheduling

A small, readable Python tool for one question: **given limited capacity and uncertain need, where and when should a 911 center put its call takers?**

It turns an hourly call forecast into a staffing requirement (queueing math) and then into a 10-hour shift plan (integer programming). It accompanies a case study on [adeleon.netlify.app](https://adeleon.netlify.app/work/opd-911/index.html).

## The method

1. **Demand.** Hourly calls, planned at the P90 (the level 9 of 10 hours stay under), because an unanswered call costs far more than an idle seat.
2. **Erlang C.** The smallest number of call takers that answers 90% of calls within 15 seconds (the California standard) at each hour's volume.
3. **Shrinkage.** Breaks, meals, training and leave mean scheduled staff = needed staff / (1 - 30%).
4. **Shift plan.** Call takers work four 10-hour shifts a week, so one hire covers a block of the day. An integer program picks shift start times (at most five different starts, never fewer than three people on any hour) for 11, 13 or 16 shifts a day.

## Run it

```bash
pip install -r requirements.txt
python scheduling.py --plot example_output.png
```

Use your own demand with `--demand my.csv` (24 rows; columns `hour,mean,p90`).

## What is not here, and why

**No agency data, forecasts or call records are published here**, for the safety and privacy of the government agency and the people it serves. The default demand is a **synthetic** curve, so the numbers it prints are illustrative and are not findings about any real agency. The real analysis used a teammate's forecasting model and agency data that stay private; the case study reports only rounded, aggregated results.

Two inputs are assumptions, not measurements: 30 seconds of after-call work and 30% shrinkage. In the real analysis they were calibrated so the model reproduced today's observed answer rate, which is calibration, not validation. The estimated current minimum staffing comes from a published City Auditor audit.

## Limits

- Erlang C assumes callers never hang up and calls arrive at random; real waits are burstier.
- It models the 911 line only, not non-emergency lines, and "call takers" are an auditor's estimate of 911-allocated positions, not dispatcher hires.
- Averages hide bad hours: check the worst-hour figure, not just the daily number.

MIT licensed. Built by Ashley Deleon Gomez.
