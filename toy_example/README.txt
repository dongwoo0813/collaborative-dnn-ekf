Local angle-only acceleration observability experiment

Files
-----
run_local_accel_observability_experiment.m
demo_local_accel_observability_experiment.m

Default experiment
------------------
- Independent local EKFs with state [r; v; a]
- Unknown constant target acceleration
- One active angle-only watcher
- Same angle-noise realization in:
    no watcher maneuver
    finite transverse pulse pair
- Initial local state lies on the correct initial LOS but has an incorrect
  positive range scale
- Small white-jerk process noise

Supported active-watcher modes
------------------------------
single
parallel_pair
all
dropout

Important interpretation
------------------------
With one active watcher, sum(P_perp) is rank deficient. The geometry-aware
fusion output contains only the transverse acceleration component and is not
a complete acceleration estimate. The active local EKF estimate is the main
quantity for the first single-watcher experiment.

Run
---
demo_local_accel_observability_experiment
