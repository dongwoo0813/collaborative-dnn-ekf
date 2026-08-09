# Simulation Architecture and Build Handoff

## Collaborative Block-Structured DNN-EKF with GS Branch Sharing, FOV Gating, General MLP Branches, and Observability-Aware Watcher Motion

**Document status:** Full standalone build specification  
**Primary implementation:** MATLAB  
**Current development stage:** Step 09-J.6 complete; Step 10-A pending  
**Reference diagnostic:** 600-s bearing-only matched-coast active-sensing run

---

# 0. Purpose of This Document

This is the complete simulation architecture, implementation ledger,
experimental interpretation guide, and forward build plan for the
collaborative block-structured DNN-EKF MATLAB project. It is intended to stand
on its own: a new development session should be able to recover the research
objective, understand the current code state, reproduce the latest reference
experiment, and continue the implementation using only this document and the
source tree.

The document includes:

1. completed residual norm/cosine diagnostics;
2. cumulative per-branch FIM-weighted-additive residual composition;
3. bearing-only transverse observability experiments;
4. a receding-horizon observability-seeking watcher controller;
5. realistic watcher thrust limiting;
6. long-duration angle-only range/position convergence experiments;
7. the complete Step 10 build track for observability-aware collaborative
   learning;
8. required targeted checks, experiment-separation rules, file-level work, and
   continuation instructions.

The recent observability-seeking run is explicitly treated as a **co-moving,
free-translation diagnostic experiment**. It is not yet an active-control
implementation inside the original prescribed 1000-m circular watcher
architecture. Closed-loop estimator/controller co-design results are also
kept separate from estimator-only results under an identical replayed watcher
trajectory.

The main research objective remains collaborative target tracking and unknown
target-dynamics learning. Watcher maneuvering is an enabling active-sensing
layer, not a replacement for the block-structured DNN-EKF and GS branch-sharing
architecture.

Before adding a new feature:

- check this handoff;
- preserve the existing local/GS estimator interfaces;
- isolate new controller, estimator, and communication effects through
  targeted ablations.

The project must now maintain three explicit experiment axes:

1. **geometry/motion source** — prescribed circular reference, matched-velocity
   coast, prescribed excitation, or controlled excitation about a reference;
2. **estimator/model source** — known dynamics, Oracle residual, Local DNN,
   GS additive, or GS FIM-weighted-additive;
3. **communication source** — frequent, event-triggered, heartbeat-only, or
   disabled.

Only one axis should change in a primary ablation. A joint closed-loop
controller-estimator comparison is still useful, but it must be labeled as a
co-design comparison rather than a pure estimator comparison.

After adding a meaningful feature:

- update the status checklist;
- add the exact runner and targeted check;
- record the seed, geometry, thrust, residual, and covariance settings;
- distinguish requested baseline from realized watcher displacement.

---

# 1. Status Legend

| Symbol | Meaning |
|---|---|
| 🟩 | Implemented and targeted-test or simulation verified |
| 🟨 | Implemented partially or needs stronger validation |
| 🟥 | Not implemented |
| 🟦 | Planned extension or optional ablation |
| ⬜ | Documentation or convenience layer |

---

# 2. Current Handoff Snapshot

## 2.1 Current simulation stage

The codebase now contains a complete local and GS-assisted block-structured
DNN-EKF trunk with:

- angle-only measurement updates;
- general MLP residual branches;
- adaptive \(Q_\theta\) and \(Q_{\epsilon,c}\);
- GS branch repository, upload, and broadcast;
- additive, legacy bearing-FIM-gated, and new FIM-weighted-additive composite
  modes;
- nonlocal branch covariance injection;
- event-triggered and heartbeat upload infrastructure;
- FOV/dropout handling;
- observability diagnostics;
- controlled watcher translation;
- a lightweight receding-horizon observability-seeking direction planner.

The most recent **diagnostic-prototype** experiment is:

```matlab
[outActive, diagActive, figActive] = ...
    run_step09J6_transverse_additive_vs_fim( ...
    true, 100, 600, "observability_seeking");
```

Reference settings:

```matlab
cfg.T = 600.0;
cfg.dt = 0.1;
cfg.meas.type = "bearing";
cfg.meas.sigmaBearing = deg2rad(0.01);
cfg.meas.availabilityMode = "always";

cfg.watchers.motionMode = "controlled";
cfg.scenario.watcherModel = "matched_velocity_coast";
cfg.watchers.maxThrust = 0.02;          % 20 mN
cfg.control.translationMode = "observability_seeking";
cfg.control.obs.startTime = 40.0;
cfg.control.obs.burnDuration = 560.0;   % active until t = 600 s
cfg.control.obs.numCandidateDirections = 8;
cfg.control.obs.planningHorizon = 30.0;
cfg.control.obs.planningDt = 0.5;
cfg.control.obs.replanInterval = 5.0;

cfg.dnn.adaptQThetaEnabled = true;
cfg.dnn.adaptQEpsilonEnabled = true;
cfg.dnn.predictionResidualSource = "GS_composite";

cfg.gs.fimGate.accumulationMode = "cumulative_sum";
```

For a 20-kg watcher and a 100-m straight-line-equivalent displacement over
560 s, the commanded magnitude is

$$
a=6.3776\times10^{-4}\ \mathrm{m/s^2},
$$

$$
F=ma=0.012755\ \mathrm{N},
$$

which is below the configured 20-mN thrust limit.

Important: 100 m is a straight-line-equivalent displacement. Because the
controller replans its direction every 5 s, the realized net displacement can
be smaller and must be logged explicitly in the next build step.

Equally important: this runner deliberately replaces the original prescribed
circular motion by `matched_velocity_coast` before enabling the controlled
double-integrator watcher. In the original architecture,

$$
r_{w,i}^{\mathrm{ref}}(t)
=
R\begin{bmatrix}
\cos(\omega t+\psi_i)\\
\sin(\omega t+\psi_i)
\end{bmatrix},
\qquad
R=1000\ \mathrm{m},
\qquad
\omega=\frac{2\pi}{2000}\ \mathrm{rad/s}.
$$

The active diagnostic does not track this reference and does not include its
nominal centripetal acceleration. Therefore, its result establishes that
controlled LOS diversity can help in the diagnostic geometry; it does not yet
establish the benefit of active motion on top of the original circular watcher
architecture.

## 2.2 Current main finding

In the matched-velocity-coast diagnostic, weak or short-duration transverse
motion did not make the angle-only range estimate converge. Increasing the
available watcher thrust and keeping the observability-seeking controller
active over the long horizon produced substantially smaller late-run position,
range, velocity, and residual errors than the previous weak-excitation runs.

For seed 101, \(T=600\) s, requested equivalent baseline 100 m:

| Case | Whole-run position RMSE [m] | Final position RMSE [m] | Final range RMSE [m] | Final velocity RMSE [m/s] | Position convergence ratio |
|---|---:|---:|---:|---:|---:|
| GS additive | 25.8488 | 18.2291 | 18.2291 | 0.07345 | 0.4912 |
| GS FIM-weighted-additive | 26.3986 | **15.5237** | **15.5236** | **0.05746** | **0.4205** |

Residual diagnostics:

| Case | Whole-run residual RMSE [m/s²] | Final residual RMSE [m/s²] | Final cosine | Final norm ratio |
|---|---:|---:|---:|---:|
| GS additive | \(1.1786\times10^{-4}\) | \(5.6891\times10^{-5}\) | 0.8602 | 1.6824 |
| GS FIM-weighted-additive | **\(8.9237\times10^{-5}\)** | **\(3.2731\times10^{-5}\)** | **0.9838** | **1.5149** |

Permitted interpretation:

- persistent LOS-profile excitation is consistent with improved radial
  information in this diagnostic scenario;
- the closed-loop FIM-weighted run finishes better than the closed-loop
  additive run in position, range, velocity, residual direction, and
  residual-vector error;
- residual magnitude remains too large;
- error reaches a minimum around the later-middle interval and rises slightly
  near the end;
- positive final-window log slopes mean the estimator has not reached stable
  monotone convergence;
- actual controller motion and information gain have not yet been logged well
  enough to claim maneuver optimality.

Not yet permitted:

- attributing the additive/FIM difference solely to the residual-composition
  rule, because each case closes the controller around its own estimate and may
  generate a different watcher trajectory;
- claiming improvement relative to the original prescribed circular watcher
  architecture;
- calling the planner score a calibrated future Fisher-information prediction
  before its measurement-time scaling is verified;
- claiming asymptotic range convergence from one seed with positive final
  log-error slope.

## 2.3 Current technical bottleneck

The immediate bottleneck is no longer only branch interference. It is the
coupled diagnosis of:

1. scenario identity: original circular reference versus matched-coast
   diagnostic;
2. trajectory identity across estimator comparisons;
3. realized watcher geometry and correctly scaled finite-window information;
4. angle-only radial observability versus finite-noise filter convergence;
5. DNN residual magnitude bias;
6. continued maneuver effort after sufficient information has been acquired;
7. the distinction between geometry improvement and collaborative-learning
   improvement.

Do not add a more complicated joint DNN/controller optimizer before separating
these effects.

---

# 3. Architecture Summary

## 3.1 Target truth and residual model

The target physical state is

$$
\eta_t=
\begin{bmatrix}
r_t\\
v_t
\end{bmatrix},
$$

with

$$
\dot r_t=v_t,
\qquad
\dot v_t=a_{\mathrm{nom}}(r_t,v_t,t)+d_{\mathrm{true}}(r_t,v_t,t).
$$

The current main benchmark remains `feedback_sat_disturbance`, a bounded global
nonlinear residual with feedback-like, swirl, periodic, state-dependent, and
slow-bias components.

It is not a uniquely branch-decomposed truth. Therefore, branch overlap and
residual-composition interpretation remain central research issues.

## 3.2 Watcher-local estimator

Watcher \(i\) estimates

$$
X_i=
\begin{bmatrix}
\eta_i\\
\theta_i
\end{bmatrix},
$$

with covariance

$$
P_i=
\begin{bmatrix}
P_{\eta_i\eta_i} & P_{\eta_i\theta_i}\\
P_{\theta_i\eta_i} & P_{\theta_i\theta_i}
\end{bmatrix}.
$$

The measurement Jacobian has the form

$$
H_X=[H_\eta,0],
$$

so \(\theta_i\) is learned indirectly through
\(P_{\theta_i\eta_i}H_\eta^\top S_i^{-1}\nu_i\).

## 3.3 GS branch sharing

The GS is a versioned parameter-library repository, not a centralized
target-state filter. Watcher \(i\) owns its local \(\eta_i\) and \(\theta_i\).
Nonlocal branch copies are cached and used in prediction but are not appended
to the local EKF state.

The nominal block-additive predictor is

$$
\hat d^{(i)}
=
\hat d_i^{\mathrm{local}}
+
\sum_{j\ne i}\hat d_{j,\mathrm{GS}}.
$$

## 3.4 Watcher active-sensing layer

The architecture now requires two distinct controlled-motion models.

### 3.4.1 Diagnostic free-translation model

The current prototype uses

$$
\dot r_{w,i}=v_{w,i},
\qquad
\dot v_{w,i}=\frac{u_i}{m_{w,i}},
\qquad
\|u_i\|\le F_{\max,i}.
$$

This is appropriate for the matched-velocity-coast observability diagnostic.
It must not be silently interpreted as continuation of the original prescribed
circular orbit.

### 3.4.2 Nominal-reference-plus-excitation model

For integration with the original architecture, define a nominal watcher
reference ((r_{w,i}^{\mathrm{ref}},v_{w,i}^{\mathrm{ref}})) and an active
deviation ((\delta r_{w,i},\delta v_{w,i})):

$$
r_{w,i}=r_{w,i}^{\mathrm{ref}}+\delta r_{w,i},
\qquad
v_{w,i}=v_{w,i}^{\mathrm{ref}}+\delta v_{w,i}.
$$

The force command should be decomposed conceptually as

$$
u_i=u_{i}^{\mathrm{track}}+u_{i}^{\mathrm{obs}},
\qquad
\|u_i\|\le F_{\max,i},
$$

where (u_i^{\mathrm{track}}) maintains the selected nominal relative-motion
reference and (u_i^{\mathrm{obs}}) supplies bounded information-seeking
excitation. In the first integration, the prescribed circular trajectory may
remain a kinematic reference while only the perturbation dynamics are
propagated. A later model can replace it with orbital/CW dynamics and explicit
station keeping.

This separation is necessary because “20 mN available for observability” and
“20 mN total actuator limit including nominal tracking” are different physical
assumptions.

### 3.4.3 Current planning objective

The current planner evaluates a discrete set of candidate directions using
only local \(\hat\eta_i\) and \(P_{\eta_i\eta_i}\). It rolls out lightweight
physical-state bearing information and selects the direction with minimum
predicted terminal radial variance.

The current information accumulation is implemented as

$$
G_H
=
\sum_{\tau\in\mathcal T_H}
\Phi(\tau)^\top H(\tau)^\top R^{-1}H(\tau)\Phi(\tau)
\,\Delta t_{\mathrm{plan}}.
$$

Before treating the score magnitude as a predicted covariance, verify this
scaling. For discrete measurements, the natural information sum contains one
term per predicted measurement and does not automatically require
\(\Delta t_{\mathrm{plan}}\). Multiplication by a time increment is valid only
under an explicitly defined continuous measurement-information rate. Candidate
ranking may remain similar under a common scale, but predicted variance values
and comparisons between different planning grids will not be calibrated until
this convention is fixed.

Current planner limitations:

- no zero-thrust candidate;
- no control-effort penalty;
- no on/off hysteresis;
- no trajectory-safety or FOV constraint;
- constant-velocity nominal target rollout;
- measurement/information-time scaling not yet calibrated;
- no logging of every candidate score;
- no direct parameter-information objective;
- no proof that the selected direction is globally optimal.

The current planner is therefore best described as a **finite-horizon
bearing-information heuristic**. The physical information matrix it builds is
more complete than the branch geometry projector used by
`fim_weighted_additive`, but it is not yet a full augmented DNN-EKF FIM.

---

# 4. Distinguish the Geometry/FIM Modes

Two different geometry-based residual-composition paths exist and must not be
conflated.

The stored branch quantity

$$
\Omega_j
=
\sum_k\left(I-u_{j,k}u_{j,k}^{\top}\right)
$$

is a direction-only geometry-support matrix. It omits the bearing Jacobian's
range scaling, measurement covariance, state-transition dynamics, and DNN
parameter sensitivity. It is therefore safer in formal writing to call it a
**LOS geometry information surrogate** rather than the full Fisher information
matrix of the measurement likelihood. Existing configuration names containing
`fim` can remain for code compatibility.

## 4.1 Legacy `bearing_fim_gated`

The earlier gate is

$$
B_{j|m}
=
\left(
\sum_{\ell\in\mathcal A_m}\bar\Omega_\ell+\epsilon I
\right)^{-1}
\bar\Omega_j,
$$

with approximately

$$
\sum_jB_{j|m}\approx I.
$$

This behaves like an all-branch directional ensemble and was introduced to
mitigate overlapping branch corrections.

## 4.2 New `fim_weighted_additive`

The current transverse experiments use

$$
W_j
=
\frac{\bar\Omega_j}{\lambda_{\max}(\bar\Omega_j)},
$$

$$
\hat d^{(m)}
=
\sum_jW_{j|m}\hat d_j.
$$

Properties:

- normalization is internal to each branch information matrix;
- there is no watcher-to-watcher normalization;
- \(\sum_jW_j=I\) is not required;
- cumulative LOS projectors are used in the current experiment;
- adaptive \(Q_\theta\) and \(Q_{\epsilon,c}\) remain enabled.

## 4.3 Theoretical status

The nominal problem formulation treats branches as complementary additive
components. FIM-weighted residual composition changes that mean model and is
therefore an extension/ablation until a matching branch interpretation is
adopted.

Geometry/FIM-like quantities have three distinct architecture roles:

1. finite-horizon physical-state bearing information for watcher-motion
   selection;
2. geometry confidence for learning/upload validation;
3. experimental LOS-geometry weighting of residual composition.

The Young coefficients in \(Q_{\mathrm{nonlocal}}\) are separate PSD bounding
coefficients and are not FIM trust weights.

---

# 5. What Has Been Implemented Since the Previous Handoff

## 5.1 Step 09-J.6 residual estimate diagnostics — 🟩

Implemented:

- truth/estimate residual norm;
- norm ratio;
- cosine alignment;
- time-window summaries;
- additive/FIM comparison.

Main finding:

- branch-composition errors include both direction and magnitude effects;
- later observability experiments can achieve strong residual direction
  alignment while retaining magnitude overestimation.

## 5.2 Cumulative per-branch FIM-weighted additive — 🟩

Implemented:

```matlab
cfg.gs.compositeMode = "fim_weighted_additive";
cfg.gs.fimGate.accumulationMode = "cumulative_sum";
```

Weight:

$$
W_j=\bar\Omega_j/\lambda_{\max}(\bar\Omega_j).
$$

Verified properties:

- finite symmetric PSD weights;
- no across-watcher normalization;
- consistent mean/covariance transformation path;
- additive and FIM cases use identical truth, noise, seed, initialization,
  adaptive-Q settings, and non-composite configuration;
- in controlled closed-loop mode, identical configuration does **not** imply
  identical realized motion, because the controller uses each case's local
  estimate and covariance.

## 5.3 Extended transverse comparison runner — 🟩

Runner:

```matlab
run_step09J6_transverse_additive_vs_fim( ...
    makePlots, desiredBaseline, simulationTime, maneuverMode)
```

Supported maneuver modes:

- `"transverse"`: existing prescribed maneuver;
- `"observability_seeking"`: controlled receding-horizon direction selection.

Added output metrics:

- initial-to-final position convergence ratio;
- initial-to-final range convergence ratio;
- final-window position log slope;
- final-window range log slope;
- simulation time;
- requested/equivalent baseline;
- commanded acceleration, \(\Delta v\), required thrust, and thrust limit.

## 5.4 Controlled watcher motion — 🟩

Updated:

- `watcher/initWatcherTruth.m`
- `watcher/propagateWatcherStep.m`
- `control/watcherController.m`
- `simulation/simulate_GS_DNN_EKF.m`

Controller memory stores:

- selected direction;
- next replan time;
- selected candidate index;
- selected score.

The simulator passes local \(\hat\eta_i\) and \(P_{\eta_i\eta_i}\) to the
controller. Target truth remains diagnostic-only.

## 5.5 Lightweight observability-seeking planner — 🟩

New helper:

```text
control/selectObservabilitySeekingDirection.m
```

Current algorithm:

1. generate eight candidate directions;
2. roll out watcher motion over 30 s at 0.5-s resolution;
3. propagate the target with a local constant-velocity nominal model;
4. accumulate a four-state bearing-information matrix;
5. form a predicted posterior covariance;
6. select minimum terminal radial variance;
7. apply the direction for 5 s and replan.

Targeted tests completed:

- MATLAB static analysis of changed files;
- planner finite-direction/finite-score smoke test;
- thrust-saturation test;
- short additive/FIM integration test;
- long-burn configuration test for 100-m equivalent displacement.

Not yet tested:

- Monte Carlo controller consistency;
- candidate-score correctness against nonlinear covariance rollout;
- planner information scaling versus rollout grid and measurement cadence;
- safety/FOV-constrained planning;
- actual-baseline logging;
- identical trajectory replay across estimator cases;
- active deviation about the original prescribed circular reference;
- coast/re-excitation switching;
- parameter-information optimization.

---

# 6. Updated Architecture Checklist

## A. Truth / Measurement Layer

| ID | Item | Status | Notes |
|---|---|---|---|
| A1 | Target truth dynamics | 🟩 | Nominal + residual |
| A2 | Bearing-only measurement | 🟩 | 2D current model |
| A3 | Measurement availability wrapper | 🟩 | Always/FOV |
| A4 | FOV/dropout diagnostics | 🟩 | Prediction-only path validated |
| A5 | `feedback_sat_disturbance` | 🟩 | Current main learning benchmark |
| A6 | Realistic force/thruster target residual | 🟥 | Later Step 10-D |
| A7 | Normalized acceleration learning output | 🟥 | Pair with force-level truth model |

## B. Local Estimator Layer

| ID | Item | Status | Notes |
|---|---|---|---|
| B1 | Physical EKF | 🟩 | Existing baseline |
| B2 | Local block DNN-EKF | 🟩 | Fixed feature and MLP |
| B3 | Adaptive \(Q_\theta\) | 🟩 | Covariance matching |
| B4 | Adaptive \(Q_{\epsilon,c}\) | 🟩 | Enabled in current experiments |
| B5 | Low-rank measurement update | 🟩 | \(H_X=[H_\eta,0]\) |
| B6 | State-parameter cross covariance | 🟩 | Indirect learning path |
| B7 | General MLP Jacobians | 🟩 | Analytic and checked |
| B8 | Residual norm/cosine diagnostic | 🟩 | Step 09-J.6 |
| B9 | Split physical process and approximation noise | 🟦 | Future refinement |
| B10 | Geometry-dependent learning confidence | 🟥 | Step 10-D after replayed validation |

## C. GS Backend

| ID | Item | Status | Notes |
|---|---|---|---|
| C1 | Versioned GS repository | 🟩 | Parameter library, no state filter |
| C2 | Upload/broadcast/cache | 🟩 | MLP compatible |
| C3 | Additive composite residual | 🟩 | Nominal block baseline |
| C4 | Nonlocal covariance injection | 🟩 | Young conservative surrogate |
| C5 | Event-triggered upload | 🟩 | Contribution/dwell infrastructure |
| C6 | Maximum-silence upload | 🟩 | Heartbeat path exists |
| C7 | Legacy bearing-FIM-gated mode | 🟩 | Across-branch ensemble gate |
| C8 | Cumulative FIM-weighted-additive | 🟩 | Per-branch eigenvalue normalization |
| C9 | Geometry metadata in GS records | 🟩 | OmegaBar plumbing exists |
| C10 | Geometry age/validity cutoff | 🟥 | Needed for long missions |
| C11 | Geometry-supported GS acceptance | 🟥 | Planned extension |
| C12 | Innovation-supported final acceptance | 🟨 | Theory exists; verify full implementation path |
| C13 | Branch novelty/overlap acceptance | 🟦 | Diagnostic/extension |

## D. Watcher Motion / Active Sensing Layer

| ID | Item | Status | Notes |
|---|---|---|---|
| M1 | Prescribed watcher trajectories | 🟩 | Original circular, matched coast, transverse paths |
| M2 | Controlled translational dynamics | 🟩 | Free double integrator with force saturation |
| M3 | Local-estimate controller interface | 🟩 | \(\hat\eta_i,P_{\eta_i\eta_i}\) passed |
| M4 | Candidate-direction planner | 🟩 | Eight 2D directions |
| M5 | Finite-horizon bearing Gramian | 🟨 | Four-state rollout; time scaling needs calibration |
| M6 | Radial-variance objective | 🟩 | Current selected score |
| M7 | 20-mN thrust envelope | 🟩 | Active-mode reference setting |
| M8 | Direction/index/score controller memory | 🟩 | Logged to results |
| M9 | Full controller diagnostics | 🟩 | Step 10-A.1 implemented |
| M10 | Zero-thrust candidate | 🟥 | Step 10-C |
| M11 | Effort penalty | 🟥 | Step 10-C |
| M12 | On/off hysteresis | 🟥 | Step 10-C |
| M13 | Observable coast/re-excitation policy | 🟥 | Step 10-C |
| M14 | Bounded target-relative orbit reference | 🟥 | Integrate active deviation with original architecture |
| M15 | FOV/collision/safety constraints | 🟦 | Later MPC extension |
| M16 | Parameter-information objective | 🟦 | Step 10-D after physical validation |
| M17 | Nominal-reference + active-deviation dynamics | 🟥 | Required before architecture-level active-motion claim |
| M18 | Tracking-force versus sensing-force budget | 🟥 | Required for physical thrust accounting |

## E. Experiment / Analysis Layer

| ID | Item | Status | Notes |
|---|---|---|---|
| D1 | Multi-metric evaluator | 🟩 | Tracking/residual metrics |
| D2 | FOV sweeps | 🟩 | Existing |
| D3 | Paired Monte Carlo wrapper | 🟨 | Final MC deferred |
| D4 | Branch contribution alignment | 🟩 | Existing |
| D5 | Residual norm/cosine | 🟩 | Current diagnostic |
| D6 | Transverse baseline sweep | 🟩 | Physical EKF baseline exists |
| D7 | Long-time additive/FIM comparison | 🟩 | 600-s experiment completed |
| D8 | Observability-seeking experiment | 🟩 | Seed-101 reference result |
| D9 | Actual baseline and \(\Delta v\) metrics | 🟩 | Step 10-A.1 implemented |
| D10 | Planner score-margin diagnostic | 🟩 | Step 10-A.1 implemented |
| D11 | Known/Oracle/DNN frozen-motion ablation | 🟥 | Step 10-A.4 |
| D12 | Coast/re-excitation comparison | 🟥 | Step 10-C |
| D13 | Event-trigger under active motion | 🟥 | Step 10-F |
| D14 | Final paired Monte Carlo | 🟥 | Step 10-G |
| D15 | Watcher trajectory export/replay | 🟥 | Next Step 10-A.2 |
| D16 | Original circular versus active-about-circular ablation | 🟥 | Architecture integration test |
| D17 | Planner grid/time-scaling invariance | 🟥 | Required before calibrated score claims |

## F. P2P Backend

| ID | Item | Status | Notes |
|---|---|---|---|
| P1 | P2P graph/cache | 🟦 | Deferred |
| P2 | GS vs P2P comparison | 🟦 | After GS active-sensing benchmark matures |

---

# 7. Recommended Next Build Order

The build order is organized around **identifiability of experimental causes**.
Motion generation, estimator evaluation, and communication evaluation are
separated before they are recombined.

## Step 10-A — Experimental Decoupling and Trajectory Replay

### Step 10-A.1 — Controller, realized-motion, and geometry logging

**Status:** Implemented and smoke-tested

Add per-watcher logs:

```text
watcherR(:,k,i), watcherV(:,k,i), watcherA(:,k,i)
commandForce(:,k,i), selectedDirection(:,k,i)
selectedCandidateIndex(k,i), selectedScore(k,i)
candidateScores(:,k,i), replanFlag(k,i), controllerActive(k,i)
cumulativeImpulse(k,i), cumulativeDeltaV(k,i)
referenceR(:,k,i), referenceV(:,k,i)
deviationR(:,k,i), deviationV(:,k,i)
actualLOSChange(k,i), predictedRadialVariance(k,i)
```

Required summaries:

- vector displacement, path length, and maximum reference deviation;
- thrust-on time, total impulse, and \(\Delta v\);
- number of replans and direction switches;
- best/second-best candidate score margin;
- realized LOS angular change and LOS angular-rate history;
- windowed information eigenvalues and condition number;
- information improvement per impulse or \(\Delta v\).

The targeted check
`sanity_check/check_step10A1_observability_controller_logging.m` must verify
dimensions, finite values, force saturation, monotone cumulative impulse,
score/index consistency, reference/deviation reconstruction, and absence of
target truth from controller inputs.

Implemented outputs include `selectedDirection`, `selectedCandidateIndex`,
`selectedScore`, `candidateScores`, candidate information eigenvalue and
condition diagnostics, `replanFlag`, `controllerActive`, cumulative impulse,
cumulative Δv, path length, realized displacement, nominal reference,
LOS-change telemetry, and predicted radial variance. The case runner also
returns `controllerSummaryAdd` and `controllerSummaryFIM`.

### Step 10-A.2 — Trajectory export and deterministic replay

**Status:** Next implementation

Add a watcher-motion mode such as

```matlab
cfg.watchers.motionMode = "replay";
cfg.watchers.replay = savedMotion;
```

The replay record must contain time, position, velocity, acceleration, and any
attitude/FOV state that affects measurement availability. During replay, the
estimator must not call the observability controller.

Generate a reference motion once, then rerun all estimators with exactly that
motion and the same truth/noise draws. Prefer an estimator-independent source,
such as the Oracle or a fixed reference estimator, for the primary benchmark.
A motion generated by a learned estimator may also be replayed, but it must be
labeled by its generating policy.

Acceptance criteria:

- watcher trajectories match sample-by-sample across replayed cases;
- noiseless bearing measurements match sample-by-sample;
- availability/FOV flags match sample-by-sample;
- only the selected estimator/composite mode differs.

### Step 10-A.3 — Planner information-scaling calibration

**Status:** Not implemented — required before interpreting score magnitude

Choose one explicit interpretation:

1. discrete measurements at a specified cadence, using
   \(G=\sum_k\Phi_k^\top H_k^\top R^{-1}H_k\Phi_k\); or
2. a continuous measurement-information rate, using a dimensionally defined
   integral and spectral-density convention.

The current implementation multiplies each discrete contribution by
`planningDt`. Test `planningDt = [0.25, 0.5, 1.0]` while keeping the assumed
physical measurement cadence fixed. Candidate rankings and predicted posterior
covariance must not change merely because the numerical rollout grid changes.

For selected replans, compare the linear information prediction with a
nonlinear covariance or short Monte Carlo rollout. Exact equality is not
required, but the candidate ranking should be sufficiently reliable.

### Step 10-A.4 — Frozen-motion estimator ablation

Use one replayed watcher trajectory, truth, measurement/noise realization,
availability sequence, initial condition, and process-noise setting for:

1. nominal physical EKF without learned residual;
2. Oracle residual EKF;
3. Local DNN-EKF;
4. GS additive DNN-EKF;
5. GS FIM-weighted-additive DNN-EKF.

This separates four questions: whether the geometry supports range estimation,
how much residual-model error degrades it, whether GS sharing helps Local, and
whether LOS-geometry weighting helps additive under identical measurements.
The current 18.23 m versus 15.52 m result remains a closed-loop co-design result
until this replayed comparison is completed.

## Step 10-B — Reconnect Active Sensing to the Original Watcher Architecture

### Step 10-B.1 — Original circular-reference baseline

Run the original prescribed 1000-m circular geometry for the same 600-s
horizon, bearing noise, residual truth, and estimator cases. Report its LOS
change and windowed physical observability beside the matched-coast diagnostic.

This determines whether the original circular geometry already provides most
of the necessary excitation. If it does, active control should be framed as
information maintenance or recovery under weak geometry, FOV loss, and mission
constraints—not as continuous replacement of nominal motion.

### Step 10-B.2 — Active deviation about a nominal reference

Implement

$$
r_w=r_w^{\mathrm{ref}}+\delta r_w,
\qquad
v_w=v_w^{\mathrm{ref}}+\delta v_w,
$$

with bounded deviation and explicit allocation of total force between
reference tracking and observability excitation. Initially, the analytic
circle may remain the kinematic nominal reference; do not yet claim orbital
station-keeping fidelity.

### Step 10-B.3 — Geometry-source ablation

Compare:

1. original prescribed circular reference;
2. matched-velocity coast;
3. prescribed fixed transverse excitation;
4. observability-seeking free translation;
5. observability-seeking deviation about the circular reference.

Match actuator assumptions and report impulse, \(\Delta v\), reference
deviation, FOV availability, and realized information. Do not compare only the
input `desiredBaseline` values.

## Step 10-C — Information Maintenance: Coast and Re-Excitation

### Step 10-C.1 — Zero-thrust candidate

Add \(u=0\) to the candidate set and log its score.

### Step 10-C.2 — Effort and reference-deviation penalties

Use initially

$$
J_i(u)
=
\sigma_{\rho,i,H}^2(u)
+
\lambda_u\mathcal C_u(u)
+
\lambda_r\mathcal C_{\mathrm{dev}}(u).
$$

Sweep \(\lambda_u\) and \(\lambda_r\); do not select them from one seed only.

### Step 10-C.3 — Acquisition/coast/re-excitation hysteresis

Use separate start and stop thresholds. The intended operation is:

```text
acquire information -> coast/track nominal reference -> monitor degradation
-> re-excite only when needed
```

## Step 10-D — Geometry-Supported Collaborative Learning

### Step 10-D.1 — Geometry confidence metadata

Correlate geometry support with parameter update norm, innovation improvement,
branch-output error, upload acceptance, and later nonlocal usefulness. Keep it
diagnostic first.

### Step 10-D.2 — Geometry-aware GS validation

Test covariance-margin or quarantine policies before hard rejection. Poor
geometry lowers confidence but does not prove that an update is wrong.

### Step 10-D.3 — Parameter-information motion objective

Only after physical-state observability and replayed estimator comparisons are
validated, add parameter/feature information to the motion objective. Avoid a
full augmented DNN-EKF rollout for every candidate in the first implementation.

## Step 10-E — Realistic Target Force/Thruster Residual

**Status:** 🟥 Preserved from previous Step 09-K.1 plan

Candidate truth model:

$$
F_{\mathrm{cmd}}
=
\operatorname{sat}_{F_{\max,T}}
\left(
-K_r(r_T-r_{\mathrm{ref}})
-K_v(v_T-v_{\mathrm{ref}})
+F_{\mathrm{bias}}
+F_{\mathrm{pulse}}(t)
\right),
$$

$$
\tau_F\dot F_{\mathrm{res}}
=
-F_{\mathrm{res}}+F_{\mathrm{cmd}},
$$

$$
d_{\mathrm{true}}=F_{\mathrm{res}}/m_T.
$$

Recommended learning output:

$$
\hat d_{\mathrm{DNN}}
=
a_{\mathrm{scale}}\tanh(\varphi_\theta).
$$

Keep target-control residual modeling separate from watcher active-sensing
control in configuration names and logs.

## Step 10-F — Event-Triggered Sharing under Persistent Motion

After the active-motion frequent-upload reference is mature:

1. re-enable contribution-change event triggering;
2. retain dwell and maximum-silence conditions;
3. compare frequent upload, event-triggered upload, and heartbeat-only cases;
4. report communication reduction versus tracking and residual degradation;
5. study whether maneuver-induced innovation transients cause unnecessary
   uploads.

## Step 10-G — Robustness and Monte Carlo

Final selected benchmark sweeps:

- seed;
- bearing noise;
- initial range error;
- FOV/dropout;
- watcher thrust limit;
- planning horizon and replan interval;
- residual family;
- additive versus FIM-weighted-additive;
- frequent versus event-triggered communication.

Use paired truth/noise draws for all compared cases.

Report two separate result families:

1. **frozen-motion estimator comparisons**, which support estimator-specific
   claims;
2. **closed-loop co-design comparisons**, which measure total system
   performance but combine estimator and controller effects.

---

# 8. Next File-Level Implementation Plan

## Immediate files to modify

```text
simulation/simulate_GS_DNN_EKF.m
watcher/propagateWatcherStep.m
watcher/initWatcherTruth.m
control/watcherController.m
control/selectObservabilitySeekingDirection.m
main/run_step09J6_transverse_additive_vs_fim.m
```

Implementation sequence inside Step 10-A:

1. expose existing controller memory and force/motion histories in simulator
   output without changing estimator mathematics;
2. add summary/plot functions and validate logs;
3. add a replay motion path and sample-by-sample identity check;
4. calibrate planner information scaling;
5. only then run the frozen-motion estimator ablation.

## Recommended new files

```text
analysis/computeWatcherObservabilityDiagnostics.m
analysis/summarizeObservabilityController.m
plotting/plotObservabilityControllerDiagnostics.m
sanity_check/check_step10A1_observability_controller_logging.m
watcher/validateWatcherReplay.m
sanity_check/check_step10A2_watcher_motion_replay.m
sanity_check/check_step10A3_planner_information_scaling.m
main/run_step10A4_frozen_motion_estimator_ablation.m
main/run_step10B1_original_circular_reference_baseline.m
main/run_step10B3_geometry_source_ablation.m
```

Do not place analysis-only calculations inside the estimator prediction or
measurement-update functions.

---

# 9. Required Targeted Checks

## 9.1 Controller logging consistency

Verify:

- selected candidate index maps to selected direction and score;
- replan flags occur at configured intervals;
- command force does not exceed \(F_{\max}\);
- cumulative impulse and path-length \(\Delta v\) are nondecreasing;
- controller output is finite;
- planner input contains local estimate/covariance and not target truth.

## 9.2 Actual baseline consistency

For a fixed-direction constant-acceleration test, verify the logged displacement
against

$$
B
=
a
\left(
\frac12T_b^2+T_bT_c
\right).
$$

This check applies only to fixed direction. The adaptive planner must report
realized vector displacement separately.

## 9.3 Observability metric reconstruction

For one stored replan event, manually reconstruct the candidate Gramian,
posterior covariance, terminal radial direction, and selected score. Compare
against the planner log.

## 9.4 Paired-case motion identity

When comparing additive and FIM residual composition, verify whether controller
motion is intended to be:

- closed-loop and estimator-dependent for each case; or
- frozen/replayed identically across estimator cases.

Both experiments are useful but answer different questions. A fair pure
estimator comparison should support frozen motion replay.

## 9.5 Replay determinism

For two estimators using one replay record, require exact or tolerance-level
identity of watcher position, velocity, acceleration, noiseless bearing,
measurement-availability flag, and measurement-noise draw. Confirm that the
controller call count is zero in replay mode.

## 9.6 Planner grid and measurement-cadence consistency

Hold physical measurement cadence fixed while refining the candidate rollout
integration grid. Verify that the information model does not create artificial
information merely by adding numerical grid points. Separately vary the
assumed measurement cadence and confirm the expected increase or decrease in
information.

## 9.7 Original-reference reconstruction

With active deviation disabled, the nominal-reference-plus-deviation model must
reconstruct `watcherTrajectory.m` for the prescribed circular case to numerical
tolerance. With active deviation enabled, verify

$$
r_w-r_w^{\mathrm{ref}}=\delta r_w,
\qquad
v_w-v_w^{\mathrm{ref}}=\delta v_w,
$$

and account separately for reference-maintenance and sensing-control effort.

---

# 10. Core MATLAB Commands

## Current long observability-seeking comparison

```matlab
addpath(genpath(pwd));
[outActive, diagActive, figActive] = ...
    run_step09J6_transverse_additive_vs_fim( ...
    true, 100, 600, "observability_seeking");
```

## Existing prescribed comparison

```matlab
addpath(genpath(pwd));
[outPrescribed, diagPrescribed, figPrescribed] = ...
    run_step09J6_transverse_additive_vs_fim( ...
    true, 100, 600, "transverse");
```

Do not interpret these as equal-thrust comparisons unless the prescribed
trajectory is explicitly constrained to the same \(F_{\max}\).

## Existing FIM-weight check

```matlab
addpath(genpath(pwd));
check_fim_weighted_additive
```

## Existing geometry support checks

```matlab
addpath(genpath(pwd));
check_step09J1_omega_bar_update
check_step09J2_gs_omega_bar_metadata
```

## Existing MLP and covariance checks

```matlab
addpath(genpath(pwd));
check_step09H2_general_branch_MLP_jacobians
check_step09H3a_branch_model_wrapper
check_step09H3b_init_local_dnn_ekf_branch_models
check_step09H3c_predict_local_branch_models
check_step09H3d_update_local_branch_models
check_step09H3e_theta_learning_path
check_step09I3_mlp_nonlocal_covariance_injection
```

---

# 11. Interpretation Rules for Future Results

1. Do not infer range convergence from bearing NIS alone.
2. Report position and exact-range errors separately, even when they are nearly
   equal because the error is predominantly radial.
3. A convergence ratio below one indicates improvement from the initial to the
   final window; it does not prove asymptotic convergence.
4. A positive final log-error slope indicates late-window error growth even if
   final RMSE is below initial RMSE.
5. Report residual direction and magnitude separately using cosine and norm
   ratio.
6. Report requested, straight-line-equivalent, and realized baseline
   separately.
7. Report thrust limit, commanded thrust, thrust-on time, impulse, and \(\Delta
   v\).
8. Separate structural observability, finite-noise detectability, filter
   convergence, and DNN parameter identifiability.
9. Do not call Young coefficients FIM weights or trust weights.
10. State explicitly whether compared estimators experienced identical replayed
    motion or different closed-loop estimator-dependent motion.
11. State explicitly whether watcher motion is the original prescribed circle,
    matched-velocity coast, free controlled translation, or active deviation
    about a nominal reference.
12. Treat \(\Omega=\sum(I-uu^\top)\) as a LOS geometry-information surrogate,
    not the complete likelihood FIM, unless range, noise, dynamics, and relevant
    sensitivities are included.
13. Do not compare planner-score magnitudes across planning grids until the
    measurement-cadence convention is calibrated.
14. Report nominal reference-maintenance effort and information-seeking effort
    separately whenever both are represented.

---

# 12. Updated Continuation Prompt

```text
나는 MATLAB에서 observability-aware collaborative block-structured DNN-EKF
simulation을 만들고 있다.

첨부한 Updated Simulation Architecture and Build Handoff 문서가 현재
architecture와 build status이다.

응답 스타일:
- 한국어로 설명해줘.
- 기존 구조를 최대한 유지해줘.
- MATLAB patch는 한 단계씩 적용해줘.
- dimension/Jacobian/covariance/controller logic이 바뀌면 targeted check를
  추가해줘.
- truth를 controller 입력으로 사용하지 마.
- prescribed baseline, straight-line-equivalent baseline, realized baseline을
  구분해줘.
- additive와 FIM 비교에서 motion이 replay-identical인지 closed-loop인지
  명시해줘.
- original prescribed circular geometry, matched-velocity coast diagnostic,
  free controlled translation, nominal-reference active deviation을 구분해줘.
- Omega=sum(I-uu')는 full likelihood FIM이 아니라 LOS geometry information
  surrogate라고 구분해줘.

현재 핵심 상태:
1. Local/GS block-structured DNN-EKF, general MLP, adaptive Qtheta/Qepsilon,
   event-trigger infrastructure, Qnonlocal, FOV/dropout이 구현되어 있다.
2. GS는 target-state filter가 아니라 versioned branch repository이다.
3. additive, legacy bearing_fim_gated, cumulative fim_weighted_additive가
   구현되어 있다.
4. fim_weighted_additive는 W_j=Omega_j/lambda_max(Omega_j)이며 watcher 간
   normalization은 없다.
5. matched_velocity_coast 진단 시나리오용 observability-seeking free
   controlled watcher motion이 구현되어 있다.
6. planner는 local etaHat와 PEta만 사용하고 target truth는 사용하지 않는다.
7. 현재 reference는 T=600 s, desired equivalent baseline=100 m,
   Fmax=20 mN, controller active 40--600 s이다.
8. 이 closed-loop reference에서 final position/range RMSE는 additive 약
   18.23 m, FIM-weighted 약 15.52 m이다. 두 case의 realized motion identity는
   아직 보장되지 않으므로 pure estimator comparison 결과는 아니다.
9. original architecture는 R=1000 m, omega=2*pi/2000인 prescribed circular
   watcher reference이다. 현재 active prototype은 이것을 추종하지 않는다.
10. Step 10-A.1 controller/realized-motion logging과 summary/check가 구현되어
    있다. 현재 가장 큰 누락은 motion replay, planner information scaling
    calibration, original reference integration이다.

지금 가장 먼저 할 일:
Step 10-A.2 Trajectory Export and Deterministic Replay를 구현하자.

이미 구현된 로그:
- selected direction/index/score and all candidate scores
- replan flag and active flag
- cumulative impulse and Delta-v
- reference/deviation displacement and realized baseline
- LOS change and LOS-change/noise ratio
- predicted radial variance
- finite-window observability eigenvalue/condition diagnostics

그 다음 순서는:
1. Step 10-A.2 trajectory export/replay
2. Step 10-A.3 planner information-scaling calibration
3. Step 10-A.4 known/Oracle/Local/GS additive/GS FIM frozen-motion ablation
4. Step 10-B original circular baseline 및 active-deviation integration
이다.
```

---

# 13. One-Sentence Current Status

The simulation contains a complete local/GS block-structured DNN-EKF and a
promising matched-coast active-sensing prototype with Step 10-A.1 telemetry;
the recent additive/FIM result is still a closed-loop co-design diagnostic,
and the immediate priority is deterministic trajectory replay, information-
score calibration, and integration as a bounded active deviation about the
original circular watcher reference.
