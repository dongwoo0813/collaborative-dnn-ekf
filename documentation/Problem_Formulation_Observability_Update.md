# Problem Formulation Update: Observability-Aware Collaborative Learning and Target Tracking

## Purpose of this update

This document contains the material that should be inserted into the existing
problem-formulation note. It does not replace the existing derivations for the
block-local DNN-EKF, event-triggered GS sharing, stale covariance aging, or the
Young-inequality covariance surrogate.

The update makes watcher motion an explicit part of the collaborative learning
and tracking architecture. The maneuver is not a separate end objective. It is
an active-sensing mechanism that improves the angle-only geometry needed for
both target-state tracking and indirect DNN-parameter learning.

## Recommended edits to the existing document

1. Add the short paragraph in **Core-Idea Addition** at the end of Section 1.
2. Insert **Observability-Aware Watcher Motion and Active Sensing** after the
   current local measurement/EKF section and before event-triggered parameter
   sharing.
3. Insert **Geometry Information and the Three Uses of FIM** after the new
   active-sensing section.
4. Keep the nominal additive composite residual in the main formulation.
   Present FIM-weighted residual composition as an extension/ablation unless
   the branch interpretation is explicitly changed from complementary blocks
   to overlapping residual estimators.
5. Add the geometry metadata in **GS Metadata Addition** to the branch record
   and upload packet.
6. Replace the existing revised/suggested research-framing paragraphs with
   **Updated Research Framing**.
7. Append the new advantages, caveats, assumptions, theory items, and simulation
   questions to their corresponding sections.
8. Correct the duplicated section numbers near `Main Advantages`,
   `Main Caveats`, and `Suggested Research Framing` during final consolidation.

---

## Core-Idea Addition

Because the watchers use angle-only measurements, the quality of the local
state estimate and the indirect learning signal for each DNN branch depend on
the watcher-target relative geometry. Watcher motion is therefore included as
an active-sensing component of the architecture. Each watcher may coast or
execute a bounded, calibrated maneuver selected to improve predicted
observability using only its local target-state estimate and covariance. The
resulting geometry supports target tracking and dynamics approximation; it does
not replace the block-structured collaborative learning architecture.

---

## Observability-Aware Watcher Motion and Active Sensing

### Watcher dynamics and constraints

Let watcher (i) have position (r_{w,i}), velocity (v_{w,i}), mass
(m_{w,i}), and commanded force (u_i). Its translational dynamics are

$$
\dot r_{w,i}=v_{w,i},
\qquad
\dot v_{w,i}=\frac{1}{m_{w,i}}u_i,
$$

subject to

$$
\|u_i(t)\|\le F_{\max,i}.
$$

The watcher state and applied control are assumed known to watcher (i). The
relative target position and LOS direction are

$$
q_i=r_t-r_{w,i},
\qquad
\ell_i=\frac{q_i}{\|q_i\|}.
$$

The maneuver planner must use the local estimate

$$
\hat q_i=\hat r_{t,i}-r_{w,i}
$$

and must not use the true target state. Target truth is used only for offline
simulation evaluation.

### Why maneuvering is required

For linear relative dynamics without a calibrated maneuver, positive scaling
of the initial relative position and velocity can generate the same LOS
measurement profile. This produces the familiar range ambiguity of
angles-only navigation.

Let the natural relative position profile be

$$
q_{i,\mathrm{nat}}(t)
=
\Phi_{rr}(t,t_0)q_i(t_0)
+
\Phi_{rv}(t,t_0)\dot q_i(t_0),
$$

and let the maneuver-induced relative displacement be

$$
\Delta q_{i,u}(t)
=
-\int_{t_0}^{t}
\Phi_{rv}(t,\tau)
\frac{u_i(\tau)}{m_{w,i}},d\tau.
$$

A sufficient observability condition is that the calibrated maneuver alter the
natural LOS profile. Equivalently, over the observation interval,

$$
\boxed{
\Delta q_{i,u}(t)
\neq
\alpha_i(t)q_{i,\mathrm{nat}}(t)
}
$$

for any scalar function \(\alpha_i(t)\) over at least part of the interval.
Purely LOS-parallel displacement generally fails to remove the range-scale
ambiguity, whereas a maneuver with a LOS-normal component changes the bearing
profile and can make range observable.

Structural observability alone does not guarantee accurate estimation in the
presence of bearing noise, model mismatch, and DNN approximation error.
Therefore, the controller evaluates observability strength rather than only a
binary observable/unobservable condition.

### Predicted finite-horizon information

At planning time (t_k), watcher (i) generates a finite set of admissible
control candidates

$$
\mathcal U_i(k)=\{u_i^{(0)},u_i^{(1)},\ldots,u_i^{(N_c)}\},
$$

where (u_i^{(0)}=0) may be included as a coast candidate. For every candidate,
the watcher rolls out its own trajectory and a nominal target trajectory from
\((\hat\eta_{i,k},P_{\eta_i\eta_i,k})\). For a planning horizon of (H)
steps, define

$$
\mathcal O_i(u)
=
\sum_{h=1}^{H}
\Phi_{i,h}^{\top}
H_{i,h}^{\top}R_i^{-1}H_{i,h}
\Phi_{i,h},
$$

where (H_{i,h}) is the predicted angle-measurement Jacobian and
\(\Phi_{i,h}\) maps the current physical-state perturbation to prediction step
\(h\). A prior-scaled information matrix is

$$
\widetilde{\mathcal O}_i(u)
=
P_{\eta_i\eta_i,k}^{1/2}
\mathcal O_i(u)
P_{\eta_i\eta_i,k}^{1/2}.
$$

Possible observability objectives include

$$
\max_{u\in\mathcal U_i(k)}
\lambda_{\min}
\left(\widetilde{\mathcal O}_i(u)\right),
$$

or

$$
\max_{u\in\mathcal U_i(k)}
\log\det
\left(I+\widetilde{\mathcal O}_i(u)\right).
$$

For angle-only range recovery, the controller may instead minimize the
predicted terminal radial variance. If

$$
e_{\rho,i,H}
=
\frac{\hat q_{i,H}}{\|\hat q_{i,H}\|},
$$

then

$$
\sigma_{\rho,i,H}^2(u)
=
e_{\rho,i,H}^{\top}
P_{rr,i,H}(u)
e_{\rho,i,H}.
$$

A control-effort-aware objective is

$$
\boxed{
J_i(u)
=
\sigma_{\rho,i,H}^2(u)
+
\lambda_u
\sum_{h=0}^{H-1}
\|u_{i,h}\|^2
}
$$

and the selected command is

$$
u_i^*(k)
=
\arg\min_{u\in\mathcal U_i(k)}J_i(u).
$$

The candidate rollout uses the local physical estimate and covariance only.
The first implementation may use a four-state position-velocity rollout rather
than duplicating the complete augmented DNN-EKF for every control candidate.

### Receding-horizon implementation

The selected command is applied for a finite replanning interval and then
recomputed:

$$
u_i(t)=u_i^*(k),
\qquad
t\in[t_k,t_k+T_{\mathrm{replan}}).
$$

This receding-horizon structure allows the watcher to respond to changing LOS
geometry and state uncertainty. A practical initial implementation uses a
small discrete set of thrust directions, a short physical-state rollout, and
a zero-thrust candidate.

### Acquisition, coast, and re-excitation modes

Long-duration operation should not require maximum thrust at every time. The
active-sensing policy is divided into three modes:

1. **Acquisition:** maneuver while radial uncertainty is large or geometry is
   weak.
2. **Observable coast:** coast on a naturally informative or bounded relative
   trajectory while LOS variation remains sufficient.
3. **Re-excitation:** execute a bounded maneuver when predicted information
   falls below a threshold or radial uncertainty grows above a threshold.

Hysteresis can prevent control chatter. For example,

$$
\sigma_{\rho,i}>\sigma_{\mathrm{on}}
\quad\Longrightarrow\quad
\text{maneuver},
$$

$$
\sigma_{\rho,i}<\sigma_{\mathrm{off}}
\quad\Longrightarrow\quad
\text{coast},
\qquad
\sigma_{\mathrm{off}}<\sigma_{\mathrm{on}}.
$$

The same logic may be expressed using the smallest eigenvalue of a finite-window
observability Gramian.

### Relationship to dynamics learning

The DNN parameter block is updated indirectly through

$$
P_{\theta_i\eta_i}H_i^{\top}S_i^{-1}\nu_i.
$$

Weak physical-state observability can cause radial state error, model mismatch,
and branch-parameter error to become difficult to distinguish. The
observability-aware maneuver improves the physical-state learning channel and
thereby makes the innovation-supported branch update more meaningful.

State observability is necessary but not sufficient for identifying a useful
residual approximation. The target trajectory must also visit a sufficiently
rich operating region for the selected DNN features. A later extension may
augment the motion objective with a parameter-information or feature-excitation
term. The first implementation should optimize physical-state/range
observability so that controller and learning failures can be diagnosed
separately.

---

## Geometry Information and the Three Uses of FIM

### Direction-only geometry support

For watcher (j), define the instantaneous direction-only bearing information
support

$$
\Omega_{j,k}=I-\ell_{j,k}\ell_{j,k}^{\top}.
$$

A cumulative support matrix is

$$
\bar\Omega_{j,k}
=
\sum_{\tau\le k:\,\delta_j^m(\tau)=1}
\left(I-\ell_{j,\tau}\ell_{j,\tau}^{\top}\right).
$$

This matrix records the output directions supported by watcher (j)'s bearing
history. It is geometry metadata, not a centralized target-state estimate.

Three distinct uses of geometry/FIM must be separated.

### 1. Sensing FIM

The first use selects watcher motion to improve future angle-only information.
It acts on (u_i) through the predicted measurement Jacobians and physical-state
Gramian. This is the primary role of FIM in the active-sensing layer.

### 2. Learning and communication confidence

The second use evaluates whether a local branch update was produced under
informative geometry. Define, for example,

$$
g_i(k)
=
\frac{
\lambda_{\min}(\bar\Omega_{i,k})
}{
\lambda_{\min}(\bar\Omega_{i,k})+\gamma_\Omega
}.
$$

This confidence may be used as an additional GS validation condition,

$$
\delta_i^c(k)=1
$$

only if

$$
\Delta_i(k)\ge\alpha_\Delta,
\qquad
\mathcal R_i(k)\ge\alpha_R,
\qquad
g_i(k)\ge g_{\min},
\qquad
k-k_i^{\mathrm{last}}\ge N_{\min},
$$

or it may determine a geometry-dependent acceptance covariance margin. A branch
update generated under weak geometry can be quarantined or accepted with larger
uncertainty rather than treated as a high-confidence library update.

### 3. Residual-composition FIM

An experimental residual-composition extension uses

$$
W_j
=
\frac{\bar\Omega_j}{\lambda_{\max}(\bar\Omega_j)},
$$

and

$$
\hat d_{\mathrm{FIM}}^{(m)}
=
\sum_{j\in\mathcal A_m}
W_{j|m}\hat d_j.
$$

The normalization is performed within each watcher/branch information matrix;
there is no normalization across watchers and no requirement that
\(\sum_jW_j=I\).

This mode changes the interpretation of the residual model. The nominal
block-structured formulation assumes complementary additive components,

$$
d_{\mathrm{unk}}\approx\sum_jd_j,
$$

whereas FIM-weighted composition treats the learned branch outputs as
direction-dependent corrections. Therefore:

- the unweighted additive composite remains the main block-structured model;
- FIM-weighted composition is reported as an extension or ablation;
- its mean and covariance paths must use consistent Jacobian transformations;
- empirical tracking improvement does not by itself prove unique branch
  decomposition or unbiased residual reconstruction.

The Young coefficients used in the conservative nonlocal covariance surrogate
remain distinct from all three FIM uses. They are PSD bounding coefficients,
not geometry trust weights.

---

## GS Metadata Addition

Extend the branch record to include geometry and learning-support metadata:

$$
\mathcal B_{i,\mathrm{GS}}
=
\left(
\hat\theta_{i,\mathrm{GS}},
P_{\theta_i\theta_i,\mathrm{GS}},
t_i^{\mathrm{last}},
\mathrm{ver}_i,
c_i,
\mathcal Z_i,
\mathrm{status}_i,
\bar\Omega_i,
n_{\Omega,i},
t_{\Omega,i}^{\mathrm{last}}
\right).
$$

Here, \(n_{\Omega,i}\) is the number of valid geometry updates and
\(t_{\Omega,i}^{\mathrm{last}}\) is the time of the latest valid LOS update.
The upload packet may carry the same geometry metadata. Recipients use it as
branch-associated sensing support; they do not interpret it as a centralized
physical-state estimate.

Geometry records should eventually include validity, age, and frame metadata.
An old \(\bar\Omega_i\) can remain numerically well conditioned while no longer
representing the current operating region, so geometry age must not be confused
with information quality.

---

## Updated Research Framing

We propose an observability-aware, communication-efficient collaborative
block-structured DNN-EKF for cooperative angle-only target tracking and unknown
target-dynamics learning. The unknown residual dynamics are represented by a
shared neural residual approximator partitioned into branch-wise parameter
blocks. Each watcher estimates its own local target physical state and only its
assigned DNN parameter block. Prediction uses the local branch together with
validated ground-station copies of the remaining branches, preserving a small
watcher-local augmented EKF while using the full distributed residual library.

Because angle-only measurements provide weak radial information, each watcher
also runs a bounded active-sensing policy. Using only its local physical-state
estimate, covariance, known watcher state, and admissible thrust set, it selects
coast or maneuver commands that improve predicted finite-horizon observability.
The resulting LOS diversity improves target-state estimation and strengthens
the indirect state-parameter learning channel. Watcher motion is therefore an
enabling sensing layer for collaborative dynamics learning rather than a
separate centralized tracking mechanism.

The ground station remains a validated, versioned parameter-library repository
and does not run a centralized target-state filter. Watchers transmit branch
updates using contribution-change, innovation-improvement, dwell-time, and
maximum-silence conditions. Geometry support may be included as validation
metadata. Accepted branch updates replace stale GS copies rather than being
naively fused with them, avoiding direct double counting of information already
embedded in the local posterior.

Nonlocal branch uncertainty is handled through covariance aging, branch-output
uncertainty projection, and a Young-inequality-based conservative covariance
surrogate. Direction-only FIM information is used primarily for active sensing
and geometry confidence. FIM-weighted residual composition is retained as a
separate extension because it changes the interpretation of complementary
additive branches.

The proposed architecture jointly studies four coupled tradeoffs:

1. angle-only observability versus maneuver effort;
2. tracking accuracy versus residual-model approximation error;
3. collaborative learning benefit versus branch overlap and stale copies;
4. communication reduction versus GS-library mismatch.

---

## Additions to Main Advantages

Add the following items:

8. It explicitly addresses the weak radial observability of angle-only sensing
   through bounded watcher motion.
9. It separates sensing-information design from residual-composition weighting
   and conservative covariance inflation.
10. It supports acquisition, observable coast, and re-excitation instead of
    requiring continuous maximum thrust.
11. It provides motion-cost metrics such as cumulative impulse, \(\Delta v\),
    and information gain per unit control effort.
12. It creates a principled path toward joint tracking and dynamics-learning
    excitation without centralizing target-state estimates.

---

## Additions to Main Caveats

Add the following items:

7. Structural observability does not guarantee accurate finite-noise range
   estimation or DNN-parameter convergence.
8. A controller planned from a biased local state estimate can select a
   suboptimal maneuver direction.
9. Persistent maneuvering can improve geometry while increasing propellant use,
   estimator nonlinearity, and trajectory-safety requirements.
10. A straight-line-equivalent baseline is not necessarily the actual baseline
    when the controller replans and changes direction.
11. Good bearing NIS does not imply small radial position error.
12. FIM-weighted residual composition can bias a genuinely complementary
    additive branch decomposition.
13. Persistent physical-state observability does not automatically provide
    persistent excitation of all DNN features.
14. Target-relative coasting or circumnavigation requires an appropriate
    relative-motion model or bounded trajectory controller; it does not arise
    naturally from the current matched-velocity double-integrator coast model.

---

## Additional Assumptions for Analysis

### Assumption: bounded and known watcher actuation

Each watcher applies a calibrated command satisfying

$$
\|u_i(k)\|\le F_{\max,i},
$$

and its own position, velocity, and applied command are known with bounded
error.

### Assumption: finite-window physical-state information

During each acquisition or re-excitation interval, there exist an integer
\(H_o>0\) and constant \(\alpha_o>0\) such that

$$
\sum_{h=k}^{k+H_o}
\Phi_{i,h,k}^{\top}
H_{i,h}^{\top}R_i^{-1}H_{i,h}
\Phi_{i,h,k}
\succeq
\alpha_o I
$$

on the physical-state subspace of interest, or an equivalent prior-scaled
finite-window information condition holds.

### Assumption: safe compact operation

The target estimate, watcher state, and candidate rollouts remain in compact
operating and safety regions, with nonzero target-watcher range and valid
measurement Jacobians.

### Assumption: bounded planner mismatch

The difference between the nominal candidate rollout and the realized relative
trajectory is bounded over each finite planning horizon.

These assumptions should initially support practical boundedness claims. They
should not be presented as a proof of DNN parameter convergence.

---

## Additions to the Theory Plan

Insert the following items after target-state practical boundedness:

| Priority | Theoretical guarantee | Purpose | Recommendation |
|---:|---|---|---|
| 2 | Finite-window observability under bounded calibrated maneuvers | Connects watcher motion to removal of angle-only range ambiguity | Essential |
| 3 | Closed-loop estimator-motion practical boundedness | Shows bounded planner/model error does not destabilize tracking | Essential once active sensing is part of the main method |
| 4 | Coast/re-excitation hysteresis and no-chattering property | Prevents arbitrarily fast maneuver switching | Strongly recommended |
| 5 | Maneuver-cost bound | Bounds cumulative impulse or \(\Delta v\) over a finite mission horizon | Recommended |
| 6 | Geometry-supported branch-update validity | Relates local branch confidence to finite-window sensing information | Recommended; avoid claiming parameter convergence |

Renumber the original lower-priority theory items after inserting these rows.

---

## Simulation Questions Introduced by the Updated Formulation

The updated simulation should answer the following questions in order:

1. Does the observability-seeking maneuver make the physical angle-only system
   converge when the target dynamics are known?
2. Does an Oracle-residual EKF converge under the same realized watcher motion?
3. How much tracking degradation is caused by residual learning rather than by
   weak geometry?
4. Does GS branch sharing improve Local DNN-EKF tracking under identical motion,
   truth, noise, and initialization?
5. Does FIM-weighted residual composition improve tracking without unacceptable
   residual-magnitude bias?
6. How much information gain is obtained per unit \(\Delta v\) or impulse?
7. Can a zero-thrust/coast candidate preserve accuracy while reducing maneuver
   cost?
8. Does event-triggered branch sharing retain the frequent-upload benefit under
   persistent relative motion?
9. Are accepted branch updates supported by both innovation improvement and
   adequate sensing geometry?
10. Do the conclusions persist over seeds, bearing noise, FOV dropout, thrust
    limits, and target residual families?

---

## Compact Updated Model Statement

The nominal collaborative residual predictor remains

$$
\boxed{
\hat d^{(i)}(\hat\eta_i)
=
\hat d_i(\hat\eta_i;\hat\theta_i^{\mathrm{local}})
+
\sum_{j\neq i}
\hat d_j(\hat\eta_i;\hat\theta_{j,\mathrm{GS}})
}
$$

with local augmented state

$$
\boxed{
X_i=
\begin{bmatrix}
\eta_i\\
\theta_i
\end{bmatrix}.
}
$$

The watcher control is selected locally by

$$
\boxed{
u_i^*(k)
=
\arg\min_{u\in\mathcal U_i(k)}
\left[
\sigma_{\rho,i,H}^2(u)
+
\lambda_u\mathcal C_u(u)
\right],
\qquad
\|u_i\|\le F_{\max,i}.
}
$$

The event-triggered GS upload rule may be augmented with geometry confidence:

$$
\boxed{
\delta_i^c(k)=1
}
$$

if

$$
\Delta_i(k)\ge\alpha_\Delta,
\qquad
\mathcal R_i(k)\ge\alpha_R,
\qquad
g_i(k)\ge g_{\min},
\qquad
k-k_i^{\mathrm{last}}\ge N_{\min},
$$

with a forced upload when the maximum silence interval is reached.

This compact statement preserves the original block-structured learning and GS
sharing architecture while adding an explicit observability-aware sensing layer.
