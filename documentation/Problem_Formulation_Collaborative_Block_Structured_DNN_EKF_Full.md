# Problem Formulation

## Observability-Aware Collaborative Block-Structured DNN-EKF with Event-Triggered Ground-Station Branch Sharing

**Document type:** Full standalone Obsidian Markdown source  
**Primary architecture:** Ground-station-assisted distributed branch library  
**Primary sensing mode:** Intermittent angle-only measurements  
**Primary learning task:** Unknown target residual-dynamics approximation  
**Active-sensing role:** Improve physical-state observability and the indirect state-parameter learning channel

![[Local Area Awareness (Ground Station).png]]

---

## Literature Review

The formulation combines four research areas.

First, angles-only navigation shows that bearing measurements can leave range
weakly observable or scale-ambiguous unless natural relative motion or a
calibrated observer maneuver generates sufficient LOS diversity. Observability
must be distinguished from finite-noise conditioning and practical filter
convergence.

Second, adaptive and neural-augmented Kalman filtering treats unknown dynamics
as learned model corrections. When neural parameters are included in an EKF,
their measurement update occurs through state-parameter cross-covariance.
Large monolithic augmented filters can become computationally expensive and
poorly conditioned.

Third, distributed and federated estimation reduces local state dimension and
communication by assigning information ownership across agents. Unknown
cross-correlation and information recycling make naive posterior fusion
unsafe, motivating conservative covariance handling, versioned records, and
covariance intersection in decentralized extensions.

Fourth, event-triggered communication replaces fixed-rate transmission by
state-, innovation-, or function-change conditions. Dwell time limits
communication rate, while maximum silence bounds staleness. For neural models,
function-space change is more meaningful than Euclidean parameter change.

Exact bibliographic citations should be attached to these four themes in the
paper manuscript. The present document defines the architecture and claims; it
does not use uncited literature statements as proof.

---

## 0. Scope and Research Question

The objective is to track an uncooperative target and learn its unknown residual
dynamics using multiple watcher spacecraft equipped with angle-only cameras.
The architecture must satisfy four requirements:

1. no watcher carries the complete DNN parameter vector inside its EKF;
2. each watcher still predicts with the complete distributed residual model;
3. branch communication is reduced through event-triggered sharing;
4. watcher motion can be used to improve weak angle-only radial observability.

The central research question is:

> Can a set of angle-only watcher spacecraft collaboratively learn a shared
> residual-dynamics model using small branch-local DNN-EKFs, while maintaining
> conservative uncertainty, reducing communication, and actively preserving
> enough geometry for range and dynamics learning?

The method does not claim unique identification of individual branch
functions. It seeks a useful composite residual approximation, bounded local
estimation errors, conservative treatment of stale nonlocal branches, and a
measurable communication-accuracy tradeoff.

The ground station is not a centralized target-state filter. It is a
validated, versioned repository for learned branch records.

---

## 1. Target State and Unknown Residual Dynamics

### 1.1 Physical target state

Let the target physical state be

$$
\eta
=
\begin{bmatrix}
r_t^\top & v_t^\top
\end{bmatrix}^\top
\in\mathbb R^{2n_r},
$$

where \(r_t\) and \(v_t\) are the target position and velocity. The target
dynamics are

$$
\dot r_t=v_t,
$$

$$
\dot v_t
=
a_0(\eta,t)
+
d_{\mathrm{unk}}(\eta,t)
+
w_a(t),
$$

where \(a_0\) is the known nominal acceleration, \(d_{\mathrm{unk}}\) is the
unknown residual acceleration, and \(w_a\) is physical process noise.

Equivalently,

$$
\dot\eta
=
f_0(\eta,t)
+
G_d d_{\mathrm{unk}}(\eta,t)
+
G_w w(t),
$$

with

$$
G_d
=
\begin{bmatrix}
0\\
I
\end{bmatrix}.
$$

### 1.2 Distributed residual approximation

Over a compact operating region \(\mathcal D_{\mathrm{op}}\), represent the
unknown residual acceleration as

$$
d_{\mathrm{unk}}(\eta)
=
\sum_{i=1}^{N_s}
d_i(\eta;\theta_i)
+
\epsilon(\eta),
\qquad
\eta\in\mathcal D_{\mathrm{op}},
$$

where \(d_i\) is the contribution of branch \(i\), \(\theta_i\) is its
parameter block, and \(\epsilon\) is the remaining approximation error.
Assume

$$
\sup_{\eta\in\mathcal D_{\mathrm{op}}}
\|\epsilon(\eta)\|
\le
\bar\epsilon.
$$

The nominal learned residual model is

$$
\boxed{
\hat d(\eta)
=
\sum_{i=1}^{N_s}
\hat d_i(\eta;\hat\theta_i)
}.
$$

Each branch is part of one shared approximator. It is not an independent full
residual estimator.

### 1.3 Safe interpretation

The sum of branch outputs may be identifiable even when the individual branch
parameters are not. Therefore:

- the composite residual is the main modeled quantity;
- branch parameters are not assigned unique physical meanings;
- parameter boundedness is a defensible goal;
- branch-wise parameter convergence requires stronger excitation and
  identifiability assumptions and is not claimed by default.

---

## 2. Watcher Dynamics and Angle-Only Measurements

### 2.1 Watcher state

Watcher \(i\) has position \(r_{w,i}\), velocity \(v_{w,i}\), mass \(m_{w,i}\),
and commanded force \(u_i\). For a free-translation diagnostic,

$$
\dot r_{w,i}=v_{w,i},
\qquad
\dot v_{w,i}=\frac{1}{m_{w,i}}u_i,
\qquad
\|u_i(t)\|\le F_{\max,i}.
$$

The watcher state and applied command are locally known, possibly with bounded
navigation and actuation error.

For integration with a nominal watcher trajectory, decompose

$$
r_{w,i}
=
r_{w,i}^{\mathrm{ref}}
+
\delta r_{w,i},
$$

$$
v_{w,i}
=
v_{w,i}^{\mathrm{ref}}
+
\delta v_{w,i},
$$

and conceptually split the force as

$$
u_i
=
u_i^{\mathrm{track}}
+
u_i^{\mathrm{obs}},
\qquad
\|u_i\|\le F_{\max,i}.
$$

The nominal-reference tracking effort and observability-seeking effort must be
reported separately whenever both are modeled.

### 2.2 Relative geometry

Define

$$
q_i=r_t-r_{w,i},
\qquad
\rho_i=\|q_i\|,
\qquad
\ell_i=\frac{q_i}{\rho_i}.
$$

The online controller uses the local estimate

$$
\hat q_i=\hat r_{t,i}-r_{w,i}
$$

and never uses target truth. Truth is reserved for offline simulation
evaluation.

### 2.3 Angle-only measurement

Watcher \(i\) measures

$$
z_{i,k}
=
h_i(\eta_k,r_{w,i,k})
+
v_{i,k},
$$

with measurement covariance

$$
\mathbb E[v_{i,k}v_{i,k}^\top]=R_i.
$$

In two dimensions,

$$
h_i
=
\operatorname{atan2}
\left(
r_{t,y}-r_{w,i,y},
r_{t,x}-r_{w,i,x}
\right).
$$

Let the measurement-availability indicator be

$$
\delta_i^m(k)
=
\begin{cases}
1, & \text{a valid angle measurement is available},\\
0, & \text{otherwise}.
\end{cases}
$$

Availability may depend on field of view, occultation, sensor scheduling,
dropout, or communication-independent sensor validity.

Measurement availability and communication triggering are different events:

- \(\delta_i^m\) controls the local EKF measurement correction;
- \(\delta_i^c\) controls branch upload to the ground station.

---

## 3. Block-Structured Residual Model

### 3.1 Feature-block form

A simple branch model is

$$
\hat d_i(\eta;\theta_i)
=
W_i\phi_i(\eta),
\qquad
\theta_i=\operatorname{vec}(W_i).
$$

The full residual approximator is

$$
\boxed{
\hat d(\eta)
=
\sum_{i=1}^{N_s}
W_i\phi_i(\eta)
}.
$$

The feature block \(\phi_i\) is part of the shared residual model, not a
watcher measurement. Different branches should have different function-space
roles, for example through:

- different feature masks;
- different local basis functions;
- different random-feature realizations;
- different frequency bands;
- different operating-region centers;
- different subnetworks of a larger MLP.

### 3.2 General branch model

More generally,

$$
\hat d_i(\eta;\theta_i)
=
\mathcal N_i(\eta;\theta_i),
$$

where \(\mathcal N_i\) is a differentiable MLP branch. The local EKF requires

$$
\frac{\partial \hat d_i}{\partial \eta},
\qquad
\frac{\partial \hat d_i}{\partial \theta_i}.
$$

Analytic Jacobians are preferred and must be checked against finite
differences.

### 3.3 Watcher-local composite prediction

Watcher \(i\) owns its local branch \(\theta_i\). For other branches, it uses
the latest accepted ground-station copies:

$$
\boxed{
\hat d^{(i)}(\hat\eta_i)
=
\hat d_i(\hat\eta_i;\hat\theta_i^{\mathrm{local}})
+
\sum_{j\ne i}
\hat d_j(\hat\eta_i;\hat\theta_{j,\mathrm{GS}})
}.
$$

Thus, watcher \(i\) learns in the context of the current composite library.
Its local learning signal is approximately residualized against the other
branch contributions:

$$
d_{\mathrm{unk}}(\eta)
-
\left[
\hat d_i(\eta;\hat\theta_i^{\mathrm{local}})
+
\sum_{j\ne i}
\hat d_j(\eta;\hat\theta_{j,\mathrm{GS}})
\right].
$$

This discourages, but does not eliminate, duplicated branch learning.

---

## 4. Watcher-Local DNN-EKF

### 4.1 Local augmented state

Watcher \(i\) estimates the physical target state and only its assigned branch:

$$
\boxed{
X_i
=
\begin{bmatrix}
\eta_i\\
\theta_i
\end{bmatrix}
}.
$$

Its covariance is

$$
P_i
=
\begin{bmatrix}
P_{\eta_i\eta_i} & P_{\eta_i\theta_i}\\
P_{\theta_i\eta_i} & P_{\theta_i\theta_i}
\end{bmatrix}.
$$

The cross-covariance \(P_{\eta_i\theta_i}\) is essential. The parameter block
does not appear directly in the angle measurement; it is updated through its
correlation with the physical state.

### 4.2 Local prediction

For a discrete-time propagation,

$$
\hat X_{i,k+1}^-
=
F_i^d
\left(
\hat X_{i,k}^+,
\{\hat\theta_{j,\mathrm{GS}}\}_{j\ne i}
\right),
$$

where the physical prediction uses the composite residual
\(\hat d^{(i)}\). A branch parameter model may be random walk,

$$
\theta_{i,k+1}
=
\theta_{i,k}
+
w_{\theta_i,k},
$$

or first-order Gauss-Markov,

$$
\dot\theta_i
=
-\Lambda_{\theta_i}^{-1}\theta_i
+
w_{\theta_i}.
$$

Let \(\Phi_{i,k}\) be the local augmented transition Jacobian. The covariance
prediction is

$$
\boxed{
P_{i,k+1}^-
=
\Phi_{i,k}P_{i,k}^+\Phi_{i,k}^\top
+
Q_{i,k}^{\mathrm{base}}
+
Q_{k,-i}^{X}
}.
$$

The term \(Q_{k,-i}^{X}\) represents uncertainty from nonlocal branch copies
that affect the mean prediction but are not included in \(X_i\).

### 4.3 Measurement correction

The innovation and innovation covariance are

$$
\nu_{i,k}
=
z_{i,k}
-
h_i(\hat\eta_{i,k}^-),
$$

$$
S_{i,k}
=
H_{i,k}P_{\eta_i\eta_i,k}^-H_{i,k}^\top
+
R_i.
$$

The normalized innovation squared is

$$
\epsilon_i(k)
=
\nu_{i,k}^\top S_{i,k}^{-1}\nu_{i,k}.
$$

When \(\delta_i^m(k)=1\),

$$
\hat\eta_{i,k}^+
=
\hat\eta_{i,k}^-
+
P_{\eta_i\eta_i,k}^-
H_{i,k}^\top
S_{i,k}^{-1}
\nu_{i,k},
$$

and

$$
\boxed{
\hat\theta_{i,k}^+
=
\hat\theta_{i,k}^-
+
P_{\theta_i\eta_i,k}^-
H_{i,k}^\top
S_{i,k}^{-1}
\nu_{i,k}
}.
$$

When \(\delta_i^m(k)=0\), the watcher performs prediction only.

The covariance update should use a numerically stable Joseph or equivalent
PSD-preserving form. Covariance symmetry and eigenvalue floors must be
monitored.

### 4.4 Adaptive process uncertainty

Adaptive \(Q_{\theta_i}\) and residual approximation uncertainty
\(Q_{\epsilon,i}\) may be driven by innovation covariance matching. These
adaptations are implementation mechanisms, not observability proofs. Their
gains, bounds, update cadence, and use in compared cases must be identical in
paired experiments unless they are the explicit ablation variable.

---

## 5. Observability-Aware Watcher Motion

### 5.1 Angle-only scale ambiguity

Angle measurements strongly constrain cross-LOS position but may weakly
constrain range. Under simplified relative dynamics, scaled initial relative
position and velocity can generate the same LOS history.

Let the natural relative trajectory be

$$
q_{i,\mathrm{nat}}(t)
=
\Phi_{rr}(t,t_0)q_i(t_0)
+
\Phi_{rv}(t,t_0)\dot q_i(t_0),
$$

and the maneuver-induced relative displacement be

$$
\Delta q_{i,u}(t)
=
-
\int_{t_0}^{t}
\Phi_{rv}(t,\tau)
\frac{u_i(\tau)}{m_{w,i}}
\,d\tau.
$$

A maneuver that remains everywhere parallel to the natural LOS may preserve
the range-scale ambiguity. A useful scale-breaking condition is that, over a
nonzero part of the observation window,

$$
\Delta q_{i,u}(t)
\ne
\alpha_i(t)q_{i,\mathrm{nat}}(t)
$$

for every scalar function \(\alpha_i(t)\). Under a specific linear model, the
formal condition is the required rank or positive-definiteness condition of
the finite-window observability Gramian. The non-collinearity condition is a
geometric diagnostic, not a universal theorem for all nonlinear target models.

Structural observability does not guarantee accurate estimation under finite
bearing noise, model mismatch, DNN approximation error, or poor conditioning.
The controller should therefore optimize observability strength.

### 5.2 Scenario distinction

Two watcher-motion scenarios must remain separate:

1. **Matched-velocity free-translation diagnostic:** used to isolate whether
   active LOS excitation can remove range ambiguity.
2. **Nominal-reference mission architecture:** the watcher follows a bounded
   relative trajectory, such as the prescribed circular reference, and active
   sensing generates a bounded deviation about it.

The present diagnostic controller must not be interpreted as already
maintaining the original circular reference. The architecture-level controller
should use

$$
r_{w,i}
=
r_{w,i}^{\mathrm{ref}}
+
\delta r_{w,i}
$$

with bounds on \(\delta r_{w,i}\), \(\delta v_{w,i}\), total force, field of
view, and safety.

### 5.3 Finite-horizon physical-state information

At time \(k\), watcher \(i\) considers admissible control candidates

$$
\mathcal U_i(k)
=
\{u_i^{(0)},u_i^{(1)},\ldots,u_i^{(N_c)}\},
$$

where \(u_i^{(0)}=0\) is the coast candidate.

For discrete predicted measurements,

$$
\boxed{
\mathcal O_i(u)
=
\sum_{h=1}^{H}
\Phi_{i,h,k}^\top
H_{i,h}^\top
R_i^{-1}
H_{i,h}
\Phi_{i,h,k}
}.
$$

The measurement cadence used by this sum must be explicit. A numerical
planning time step must not be multiplied into or removed from the information
sum without defining whether the model is discrete measurement information or
a continuous information-rate integral.

A prior-scaled information matrix is

$$
\widetilde{\mathcal O}_i(u)
=
P_{\eta_i\eta_i,k}^{1/2}
\mathcal O_i(u)
P_{\eta_i\eta_i,k}^{1/2}.
$$

Candidate objectives include

$$
\max_{u\in\mathcal U_i(k)}
\lambda_{\min}
\left(
\widetilde{\mathcal O}_i(u)
\right),
$$

or

$$
\max_{u\in\mathcal U_i(k)}
\log\det
\left(
I+\widetilde{\mathcal O}_i(u)
\right).
$$

For range recovery, define the predicted terminal radial direction

$$
e_{\rho,i,H}
=
\frac{\hat q_{i,H}}{\|\hat q_{i,H}\|},
$$

and radial variance

$$
\sigma_{\rho,i,H}^2(u)
=
e_{\rho,i,H}^\top
P_{rr,i,H}(u)
e_{\rho,i,H}.
$$

An effort- and deviation-aware objective is

$$
\boxed{
J_i(u)
=
\sigma_{\rho,i,H}^2(u)
+
\lambda_u\mathcal C_u(u)
+
\lambda_r\mathcal C_{\mathrm{dev}}(u)
}.
$$

Then

$$
\boxed{
u_i^\star(k)
=
\arg\min_{u\in\mathcal U_i(k)}
J_i(u)
},
\qquad
\|u_i^\star(k)\|\le F_{\max,i}.
$$

The first implementation should roll out only the physical target covariance.
A full augmented DNN-EKF rollout for every candidate is deferred.

### 5.4 Receding-horizon operation

Apply the selected command over a finite replanning interval:

$$
u_i(t)=u_i^\star(k),
\qquad
t\in[t_k,t_k+T_{\mathrm{replan}}).
$$

Long-duration operation should have three modes:

1. **Acquisition:** maneuver while radial uncertainty is large or geometry is
   weak.
2. **Observable coast:** track the nominal reference or coast while adequate
   information persists.
3. **Re-excitation:** maneuver again when predicted information drops or
   radial uncertainty grows.

Use hysteresis, for example

$$
\sigma_{\rho,i}>\sigma_{\mathrm{on}}
\Longrightarrow
\text{maneuver},
$$

$$
\sigma_{\rho,i}<\sigma_{\mathrm{off}}
\Longrightarrow
\text{coast},
\qquad
\sigma_{\mathrm{off}}<\sigma_{\mathrm{on}}.
$$

### 5.5 Relationship to dynamics learning

The branch update occurs through

$$
P_{\theta_i\eta_i}
H_i^\top
S_i^{-1}
\nu_i.
$$

Weak physical-state observability makes radial state error, model mismatch,
and parameter error difficult to distinguish. Active sensing improves the
physical-state channel and can make innovation-supported branch updates more
meaningful.

Physical-state observability remains insufficient for branch identification.
The target must also visit a feature-rich operating region. A later extension
may add parameter-information or feature-excitation terms, but only after the
physical observability layer is validated independently.

---

## 6. Geometry Information and Its Three Roles

### 6.1 LOS geometry-support matrix

For watcher \(j\), define

$$
\Omega_{j,k}
=
I-\ell_{j,k}\ell_{j,k}^\top.
$$

With valid measurements,

$$
\bar\Omega_{j,k}
=
\sum_{\tau\le k:\,\delta_j^m(\tau)=1}
\left(
I-\ell_{j,\tau}\ell_{j,\tau}^\top
\right).
$$

This matrix records supported output directions. It is a
**direction-only LOS geometry-information surrogate**. It omits range scaling,
measurement covariance, state-transition dynamics, and DNN parameter
sensitivity. It must not be called the complete likelihood Fisher information
matrix without those elements.

Online implementations should construct the LOS direction from available
measurements or local estimates, not from target truth.

All \(\bar\Omega_j\) matrices and branch acceleration outputs must be expressed
in the same coordinate frame before they are accumulated, compared, or used
as residual-composition matrices.

### 6.2 Role 1: active-sensing information

The physical-state information matrix in Section 5 uses predicted
measurement Jacobians, \(R_i^{-1}\), and state transitions. Its purpose is to
choose future watcher motion.

### 6.3 Role 2: learning and communication confidence

A geometry-confidence diagnostic can be defined as

$$
g_i(k)
=
\frac{
\lambda_{\min}(\bar\Omega_{i,k})
}{
\lambda_{\min}(\bar\Omega_{i,k})
+
\gamma_\Omega
}.
$$

Geometry confidence may accompany an uploaded branch record and influence a
covariance margin or quarantine decision. It should initially be diagnostic
rather than a hard rejection gate. Weak geometry lowers confidence but does
not prove that an update is wrong.

### 6.4 Role 3: experimental residual-composition weighting

The experimental geometry-weighted additive mode uses

$$
\boxed{
W_j
=
\frac{\bar\Omega_j}
{\lambda_{\max}(\bar\Omega_j)}
}
$$

and

$$
\boxed{
\hat d_{\mathrm{geom}}^{(m)}
=
\sum_{j\in\mathcal A_m}
W_{j|m}\hat d_j
}.
$$

Normalization occurs within each watcher or branch matrix. There is no
normalization across watchers and no requirement that

$$
\sum_jW_j=I.
$$

This mode changes the nominal complementary-additive interpretation. It is
therefore an extension or ablation, not the default block-structured model.
The mean Jacobian and covariance transformation paths must apply the same
weights consistently.

The earlier ensemble gate

$$
B_{j|m}
=
\left(
\sum_{\ell\in\mathcal A_m}\bar\Omega_\ell+\varepsilon I
\right)^{-1}
\bar\Omega_j
$$

has the approximate partition property

$$
\sum_jB_{j|m}\approx I.
$$

It is conceptually different from per-branch
\(\lambda_{\max}\)-normalization.

The Young coefficients introduced later are covariance-bound coefficients.
They are not FIM weights, geometry trust weights, or branch-mean gains.

---

## 7. Event-Triggered Branch Sharing

### 7.1 Communication event

Define

$$
\delta_i^c(k)
=
\begin{cases}
1, & \text{watcher }i\text{ uploads its branch},\\
0, & \text{otherwise}.
\end{cases}
$$

The communication event is evaluated separately from
\(\delta_i^m(k)\).

### 7.2 Function-space contribution change

Let

$$
\mathcal Z_{i,k}
=
\{x_{i,k}^{(1)},\ldots,x_{i,k}^{(M)}\}
$$

be a representative set of states near the local operating region. Define

$$
\boxed{
\Delta_i(k)
=
\frac{1}{M}
\sum_{\ell=1}^{M}
\left\|
\hat d_i
\left(
x_{i,k}^{(\ell)};
\hat\theta_i^{\mathrm{local}}
\right)
-
\hat d_i
\left(
x_{i,k}^{(\ell)};
\hat\theta_{i,\mathrm{GS}}
\right)
\right\|^2
}.
$$

This metric is preferred to a parameter-vector norm because neural parameters
are not unique and parameter distance does not directly measure change in the
represented acceleration.

### 7.3 Innovation support

Define a measurement-weighted NIS average over a window of length \(L\):

$$
\bar\epsilon_i(k)
=
\frac{
\sum_{\ell=k-L+1}^{k}
\delta_i^m(\ell)\epsilon_i(\ell)
}{
\sum_{\ell=k-L+1}^{k}
\delta_i^m(\ell)
+
\varepsilon
}.
$$

A stronger diagnostic compares a local branch against the current GS copy:

$$
\boxed{
\mathcal R_i(k)
=
\frac{1}{n_i^m(k)}
\sum_{\ell=k-L+1}^{k}
\delta_i^m(\ell)
\left[
\epsilon_i^{\mathrm{GS}}(\ell)
-
\epsilon_i^{\mathrm{local}}(\ell)
\right]
},
$$

where

$$
n_i^m(k)
=
\sum_{\ell=k-L+1}^{k}\delta_i^m(\ell)
$$

is the number of valid measurements in the window. This comparison should be
implemented with a shadow predictor or stored validation sequence so that the
two NIS values are computed on comparable data.

### 7.4 Trigger with dwell and maximum silence

A nominal trigger is

$$
\delta_i^c(k)=1
$$

if

$$
\Delta_i(k)\ge\alpha_\Delta,
\qquad
\mathcal R_i(k)\ge\alpha_R,
\qquad
k-k_i^{\mathrm{last}}\ge N_{\min}.
$$

Geometry support may be included as metadata or, after validation, as

$$
g_i(k)\ge g_{\min}.
$$

A forced heartbeat upload occurs when

$$
k-k_i^{\mathrm{last}}\ge N_{\max},
\qquad
N_{\max}\ge N_{\min}.
$$

The maximum-silence rule prevents arbitrarily stale repository records. In
discrete time, the positive dwell \(N_{\min}\) also prevents chattering and
provides a direct upload-rate bound.

---

## 8. Ground-Station Repository and Protocol

![[Pasted image 20260623102813.png]]

### 8.1 Information ownership

Watcher \(i\) owns

$$
\hat\eta_i,
\quad
P_{\eta_i\eta_i},
\quad
\hat\theta_i^{\mathrm{local}},
\quad
P_{\theta_i\theta_i},
\quad
P_{\eta_i\theta_i},
$$

as well as its local innovation history.

The GS stores a versioned branch record

$$
\boxed{
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
t_{\Omega,i}^{\mathrm{last}},
\mathcal F_i
\right)
}.
$$

Here:

- \(c_i\) is confidence or validation metadata;
- \(\mathcal Z_i\) describes the operating region;
- \(\mathrm{status}_i\) is active, stale, quarantined, or rejected;
- \(n_{\Omega,i}\) is the number of geometry samples;
- \(t_{\Omega,i}^{\mathrm{last}}\) is the latest geometry timestamp;
- \(\mathcal F_i\) identifies the coordinate frame used for geometry metadata.

An old \(\bar\Omega_i\) may remain well conditioned while no longer
representing the current region. Geometry age, operating region, and reference
frame must therefore accompany the matrix.

### 8.2 Watcher-to-GS packet

When triggered, watcher \(i\) sends

$$
\boxed{
\mathcal P_i(k)
=
\left(
\hat\theta_i^{\mathrm{local}},
P_{\theta_i\theta_i}^{\mathrm{local}},
t_k,
\mathrm{ver}_{\mathrm{used}},
\mathcal Z_i,
\Delta_i,
\mathcal R_i,
n_i^m,
\bar\Omega_i,
n_{\Omega,i},
t_{\Omega,i}^{\mathrm{last}},
\mathcal F_i
\right)
}.
$$

The field \(\mathrm{ver}_{\mathrm{used}}\) records the nonlocal library
versions used to construct the local posterior. This metadata helps reveal
information recycling and hidden dependence.

Raw innovation sequences are not required in the baseline protocol. They may
be included only for diagnostic studies.

### 8.3 GS validation

The GS evaluates the branch in function space. Baseline checks include

$$
\Delta_i\ge\alpha_\Delta,
\qquad
\mathcal R_i\ge\alpha_R,
\qquad
k-k_i^{\mathrm{last}}\ge N_{\min}.
$$

Additional checks may include:

$$
P_{\theta_i\theta_i}^{\mathrm{local}}
\succeq
P_{\min,i},
$$

$$
\mathcal N_i\ge\alpha_N,
$$

$$
\rho_{ij}\le\rho_{\max},
$$

and valid geometry age, frame, and operating-region overlap.

The GS may:

1. accept and broadcast;
2. accept with a conservative covariance margin;
3. quarantine without broadcast;
4. reject and retain the previous record.

### 8.4 Replacement, not naive fusion

If accepted,

$$
\hat\theta_{i,\mathrm{GS}}^+
=
\hat\theta_i^{\mathrm{local}},
$$

$$
P_{\theta_i\theta_i,\mathrm{GS}}^+
=
P_{\theta_i\theta_i}^{\mathrm{local}}
+
P_{\mathrm{acc},i},
\qquad
P_{\mathrm{acc},i}\succeq0,
$$

$$
t_i^{\mathrm{last},+}=t_k,
\qquad
\mathrm{ver}_i^+=\mathrm{ver}_i+1.
$$

The GS does not fuse the stale stored branch posterior with the fresh local
posterior as if they were independent. The stale GS copy was already used in
the watcher's prior or composite prediction. Naive information fusion would
double count prior information and create overconfident covariance.

### 8.5 Broadcast and covariance aging

The GS broadcasts nonlocal records to watcher \(m\):

$$
\mathcal G_m(k)
=
\left\{
\hat\theta_{j,\mathrm{GS}},
\widetilde P_{\theta_j\theta_j,\mathrm{GS}},
t_j^{\mathrm{last}},
\mathrm{ver}_j,
c_j,
\mathcal Z_j,
\mathrm{status}_j,
\bar\Omega_j,
t_{\Omega,j}^{\mathrm{last}},
\mathcal F_j
\right\}_{j\ne m}.
$$

The aged covariance satisfies

$$
\dot{\widetilde P}_{\theta_j\theta_j,\mathrm{GS}}
=
F_{\theta_j}
\widetilde P_{\theta_j\theta_j,\mathrm{GS}}
+
\widetilde P_{\theta_j\theta_j,\mathrm{GS}}
F_{\theta_j}^\top
+
Q_{\theta_j}^{\mathrm{age}}
+
S_{\mathrm{stale},j}.
$$

Equivalently,

$$
\begin{aligned}
\widetilde P_{\theta_j\theta_j,\mathrm{GS}}(t)
=&
\Phi_{\theta_j}(t,t_j^{\mathrm{last}})
P_{\theta_j\theta_j,\mathrm{GS}}(t_j^{\mathrm{last}})
\Phi_{\theta_j}^\top(t,t_j^{\mathrm{last}})
\\
&+
\int_{t_j^{\mathrm{last}}}^{t}
\Phi_{\theta_j}(t,\tau)
\left(
Q_{\theta_j}^{\mathrm{age}}(\tau)
+
S_{\mathrm{stale},j}(\tau)
\right)
\Phi_{\theta_j}^\top(t,\tau)
\,d\tau.
\end{aligned}
$$

A first implementation may use

$$
\widetilde P_{\theta_j\theta_j,\mathrm{GS}}(t)
\approx
P_{\theta_j\theta_j,\mathrm{GS}}(t_j^{\mathrm{last}})
+
S_{\mathrm{age},j}
\left(
t-t_j^{\mathrm{last}}
\right).
$$

Watcher \(m\) uses the received nonlocal branch in mean prediction but does not
append it to \(X_m\) and does not update it with local measurements.

### 8.6 Responsibility summary

| Entity | Owns or stores | Updates | Communicates |
|---|---|---|---|
| Watcher \(i\) | \(\hat\eta_i,\hat\theta_i,P_i\) | Local augmented EKF | Candidate branch packet |
| Watcher \(i\) | Innovation/NIS window | \(\nu_i,S_i,\epsilon_i,\mathcal R_i\) | Scalar validation metadata |
| Watcher \(i\) | Nonlocal GS cache | Ages locally if needed; does not estimate nonlocal parameters | Requests/acknowledgments |
| Ground station | Versioned branch records | Validates, replaces, ages covariance | Validated branch library |
| Ground station | Status, confidence, geometry metadata | Accepts, quarantines, or rejects | Decision and broadcast |
| Ground station | No target-state posterior | Does not run target-state fusion | No centralized target estimate |

---

## 9. Nonlocal Branch Uncertainty

### 9.1 Parameter-to-output projection

For watcher \(m\), define

$$
B_{j|m}(t)
=
\left.
\frac{\partial\hat d_j(\eta;\theta_j)}
{\partial\theta_j}
\right|_{
\eta=\hat\eta_m(t),
\theta_j=\hat\theta_{j,\mathrm{GS}}(t)
}.
$$

The uncertainty of branch \(j\) induces residual-acceleration covariance

$$
S_{j|m}(t)
=
B_{j|m}(t)
\widetilde P_{\theta_j\theta_j,\mathrm{GS}}(t)
B_{j|m}^\top(t).
$$

If the branch output is acceleration,

$$
[S_{j|m}]
=
\mathrm{m^2\,s^{-4}}.
$$

### 9.2 Missing nonlocal cross-covariances

Let

$$
y_j=B_{j|m}\delta\theta_j.
$$

For each pair \(i<j\), Young's inequality gives

$$
y_i y_j^\top
+
y_j y_i^\top
\preceq
\mu_{ij|m}y_i y_i^\top
+
\mu_{ij|m}^{-1}y_j y_j^\top,
\qquad
\mu_{ij|m}>0.
$$

Define

$$
a_{j|m}
=
1
+
\sum_{\ell:j<\ell}\mu_{j\ell|m}
+
\sum_{\ell:\ell<j}\mu_{\ell j|m}^{-1}.
$$

Then a conservative nonlocal output-covariance surrogate is

$$
\boxed{
\bar S_{d,-m}
=
\sum_{j\ne m}
a_{j|m}
B_{j|m}
\widetilde P_{\theta_j\theta_j,\mathrm{GS}}
B_{j|m}^\top
+
S_{\mathrm{res}}
}.
$$

The term \(S_{\mathrm{res}}\succeq0\) covers approximation error, unrepresented
local-nonlocal dependence, and remaining residual-model uncertainty.

With \(\mu_{ij|m}=1\),

$$
a_{j|m}=N_{-m},
\qquad
N_{-m}=|\{j:j\ne m\}|.
$$

A trace-balanced choice is

$$
\mu_{ij|m}
=
\sqrt{
\frac{\operatorname{tr}(S_{j|m})+\varepsilon}
{\operatorname{tr}(S_{i|m})+\varepsilon}
}.
$$

Any positive choice preserves the pairwise PSD bound; the trace-balanced
choice may reduce unnecessary conservatism.

### 9.3 Discrete-time injection

Let

$$
G_d^m
=
\begin{bmatrix}
0\\
I\\
0
\end{bmatrix}
$$

inject residual acceleration into the physical velocity dynamics but not
directly into the local parameter state. Define

$$
M_{k,m}
=
\int_{t_k}^{t_{k+1}}
\Phi_{X,m}(t_{k+1},\tau)
G_d^m(\tau)
\,d\tau.
$$

Then

$$
\boxed{
Q_{k,-m}^{X}
=
M_{k,m}
\bar S_{d,-m,k}
M_{k,m}^\top
}.
$$

For a constant-acceleration double-integrator approximation,

$$
M_{k,m}
\approx
\begin{bmatrix}
\frac12\Delta t_k^2\beta_{\mathrm{DNN}}I\\
\Delta t_k\beta_{\mathrm{DNN}}I\\
0
\end{bmatrix}.
$$

This is the preferred discrete implementation because
\(\bar S_{d,-m,k}\) is interpreted as a one-step acceleration-bias covariance
and time integration is explicit.

### 9.4 Continuous-time alternative and units

An acceleration covariance with units
\(\mathrm{m^2\,s^{-4}}\) cannot be inserted directly into a continuous-time
Riccati equation as white-noise intensity. Introduce an effective correlation
time \(T_{d,j}\):

$$
Q_{j|m}^c
=
T_{d,j}S_{j|m},
$$

so

$$
[Q_{j|m}^c]
=
\mathrm{m^2\,s^{-3}}.
$$

A dimensionally consistent continuous intensity is

$$
Q_{d,-m}^{c}
=
\beta_{\mathrm{DNN}}^2
\left[
\sum_{j\ne m}
a_{j|m}T_{d,j}S_{j|m}
+
Q_{\mathrm{res}}^c
\right].
$$

The discrete bias-covariance interpretation is the nominal implementation.
The continuous form is an alternative model, not an additional term to be
applied simultaneously.

### 9.5 Interpretation

This construction is a conservative surrogate. It does not reconstruct the
exact marginal covariance of a centralized augmented DNN-EKF and does not
recover all local-nonlocal or inter-branch cross-covariances.

The local branch \(\theta_m\) already appears in \(X_m\), so its covariance is
not reinjected through \(Q_{k,-m}^{X}\). Only nonlocal branches are included.

---

## 10. Branch Overlap and Novelty

The decomposition

$$
d_{\mathrm{unk}}
\approx
\sum_i d_i
$$

is generally nonunique. Composite prediction discourages redundant learning
but cannot guarantee nonoverlap.

For a representative state set, define

$$
D_i(k)
=
\begin{bmatrix}
\hat d_i(x_k^{(1)}) &
\cdots &
\hat d_i(x_k^{(M)})
\end{bmatrix}.
$$

The functional overlap is

$$
\rho_{ij}(k)
=
\frac{
\langle D_i(k),D_j(k)\rangle_F
}{
\|D_i(k)\|_F
\|D_j(k)\|_F
+
\varepsilon
}.
$$

Let \(\Pi_{-i}\) project onto the span of the other branch-output matrices.
Define novelty

$$
\mathcal N_i(k)
=
\frac{
\|(I-\Pi_{-i})D_i(k)\|_F
}{
\|D_i(k)\|_F+\varepsilon
}.
$$

If \(\rho_{ij}\approx1\), two branches are functionally similar on the
representative region. If \(\mathcal N_i\approx0\), the candidate branch is
mostly redundant. These metrics support quarantine, covariance-margin, or
diagnostic decisions. They do not establish globally unique branch
identification.

---

## 11. Peer-to-Peer Extension

The GS-assisted architecture is the primary method. A peer-to-peer alternative
replaces the global repository with asynchronous local caches.

![[Pasted image 20260623103341.png]]

### 11.1 Communication graph and cache

Let \(\mathcal N_i\) be watcher \(i\)'s neighbor set. Its cache is

$$
\mathcal C_i(k)
=
\left\{
\hat\theta_{j\to i},
P_{\theta_j\theta_j\to i},
t_{j\to i}^{\mathrm{stamp}},
\mathrm{ver}_{j\to i},
h_{j\to i}
\right\}_{j\ne i},
$$

where \(h_{j\to i}\) is hop count.

The aged peer covariance may be

$$
\widetilde P_{\theta_j\theta_j}^{(i)}(t)
=
P_{\theta_j\theta_j\to i}
+
S_{\mathrm{age},j}
\left(
t-t_{j\to i}^{\mathrm{stamp}}
\right)
+
h_{j\to i}S_{\mathrm{hop},j}.
$$

### 11.2 Peer communication trigger

Without a GS copy, watcher \(i\) compares its local branch with the last branch
version it transmitted:

$$
\Delta_i^{\mathrm{tx}}(k)
=
\frac{1}{M}
\sum_{\ell=1}^{M}
\left\|
\hat d_i
\left(
x_k^{(\ell)};
\hat\theta_i^{\mathrm{local}}(k)
\right)
-
\hat d_i
\left(
x_k^{(\ell)};
\hat\theta_i^{\mathrm{last,tx}}
\right)
\right\|^2.
$$

An optional peer innovation score is

$$
\mathcal R_i^{\mathrm{tx}}(k)
=
\frac{1}{n_i^m(k)}
\sum_{\ell=k-L+1}^{k}
\delta_i^m(\ell)
\left[
\epsilon_i^{\mathrm{last,tx}}(\ell)
-
\epsilon_i^{\mathrm{local}}(\ell)
\right].
$$

Watcher \(i\) transmits to its neighbors if

$$
\Delta_i^{\mathrm{tx}}\ge\alpha_\Delta,
\qquad
\mathcal R_i^{\mathrm{tx}}\ge\alpha_R,
\qquad
k-k_i^{\mathrm{last,tx}}\ge N_{\min},
$$

with an optional maximum-silence heartbeat.

### 11.3 Version replacement

A newer branch version replaces an older cached version. Equal versions are
duplicates; older versions are rejected. Excessively stale or high-hop records
may be rejected or retained with covariance inflation.

### 11.4 Peer composite prediction

Watcher \(m\) uses

$$
\hat d_{\mathrm{P2P}}^{(m)}(\hat\eta_m)
=
\hat d_m(\hat\eta_m;\hat\theta_m)
+
\sum_{j\in\mathcal C_m,\,j\ne m}
\hat d_j(\hat\eta_m;\hat\theta_{j\to m}).
$$

Different watchers may use different library versions at the same time.

### 11.5 Parameter sharing is not parameter fusion

Branch \(j\) and branch \(i\) represent different blocks. Watcher \(i\) does
not fuse \(\theta_j\) into \(\theta_i\). A received branch is cached and used
in composite prediction only.

### 11.6 Optional physical-state sharing

If physical target posteriors are exchanged and their cross-correlations are
unknown, naive averaging is not allowed. Covariance intersection may be used:

$$
Y_{ij}^{\mathrm{CI}}
=
\omega P_{\eta_i\eta_i}^{-1}
+
(1-\omega)P_{\eta_j\eta_j}^{-1},
$$

$$
y_{ij}^{\mathrm{CI}}
=
\omega P_{\eta_i\eta_i}^{-1}\hat\eta_i
+
(1-\omega)P_{\eta_j\eta_j}^{-1}\hat\eta_j,
$$

$$
P_{\eta_i\eta_i}^{\mathrm{CI}}
=
\left(
Y_{ij}^{\mathrm{CI}}
\right)^{-1},
\qquad
\hat\eta_i^{\mathrm{CI}}
=
P_{\eta_i\eta_i}^{\mathrm{CI}}y_{ij}^{\mathrm{CI}}.
$$

Physical-state fusion is optional and should be disabled in the first P2P
branch-sharing implementation.

### 11.7 P2P nonlocal covariance

Use the same output-projection and Young-bound construction with cached peer
covariances:

$$
S_{j|m}^{\mathrm{peer}}
=
B_{j|m}
\widetilde P_{\theta_j\theta_j}^{(m)}
B_{j|m}^\top,
$$

$$
\bar S_{d,-m}^{\mathrm{peer}}
=
\sum_j a_{j|m}S_{j|m}^{\mathrm{peer}}
+
S_{\mathrm{res}}^{\mathrm{peer}},
$$

$$
Q_{k,-m}^{X,\mathrm{peer}}
=
M_{k,m}
\bar S_{d,-m,k}^{\mathrm{peer}}
M_{k,m}^\top.
$$

P2P consistency is an extension and should not obscure validation of the
primary GS architecture.

---

## 12. Assumptions for Analysis

The following assumptions define a defensible practical-boundedness analysis.
They should be weakened only when a replacement proof is supplied.

### Assumption A1: compact operation

The target state estimate, watcher states, branch parameters, and candidate
rollouts remain in compact sets:

$$
\hat\eta_i(k)\in\mathcal D_\eta,
\qquad
\hat\theta_i(k)\in\mathcal D_{\theta_i},
\qquad
r_{w,i}(k)\in\mathcal D_{w,i}.
$$

Target-watcher range remains nonzero and all required Jacobians are bounded.

### Assumption A2: bounded approximation and noise

The residual approximation error is bounded on
\(\mathcal D_{\mathrm{op}}\), and process and measurement noises have bounded
second moments.

### Assumption A3: bounded watcher actuation

Each watcher applies a calibrated command satisfying

$$
\|u_i(k)\|\le F_{\max,i},
$$

and its own motion and applied command are known with bounded error.

### Assumption A4: finite-window physical-state information

During each acquisition or re-excitation interval, there exist
\(H_o>0\) and \(\alpha_o>0\) such that, on the physical-state subspace of
interest,

$$
\sum_{h=k}^{k+H_o}
\Phi_{i,h,k}^\top
H_{i,h}^\top
R_i^{-1}
H_{i,h}
\Phi_{i,h,k}
\succeq
\alpha_o I,
$$

or an equivalent prior-scaled condition holds.

For bearing-only problems with gauge freedoms or intentionally unobservable
subspaces, the condition must be stated on the observable subspace.

### Assumption A5: bounded planner mismatch

The difference between nominal candidate rollouts and realized relative motion
is bounded over each finite planning horizon.

### Assumption A6: bounded branch-output drift

For each branch,

$$
e_j(\eta,k)
=
\hat d_j(\eta;\hat\theta_j^{\mathrm{local}}(k))
-
\hat d_j(\eta;\hat\theta_{j,\mathrm{GS}}(k))
$$

satisfies

$$
\|e_j(\eta,k+1)-e_j(\eta,k)\|
\le
\rho_j\Delta t
$$

for \(\eta\in\mathcal D_\eta\).

### Assumption A7: bounded communication delay and silence

Communication delay is bounded, trigger checks occur at least every
\(N_{\mathrm{chk}}\) steps, and each branch has

$$
N_{\min}\le N_{\max}<\infty.
$$

### Assumption A8: frequent-GS practical ISS

The frequent-update GS-DNN-EKF error is mean-square input-to-state practically
stable with respect to an additive residual perturbation \(w_{d,m}\):

$$
\mathbb E\|e_m(k)\|^2
\le
c_0\lambda^k\mathbb E\|e_m(0)\|^2
+
c_w
+
c_d
\sup_{0\le\ell\le k}
\mathbb E\|w_{d,m}(\ell)\|^2,
$$

where \(c_0>0\), \(0<\lambda<1\), \(c_w>0\), and \(c_d>0\).

This is a strong baseline assumption. Any event-triggered boundedness theorem
that invokes it is conditional and does not itself prove stability of the
underlying frequent-GS nonlinear DNN-EKF.

---

## 13. Conditional Event-Triggered Sharing Results

### Proposition 1: communication upper bound

Over a horizon of \(K\) steps, if watcher \(i\) uploads at most once every
\(N_{\min}\) steps after bootstrap, then

$$
N_i^{\mathrm{up}}(K)
\le
1+
\left\lfloor
\frac{K}{N_{\min}}
\right\rfloor.
$$

For \(N_s\) watchers,

$$
\boxed{
N_{\mathrm{tot}}^{\mathrm{up}}(K)
\le
N_s
+
N_s
\left\lfloor
\frac{K}{N_{\min}}
\right\rfloor
}.
$$

This is a worst-case rate bound, not an expected communication count.

### Lemma 1: bounded local-to-GS branch mismatch

Under Assumptions A1, A6, and A7, branch-output mismatch is bounded by

$$
\|e_j(\eta,k)\|
\le
\bar e_j.
$$

A conservative maximum-silence bound is

$$
\bar e_j
=
\rho_jN_{\max}\Delta t.
$$

When trigger checks occur at most \(N_{\mathrm{chk}}\) steps apart, an
event-dependent bound is

$$
\boxed{
\bar e_j
=
\min
\left\{
\rho_jN_{\max}\Delta t,
\max
\left[
\rho_jN_{\min}\Delta t,
\sqrt{\alpha_\Delta}
+
\rho_jN_{\mathrm{chk}}\Delta t
\right]
\right\}
}.
$$

The dwell term covers mismatch growth while uploading is prohibited. The
threshold/check term covers the post-dwell interval. The maximum-silence term
provides a global cap.

### Lemma 2: bounded composite perturbation

Let

$$
\hat d_m^{\mathrm{freq}}
=
\hat d_m^{\mathrm{local}}
+
\sum_{j\ne m}
\hat d_{j,\mathrm{GS}}^{\mathrm{freq}},
$$

and

$$
\hat d_m^{\mathrm{ET}}
=
\hat d_m^{\mathrm{local}}
+
\sum_{j\ne m}
\hat d_{j,\mathrm{GS}}^{\mathrm{ET}}.
$$

Then

$$
e_{d,m}
=
\hat d_m^{\mathrm{freq}}
-
\hat d_m^{\mathrm{ET}}
$$

satisfies

$$
\boxed{
\|e_{d,m}(k)\|
\le
\sum_{j\ne m}\bar e_j
},
$$

and

$$
\|e_{d,m}(k)\|^2
\le
(N_s-1)
\sum_{j\ne m}\bar e_j^2.
$$

### Conditional Theorem 1: practical boundedness

Under Assumptions A1-A8 and the bounds above,

$$
\mathbb E\|e_m^{\mathrm{ET}}(k)\|^2
\le
c_0\lambda^k
\mathbb E\|e_m^{\mathrm{ET}}(0)\|^2
+
c_w
+
c_d
\left(
\sum_{j\ne m}\bar e_j
\right)^2.
$$

Therefore,

$$
\boxed{
\limsup_{k\to\infty}
\mathbb E\|e_m^{\mathrm{ET}}(k)\|^2
\le
c_w
+
c_d
\left(
\sum_{j\ne m}\bar e_j
\right)^2
}.
$$

This theorem formalizes the additional error floor caused by stale nonlocal
branches, conditional on practical ISS of the frequent-update estimator.

### Corollary 1: communication-accuracy tradeoff

Larger \(N_{\min}\) reduces the worst-case upload rate but can enlarge
post-upload mismatch during the dwell interval. Smaller
\(\alpha_\Delta\) makes the trigger more sensitive and can reduce stale-model
error while increasing communication. Finite \(N_{\max}\) caps worst-case
staleness.

### No-chattering property

Because the implementation is discrete time and an upload is prohibited for
at least \(N_{\min}>0\) steps after an event, infinitely many uploads cannot
occur in a finite number of simulation steps. This is a discrete dwell-time
property, not a continuous-time no-Zeno theorem.

---

## 14. Observability and Closed-Loop Theory Plan

The following results remain a proof roadmap unless separately established.

| Priority | Theoretical result | Purpose | Status |
|---:|---|---|---|
| 1 | Frequent-GS target-state mean-square practical boundedness | Supplies the baseline needed by Conditional Theorem 1 | Essential, open |
| 2 | Finite-window observability under bounded calibrated maneuvers | Connects watcher motion to angle-only scale recovery | Essential, open |
| 3 | Closed-loop estimator-motion practical boundedness | Handles planner and rollout mismatch | Essential once active sensing is primary |
| 4 | Local parameter and covariance boundedness | Prevents DNN-EKF divergence without claiming parameter convergence | Essential, open |
| 5 | Conservative nonlocal covariance surrogate | Justifies PSD upper bounding of missing nonlocal correlations | Young-bound structure available; full filter consistency open |
| 6 | Stale covariance aging consistency | Prevents old branch copies from retaining unrealistic confidence | Strongly recommended |
| 7 | GS replacement avoids direct double counting | Justifies replacement rather than naive posterior fusion | Recommended |
| 8 | Coast/re-excitation no-chattering and maneuver-cost bound | Supports persistent operation | Recommended |
| 9 | Innovation-supported acceptance property | Connects accepted updates to validation improvement | Recommended |
| 10 | Geometry-supported update validity | Relates sensing support to branch confidence | Diagnostic first |
| 11 | Branch novelty/overlap guarantee | Limits purely redundant accepted updates on \(\mathcal Z_i\) | Recommended |
| 12 | Chance-constrained FOV and safety | Uses conservative covariance in control constraints | Later extension |
| 13 | P2P cache and CI consistency | Supports decentralized extension | Deferred |

The analysis should not infer DNN parameter convergence from physical-state
observability alone. Parameter convergence requires feature excitation and an
appropriate identifiability condition for the selected branch model.

---

## 15. Experimental Separation and Validation Logic

### 15.1 Three independent experiment axes

Primary ablations should change one axis at a time:

1. **Motion/geometry:** prescribed circular reference, matched-velocity coast,
   prescribed transverse excitation, free active sensing, or bounded active
   deviation about a nominal reference.
2. **Estimator/model:** nominal physical EKF, Oracle residual, Local DNN-EKF,
   GS additive, or GS geometry-weighted additive.
3. **Communication:** disabled, frequent upload, event-triggered, or
   heartbeat-only.

### 15.2 Frozen-motion estimator comparison

To isolate estimator effects:

1. generate and save one watcher trajectory;
2. replay the same position, velocity, acceleration, FOV state, truth, and
   noise draws for every estimator;
3. disable controller calls during replay;
4. verify sample-by-sample measurement identity.

This experiment supports estimator-specific claims.

### 15.3 Closed-loop co-design comparison

When every estimator controls watcher motion from its own estimate and
covariance, realized trajectories may differ. Such results measure total
closed-loop system performance and cannot attribute differences solely to
residual composition.

Both experiment families are useful but must be reported separately.

### 15.4 Required observability sequence

The simulations should answer, in order:

1. Does the motion make the known-dynamics angle-only filter recover range?
2. Does the Oracle-residual filter recover range under identical motion?
3. How much degradation is caused by learned residual approximation?
4. Does GS sharing improve Local under identical motion and measurements?
5. Does geometry weighting improve additive under identical motion?
6. Does the original circular watcher reference already provide sufficient
   information?
7. What information gain is obtained per unit impulse and \(\Delta v\)?
8. Can zero-thrust and hysteresis preserve accuracy while reducing effort?
9. Does event-triggered sharing retain the frequent-upload benefit?
10. Are accepted branches supported by innovation, geometry, novelty, and
    operating-region relevance?

### 15.5 Required metrics

Report:

- position, exact range, and velocity RMSE;
- convergence ratios and final-window log-error slopes;
- bearing innovation and NIS;
- residual-vector RMSE;
- residual cosine and norm ratio;
- watcher displacement, path length, reference deviation, impulse, and
  \(\Delta v\);
- LOS angular change and LOS angular rate;
- finite-window information eigenvalues and condition number;
- upload count, communication reduction, branch age, and maximum silence;
- geometry confidence, overlap, novelty, and later nonlocal usefulness;
- covariance consistency diagnostics such as NEES when meaningful.

Good NIS alone does not imply small radial error. A convergence ratio below one
does not establish asymptotic convergence. Positive final-window log slope
indicates late error growth even when the final RMSE improved.

### 15.6 Planner calibration

The assumed measurement cadence and numerical rollout grid must be separated.
Refining the planning integration grid must not create artificial information.
Candidate rankings should be checked against nonlinear covariance or short
Monte Carlo rollouts.

---

## 16. Main Advantages

1. The complete DNN parameter vector is not placed inside one augmented EKF.
2. Each watcher estimates only its assigned DNN block.
3. Every watcher still predicts with the distributed composite residual model.
4. Composite prediction residualizes local learning against current nonlocal
   branch copies.
5. Function-space event triggers are more meaningful than parameter norms.
6. Innovation support reduces uploads that changed without improving
   measurement prediction.
7. GS replacement avoids direct fusion with a stale posterior already embedded
   in the local information history.
8. Covariance aging prevents old nonlocal branches from remaining
   unrealistically precise.
9. Nonlocal output uncertainty is conservatively projected into the local
   physical covariance.
10. Bounded watcher motion explicitly addresses weak angle-only radial
    information.
11. Acquisition, coast, and re-excitation avoid assuming continuous maximum
    thrust.
12. Sensing information, geometry metadata, residual weighting, and Young
    covariance bounds are conceptually separated.
13. The architecture supports future FOV, collision, and chance-constrained
    control without centralizing target-state estimates.

---

## 17. Main Caveats

1. Individual branch parameters and functions are generally not uniquely
   identifiable.
2. Composite residual accuracy does not imply correct branch decomposition.
3. Similar feature blocks can still produce redundant learning.
4. Ground-station copies may be stale and statistically dependent on local
   posteriors.
5. The conservative covariance surrogate is not the exact covariance of a
   centralized augmented DNN-EKF.
6. Young bounds may be conservative.
7. Angle-only measurements are weak in the radial direction.
8. Structural observability does not guarantee useful finite-noise accuracy.
9. A biased local estimate can select a suboptimal active-sensing maneuver.
10. Maneuvering consumes propellant and may violate FOV or safety constraints.
11. Straight-line-equivalent displacement is not realized net displacement
    when thrust direction is replanned.
12. Good bearing NIS does not imply accurate range.
13. Physical-state observability does not imply persistent excitation of every
    DNN feature.
14. Geometry-weighted residual composition can bias a truly complementary
    additive decomposition.
15. \(\bar\Omega_i\) is a direction-only surrogate, not a complete Fisher
    information matrix.
16. The event-triggered practical-boundedness result is conditional on a strong
    frequent-GS ISS assumption.
17. Target-relative coasting or circumnavigation requires a bounded
    relative-motion model; it does not arise automatically from a free
    double-integrator coast.

---

## 18. Compact Architecture Statement

The local state is

$$
\boxed{
X_i
=
\begin{bmatrix}
\eta_i\\
\theta_i
\end{bmatrix}
}.
$$

The nominal composite residual is

$$
\boxed{
\hat d^{(i)}(\hat\eta_i)
=
\hat d_i(\hat\eta_i;\hat\theta_i^{\mathrm{local}})
+
\sum_{j\ne i}
\hat d_j(\hat\eta_i;\hat\theta_{j,\mathrm{GS}})
}.
$$

The local active-sensing decision is

$$
\boxed{
u_i^\star(k)
=
\arg\min_{u\in\mathcal U_i(k)}
\left[
\sigma_{\rho,i,H}^2(u)
+
\lambda_u\mathcal C_u(u)
+
\lambda_r\mathcal C_{\mathrm{dev}}(u)
\right]
},
$$

subject to force, FOV, safety, and reference-deviation constraints.

The nominal event-trigger is

$$
\boxed{
\delta_i^c(k)=1
}
$$

when

$$
\Delta_i(k)\ge\alpha_\Delta,
\qquad
\mathcal R_i(k)\ge\alpha_R,
\qquad
k-k_i^{\mathrm{last}}\ge N_{\min},
$$

with a forced upload at \(N_{\max}\). Geometry confidence is metadata first and
may later enter validation after empirical calibration.

The nonlocal covariance injection is

$$
\boxed{
Q_{k,-i}^{X}
=
M_{k,i}
\left[
\sum_{j\ne i}
a_{j|i}
B_{j|i}
\widetilde P_{\theta_j\theta_j,\mathrm{GS}}
B_{j|i}^\top
+
S_{\mathrm{res}}
\right]
M_{k,i}^\top
}.
$$

---

## 19. Recommended Research Framing

We propose an observability-aware, communication-efficient collaborative
block-structured DNN-EKF for cooperative angle-only target tracking and
unknown target-dynamics learning. The unknown residual dynamics are represented
by a shared neural approximator partitioned into branch-wise parameter blocks.
Each watcher estimates its own target physical state and only its assigned
branch parameter block. Prediction uses the local branch together with
validated ground-station copies of the remaining branches, preserving a small
watcher-local augmented EKF while using the full distributed residual library.

Because angle-only measurements provide weak radial information, each watcher
also executes a bounded active-sensing policy. Using only its local target
estimate, covariance, known watcher state, and admissible command set, it
selects coast or maneuver actions that improve predicted finite-horizon
physical-state information. Watcher motion is therefore an enabling sensing
layer for collaborative dynamics learning, not a centralized target-tracking
mechanism.

The ground station remains a validated, versioned parameter repository and
does not estimate the target state. Watchers upload branch candidates according
to function-space contribution change, innovation improvement, dwell time, and
maximum silence. Accepted candidates replace stale repository records rather
than being naively fused with them. Geometry, operating-region, novelty, and
confidence metadata support validation.

Nonlocal branch uncertainty is handled through covariance aging,
parameter-to-output uncertainty projection, and a Young-inequality-based
conservative covariance surrogate. Direction-only LOS geometry is used as
metadata and as an experimental residual-composition extension, while
finite-horizon physical-state information drives active sensing. These uses
are deliberately separated.

The resulting framework studies four coupled tradeoffs:

1. angle-only information versus maneuver effort;
2. tracking accuracy versus residual approximation error;
3. collaborative learning benefit versus stale or redundant branches;
4. communication reduction versus repository mismatch.

---

## 20. Conservative Paper Claims

Appropriate claims are:

- the architecture distributes a shared residual model across branch-local
  DNN-EKFs;
- validated GS copies allow each watcher to use the composite model without
  estimating every branch;
- event triggering provides an explicit worst-case communication-rate bound;
- under bounded branch drift and baseline ISS, stale-copy degradation is
  practically bounded;
- active sensing can improve finite-horizon angle-only physical-state
  information;
- conservative covariance injection reduces overconfidence from nonlocal
  branch uncertainty.

Claims that require additional proof or should be avoided are:

- unique identification of branch parameters;
- exact equivalence to a centralized augmented DNN-EKF;
- unconditional stability of the nonlinear DNN-EKF;
- parameter convergence from physical-state observability alone;
- optimality of a finite candidate-direction planner;
- interpreting \(\bar\Omega_i\) as the full likelihood FIM;
- attributing closed-loop additive-versus-weighted performance solely to the
  estimator when watcher trajectories differ.

---

## 21. Implementation-Facing Requirements

The formulation requires the simulator to support:

1. local physical and parameter-state histories;
2. full covariance block histories;
3. GS record versions, timestamps, status, and aged covariance;
4. contribution-change and innovation-improvement histories;
5. geometry metadata with timestamp, sample count, frame, and operating region;
6. nonlocal covariance components before and after Young inflation;
7. watcher command, trajectory, reference, deviation, impulse, and
   \(\Delta v\) histories;
8. candidate planner scores and predicted radial variance;
9. deterministic watcher-trajectory export and replay;
10. explicit separation of matched-coast diagnostics and nominal-reference
    mission scenarios;
11. identical truth/noise replay for estimator-only comparisons;
12. separate frozen-motion and closed-loop result reports.

Any change to state dimension, Jacobian, covariance injection, branch
composition, controller logic, or communication packet requires a targeted
dimension and consistency check.

---

## 22. One-Sentence Summary

The proposed method is a distributed angle-only target tracker and residual
dynamics learner in which each watcher estimates one DNN branch, predicts with
a validated shared branch library, actively preserves informative geometry,
communicates only useful branch changes, and conservatively accounts for stale
nonlocal uncertainty without claiming exact centralized equivalence or unique
branch identification.
