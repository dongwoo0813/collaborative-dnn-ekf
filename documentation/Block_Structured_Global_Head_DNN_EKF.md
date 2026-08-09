# Block-Structured Global-Head DNN-EKF Toy

## Purpose

This runner is a separate experiment and does not replace the existing additive-DNN/WLS runner.
It implements one global DNN whose hidden representation is partitioned across watchers:

\[
h_i^{(1)}=\tanh(W_i^{(1)}\bar\eta+b_i^{(1)}),
\qquad
h_i^{(\ell)}=\tanh(W_i^{(\ell)}h_i^{(\ell-1)}+b_i^{(\ell)}),
\]

\[
\hat d(\eta;\Theta)
=
\begin{bmatrix}W^{\mathrm{out}}_1&\cdots&W^{\mathrm{out}}_{N_w}\end{bmatrix}
\begin{bmatrix}h_1\\ \vdots\\ h_{N_w}\end{bmatrix}
=
\sum_{i=1}^{N_w}W^{\mathrm{out}}_i h_i(\eta).
\]

Watcher \(i\) owns exactly one coordinate block: all hidden-layer weights and biases of
\(h_i\), together with its final output-layer weight \(W^{\mathrm{out}}_i\). The final output is linear and has no
duplicated per-watcher output bias. Therefore the branches are feature blocks of one wider
network, rather than independent full-residual predictors whose outputs happen to be summed.

## Local filtering and global model use

Every watcher maintains its own angle-only augmented EKF state

\[
z_i=\begin{bmatrix}\eta_i\\ \theta_i\end{bmatrix},
\qquad \eta_i=\begin{bmatrix}r_i\\v_i\end{bmatrix},
\]

where \(\theta_i\) contains only the parameters owned by watcher \(i\). At prediction time it
evaluates the complete cached global model—its local block plus the latest received remote
blocks—at its own state estimate. Thus

\[
J_\eta
=
\frac{\partial\hat d}{\partial\eta}
=
\sum_{j=1}^{N_w} W^{\mathrm{out}}_j\frac{\partial h_j}{\partial\eta},
\]

while the local parameter Jacobian is

\[
J_{\theta_i}
=
\frac{\partial(W^{\mathrm{out}}_i h_i)}{\partial\theta_i}
=
\begin{bmatrix}
\dfrac{\partial(W^{\mathrm{out}}_i h_i)}{\partial\theta_{i,\mathrm{hidden}}}
& h_i^\mathsf{T}\otimes I_2
\end{bmatrix}.
\]

For continuous-time target dynamics

\[
\dot r=v,\qquad
\dot v=a_{\mathrm{nom}}(r,v)+\hat d(\eta;\Theta),
\]

the augmented dynamics Jacobian used by the discretized toy is formed from

\[
F_i=
\begin{bmatrix}
A_0+LJ_\eta & LJ_{\theta_i}\\
0&I
\end{bmatrix},
\qquad
L=\begin{bmatrix}0_{2\times2}\\I_2\end{bmatrix}.
\]

For bearing \(y_i=\operatorname{atan2}(r_y-r_{w,i,y},r_x-r_{w,i,x})\),

\[
H_i=
\begin{bmatrix}
-\Delta y/\rho^2 & \Delta x/\rho^2 & 0 & 0 & 0_{\theta_i}
\end{bmatrix}.
\]

Although the measurement has no direct parameter derivative, propagation through
\(J_{\theta_i}\) creates state–parameter cross covariance, so bearing innovations can update
the local DNN block.

## Event-triggered parameter sharing

At a communication event, watcher \(i\) uploads its complete owned parameter posterior and
diagonal covariance to the ground-station cache. The cache broadcasts that block to all
watchers. Between events, remote blocks are held at their last received values. The event score
is a normalized diagonal Mahalanobis change with minimum-interval and maximum-silence guards.

## Cases in the runner

- `Nominal EKF`: no learned residual.
- `Local branch only`: each watcher uses and adapts only its own block.
- `Shared block-structured DNN`: each watcher uses the complete cached global model, adapts
  only its owned block, and shares that whole block according to the selected communication
  mode.

All hidden layers and output-head slices are initialized from small random values.  No
current-mission truth state or acceleration sample is used in initialization.  Online updates
therefore use angle measurements only; this makes the runner an online-learning experiment
rather than a supervised warm-start experiment.

## Run

```matlab
addpath(genpath(pwd))

out = run_toy_block_structured_global_head_dnn_ekf( ...
    101, 600, true, 4, 0.1, 3, "event_triggered");

out.summary
out.architecture
```

The arguments are seed, duration, plotting flag, number of watchers, time step, number of
hidden layers per watcher block, and communication mode. Supported communication modes are
`"event_triggered"`, `"instantaneous"`, and `"never"`.

## Interpretation

Adding watchers widens the global feature vector and therefore increases representational
capacity. It does not mathematically guarantee monotonic improvement in closed-loop EKF
accuracy: angle-only excitation, conditioning, parameter identifiability, finite data, and
event staleness still matter. Watcher-count scaling must therefore be evaluated over paired
random seeds rather than inferred from a single trajectory.
