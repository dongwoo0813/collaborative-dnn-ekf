%{
check_step09F6_Hybrid_Theta_Sensitivity_Covariance_Cross_Block.m

Purpose:
    Targeted structural check for the hybrid GS composite mode:

        cfg.gs.compositeMode = "local_full_plus_gated_nonlocal"

    This check verifies that the local theta_i-to-velocity covariance
    channel remains full/raw in the hybrid mode.

Hybrid residual model:
    For watcher i,

        d_i^GS(eta)
            = d_i(eta; theta_i)
              + sum_{j neq i} B_j(eta) d_j(eta; theta_j).

    The local branch is intentionally not gated.

Expected local theta sensitivity:
    Since the local branch is full,

        F_{v,theta}
            = partial dot{v} / partial theta_i
            = beta_DNN * partial d_i / partial theta_i.

    In other words, the expected sensitivity is raw, not

        beta_DNN * B_i(eta) * partial d_i / partial theta_i.

What this file compares:
    The file compares the actual covariance cross-block after one
    prediction step,

        P_vtheta_actual = P^+_{v,theta},

    against

        P_vtheta_expected = dt * beta_DNN * branchJacobianTheta(i, eta, cfg).

Why this works:
    The initial covariance is artificially set as

        P_{theta,theta} = I,
        P_{v,theta}     = 0,
        all other blocks = 0.

    With q_acc = 0, qTheta = 0, and block covariance prediction disabled,
    the first-order covariance propagation gives

        P^+_{v,theta} = dt * F_{v,theta}.

Expected result:
    The raw/full comparison should be at machine precision level.

    The optional gated comparison should generally be nonzero, confirming
    that hybrid mode did not accidentally use the fully gated local theta
    sensitivity.
%}

addpath(genpath(pwd));
rehash;

cfg = config_step04_GS_DNN_EKF();

cfg.dim = 2;
cfg.Nw  = 4;

cfg.dt = 0.5;

% Use the GS composite path because the hybrid mode is a GS composite mode.
cfg.dnn.predictionResidualSource = "GS_composite";
cfg.dnn.useResidualInPrediction = true;
cfg.dnn.residualInjectionGain = 1.0;

% Remove theta process effects so the cross-block is created only by
% F_{v,theta}.
cfg.dnn.thetaDynamics = "random_walk";
cfg.dnn.qTheta = 0.0;
cfg.dnn.minCovDiag = 0.0;

% Remove physical process noise and use the simple covariance propagation
% path to isolate the theta-to-velocity channel.
cfg.ekf.q_acc = 0.0;
cfg.ekf.useBlockCovPrediction = false;

% Hybrid mode under test.
cfg.gs.compositeMode = "local_full_plus_gated_nonlocal";

% Disable nonlocal covariance injection. This check is only about the local
% theta_i-to-velocity covariance channel.
cfg.gs.useNonlocalBranchCovariance = false;
cfg.gs.nonlocalWeightMode = "none";

% Gate fields are still needed because the hybrid mean/Jeta model gates
% nonlocal branches, even though this particular check uses no nonlocal
% branch records.
cfg.gate.mode = "tight_frame_2d_rt";
cfg.gate.minRange = 1e-12;

eta0 = [3; 4; 0.1; -0.2];

rng(101);
watcher = initLocalDNNEKF(1, eta0, cfg);

watcher.xhat(watcher.idxEta) = eta0;
watcher.xhat(watcher.idxTheta) = zeros(watcher.nTheta, 1);

% -------------------------------------------------------------------------
% Isolate the theta-to-velocity covariance channel.
%
% Initial condition:
%     P_{theta,theta} = I,
%     all other covariance blocks = 0.
%
% Then the one-step covariance prediction should produce
%
%     P^+_{v,theta} = dt * beta_DNN * partial d_i / partial theta_i
%
% in hybrid mode.
% -------------------------------------------------------------------------
watcher.P = zeros(watcher.nX, watcher.nX);
watcher.P(watcher.idxTheta, watcher.idxTheta) = eye(watcher.nTheta);

% No nonlocal branch records are needed for this local theta sensitivity
% check.
watcher.gsBranches = [];

watcherPred = DNN_EKF_Predict_Local(watcher, 0.0, cfg);

idxV = watcher.idxEta(cfg.dim + (1:cfg.dim));

BthetaRawExpected = cfg.dnn.residualInjectionGain * ...
    branchJacobianTheta(watcher.localBranchID, eta0, cfg);

P_vtheta_expected_raw = cfg.dt * BthetaRawExpected;

P_vtheta_actual = watcherPred.P(idxV, watcher.idxTheta);

fprintf("||P_vtheta_actual - dt*BthetaRawExpected||_F = %.3e\n", ...
    norm(P_vtheta_actual - P_vtheta_expected_raw, "fro"));

% Optional contrast:
% This should generally not be zero. It checks that hybrid mode did not
% accidentally use the fully gated local theta sensitivity.
Bgate = branchGateMatrix(watcher.localBranchID, eta0, cfg);

BthetaGatedContrast = cfg.dnn.residualInjectionGain * ...
    (Bgate * branchJacobianTheta(watcher.localBranchID, eta0, cfg));

P_vtheta_expected_gated = cfg.dt * BthetaGatedContrast;

fprintf("contrast ||P_vtheta_actual - dt*BthetaGatedExpected||_F = %.3e\n", ...
    norm(P_vtheta_actual - P_vtheta_expected_gated, "fro"));