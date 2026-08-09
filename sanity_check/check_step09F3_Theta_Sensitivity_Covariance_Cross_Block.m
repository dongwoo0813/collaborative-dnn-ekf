%{
check_step09F3_Theta_Sensitivity_Covariance_Cross_Block.m

Purpose:
    Targeted structural check for Step 09-F.3.

    This check verifies that, when

        cfg.gs.compositeMode = "gated_additive",

    the local DNN parameter sensitivity used inside covariance prediction is
    consistent with the gated residual model.

Background:
    In gated_additive mode, the local branch residual entering the velocity
    dynamics is

        d_i^gate(eta, theta_i) = B_i(eta) d_i(eta, theta_i).

    Therefore, the velocity dynamics sensitivity with respect to the local
    DNN parameter theta_i should be

        F_{v,theta}
            = partial dot{v} / partial theta_i
            = beta_DNN * B_i(eta) * partial d_i / partial theta_i.

    Since B_i(eta) depends on eta but not on theta_i, there is no
    dB_i/dtheta_i chain-rule term.

What this file compares:
    The file compares the actual covariance cross-block produced by
    DNN_EKF_Predict_Local,

        P_vtheta_actual = P^+_{v,theta},

    against the expected first-order covariance cross-block,

        P_vtheta_expected
            = dt * beta_DNN * B_i(eta) * branchJacobianTheta(i, eta, cfg).

Why this works:
    The initial covariance is chosen artificially as

        P_{theta,theta} = I,
        P_{v,theta}     = 0,
        all other blocks = 0.

    With q_acc = 0, qTheta = 0, and block covariance prediction disabled,
    the only first-order mechanism that creates P_{v,theta} is the
    theta-to-velocity sensitivity F_{v,theta}. Therefore,

        P^+_{v,theta} = dt * F_{v,theta}

    for this isolated check.

Expected result:
    The printed Frobenius norm

        ||P_vtheta_actual - dt*BthetaExpected||_F

    should be at machine precision level.

Interpretation:
    Passing this check does not show that tracking performance improved.
    It only confirms that the EKF covariance / learning channel sees the
    same gated local residual model used by the mean prediction.
%}

addpath(genpath(pwd));
rehash;

cfg = config_step04_GS_DNN_EKF();

cfg.dim = 2;
cfg.Nw  = 4;

cfg.dt = 0.5;

% Use the GS composite residual path, because gated_additive is implemented
% in the GS composite residual model.
cfg.dnn.predictionResidualSource = "GS_composite";
cfg.dnn.useResidualInPrediction = true;
cfg.dnn.residualInjectionGain = 1.0;

% Use random-walk theta dynamics with zero theta process noise so that the
% covariance cross-block is created only by F_{v,theta}, not by extra theta
% dynamics or process noise.
cfg.dnn.thetaDynamics = "random_walk";
cfg.dnn.qTheta = 0.0;
cfg.dnn.minCovDiag = 0.0;

% Remove physical process noise and disable block covariance prediction so
% this check isolates the standard one-step covariance propagation path.
cfg.ekf.q_acc = 0.0;
cfg.ekf.useBlockCovPrediction = false;

% Activate the gated additive mode under test.
cfg.gs.compositeMode = "gated_additive";

% Disable nonlocal covariance injection. This check is only about the local
% theta_i-to-velocity covariance channel.
cfg.gs.useNonlocalBranchCovariance = false;
cfg.gs.nonlocalWeightMode = "none";

% Gate configuration used by Step 09-F.
cfg.gate.mode = "tight_frame_2d_rt";
cfg.gate.minRange = 1e-12;

% Fixed eta away from the singular r = 0 case.
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
% Then, after one prediction step,
%
%     P^+_{v,theta} = dt * F_{v,theta}
%
% if the covariance propagation uses the intended local theta sensitivity.
% -------------------------------------------------------------------------
watcher.P = zeros(watcher.nX, watcher.nX);
watcher.P(watcher.idxTheta, watcher.idxTheta) = eye(watcher.nTheta);

% No nonlocal branches are needed because this check only inspects the
% local branch theta sensitivity. Nonlocal branch uncertainty is handled by
% a separate Qnonlocal injection path.
watcher.gsBranches = [];

watcherPred = DNN_EKF_Predict_Local(watcher, 0.0, cfg);

idxV = watcher.idxEta(cfg.dim + (1:cfg.dim));

% Expected gated theta sensitivity:
%
%     F_{v,theta}
%       = beta_DNN * B_i(eta) * partial d_i / partial theta_i.
%
% Here branchJacobianTheta returns partial d_i / partial theta_i for the raw
% local branch output. The gate B_i must be applied on the left.
Bgate = branchGateMatrix(watcher.localBranchID, eta0, cfg);

BthetaExpected = cfg.dnn.residualInjectionGain * ...
    (Bgate * branchJacobianTheta(watcher.localBranchID, eta0, cfg));

P_vtheta_expected = cfg.dt * BthetaExpected;

% Actual covariance cross-block created by DNN_EKF_Predict_Local.
P_vtheta_actual = watcherPred.P(idxV, watcher.idxTheta);

fprintf("||P_vtheta_actual - dt*BthetaExpected||_F = %.3e\n", ...
    norm(P_vtheta_actual - P_vtheta_expected, "fro"));