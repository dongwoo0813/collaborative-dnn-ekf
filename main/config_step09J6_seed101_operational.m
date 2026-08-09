function [cfgBase, seed, meta] = config_step09J6_seed101_operational()
%CONFIG_STEP09J6_SEED101_OPERATIONAL Shared configuration for 09-J.6 runs.
% Keeping the configuration in one function ensures the operational run and
% the parameter-mobility ablation use identical truth, measurement, DNN,
% GS, and discretization settings.

    seed = 101;
    thetaStd = 7.5e-5;
    thetaInitStd = thetaStd;

    cfgBase = config_step04_GS_DNN_EKF();
    cfgBase.T = 200.0;
    cfgBase.dt = 0.1;
    cfgBase.time = 0:cfgBase.dt:cfgBase.T;
    cfgBase.N = numel(cfgBase.time);

    cfgBase.truth.useResidual = true;
    cfgBase.truth.residualFamily = "feedback_sat_disturbance";
    cfgBase.truth.residualModel = "feedback_sat_disturbance";
    cfgBase.truth.residualAmp = 5e-4;

    cfgBase.meas.availabilityMode = "always";
    if isfield(cfgBase, "fov")
        cfgBase.fov.enabled = false;
        cfgBase.fov.guardUnimplementedMode = true;
    end

    cfgBase.ekf.useBlockCovPrediction = true;
    cfgBase.ekf.transitionDiscretization = "second_order_block";

    cfgBase.dnn.branchModel = "mlp_general";
    cfgBase.dnn.mlp.hiddenSizes = [6 6];
    cfgBase.dnn.mlp.activations = ["softplus", "tanh"];
    % The operational benchmark models unknown target feedback dynamics as
    % a state-feedback law d_unk = d(r,v).  Do not inject branch identity
    % into this function approximator: each watcher retains an independent
    % parameter vector and learns from its own measurement history, while
    % the represented residual function has the same physical inputs for
    % every branch.
    cfgBase.dnn.mlp.inputMode = "eta_only";
    cfgBase.dnn.mlp.rScale = 1000.0;
    cfgBase.dnn.mlp.vScale = 0.1;
    cfgBase.dnn.mlp.thetaInitMode = "small_random";
    cfgBase.dnn.mlp.theta0Std = thetaInitStd;
    cfgBase.dnn.theta0_std = thetaInitStd;
    cfgBase.dnn.residualInjectionGain = 1.0;
    cfgBase.dnn.Ptheta0 = thetaInitStd^2;
    cfgBase.dnn.thetaSigmaSS = thetaStd;
    cfgBase.dnn.predictionResidualSource = "GS_composite";

    cfgBase.estimator.type = "GS_DNN_EKF";
    cfgBase.gs.enabled = true;
    cfgBase.gs.bootstrapUpload = true;
    cfgBase.gs.nonlocalWeightMode = "none";
    cfgBase.gs.nonlocalWeight = 1.0;
    cfgBase.gs.uploadMode = "after_measurement_update";
    cfgBase.gs.broadcastMode = "every_step";
    cfgBase.gs.useNonlocalBranchCovariance = true;
    cfgBase.gs.youngMode = "uniform";
    cfgBase.gs.fimGate.enabled = true;
    cfgBase.gs.fimGate.lambdaOmega = 0.02;
    cfgBase.gs.fimGate.epsilon = 1e-6;
    cfgBase.gs.fimGate.normalizeTrace = false;
    cfgBase.gs.fimGate.outputFrame = "inertial";
    cfgBase.gate.mode = "tight_frame_2d_rt";
    cfgBase.gate.minRange = 1e-12;

    meta = struct;
    meta.thetaStd = thetaStd;
    meta.thetaInitStd = thetaInitStd;
end
