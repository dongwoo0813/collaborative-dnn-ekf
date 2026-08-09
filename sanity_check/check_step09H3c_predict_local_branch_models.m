function check_step09H3c_predict_local_branch_models()
%{
File:
    checks/check_step09H3c_predict_local_branch_models.m

Purpose:
    Check that DNN_EKF_Predict_Local works with both branch models:

        1. fixed_feature_lip
        2. mlp_general

What this verifies:
    - initLocalDNNEKF creates the correct augmented state dimension.
    - DNN_EKF_Predict_Local runs one prediction step.
    - xhat and P keep the correct size.
    - P remains finite and symmetric.
    - local_DNN prediction no longer depends on fixed-feature-only helpers
      such as branchOutput or branchJacobianTheta.

This is a structural prediction check, not a tracking performance test.
%}

    addpath(genpath(pwd));
    rehash;

    rng(101);

    cfg0 = config_step04_GS_DNN_EKF();

    dim = cfg0.dim;

    if dim == 2
        etaTrue0 = [
            800.0;
           -150.0;
              0.08;
             -0.04
        ];
    elseif dim == 3
        etaTrue0 = [
            800.0;
           -150.0;
            120.0;
              0.08;
             -0.04;
              0.02
        ];
    else
        error("check_step09H3c:UnsupportedDim", ...
            "This check supports cfg.dim = 2 or 3.");
    end

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.3c check: DNN_EKF_Predict_Local branch models\n");
    fprintf("============================================================\n");

    % ---------------------------------------------------------------------
    % Case 1: fixed_feature_lip
    % ---------------------------------------------------------------------
    cfg = cfg0;
    cfg.dnn.branchModel = "fixed_feature_lip";
    cfg.dnn.predictionResidualSource = "local_DNN";
    cfg.dnn.useResidualInPrediction = true;

    watcher0 = initLocalDNNEKF(1, etaTrue0, cfg);
    watcher1 = DNN_EKF_Predict_Local(watcher0, 0.0, cfg);

    assertPredictionOutput_step09h3c(watcher1, cfg, watcher0.nTheta);

    fprintf("\nCase: fixed_feature_lip\n");
    fprintf("    nTheta        = %d\n", watcher1.nTheta);
    fprintf("    nX            = %d\n", watcher1.nX);
    fprintf("    ||dxhat||     = %.6e\n", norm(watcher1.xhat - watcher0.xhat));
    fprintf("    trace(Ppred)  = %.6e\n", trace(watcher1.P));

    % ---------------------------------------------------------------------
    % Case 2: mlp_general
    % ---------------------------------------------------------------------
    cfg = cfg0;
    cfg.dnn.branchModel = "mlp_general";
    cfg.dnn.predictionResidualSource = "local_DNN";
    cfg.dnn.useResidualInPrediction = true;

    cfg.dnn.mlp.hiddenSizes = [12 8 6];
    cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
    cfg.dnn.mlp.inputMode = "eta_phase";
    cfg.dnn.mlp.rScale = 1000.0;
    cfg.dnn.mlp.vScale = 0.1;

    cfg.dnn.mlp.thetaInitMode = "random_hidden_zero_output";
    cfg.dnn.mlp.hiddenWeightStd = 0.1;
    cfg.dnn.mlp.hiddenBiasStd = 0.0;
    cfg.dnn.mlp.outputWeightStd = 0.0;
    cfg.dnn.mlp.outputBiasStd = 0.0;
    cfg.dnn.mlp.thetaInitSeed = 9100;

    watcher0 = initLocalDNNEKF(1, etaTrue0, cfg);
    watcher1 = DNN_EKF_Predict_Local(watcher0, 0.0, cfg);

    assertPredictionOutput_step09h3c(watcher1, cfg, watcher0.nTheta);

    theta0 = watcher0.xhat(watcher0.idxTheta);
    eta0 = watcher0.xhat(watcher0.idxEta);

    [d0, Jeta0, Jtheta0] = evaluateBranchResidualModel( ...
        watcher0.localBranchID, eta0, theta0, cfg);

    fprintf("\nCase: mlp_general\n");
    fprintf("    nTheta        = %d\n", watcher1.nTheta);
    fprintf("    nX            = %d\n", watcher1.nX);
    fprintf("    ||d0||        = %.6e\n", norm(d0));
    fprintf("    size(Jeta0)   = %s\n", mat2str(size(Jeta0)));
    fprintf("    size(Jtheta0) = %s\n", mat2str(size(Jtheta0)));
    fprintf("    ||dxhat||     = %.6e\n", norm(watcher1.xhat - watcher0.xhat));
    fprintf("    trace(Ppred)  = %.6e\n", trace(watcher1.P));

    fprintf("\nPASS: Step 09-H.3c DNN_EKF_Predict_Local branch-model check passed.\n");

end

function assertPredictionOutput_step09h3c(watcher, cfg, nThetaExpected)

    dim = cfg.dim;
    nEta = 2*dim;
    nXExpected = nEta + nThetaExpected;

    assert(watcher.nEta == nEta, ...
        "watcher.nEta mismatch.");

    assert(watcher.nTheta == nThetaExpected, ...
        "watcher.nTheta mismatch.");

    assert(watcher.nX == nXExpected, ...
        "watcher.nX mismatch.");

    assert(numel(watcher.xhat) == nXExpected, ...
        "xhat length mismatch after prediction.");

    assert(isequal(size(watcher.P), [nXExpected nXExpected]), ...
        "P size mismatch after prediction.");

    assert(all(isfinite(watcher.xhat)), ...
        "Predicted xhat contains non-finite values.");

    assert(all(isfinite(watcher.P(:))), ...
        "Predicted P contains non-finite values.");

    assert(norm(watcher.P - watcher.P.', "fro") < 1e-10, ...
        "Predicted P is not symmetric.");

    eigMin = min(eig(0.5*(watcher.P + watcher.P.')));

    assert(eigMin > -1e-8, ...
        "Predicted P has a significantly negative eigenvalue: %.6e", eigMin);

end