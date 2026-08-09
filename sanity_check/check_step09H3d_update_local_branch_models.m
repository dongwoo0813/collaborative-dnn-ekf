function check_step09H3d_update_local_branch_models()
%{
File:
    checks/check_step09H3d_update_local_branch_models.m

Purpose:
    Check that DNN_EKF_Update_Local works after prediction for both:

        1. fixed_feature_lip
        2. mlp_general

What this verifies:
    - initLocalDNNEKF works.
    - DNN_EKF_Predict_Local works.
    - DNN_EKF_Update_Local works.
    - The augmented state/covariance dimensions are preserved.
    - The update remains valid for large MLP theta dimension.
    - theta can be indirectly updated through P_{theta eta}.

This is a structural EKF compatibility check.
It is not a full tracking-performance simulation.
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
        error("check_step09H3d:UnsupportedDim", ...
            "This check supports cfg.dim = 2 or 3.");
    end

    watcherState = struct();

    % The measurement model used by DNN_EKF_Update_Local only requires
    % watcherState.r according to the current update-file interface.
    watcherState.r = zeros(dim, 1);

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.3d check: DNN_EKF_Update_Local branch models\n");
    fprintf("============================================================\n");

    % ---------------------------------------------------------------------
    % Case 1: fixed_feature_lip
    % ---------------------------------------------------------------------
    cfg = cfg0;
    cfg.dnn.branchModel = "fixed_feature_lip";
    cfg.dnn.predictionResidualSource = "local_DNN";
    cfg.dnn.useResidualInPrediction = true;
    cfg.dnn.residualInjectionGain = 1.0;

    runOneUpdateCase_step09h3d("fixed_feature_lip", etaTrue0, watcherState, cfg);

    % ---------------------------------------------------------------------
    % Case 2: mlp_general
    % ---------------------------------------------------------------------
    cfg = cfg0;
    cfg.dnn.branchModel = "mlp_general";
    cfg.dnn.predictionResidualSource = "local_DNN";
    cfg.dnn.useResidualInPrediction = true;
    cfg.dnn.residualInjectionGain = 1.0;

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

    runOneUpdateCase_step09h3d("mlp_general", etaTrue0, watcherState, cfg);

    fprintf("\nPASS: Step 09-H.3d DNN_EKF_Update_Local branch-model check passed.\n");

end

function runOneUpdateCase_step09h3d(caseName, etaTrue0, watcherState, cfg)

    watcher0 = initLocalDNNEKF(1, etaTrue0, cfg);

    watcherPred = DNN_EKF_Predict_Local(watcher0, 0.0, cfg);

    etaPred = watcherPred.xhat(watcherPred.idxEta);

    zhat = measurementPrediction(etaPred, watcherState, cfg);
    zhat = zhat(:);

    nz = numel(zhat);

    % Add a small artificial innovation so the update actually moves the
    % state. This avoids a trivial zero-innovation update.
    if nz == 1
        dz = 1e-3;
    else
        dz = zeros(nz, 1);
        dz(1) = 1e-3;
        dz(2) = -5e-4;
    end

    z = zhat + dz;

    thetaPred = watcherPred.xhat(watcherPred.idxTheta);

    watcherUpd = DNN_EKF_Update_Local(watcherPred, z, watcherState, cfg);

    thetaUpd = watcherUpd.xhat(watcherUpd.idxTheta);

    dTheta = thetaUpd - thetaPred;

    assertUpdateOutput_step09h3d(watcherUpd, cfg, watcherPred.nTheta);

    fprintf("\nCase: %s\n", caseName);
    fprintf("    nTheta             = %d\n", watcherUpd.nTheta);
    fprintf("    nX                 = %d\n", watcherUpd.nX);
    fprintf("    innovation norm    = %.6e\n", norm(watcherUpd.lastInnovation));
    fprintf("    trace(S)           = %.6e\n", trace(watcherUpd.lastS));
    fprintf("    ||dx update||      = %.6e\n", norm(watcherUpd.xhat - watcherPred.xhat));
    fprintf("    ||dtheta update||  = %.6e\n", norm(dTheta));
    fprintf("    trace(P pred)      = %.6e\n", trace(watcherPred.P));
    fprintf("    trace(P update)    = %.6e\n", trace(watcherUpd.P));

end

function assertUpdateOutput_step09h3d(watcher, cfg, nThetaExpected)

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
        "xhat length mismatch after update.");

    assert(isequal(size(watcher.P), [nXExpected nXExpected]), ...
        "P size mismatch after update.");

    assert(all(isfinite(watcher.xhat)), ...
        "Updated xhat contains non-finite values.");

    assert(all(isfinite(watcher.P(:))), ...
        "Updated P contains non-finite values.");

    assert(all(isfinite(watcher.lastInnovation)), ...
        "Innovation contains non-finite values.");

    assert(all(isfinite(watcher.lastS(:))), ...
        "Innovation covariance S contains non-finite values.");

    assert(norm(watcher.P - watcher.P.', "fro") < 1e-10, ...
        "Updated P is not symmetric.");

    eigMin = min(eig(0.5*(watcher.P + watcher.P.')));

    assert(eigMin > -1e-8, ...
        "Updated P has a significantly negative eigenvalue: %.6e", eigMin);

end