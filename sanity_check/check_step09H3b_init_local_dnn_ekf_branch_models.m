function check_step09H3b_init_local_dnn_ekf_branch_models()
%{
File:
    checks/check_step09H3b_init_local_dnn_ekf_branch_models.m

Purpose:
    Check that initLocalDNNEKF is branch-model-aware.

What this verifies:
    1. fixed_feature_lip still initializes with the old small parameter size.
    2. mlp_general initializes with architecture-derived parameter size.
    3. xhat, P, idxEta, idxTheta, nX are dimensionally consistent.
    4. MLP default initialization gives near-zero initial residual output.

This check does not run the EKF simulation.
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
        error("check_step09H3b:UnsupportedDim", ...
            "This check supports cfg.dim = 2 or 3.");
    end

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.3b check: initLocalDNNEKF branch-model awareness\n");
    fprintf("============================================================\n");

    % ---------------------------------------------------------------------
    % Case 1: fixed_feature_lip
    % ---------------------------------------------------------------------
    cfg = cfg0;
    cfg.dnn.branchModel = "fixed_feature_lip";

    watcherFixed = initLocalDNNEKF(1, etaTrue0, cfg);

    [nThetaFixedExpected, infoFixed] = branchThetaNumel(cfg);

    assertWatcherDimensions_step09h3b( ...
        watcherFixed, cfg, nThetaFixedExpected);

    fprintf("\nCase: fixed_feature_lip\n");
    fprintf("    expected nTheta = %d\n", nThetaFixedExpected);
    fprintf("    watcher.nTheta  = %d\n", watcherFixed.nTheta);
    fprintf("    watcher.nX      = %d\n", watcherFixed.nX);
    fprintf("    branchModel     = %s\n", infoFixed.branchModel);

    % ---------------------------------------------------------------------
    % Case 2: mlp_general
    % ---------------------------------------------------------------------
    cfg = cfg0;
    cfg.dnn.branchModel = "mlp_general";

    cfg.dnn.mlp.hiddenSizes = [12 8 6];
    cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
    cfg.dnn.mlp.inputMode = "eta_phase";
    cfg.dnn.mlp.rScale = 1000.0;
    cfg.dnn.mlp.vScale = 0.1;

    % Default recommended MLP initialization.
    cfg.dnn.mlp.thetaInitMode = "random_hidden_zero_output";
    cfg.dnn.mlp.hiddenWeightStd = 0.1;
    cfg.dnn.mlp.hiddenBiasStd = 0.0;
    cfg.dnn.mlp.outputWeightStd = 0.0;
    cfg.dnn.mlp.outputBiasStd = 0.0;
    cfg.dnn.mlp.thetaInitSeed = 9100;

    watcherMLP = initLocalDNNEKF(1, etaTrue0, cfg);

    [nThetaMLPExpected, infoMLP] = branchThetaNumel(cfg);

    assertWatcherDimensions_step09h3b( ...
        watcherMLP, cfg, nThetaMLPExpected);

    thetaMLP0 = watcherMLP.xhat(watcherMLP.idxTheta);

    [d0, Jeta0, Jtheta0] = evaluateBranchResidualModel( ...
        watcherMLP.localBranchID, etaTrue0, thetaMLP0, cfg);

    fprintf("\nCase: mlp_general\n");
    fprintf("    layerSizes      = %s\n", mat2str(infoMLP.arch.layerSizes));
    fprintf("    expected nTheta = %d\n", nThetaMLPExpected);
    fprintf("    watcher.nTheta  = %d\n", watcherMLP.nTheta);
    fprintf("    watcher.nX      = %d\n", watcherMLP.nX);
    fprintf("    ||theta0||      = %.6e\n", norm(thetaMLP0));
    fprintf("    ||d0||          = %.6e\n", norm(d0));
    fprintf("    size(Jeta0)     = %s\n", mat2str(size(Jeta0)));
    fprintf("    size(Jtheta0)   = %s\n", mat2str(size(Jtheta0)));

    assert(norm(d0) < 1e-12, ...
        "MLP random_hidden_zero_output should give initial dHat = 0.");

    fprintf("\nPASS: Step 09-H.3b initLocalDNNEKF branch-model check passed.\n");

end

function assertWatcherDimensions_step09h3b(watcher, cfg, nThetaExpected)

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
        "watcher.xhat length mismatch.");

    assert(isequal(size(watcher.P), [nXExpected nXExpected]), ...
        "watcher.P size mismatch.");

    assert(isequal(watcher.idxEta, 1:nEta), ...
        "watcher.idxEta mismatch.");

    assert(isequal(watcher.idxTheta, nEta + (1:nThetaExpected)), ...
        "watcher.idxTheta mismatch.");

    assert(all(isfinite(watcher.xhat)), ...
        "watcher.xhat contains non-finite values.");

    assert(all(isfinite(watcher.P(:))), ...
        "watcher.P contains non-finite values.");

    assert(norm(watcher.P - watcher.P.', "fro") < 1e-12, ...
        "watcher.P is not symmetric.");

end