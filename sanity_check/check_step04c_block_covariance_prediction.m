function out = check_step04c_block_covariance_prediction(mode)
%{
File:
    sanity_check/check_step04c_block_covariance_prediction.m

Purpose:
    Sanity checks for Step 04c block-structured covariance prediction.

This check assumes the block covariance helper is named:

    predict_Cov_Block_DNN_EKF.m

Main checks:
    Check 1:
        Verify that predict_Cov_Block_DNN_EKF exists and that
        DNN_EKF_Predict_Local.m calls it.

    Check 2:
        Direct algebra test:
            P_dense = F*P*F' + Q
        versus
            P_block = predict_Cov_Block_DNN_EKF(F,P,Q,watcher,cfg)

        for an artificial block-upper-triangular F.

    Check 3:
        Full local-DNN simulation equivalence:
            cfg.ekf.useBlockCovPrediction = false
        versus
            cfg.ekf.useBlockCovPrediction = true

    Check 4:
        Full GS-composite Step 04b simulation equivalence:
            cfg.ekf.useBlockCovPrediction = false
        versus
            cfg.ekf.useBlockCovPrediction = true

How to run:
    Full check:
        out = check_step04c_block_covariance_prediction();

    Quick check only:
        out = check_step04c_block_covariance_prediction("quick");

    Local simulation check only:
        out = check_step04c_block_covariance_prediction("local");

Modes:
    "quick":
        Runs only source/existence check and direct algebra test.

    "local":
        Runs quick checks plus local-DNN full simulation equivalence.

    "full":
        Runs all checks, including GS-composite Step 04b equivalence.

Expected result:
    The dense and block covariance prediction should match to numerical
    precision. The full simulations should be essentially identical.
%}

    if nargin < 1
        mode = "full";
    end

    mode = string(mode);

    addpath(genpath(pwd));
    rehash;

    fprintf("\n============================================================\n");
    fprintf("Step 04c sanity check: block covariance prediction\n");
    fprintf("============================================================\n");

    out = struct();

    out.sourceCheck = runSourceCheck();
    out.directAlgebra = runDirectAlgebraCheck();

    if mode == "quick"
        fprintf("\nQuick mode selected. Skipping full simulation checks.\n");
        fprintf("\nAll requested Step 04c block covariance checks passed.\n");
        return;
    end

    out.localSimulation = runLocalSimulationEquivalenceCheck();

    if mode == "local"
        fprintf("\nLocal mode selected. Skipping GS full simulation check.\n");
        fprintf("\nAll requested Step 04c block covariance checks passed.\n");
        return;
    end

    out.GSSimulation = runGSSimulationEquivalenceCheck();

    fprintf("\n============================================================\n");
    fprintf("All Step 04c block covariance sanity checks passed.\n");
    fprintf("============================================================\n");

end

function out = runSourceCheck()
% Check that the helper exists and DNN_EKF_Predict_Local.m calls it.

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Check 1: Source and function existence\n");
    fprintf("------------------------------------------------------------\n");

    assert(exist("predict_Cov_Block_DNN_EKF", "file") == 2, ...
        "Missing ekf/predict_Cov_Block_DNN_EKF.m.");

    assert(exist("DNN_EKF_Predict_Local", "file") == 2, ...
        "Missing ekf/DNN_EKF_Predict_Local.m.");

    predictFile = which("DNN_EKF_Predict_Local");
    txt = fileread(predictFile);

    assert(contains(txt, "predict_Cov_Block_DNN_EKF"), ...
        "DNN_EKF_Predict_Local.m does not appear to call predict_Cov_Block_DNN_EKF.");

    assert(contains(txt, "useBlockCovPrediction"), ...
        "DNN_EKF_Predict_Local.m does not appear to use cfg.ekf.useBlockCovPrediction.");

    out = struct();
    out.passed = true;
    out.predictFile = predictFile;
    out.blockHelperFile = which("predict_Cov_Block_DNN_EKF");

    fprintf("Found helper file:\n");
    fprintf("  %s\n", out.blockHelperFile);
    fprintf("DNN_EKF_Predict_Local.m calls predict_Cov_Block_DNN_EKF.\n");
    fprintf("Check 1 passed.\n");

end

function out = runDirectAlgebraCheck()
% Directly compare block helper with dense F*P*F' + Q.

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Check 2: Direct algebra equivalence\n");
    fprintf("------------------------------------------------------------\n");

    cfg = config_step04_GS_DNN_EKF();

    if ~isfield(cfg, "ekf")
        cfg.ekf = struct();
    end

    cfg.ekf.blockPredictionTol = 1e-12;

    dim = cfg.dim;
    nEta = 2 * dim;
    nTheta = cfg.dnn.nThetaPerBranch;
    nX = nEta + nTheta;

    watcher = struct();
    watcher.idxEta = 1:nEta;
    watcher.idxTheta = nEta + (1:nTheta);
    watcher.nEta = nEta;
    watcher.nTheta = nTheta;
    watcher.nX = nX;

    rng(404);

    % Artificial block upper-triangular transition matrix.
    F = zeros(nX, nX);

    F_ee = eye(nEta) + 1e-2 * randn(nEta, nEta);
    F_et = 1e-3 * randn(nEta, nTheta);

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "thetaTau")
        dTheta = exp(-cfg.dt / cfg.dnn.thetaTau) * ones(nTheta, 1);
    else
        dTheta = ones(nTheta, 1);
    end

    F(watcher.idxEta, watcher.idxEta) = F_ee;
    F(watcher.idxEta, watcher.idxTheta) = F_et;
    F(watcher.idxTheta, watcher.idxTheta) = diag(dTheta);

    % Random symmetric positive definite covariance.
    A = randn(nX, nX);
    P = A * A.';
    P = P / norm(P, "fro");
    P = P + 1e-6 * eye(nX);
    P = 0.5 * (P + P.');

    % Random symmetric positive semidefinite process covariance.
    B = randn(nX, nX);
    Q = 1e-8 * (B * B.');
    Q = Q / max(norm(Q, "fro"), eps);
    Q = 1e-8 * Q;
    Q = 0.5 * (Q + Q.');

    Pdense = F * P * F.' + Q;
    Pdense = 0.5 * (Pdense + Pdense.');

    Pblock = predict_Cov_Block_DNN_EKF(F, P, Q, watcher, cfg);

    diffP = maxAbsDiff(Pdense, Pblock);
    relDiffP = diffP / max(1, max(abs(Pdense(:))));

    fprintf("max |P_dense - P_block| = %.3e\n", diffP);
    fprintf("relative difference      = %.3e\n", relDiffP);

    tol = 1e-10;

    assert(diffP < tol, ...
        "Direct algebra check failed: block covariance prediction does not match dense F*P*F' + Q.");

    out = struct();
    out.passed = true;
    out.diffP = diffP;
    out.relDiffP = relDiffP;

    fprintf("Check 2 passed.\n");

end

function out = runLocalSimulationEquivalenceCheck()
% Compare full local-DNN simulation with dense versus block covariance prediction.

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Check 3: Local-DNN full simulation equivalence\n");
    fprintf("------------------------------------------------------------\n");

    seed = 100;

    cfgDense = makeBaseLocalConfig();
    cfgDense.ekf.useBlockCovPrediction = false;

    cfgBlock = cfgDense;
    cfgBlock.ekf.useBlockCovPrediction = true;

    rng(seed);
    resDense = simulateLocalDNNEKF(cfgDense);

    rng(seed);
    resBlock = simulateLocalDNNEKF(cfgBlock);

    diff_xhat = maxAbsDiff(resDense.xhat, resBlock.xhat);
    diff_xaug = maxAbsDiff(resDense.xhatAug, resBlock.xhatAug);
    diff_theta = maxAbsDiff(resDense.thetaHat, resBlock.thetaHat);
    diff_Pdiag = maxAbsDiff(resDense.Pdiag, resBlock.Pdiag);
    diff_NIS = maxAbsDiffFinite(resDense.NIS, resBlock.NIS);

    fprintf("max |xhat dense - block|   = %.3e\n", diff_xhat);
    fprintf("max |xaug dense - block|   = %.3e\n", diff_xaug);
    fprintf("max |theta dense - block|  = %.3e\n", diff_theta);
    fprintf("max |Pdiag dense - block|  = %.3e\n", diff_Pdiag);
    fprintf("max |NIS dense - block|    = %.3e\n", diff_NIS);

    tol = 1e-8;

    assert(diff_xhat < tol, ...
        "Local simulation check failed: xhat differs between dense and block prediction.");

    assert(diff_xaug < tol, ...
        "Local simulation check failed: xhatAug differs between dense and block prediction.");

    assert(diff_theta < tol, ...
        "Local simulation check failed: thetaHat differs between dense and block prediction.");

    assert(diff_Pdiag < tol, ...
        "Local simulation check failed: Pdiag differs between dense and block prediction.");

    assert(diff_NIS < tol, ...
        "Local simulation check failed: NIS differs between dense and block prediction.");

    out = struct();
    out.passed = true;
    out.diff_xhat = diff_xhat;
    out.diff_xaug = diff_xaug;
    out.diff_theta = diff_theta;
    out.diff_Pdiag = diff_Pdiag;
    out.diff_NIS = diff_NIS;

    fprintf("Check 3 passed.\n");

end

function out = runGSSimulationEquivalenceCheck()
% Compare full GS-composite Step 04b simulation with dense versus block covariance prediction.

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Check 4: GS-composite Step 04b full simulation equivalence\n");
    fprintf("------------------------------------------------------------\n");

    seed = 100;

    cfgDense = makeBaseGSConfig();
    cfgDense.ekf.useBlockCovPrediction = false;

    cfgBlock = cfgDense;
    cfgBlock.ekf.useBlockCovPrediction = true;

    rng(seed);
    resDense = simulate_GS_DNN_EKF(cfgDense);

    rng(seed);
    resBlock = simulate_GS_DNN_EKF(cfgBlock);

    diff_xhat = maxAbsDiff(resDense.xhat, resBlock.xhat);
    diff_xaug = maxAbsDiff(resDense.xhatAug, resBlock.xhatAug);
    diff_theta = maxAbsDiff(resDense.thetaHat, resBlock.thetaHat);
    diff_Pdiag = maxAbsDiff(resDense.Pdiag, resBlock.Pdiag);
    diff_NIS = maxAbsDiffFinite(resDense.NIS, resBlock.NIS);

    fprintf("max |xhat dense - block|   = %.3e\n", diff_xhat);
    fprintf("max |xaug dense - block|   = %.3e\n", diff_xaug);
    fprintf("max |theta dense - block|  = %.3e\n", diff_theta);
    fprintf("max |Pdiag dense - block|  = %.3e\n", diff_Pdiag);
    fprintf("max |NIS dense - block|    = %.3e\n", diff_NIS);

    tol = 1e-8;

    assert(diff_xhat < tol, ...
        "GS simulation check failed: xhat differs between dense and block prediction.");

    assert(diff_xaug < tol, ...
        "GS simulation check failed: xhatAug differs between dense and block prediction.");

    assert(diff_theta < tol, ...
        "GS simulation check failed: thetaHat differs between dense and block prediction.");

    assert(diff_Pdiag < tol, ...
        "GS simulation check failed: Pdiag differs between dense and block prediction.");

    assert(diff_NIS < tol, ...
        "GS simulation check failed: NIS differs between dense and block prediction.");

    out = struct();
    out.passed = true;
    out.diff_xhat = diff_xhat;
    out.diff_xaug = diff_xaug;
    out.diff_theta = diff_theta;
    out.diff_Pdiag = diff_Pdiag;
    out.diff_NIS = diff_NIS;

    fprintf("Check 4 passed.\n");

end

function cfg = makeBaseLocalConfig()
% Create local-DNN configuration for dense/block prediction equivalence.

    cfg = config_step04_GS_DNN_EKF();

    cfg.truth.residualAmp = 5e-4;

    cfg.dnn.theta0_std = 0.0;
    cfg.dnn.residualInjectionGain = 1.0;
    cfg.dnn.predictionResidualSource = "local_DNN";

    if ~isfield(cfg, "gs")
        cfg.gs = struct();
    end

    cfg.gs.enabled = false;

    if ~isfield(cfg, "ekf")
        cfg.ekf = struct();
    end

    cfg.ekf.blockPredictionTol = 1e-12;

end

function cfg = makeBaseGSConfig()
% Create GS-composite Step 04b configuration for dense/block prediction equivalence.

    cfg = config_step04_GS_DNN_EKF();

    cfg.truth.residualAmp = 5e-4;

    cfg.dnn.theta0_std = 0.0;
    cfg.dnn.residualInjectionGain = 1.0;
    cfg.dnn.predictionResidualSource = "GS_composite";

    if ~isfield(cfg, "gs")
        cfg.gs = struct();
    end

    cfg.gs.enabled = true;
    cfg.gs.bootstrapUpload = true;
    cfg.gs.uploadMode = "after_measurement_update";
    cfg.gs.broadcastMode = "every_step";

    cfg.gs.useNonlocalBranchCovariance = true;
    cfg.gs.youngMode = "uniform";
    cfg.gs.SresNonlocal = 0.0;

    if ~isfield(cfg, "ekf")
        cfg.ekf = struct();
    end

    cfg.ekf.blockPredictionTol = 1e-12;

end

function d = maxAbsDiff(a, b)
% Return max absolute difference between two numeric arrays.

    d = max(abs(a(:) - b(:)));

end

function d = maxAbsDiffFinite(a, b)
% Return max absolute difference using only entries finite in both arrays.
% Useful for NIS arrays that may contain NaN when measurements are missing.

    a = a(:);
    b = b(:);

    finiteMask = isfinite(a) & isfinite(b);

    if ~any(finiteMask)
        d = 0;
        return;
    end

    d = max(abs(a(finiteMask) - b(finiteMask)));

    nanMismatch = xor(isnan(a), isnan(b));
    assert(~any(nanMismatch), ...
        "NaN pattern differs between dense and block results.");

end