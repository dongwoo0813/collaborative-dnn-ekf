%{
check_step09I3_mlp_nonlocal_covariance_injection.m

Purpose:
    Targeted check for Step 09-I.3.

    Verify that computeNonlocalBranchCovarianceInjection(...) works with
    cfg.dnn.branchModel = "mlp_general".

What this checks:
    1. Nonlocal covariance injection runs with MLP theta length nTheta = 256.
    2. The returned SdNonlocal and Qnonlocal have correct dimensions.
    3. SdNonlocal and Qnonlocal are symmetric PSD up to numerical tolerance.
    4. The computed SdNonlocal matches a manual Jtheta Ptheta Jtheta'
       construction for additive and gated composite modes.

This check does not run the full GS simulation.
%}

addpath(genpath(pwd));
rehash;

fprintf("\n");
fprintf("============================================================\n");
fprintf("Step 09-I.3 check: MLP nonlocal covariance injection\n");
fprintf("============================================================\n");

cfg = config_step04_GS_DNN_EKF();

cfg.dim = 2;
cfg.Nw = 4;
cfg.dt = 0.5;

cfg.dnn.branchModel = "mlp_general";
cfg.dnn.mlp.hiddenSizes = [12 8 6];
cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
cfg.dnn.mlp.inputMode = "eta_phase";
cfg.dnn.mlp.rScale = 1000.0;
cfg.dnn.mlp.vScale = 0.1;

cfg.dnn.residualInjectionGain = 1.0;
cfg.dnn.Ptheta0 = (7.5e-5)^2;
cfg.dnn.thetaSigmaSS = 7.5e-5;

cfg.gs.useNonlocalBranchCovariance = true;
cfg.gs.youngMode = "uniform";
cfg.gs.SresNonlocal = zeros(cfg.dim);

cfg.gate.mode = "tight_frame_2d_rt";
cfg.gate.minRange = 1e-12;

[nTheta, branchInfo] = branchThetaNumel(cfg);

fprintf("branchModel = %s\n", string(cfg.dnn.branchModel));
fprintf("nTheta      = %d\n", nTheta);
fprintf("layerSizes  = [%s]\n", sprintf("%d ", branchInfo.arch.layerSizes));

eta0 = [500.0; -250.0; 0.030; -0.020];

watcher = initLocalDNNEKF(1, eta0, cfg);
watcher.xhat(watcher.idxEta) = eta0;

thetaNonlocal = makeRandomTheta_step09I3(nTheta, 22, 2.0e-2);

% Use the tuned Local MLP covariance scale found in Step 09-H.5b.
PthetaNonlocal = (7.5e-5)^2 * eye(nTheta);

cacheTemplate = struct();
cacheTemplate.branchID = NaN;
cacheTemplate.theta = zeros(nTheta, 1);
cacheTemplate.Ptheta = NaN(nTheta, nTheta);
cacheTemplate.active = false;
cacheTemplate.usedInPrediction = false;
cacheTemplate.status = "empty";
cacheTemplate.isStale = true;

watcher.gsBranches = repmat(cacheTemplate, cfg.Nw, 1);

% Activate branch 2 as the only nonlocal branch.
%
% With one active nonlocal branch, the uniform Young coefficient is 1.
nonlocalBranchID = 2;

watcher.gsBranches(nonlocalBranchID).branchID = nonlocalBranchID;
watcher.gsBranches(nonlocalBranchID).theta = thetaNonlocal;
watcher.gsBranches(nonlocalBranchID).Ptheta = PthetaNonlocal;
watcher.gsBranches(nonlocalBranchID).active = true;
watcher.gsBranches(nonlocalBranchID).usedInPrediction = true;
watcher.gsBranches(nonlocalBranchID).status = "valid";
watcher.gsBranches(nonlocalBranchID).isStale = false;

modes = [
    "additive"
    "gated_additive"
    "local_full_plus_gated_nonlocal"
];

for iMode = 1:numel(modes)

    mode = modes(iMode);
    cfg.gs.compositeMode = mode;

    fprintf("\nMode: %s\n", mode);

    [Qnonlocal, SdNonlocal, diagInfo] = ...
        computeNonlocalBranchCovarianceInjection(watcher, eta0, cfg);

    if ~isequal(size(SdNonlocal), [cfg.dim, cfg.dim])
        error("Step09I3:BadSdSize", ...
            "SdNonlocal has wrong size for mode %s.", mode);
    end

    if ~isequal(size(Qnonlocal), [watcher.nX, watcher.nX])
        error("Step09I3:BadQSize", ...
            "Qnonlocal has wrong size for mode %s.", mode);
    end

    if diagInfo.numActiveNonlocal ~= 1
        error("Step09I3:BadNumActive", ...
            "Expected exactly one active nonlocal branch for mode %s.", mode);
    end

    if ~isequal(diagInfo.branchIDs(:), nonlocalBranchID)
        error("Step09I3:BadBranchIDs", ...
            "Unexpected active nonlocal branch IDs for mode %s.", mode);
    end

    SdManual = manualSdNonlocal_step09I3( ...
        nonlocalBranchID, eta0, thetaNonlocal, PthetaNonlocal, cfg);

    Qmanual = manualQnonlocal_step09I3(SdManual, watcher, cfg);

    errSd = norm(SdNonlocal - SdManual, "fro");
    relSd = errSd / max(1.0e-14, norm(SdManual, "fro"));

    errQ = norm(Qnonlocal - Qmanual, "fro");
    relQ = errQ / max(1.0e-14, norm(Qmanual, "fro"));

    minEigSd = min(eig(0.5 * (SdNonlocal + SdNonlocal.')));
    minEigQ = min(eig(0.5 * (Qnonlocal + Qnonlocal.')));

    fprintf("    trace(SdNonlocal) = %.3e\n", trace(SdNonlocal));
    fprintf("    trace(Qnonlocal)  = %.3e\n", trace(Qnonlocal));
    fprintf("    rel error Sd      = %.3e\n", relSd);
    fprintf("    rel error Q       = %.3e\n", relQ);
    fprintf("    min eig Sd        = %.3e\n", minEigSd);
    fprintf("    min eig Q         = %.3e\n", minEigQ);

    if relSd > 1.0e-10
        error("Step09I3:SdMismatch", ...
            "SdNonlocal manual comparison failed for mode %s.", mode);
    end

    if relQ > 1.0e-10
        error("Step09I3:QMismatch", ...
            "Qnonlocal manual comparison failed for mode %s.", mode);
    end

    if minEigSd < -1.0e-18
        error("Step09I3:SdNotPSD", ...
            "SdNonlocal is not PSD for mode %s.", mode);
    end

    if minEigQ < -1.0e-18
        error("Step09I3:QNotPSD", ...
            "Qnonlocal is not PSD for mode %s.", mode);
    end

end

fprintf("\nStep 09-I.3 MLP nonlocal covariance injection check passed.\n");

function theta = makeRandomTheta_step09I3(nTheta, seed, scale)
%MAKERANDOMTHETA_STEP09I3 Deterministic nonzero MLP theta for covariance test.

    rng(seed);
    theta = scale * randn(nTheta, 1);

end

function Sd = manualSdNonlocal_step09I3(branchID, eta, theta, Ptheta, cfg)
%MANUALSDNONLOCAL_STEP09I3 Manual Jtheta Ptheta Jtheta' construction.

    [~, ~, JthetaRaw, ~] = evaluateBranchResidualModel( ...
        branchID, eta, theta, cfg);

    compositeMode = "additive";

    if isfield(cfg, "gs") && isfield(cfg.gs, "compositeMode")
        compositeMode = string(cfg.gs.compositeMode);
    end

    switch compositeMode

        case "additive"
            JthetaEff = JthetaRaw;

        case {"gated_additive", "local_full_plus_gated_nonlocal"}
            Bgate = branchGateMatrix(branchID, eta, cfg);
            JthetaEff = Bgate * JthetaRaw;

        otherwise
            error("Step09I3:BadCompositeMode", ...
                "Unsupported composite mode = %s.", compositeMode);

    end

    Sd = JthetaEff * Ptheta * JthetaEff.';
    Sd = 0.5 * (Sd + Sd.');

end

function Q = manualQnonlocal_step09I3(Sd, watcher, cfg)
%MANUALQNONLOCAL_STEP09I3 Manual acceleration covariance mapping into X.

    dim = cfg.dim;
    dt = cfg.dt;

    Mx = zeros(watcher.nX, dim);

    idxR = 1:dim;
    idxV = dim + (1:dim);

    Mx(idxR, :) = 0.5 * dt^2 * eye(dim);
    Mx(idxV, :) = dt * eye(dim);

    Q = Mx * Sd * Mx.';
    Q = 0.5 * (Q + Q.');

end