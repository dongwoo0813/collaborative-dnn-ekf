function check_step09J4_bearing_fim_gated_Qnonlocal()
%{
File:
    sanity_check/check_step09J4_bearing_fim_gated_Qnonlocal.m

Purpose:
    Targeted Step 09-J.4 check for using the same bearing-FIM gates in
    nonlocal branch covariance injection.

Checks:
    1. computeNonlocalBranchCovarianceInjection(...) accepts
       cfg.gs.compositeMode = "bearing_fim_gated".
    2. It computes Jtheta_eff,j = B_{j|m} Jtheta_j for active nonlocal
       branches.
    3. The returned SdNonlocal matches a manual gated construction.
    4. The returned Qnonlocal matches the manual acceleration-to-state map.
    5. SdNonlocal and Qnonlocal remain symmetric PSD up to roundoff.
    6. Additive mode is unchanged relative to the raw Jtheta Ptheta Jtheta'
       construction.
%}

    addpath(genpath(pwd));
    rehash;

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-J.4 check: bearing-FIM-gated Qnonlocal\n");
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
    cfg.gs.nonlocalWeightMode = "none";
    cfg.gs.fimGate.epsilon = 1.0e-10;
    cfg.gs.fimGate.normalizeTrace = false;

    cfg.gate.mode = "tight_frame_2d_rt";
    cfg.gate.minRange = 1e-12;

    [nTheta, branchInfo] = branchThetaNumel(cfg);

    fprintf("branchModel = %s\n", string(cfg.dnn.branchModel));
    fprintf("nTheta      = %d\n", nTheta);
    fprintf("layerSizes  = [%s]\n", sprintf("%d ", branchInfo.arch.layerSizes));

    eta0 = [500.0; -250.0; 0.030; -0.020];

    watcher = initLocalDNNEKF(1, eta0, cfg);
    watcher.xhat(watcher.idxEta) = eta0;

    theta2 = makeRandomTheta_step09J4(nTheta, 22, 2.0e-2);
    theta3 = makeRandomTheta_step09J4(nTheta, 33, 2.0e-2);

    Ptheta2 = makeSPDThetaCov_step09J4(nTheta, 2.0e-9, 1.0e-10);
    Ptheta3 = makeSPDThetaCov_step09J4(nTheta, 1.5e-9, 0.8e-10);

    % Full-rank OmegaBar set for A_m = {1,2,3}. The local branch OmegaBar is
    % included in OmegaSigma_m even though only nonlocal branches contribute
    % to SdNonlocal.
    watcher.OmegaBar = bearingDirectionInfoMatrix([1.0; 0.0]);

    watcher.gsBranches = repmat(makeCacheRecord_step09J4(nTheta, cfg.dim), cfg.Nw, 1);

    watcher.gsBranches(2) = makeCacheRecord_step09J4(nTheta, cfg.dim);
    watcher.gsBranches(2).theta = theta2;
    watcher.gsBranches(2).Ptheta = Ptheta2;
    watcher.gsBranches(2).active = true;
    watcher.gsBranches(2).usedInPrediction = true;
    watcher.gsBranches(2).status = "valid";
    watcher.gsBranches(2).isStale = false;
    watcher.gsBranches(2).OmegaBar = bearingDirectionInfoMatrix([0.0; 1.0]);
    watcher.gsBranches(2).numOmegaUpdates = 20;

    watcher.gsBranches(3) = makeCacheRecord_step09J4(nTheta, cfg.dim);
    watcher.gsBranches(3).theta = theta3;
    watcher.gsBranches(3).Ptheta = Ptheta3;
    watcher.gsBranches(3).active = true;
    watcher.gsBranches(3).usedInPrediction = true;
    watcher.gsBranches(3).status = "valid";
    watcher.gsBranches(3).isStale = false;
    watcher.gsBranches(3).OmegaBar = bearingDirectionInfoMatrix([1.0; 1.0]);
    watcher.gsBranches(3).numOmegaUpdates = 20;

    activeNonlocal = [2; 3];

    % ------------------------------------------------------------------
    % New bearing-FIM-gated mode.
    % ------------------------------------------------------------------
    cfg.gs.compositeMode = "bearing_fim_gated";

    [Qnonlocal, SdNonlocal, diagInfo] = ...
        computeNonlocalBranchCovarianceInjection(watcher, eta0, cfg);

    [SdManual, Qmanual, gateDiagManual] = manualBearingFIMGatedQ_step09J4( ...
        watcher, eta0, activeNonlocal, cfg);

    errSd = norm(SdNonlocal - SdManual, "fro");
    relSd = errSd / max(1.0e-14, norm(SdManual, "fro"));

    errQ = norm(Qnonlocal - Qmanual, "fro");
    relQ = errQ / max(1.0e-14, norm(Qmanual, "fro"));

    minEigSd = min(eig(0.5 * (SdNonlocal + SdNonlocal.')));
    minEigQ = min(eig(0.5 * (Qnonlocal + Qnonlocal.')));

    fprintf("\nMode: bearing_fim_gated\n");
    fprintf("    active nonlocal branches  = %s\n", mat2str(diagInfo.branchIDs.'));
    fprintf("    gate sum identity error   = %.3e\n", diagInfo.gateSumIdentityError);
    fprintf("    manual gate identity err  = %.3e\n", gateDiagManual.sumGateIdentityError);
    fprintf("    trace(SdNonlocal)         = %.3e\n", trace(SdNonlocal));
    fprintf("    trace(Qnonlocal)          = %.3e\n", trace(Qnonlocal));
    fprintf("    rel error Sd              = %.3e\n", relSd);
    fprintf("    rel error Q               = %.3e\n", relQ);
    fprintf("    min eig Sd                = %.3e\n", minEigSd);
    fprintf("    min eig Q                 = %.3e\n", minEigQ);

    assert(diagInfo.enabled, "Qnonlocal diagnostic should be enabled.");
    assert(diagInfo.fimGateEnabled, "bearing_fim_gated mode should enable FIM gate diagnostics.");
    assert(isequal(diagInfo.branchIDs(:), activeNonlocal), "Unexpected active nonlocal branch IDs.");
    assert(isequal(size(SdNonlocal), [cfg.dim, cfg.dim]), "SdNonlocal has wrong size.");
    assert(isequal(size(Qnonlocal), [watcher.nX, watcher.nX]), "Qnonlocal has wrong size.");
    assert(relSd < 1.0e-10, "Bearing-FIM-gated SdNonlocal manual comparison failed.");
    assert(relQ < 1.0e-10, "Bearing-FIM-gated Qnonlocal manual comparison failed.");
    assert(minEigSd > -1.0e-18, "SdNonlocal is not PSD within tolerance.");
    assert(minEigQ > -1.0e-18, "Qnonlocal is not PSD within tolerance.");

    % ------------------------------------------------------------------
    % Regression: additive mode still uses raw Jtheta Ptheta Jtheta'.
    % ------------------------------------------------------------------
    cfg.gs.compositeMode = "additive";

    [Qadd, SdAdd, diagAdd] = ...
        computeNonlocalBranchCovarianceInjection(watcher, eta0, cfg);

    [SdAddManual, QaddManual] = manualAdditiveQ_step09J4( ...
        watcher, eta0, activeNonlocal, cfg);

    relSdAdd = norm(SdAdd - SdAddManual, "fro") / max(1.0e-14, norm(SdAddManual, "fro"));
    relQAdd = norm(Qadd - QaddManual, "fro") / max(1.0e-14, norm(QaddManual, "fro"));

    fprintf("\nMode: additive regression\n");
    fprintf("    active nonlocal branches  = %s\n", mat2str(diagAdd.branchIDs.'));
    fprintf("    trace(SdNonlocal)         = %.3e\n", trace(SdAdd));
    fprintf("    trace(Qnonlocal)          = %.3e\n", trace(Qadd));
    fprintf("    rel error Sd              = %.3e\n", relSdAdd);
    fprintf("    rel error Q               = %.3e\n", relQAdd);

    assert(~diagAdd.fimGateEnabled, "Additive mode should not enable FIM gate diagnostics.");
    assert(relSdAdd < 1.0e-10, "Additive SdNonlocal regression failed.");
    assert(relQAdd < 1.0e-10, "Additive Qnonlocal regression failed.");

    fprintf("\nStep 09-J.4 bearing-FIM-gated Qnonlocal check PASSED.\n");

end



function theta = makeRandomTheta_step09J4(nTheta, seed, scale)
%MAKERANDOMTHETA_STEP09J4 Deterministic nonzero MLP theta for covariance tests.

    rng(seed);
    theta = scale * randn(nTheta, 1);

end

function Ptheta = makeSPDThetaCov_step09J4(nTheta, baseScale, slopeScale)
%MAKESPDTHETACOV_STEP09J4 Simple diagonal positive covariance.

    diagVals = baseScale + slopeScale * linspace(0.0, 1.0, nTheta).';
    Ptheta = diag(diagVals);

end

function rec = makeCacheRecord_step09J4(nTheta, dim)
%MAKECACHERECORD_STEP09J4 Minimal watcher-side GS branch cache record.

    rec = struct();
    rec.theta = zeros(nTheta, 1);
    rec.Ptheta = NaN(nTheta, nTheta);
    rec.active = false;
    rec.usedInPrediction = false;
    rec.status = "empty";
    rec.isStale = true;

    rec.OmegaBar = zeros(dim, dim);
    rec.numOmegaUpdates = 0;
    rec.lastLOSUnit = NaN(dim, 1);
    rec.lastOmegaUpdateTime = NaN;
    rec.lastMeasTime = NaN;
    rec.outputFrame = "inertial";

end

function [SdManual, Qmanual, gateDiag] = manualBearingFIMGatedQ_step09J4( ...
    watcher, eta, activeNonlocal, cfg)
%MANUALBEARINGFIMGATEDQ_STEP09J4 Manual gated Jtheta Ptheta Jtheta' sum.

    dim = cfg.dim;
    Nw = cfg.Nw;

    branchUsedForGate = false(Nw, 1);
    branchUsedForGate(watcher.localBranchID) = true;
    branchUsedForGate(activeNonlocal) = true;

    [B, gateDiag] = computeBearingFIMGates(watcher, branchUsedForGate, cfg);

    Nnonlocal = numel(activeNonlocal);
    youngCoeff = Nnonlocal;

    SdManual = zeros(dim, dim);

    for k = 1:Nnonlocal

        j = activeNonlocal(k);
        theta_j = watcher.gsBranches(j).theta(:);
        Ptheta_j = watcher.gsBranches(j).Ptheta;
        Ptheta_j = 0.5 * (Ptheta_j + Ptheta_j.');

        [~, ~, JthetaRaw, ~] = evaluateBranchResidualModel( ...
            j, eta, theta_j, cfg);

        JthetaEff = B(:, :, j) * JthetaRaw;
        Sj = JthetaEff * Ptheta_j * JthetaEff.';
        Sj = 0.5 * (Sj + Sj.');

        SdManual = SdManual + youngCoeff * Sj;

    end

    Qmanual = manualQMap_step09J4(SdManual, watcher, cfg);

end

function [SdManual, Qmanual] = manualAdditiveQ_step09J4( ...
    watcher, eta, activeNonlocal, cfg)
%MANUALADDITIVEQ_STEP09J4 Manual raw additive nonlocal covariance sum.

    dim = cfg.dim;
    Nnonlocal = numel(activeNonlocal);
    youngCoeff = Nnonlocal;

    SdManual = zeros(dim, dim);

    for k = 1:Nnonlocal

        j = activeNonlocal(k);
        theta_j = watcher.gsBranches(j).theta(:);
        Ptheta_j = watcher.gsBranches(j).Ptheta;
        Ptheta_j = 0.5 * (Ptheta_j + Ptheta_j.');

        [~, ~, JthetaRaw, ~] = evaluateBranchResidualModel( ...
            j, eta, theta_j, cfg);

        Sj = JthetaRaw * Ptheta_j * JthetaRaw.';
        Sj = 0.5 * (Sj + Sj.');

        SdManual = SdManual + youngCoeff * Sj;

    end

    Qmanual = manualQMap_step09J4(SdManual, watcher, cfg);

end

function Q = manualQMap_step09J4(Sd, watcher, cfg)
%MANUALQMAP_STEP09J4 Manual acceleration covariance mapping into EKF state.

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