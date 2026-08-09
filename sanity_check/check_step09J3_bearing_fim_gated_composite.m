function check_step09J3_bearing_fim_gated_composite()
%{
File:
    sanity_check/check_step09J3_bearing_fim_gated_composite.m

Purpose:
    Targeted Step 09-J.3 check for the new GS composite mode

        cfg.gs.compositeMode = "bearing_fim_gated".

Checks:
    1. computeBearingFIMGates(...) returns B_{j|m} with the correct size.
    2. sum_j B_j is close to identity when epsilon is small and the
       accumulated OmegaBar sum is full rank.
    3. evaluateWatcherCompositeResidual(...) returns a correctly sized
       direction-gated residual and eta-Jacobian.
    4. The analytic Jeta returned by the bearing-FIM-gated composite matches
       a finite-difference Jeta check.
    5. The local-theta sensitivity implied by B_{local|m} has the correct
       size and matches finite differences with respect to selected theta
       entries.
    6. The old additive mode remains equal to the raw local + raw nonlocal
       branch sum.

Notes:
    This check does not test Qnonlocal gating. That belongs to Step 09-J.4.
%}

    addpath(genpath(pwd));
    rehash;

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-J.3 check: bearing-FIM-gated composite residual\n");
    fprintf("============================================================\n");

    cfg = config_step04_GS_DNN_EKF();
    cfg.dim = 2;
    cfg.Nw = 4;

    cfg.dnn.branchModel = "mlp_general";
    cfg.dnn.mlp.hiddenSizes = [12 8 6];
    cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
    cfg.dnn.mlp.inputMode = "eta_phase";
    cfg.dnn.mlp.rScale = 1000.0;
    cfg.dnn.mlp.vScale = 0.1;
    cfg.dnn.predictionResidualSource = "GS_composite";
    cfg.dnn.residualInjectionGain = 1.0;

    cfg.gs.nonlocalWeightMode = "none";
    cfg.gs.useNonlocalBranchCovariance = false;
    cfg.gs.fimGate.epsilon = 1.0e-10;
    cfg.gs.fimGate.normalizeTrace = false;

    [nTheta, branchInfo] = branchThetaNumel(cfg);

    fprintf("branchModel = %s\n", string(cfg.dnn.branchModel));
    fprintf("nTheta      = %d\n", nTheta);
    fprintf("layerSizes  = [%s]\n", sprintf("%d ", branchInfo.arch.layerSizes));

    eta0 = [500.0; -250.0; 0.030; -0.020];

    watcher = initLocalDNNEKF(1, eta0, cfg);
    watcher.xhat(watcher.idxEta) = eta0;

    thetaLocal = makeRandomTheta_step09J3(nTheta, 11, 2.0e-2);
    theta2 = makeRandomTheta_step09J3(nTheta, 22, 2.0e-2);
    theta3 = makeRandomTheta_step09J3(nTheta, 33, 2.0e-2);

    watcher.xhat(watcher.idxTheta) = thetaLocal;

    % Full-rank geometry set in 2-D.
    % OmegaBar for branch 1 supports mostly y-direction, branch 2 supports
    % mostly x-direction, and branch 3 adds an oblique direction. This makes
    % OmegaSigma well conditioned and gives a meaningful sum(B_j) check.
    watcher.OmegaBar = bearingDirectionInfoMatrix([1.0; 0.0]);

    watcher.gsBranches = repmat(makeCacheRecord_step09J3(nTheta, cfg.dim), cfg.Nw, 1);

    watcher.gsBranches(2) = makeCacheRecord_step09J3(nTheta, cfg.dim);
    watcher.gsBranches(2).theta = theta2;
    watcher.gsBranches(2).active = true;
    watcher.gsBranches(2).usedInPrediction = true;
    watcher.gsBranches(2).status = "valid";
    watcher.gsBranches(2).isStale = false;
    watcher.gsBranches(2).OmegaBar = bearingDirectionInfoMatrix([0.0; 1.0]);
    watcher.gsBranches(2).numOmegaUpdates = 10;

    watcher.gsBranches(3) = makeCacheRecord_step09J3(nTheta, cfg.dim);
    watcher.gsBranches(3).theta = theta3;
    watcher.gsBranches(3).active = true;
    watcher.gsBranches(3).usedInPrediction = true;
    watcher.gsBranches(3).status = "valid";
    watcher.gsBranches(3).isStale = false;
    watcher.gsBranches(3).OmegaBar = bearingDirectionInfoMatrix([1.0; 1.0]);
    watcher.gsBranches(3).numOmegaUpdates = 10;

    % Branch 4 stays inactive and must not contribute.

    branchUsedManual = [true; true; true; false];
    [Bmanual, gateManual] = computeBearingFIMGates(watcher, branchUsedManual, cfg);

    gateSumErr = norm(sum(Bmanual, 3) - eye(cfg.dim), "fro");
    minEigOmegaSigma = min(eig(gateManual.OmegaSigma));
    condOmegaSigma = cond(gateManual.OmegaSigma);

    fprintf("Manual gate sum identity error = %.3e\n", gateSumErr);
    fprintf("OmegaSigma min eig             = %.3e\n", minEigOmegaSigma);
    fprintf("OmegaSigma cond                = %.3e\n", condOmegaSigma);

    tol = 1.0e-8;

    assert(isequal(size(Bmanual), [cfg.dim, cfg.dim, cfg.Nw]), ...
        "B gate stack has wrong size.");
    assert(gateSumErr < 1.0e-8, ...
        "sum_j B_j is not close to identity for full-rank OmegaSigma.");
    assert(minEigOmegaSigma > 0.0, ...
        "OmegaSigma should be positive definite after epsilon regularization.");

    cfg.gs.compositeMode = "bearing_fim_gated";

    [dFim, JetaFim, branchContribFim, branchUsedFim, gateDiag] = ...
        evaluateWatcherCompositeResidual(watcher, eta0, thetaLocal, cfg);

    fprintf("FIM active branches             = %s\n", mat2str(find(branchUsedFim).'));
    fprintf("FIM ||dComp||                   = %.3e\n", norm(dFim));
    fprintf("FIM Jeta size                   = %dx%d\n", size(JetaFim, 1), size(JetaFim, 2));
    fprintf("FIM gate sum identity error     = %.3e\n", gateDiag.sumGateIdentityError);

    assert(isequal(branchUsedFim, branchUsedManual), ...
        "bearing_fim_gated branchUsed pattern is wrong.");
    assert(isequal(size(dFim), [cfg.dim, 1]), ...
        "bearing_fim_gated dComp has wrong size.");
    assert(isequal(size(JetaFim), [cfg.dim, 2*cfg.dim]), ...
        "bearing_fim_gated Jeta has wrong size.");
    assert(isequal(size(branchContribFim), [cfg.dim, cfg.Nw]), ...
        "bearing_fim_gated branchContrib has wrong size.");
    assert(gateDiag.enabled, "gateDiag.enabled should be true.");

    JetaFD = finiteDifferenceCompositeJeta_step09J3( ...
        watcher, eta0, thetaLocal, cfg);

    JetaAbsErr = norm(JetaFim - JetaFD, "fro");
    JetaRelErr = JetaAbsErr / max(1.0e-12, norm(JetaFD, "fro"));

    fprintf("||Jeta analytic - FD||_F        = %.3e\n", JetaAbsErr);
    fprintf("Jeta relative error             = %.3e\n", JetaRelErr);

    assert(JetaRelErr < 1.0e-5, ...
        "bearing_fim_gated Jeta finite-difference check failed.");

    [~, ~, JthetaLocalRaw, ~] = evaluateBranchResidualModel( ...
        watcher.localBranchID, eta0, thetaLocal, cfg);

    Blocal = gateDiag.B(:, :, watcher.localBranchID);
    JthetaLocalGated = Blocal * JthetaLocalRaw;

    fprintf("Local gated Jtheta size         = %dx%d\n", ...
        size(JthetaLocalGated, 1), size(JthetaLocalGated, 2));

    assert(isequal(size(JthetaLocalGated), [cfg.dim, nTheta]), ...
        "Local bearing-FIM-gated Jtheta has wrong size.");

    thetaCols = unique(round(linspace(1, nTheta, 6)));
    JthetaFD = finiteDifferenceCompositeTheta_step09J3( ...
        watcher, eta0, thetaLocal, cfg, thetaCols);

    JthetaAbsErr = norm(JthetaLocalGated(:, thetaCols) - JthetaFD, "fro");
    JthetaRelErr = JthetaAbsErr / max(1.0e-12, norm(JthetaFD, "fro"));

    fprintf("||Jtheta selected analytic-FD|| = %.3e\n", JthetaAbsErr);
    fprintf("Jtheta selected relative error  = %.3e\n", JthetaRelErr);

    assert(JthetaRelErr < 1.0e-5, ...
        "bearing_fim_gated local theta sensitivity finite-difference check failed.");

    % Check that the old additive mode still means raw local + raw nonlocal.
    cfg.gs.compositeMode = "additive";

    [dAdd, JetaAdd, ~, branchUsedAdd] = evaluateWatcherCompositeResidual( ...
        watcher, eta0, thetaLocal, cfg);

    [dRaw1, JRaw1] = evaluateBranchResidualModel(1, eta0, thetaLocal, cfg);
    [dRaw2, JRaw2] = evaluateBranchResidualModel(2, eta0, theta2, cfg);
    [dRaw3, JRaw3] = evaluateBranchResidualModel(3, eta0, theta3, cfg);

    dAddManual = dRaw1 + dRaw2 + dRaw3;
    JAddManual = JRaw1 + JRaw2 + JRaw3;

    addDerr = norm(dAdd - dAddManual);
    addJerr = norm(JetaAdd - JAddManual, "fro");

    fprintf("Additive d manual error         = %.3e\n", addDerr);
    fprintf("Additive J manual error         = %.3e\n", addJerr);

    assert(isequal(branchUsedAdd, branchUsedManual), ...
        "additive branchUsed pattern changed unexpectedly.");
    assert(addDerr < tol, "additive dComp changed unexpectedly.");
    assert(addJerr < tol, "additive Jeta changed unexpectedly.");

    fprintf("\nStep 09-J.3 bearing-FIM-gated composite check PASSED.\n");

end

function theta = makeRandomTheta_step09J3(nTheta, seed, scale)
%MAKERANDOMTHETA_STEP09J3 Deterministic nonzero MLP theta for Jacobian tests.

    rng(seed);
    theta = scale * randn(nTheta, 1);

end

function rec = makeCacheRecord_step09J3(nTheta, dim)
%MAKECACHERECORD_STEP09J3 Minimal watcher-side GS branch cache record.

    rec = struct();
    rec.theta = zeros(nTheta, 1);
    rec.Ptheta = NaN(nTheta, nTheta);
    rec.active = false;
    rec.usedInPrediction = false;
    rec.status = "empty";
    rec.isStale = true;

    % Geometry metadata fields added in Step 09-J.2.
    rec.OmegaBar = zeros(dim, dim);
    rec.numOmegaUpdates = 0;
    rec.lastLOSUnit = NaN(dim, 1);
    rec.lastOmegaUpdateTime = NaN;
    rec.lastMeasTime = NaN;
    rec.outputFrame = "inertial";

end

function Jfd = finiteDifferenceCompositeJeta_step09J3( ...
    watcher, eta0, thetaLocal, cfg)
%FINITEDIFFERENCECOMPOSITEJETA_STEP09J3 Central-difference eta-Jacobian.

    dim = cfg.dim;
    nEta = 2 * dim;
    Jfd = zeros(dim, nEta);

    for kEta = 1:nEta

        h = 1.0e-6 * max(1.0, abs(eta0(kEta)));

        etaPlus = eta0;
        etaMinus = eta0;

        etaPlus(kEta) = etaPlus(kEta) + h;
        etaMinus(kEta) = etaMinus(kEta) - h;

        dPlus = evaluateWatcherCompositeResidual( ...
            watcher, etaPlus, thetaLocal, cfg);
        dMinus = evaluateWatcherCompositeResidual( ...
            watcher, etaMinus, thetaLocal, cfg);

        Jfd(:, kEta) = (dPlus - dMinus) / (2*h);

    end

end

function Jfd = finiteDifferenceCompositeTheta_step09J3( ...
    watcher, eta0, thetaLocal, cfg, thetaCols)
%FINITEDIFFERENCECOMPOSITETHETA_STEP09J3 Central differences for selected theta columns.

    dim = cfg.dim;
    Jfd = zeros(dim, numel(thetaCols));

    for k = 1:numel(thetaCols)

        col = thetaCols(k);
        h = 1.0e-6 * max(1.0, abs(thetaLocal(col)));

        thetaPlus = thetaLocal;
        thetaMinus = thetaLocal;

        thetaPlus(col) = thetaPlus(col) + h;
        thetaMinus(col) = thetaMinus(col) - h;

        dPlus = evaluateWatcherCompositeResidual( ...
            watcher, eta0, thetaPlus, cfg);
        dMinus = evaluateWatcherCompositeResidual( ...
            watcher, eta0, thetaMinus, cfg);

        Jfd(:, k) = (dPlus - dMinus) / (2*h);

    end

end