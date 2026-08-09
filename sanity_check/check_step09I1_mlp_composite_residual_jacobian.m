%{
check_step09I1_mlp_composite_residual_jacobian.m

Purpose:
    Targeted structural check for Step 09-I.1.

    Verify that evaluateWatcherCompositeResidual(...) is compatible with
    cfg.dnn.branchModel = "mlp_general".

What this checks:
    1. Additive GS composite residual works with MLP branches.
    2. Gated additive GS composite residual works with MLP branches.
    3. Hybrid local-full + gated-nonlocal mode works with MLP branches.
    4. The returned analytic JetaComp matches finite differences.

Why this check is important:
    GS composite residual evaluates both:

        local branch:
            theta_i from watcher.xhat(idxTheta)

        nonlocal branch:
            theta_j from watcher.gsBranches(j).theta

    using the same eta input. For MLP, both theta_i and theta_j have length
    nTheta = 256 for the default architecture [6 12 8 6 2].

This check does not:
    - run the full GS simulation
    - test GS repository metadata
    - test nonlocal covariance injection

Those are later Step 09-I tasks.
%}

addpath(genpath(pwd));
rehash;

fprintf("\n");
fprintf("============================================================\n");
fprintf("Step 09-I.1 check: MLP composite residual Jacobian\n");
fprintf("============================================================\n");

cfg = config_step04_GS_DNN_EKF();

cfg.dim = 2;
cfg.Nw  = 4;

cfg.dnn.branchModel = "mlp_general";
cfg.dnn.mlp.hiddenSizes = [12 8 6];
cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
cfg.dnn.mlp.inputMode = "eta_phase";
cfg.dnn.mlp.rScale = 1000.0;
cfg.dnn.mlp.vScale = 0.1;

cfg.dnn.predictionResidualSource = "GS_composite";
cfg.dnn.residualInjectionGain = 1.0;

cfg.gs.nonlocalWeightMode = "none";
cfg.gs.nonlocalWeight = 1.0;
cfg.gs.useNonlocalBranchCovariance = false;

cfg.gate.mode = "tight_frame_2d_rt";
cfg.gate.minRange = 1e-12;

[nTheta, branchInfo] = branchThetaNumel(cfg);

fprintf("branchModel = %s\n", string(cfg.dnn.branchModel));
fprintf("nTheta      = %d\n", nTheta);
fprintf("layerSizes  = [%s]\n", sprintf("%d ", branchInfo.arch.layerSizes));

% Use eta away from r = 0 so the gate derivative is well defined.
eta0 = [500.0; -250.0; 0.030; -0.020];

rng(101);
watcher = initLocalDNNEKF(1, eta0, cfg);

watcher.xhat(watcher.idxEta) = eta0;

thetaLocal = makeRandomTheta_step09I1(nTheta, 11, 2.0e-2);
thetaNonlocal2 = makeRandomTheta_step09I1(nTheta, 22, 2.0e-2);

watcher.xhat(watcher.idxTheta) = thetaLocal;

% Minimal GS branch cache record.
%
% evaluateWatcherCompositeResidual currently needs:
%   theta
%   active
%   usedInPrediction
%   status
%   isStale
cacheTemplate = struct();
cacheTemplate.theta = zeros(nTheta, 1);
cacheTemplate.active = false;
cacheTemplate.usedInPrediction = false;
cacheTemplate.status = "empty";
cacheTemplate.isStale = true;

watcher.gsBranches = repmat(cacheTemplate, cfg.Nw, 1);

% Activate branch 2 as one valid nonlocal GS copy.
watcher.gsBranches(2).theta = thetaNonlocal2;
watcher.gsBranches(2).active = true;
watcher.gsBranches(2).usedInPrediction = true;
watcher.gsBranches(2).status = "valid";
watcher.gsBranches(2).isStale = false;

modes = [
    "additive"
    "gated_additive"
    "local_full_plus_gated_nonlocal"
];

for iMode = 1:numel(modes)

    mode = modes(iMode);
    cfg.gs.compositeMode = mode;

    fprintf("\nMode: %s\n", mode);

    [dComp, JetaComp, branchContrib, branchUsed] = ...
        evaluateWatcherCompositeResidual(watcher, eta0, thetaLocal, cfg);

    if ~isequal(size(dComp), [cfg.dim, 1])
        error("Step09I1:BadDCompSize", ...
            "dComp has wrong size for mode %s.", mode);
    end

    if ~isequal(size(JetaComp), [cfg.dim, 2*cfg.dim])
        error("Step09I1:BadJetaCompSize", ...
            "JetaComp has wrong size for mode %s.", mode);
    end

    if ~isequal(size(branchContrib), [cfg.dim, cfg.Nw])
        error("Step09I1:BadBranchContribSize", ...
            "branchContrib has wrong size for mode %s.", mode);
    end

    if ~isequal(size(branchUsed), [cfg.Nw, 1])
        error("Step09I1:BadBranchUsedSize", ...
            "branchUsed has wrong size for mode %s.", mode);
    end

    expectedUsed = false(cfg.Nw, 1);
    expectedUsed(1) = true;
    expectedUsed(2) = true;

    if ~isequal(branchUsed, expectedUsed)
        error("Step09I1:BadBranchUsedPattern", ...
            "branchUsed pattern is wrong for mode %s.", mode);
    end

    Jfd = finiteDifferenceCompositeJeta_step09I1( ...
        watcher, eta0, thetaLocal, cfg);

    absErr = norm(JetaComp - Jfd, "fro");
    relErr = absErr / max(1.0e-12, norm(Jfd, "fro"));

    fprintf("    ||Jeta analytic - Jeta FD||_F = %.3e\n", absErr);
    fprintf("    relative error                 = %.3e\n", relErr);
    fprintf("    ||dComp||                      = %.3e\n", norm(dComp));
    fprintf("    active branches                = %s\n", mat2str(find(branchUsed).'));

    if relErr > 1.0e-5
        error("Step09I1:JetaMismatch", ...
            "MLP composite Jeta finite-difference check failed for mode %s.", mode);
    end

end

fprintf("\nStep 09-I.1 MLP composite residual check passed.\n");

function theta = makeRandomTheta_step09I1(nTheta, seed, scale)
%MAKERANDOMTHETA_STEP09I1 Deterministic nonzero MLP theta for Jacobian tests.
%
% The normal MLP initialization uses a zero output layer, which can make
% some Jacobian channels too small for a useful composite residual check.
% Here we intentionally use a small fully nonzero theta vector.

    rng(seed);
    theta = scale * randn(nTheta, 1);

end

function Jfd = finiteDifferenceCompositeJeta_step09I1( ...
    watcher, eta0, thetaLocal, cfg)
%FINITEDIFFERENCECOMPOSITEJETA_STEP09I1 Central-difference JetaComp.

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