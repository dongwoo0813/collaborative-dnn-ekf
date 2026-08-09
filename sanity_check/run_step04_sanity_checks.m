function out = run_step04_sanity_checks(mode)
%{
Function:
    run_step04_sanity_checks.m

Purpose:
    Run Step 04 sanity checks for the collaborative block-structured
    GS-assisted DNN-EKF simulation.

    Put this file in

        sanity_check/run_step04_sanity_checks.m

    under the project root. Then run from MATLAB:

        out = run_step04_sanity_checks();

    or, for only the fast helper-level checks:

        out = run_step04_sanity_checks("quick");

Checks included:
    0. Required path/function existence check.
    1. Empty GS cache: GS_composite residual equals local_DNN residual.
    2. Self GS copy is not allowed to overwrite the local branch.
    3. One valid nonlocal GS branch is added exactly.
    4. Full simulation: GS_composite with empty cache equals local_DNN.
    5. Full simulation: GS bootstrap-only equals local_DNN.

Interpretation:
    These checks do not test performance improvement. They test whether the
    Step 04 GS communication/composite residual architecture preserves the
    correct limiting cases.

Modes:
    "quick" - checks 0, 1, 2, 3 only.
    "all"   - checks 0, 1, 2, 3, 4, 5. Default.

Outputs:
    out - summary structure with pass/fail information and diagnostic values.

Notes:
    - The full mode runs several full simulations, so it may take longer.
    - If any check fails, this function prints all attempted check results
      and then throws an error at the end.
%}

    if nargin < 1 || isempty(mode)
        mode = "all";
    end

    mode = string(mode);

    if mode ~= "quick" && mode ~= "all"
        error("Unsupported mode: %s. Use 'quick' or 'all'.", mode);
    end

    % ------------------------------------------------------------------
    % Add project root to path.
    % This assumes this file lives in projectRoot/sanity_check/.
    % ------------------------------------------------------------------
    thisFile = mfilename("fullpath");
    sanityDir = fileparts(thisFile);
    projectRoot = fileparts(sanityDir);

    addpath(genpath(projectRoot));
    rehash;

    fprintf("\n============================================================\n");
    fprintf("Step 04 GS-DNN-EKF sanity checks\n");
    fprintf("Mode: %s\n", mode);
    fprintf("Project root: %s\n", projectRoot);
    fprintf("============================================================\n");

    checkResults = repmat(emptyCheckResult(), 0, 1);

    checkResults(end+1) = runOneCheck("00_path_and_function_existence", @checkPathAndFunctionExistence); 
    checkResults(end+1) = runOneCheck("01_empty_GS_cache_equals_local_residual", @checkEmptyGSCacheEqualsLocalResidual); 
    checkResults(end+1) = runOneCheck("02_self_GS_copy_does_not_overwrite_local_branch", @checkSelfGSCopyDoesNotOverwriteLocalBranch); 
    checkResults(end+1) = runOneCheck("03_one_valid_nonlocal_branch_added_exactly", @checkOneValidNonlocalBranchAddedExactly); 

    if mode == "all"
        checkResults(end+1) = runOneCheck("04_full_sim_empty_GS_composite_equals_local_DNN", @checkFullSimulationEmptyGSEqualsLocalDNN); 
        checkResults(end+1) = runOneCheck("05_full_sim_bootstrap_only_equals_local_DNN", @checkFullSimulationBootstrapOnlyEqualsLocalDNN); 
    end

    passedVec = [checkResults.passed].';
    numPassed = nnz(passedVec);
    numFailed = numel(passedVec) - numPassed;

    fprintf("\n============================================================\n");
    fprintf("Step 04 sanity-check summary\n");
    fprintf("============================================================\n");

    for k = 1:numel(checkResults)
        if checkResults(k).passed
            status = "PASS";
        else
            status = "FAIL";
        end
        fprintf("[%s] %s\n", status, checkResults(k).name);
        if ~checkResults(k).passed
            fprintf("      %s\n", checkResults(k).message);
        end
    end

    fprintf("\nTotal passed: %d / %d\n", numPassed, numel(checkResults));

    out = struct();
    out.mode = mode;
    out.projectRoot = projectRoot;
    out.results = checkResults;
    out.numPassed = numPassed;
    out.numFailed = numFailed;
    out.allPassed = (numFailed == 0);

    if numFailed > 0
        error("%d Step 04 sanity check(s) failed. See printed diagnostics above.", numFailed);
    end

    fprintf("\nAll requested Step 04 sanity checks passed.\n");

end

function result = runOneCheck(checkName, checkFcn)
% Run one check and capture pass/fail state without stopping the full suite.

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Running sanity check: %s\n", checkName);
    fprintf("------------------------------------------------------------\n");

    result = emptyCheckResult();
    result.name = string(checkName);

    try
        details = checkFcn();
        result.passed = true;
        result.message = "passed";
        result.details = details;
        fprintf("PASS: %s\n", checkName);
    catch ME
        result.passed = false;
        result.message = string(ME.message);
        result.details = struct();
        fprintf("FAIL: %s\n", checkName);
        fprintf("Reason: %s\n", ME.message);
    end

end

function result = emptyCheckResult()
% Template for one sanity-check result.

    result = struct();
    result.name = "";
    result.passed = false;
    result.message = "";
    result.details = struct();

end

function details = checkPathAndFunctionExistence()
% Check 0.
%
% What this checks:
%   Verifies that the required Step 04 functions are on the MATLAB path.
%
% Expected result:
%   Every listed function should exist as an m-file.
%
% If this fails:
%   The project path, folder placement, or file names are not consistent.

    requiredFunctions = [
        "config_step04_GS_DNN_EKF"
        "initLocalDNNEKF"
        "DNN_EKF_Predict_Local"
        "DNN_EKF_Update_Local"
        "initGSRepository"
        "uploadLocalBranchToGS"
        "broadcastGSRepositoryToWatcher"
        "evaluateWatcherCompositeResidual"
        "simulateLocalDNNEKF"
        "simulate_GS_DNN_EKF"
        "branchOutput"
        "featureBlock"
        "featureJacobianEta"
        "branchJacobianTheta"
    ];

    missing = strings(0,1);

    for k = 1:numel(requiredFunctions)
        f = requiredFunctions(k);
        if exist(f, "file") ~= 2
            missing(end+1,1) = f; 
        end
    end

    if ~isempty(missing)
        disp("Missing functions:");
        disp(missing);
        error("Required Step 04 functions are missing from the MATLAB path.");
    end

    details = struct();
    details.requiredFunctions = requiredFunctions;
    details.numRequiredFunctions = numel(requiredFunctions);

    fprintf("All required Step 04 functions are on the path.\n");

end

function details = checkEmptyGSCacheEqualsLocalResidual()
% Check 1.
%
% What this checks:
%   If watcher.gsBranches is empty, GS_composite must reduce exactly to
%   local_DNN:
%
%       d_comp,m(eta) = d_m(eta;theta_m).
%
% Why this is needed:
%   This is the fallback case before GS upload/broadcast is connected.
%
% Expected result:
%   Residual and eta-Jacobian differences should be near machine precision.
%
% If this fails:
%   evaluateWatcherCompositeResidual is not handling empty GS cache correctly,
%   or the local branch/Jacobian calculation is inconsistent.

    cfg = config_step04_GS_DNN_EKF();
    cfg.dnn.predictionResidualSource = "GS_composite";
    cfg.dnn.theta0_std = 0.0;

    rng(10);

    eta0 = [cfg.target.r0; cfg.target.v0];
    watcher = initLocalDNNEKF(1, eta0, cfg);

    eta = watcher.xhat(watcher.idxEta);
    thetaLocal = makeDeterministicTheta(watcher.nTheta, 1.0e-6);

    [dComp, Jcomp, ~, branchUsed] = evaluateWatcherCompositeResidual( ...
        watcher, eta, thetaLocal, cfg);

    branchID = watcher.localBranchID;

    dLocal = branchOutput(branchID, eta, thetaLocal, cfg);

    phi = featureBlock(branchID, eta, cfg);
    W = reshape(thetaLocal, cfg.dim, numel(phi));
    JLocal = W * featureJacobianEta(branchID, eta, cfg);

    errD = norm(dComp - dLocal);
    errJ = norm(Jcomp - JLocal, "fro");

    fprintf("err_d = %.3e\n", errD);
    fprintf("err_J = %.3e\n", errJ);
    fprintf("branchUsed = ");
    disp(branchUsed.');

    assert(errD < 1e-12, "Empty GS cache residual is not equal to local residual.");
    assert(errJ < 1e-12, "Empty GS cache Jacobian is not equal to local Jacobian.");
    assert(branchUsed(branchID) == true, "Local branch was not marked as used.");
    assert(nnz(branchUsed) == 1, "Unexpected nonlocal branches were marked as used.");

    details = struct();
    details.errD = errD;
    details.errJ = errJ;
    details.branchUsed = branchUsed;

end

function details = checkSelfGSCopyDoesNotOverwriteLocalBranch()
% Check 2.
%
% What this checks:
%   Even if the GS repository contains a valid record for watcher m's own
%   branch, watcher m must not use that GS self-copy. Its local branch must
%   come from the local EKF state theta_m.
%
% Why this is needed:
%   Step 04 architecture stores nonlocal GS branch copies only. The local
%   branch remains inside X_m = [eta_m; theta_m].
%
% Expected result:
%   A deliberately wrong GS self-copy should have zero effect on d_comp,m.
%
% If this fails:
%   broadcastGSRepositoryToWatcher or evaluateWatcherCompositeResidual is
%   allowing the GS self-copy to overwrite or replace the local branch.

    cfg = config_step04_GS_DNN_EKF();
    cfg.dnn.predictionResidualSource = "GS_composite";
    cfg.gs.maxStaleTime = Inf;

    rng(11);

    eta0 = [cfg.target.r0; cfg.target.v0];
    watcher = initLocalDNNEKF(1, eta0, cfg);
    gsRepo = initGSRepository(cfg);

    eta = watcher.xhat(watcher.idxEta);
    thetaLocal = makeDeterministicTheta(watcher.nTheta, 1.0e-6);

    thetaWrongSelf = 100 * ones(watcher.nTheta, 1);

    gsRepo.branch(1).theta = thetaWrongSelf;
    gsRepo.branch(1).Ptheta = 1e-8 * eye(watcher.nTheta);
    gsRepo.branch(1).lastUpdateTime = 0;
    gsRepo.branch(1).version = 99;
    gsRepo.branch(1).status = "valid";
    gsRepo.branch(1).age = 0;
    gsRepo.branch(1).isStale = false;

    [watcher, ~] = broadcastGSRepositoryToWatcher(gsRepo, watcher, 0, cfg);

    [dComp, ~, ~, branchUsed] = evaluateWatcherCompositeResidual( ...
        watcher, eta, thetaLocal, cfg);

    dExpected = branchOutput(1, eta, thetaLocal, cfg);

    errSelfOverwrite = norm(dComp - dExpected);

    fprintf("err_self_overwrite = %.3e\n", errSelfOverwrite);
    fprintf("branchUsed = ");
    disp(branchUsed.');

    assert(errSelfOverwrite < 1e-12, "GS self-copy changed the local branch residual.");
    assert(branchUsed(1) == true, "Local branch was not marked as used.");
    assert(nnz(branchUsed) == 1, "Unexpected nonlocal branches were used.");
    assert(watcher.gsBranches(1).active == false, "GS self branch cache should be inactive.");

    details = struct();
    details.errSelfOverwrite = errSelfOverwrite;
    details.branchUsed = branchUsed;
    details.selfCacheActive = watcher.gsBranches(1).active;

end

function details = checkOneValidNonlocalBranchAddedExactly()
% Check 3.
%
% What this checks:
%   If watcher 1 receives one valid nonlocal GS branch, branch 2, then
%
%       d_comp,1 = d_1(local) + d_2(GS)
%
%   and the eta-Jacobian is the exact sum of the two branch Jacobians.
%
% Why this is needed:
%   This directly checks the Step 04 composite residual implementation.
%
% Expected result:
%   Residual and Jacobian sum errors should be near machine precision.
%
% If this fails:
%   Nonlocal GS branch cache usage, branch indexing, or feature/Jacobian
%   selection is inconsistent.

    cfg = config_step04_GS_DNN_EKF();
    cfg.dnn.predictionResidualSource = "GS_composite";
    cfg.gs.maxStaleTime = Inf;

    rng(12);

    eta0 = [cfg.target.r0; cfg.target.v0];
    watcher = initLocalDNNEKF(1, eta0, cfg);
    gsRepo = initGSRepository(cfg);

    eta = watcher.xhat(watcher.idxEta);

    thetaLocal = makeDeterministicTheta(watcher.nTheta, 1.0e-6);
    thetaGS2 = makeDeterministicTheta(watcher.nTheta, -2.0e-6);

    gsRepo.branch(2).theta = thetaGS2;
    gsRepo.branch(2).Ptheta = 1e-8 * eye(watcher.nTheta);
    gsRepo.branch(2).lastUpdateTime = 0;
    gsRepo.branch(2).version = 1;
    gsRepo.branch(2).status = "valid";
    gsRepo.branch(2).age = 0;
    gsRepo.branch(2).isStale = false;

    [watcher, pkt] = broadcastGSRepositoryToWatcher(gsRepo, watcher, 0, cfg);

    [dComp, Jcomp, ~, branchUsed] = evaluateWatcherCompositeResidual( ...
        watcher, eta, thetaLocal, cfg);

    d1 = branchOutput(1, eta, thetaLocal, cfg);
    d2 = branchOutput(2, eta, thetaGS2, cfg);

    phi1 = featureBlock(1, eta, cfg);
    phi2 = featureBlock(2, eta, cfg);

    W1 = reshape(thetaLocal, cfg.dim, numel(phi1));
    W2 = reshape(thetaGS2, cfg.dim, numel(phi2));

    J1 = W1 * featureJacobianEta(1, eta, cfg);
    J2 = W2 * featureJacobianEta(2, eta, cfg);

    dExpected = d1 + d2;
    JExpected = J1 + J2;

    errD = norm(dComp - dExpected);
    errJ = norm(Jcomp - JExpected, "fro");

    fprintf("err_d = %.3e\n", errD);
    fprintf("err_J = %.3e\n", errJ);
    fprintf("branchUsed = ");
    disp(branchUsed.');
    fprintf("includedBranchIDs from broadcast = ");
    disp(pkt.includedBranchIDs);

    assert(errD < 1e-12, "One-nonlocal-branch residual sum is incorrect.");
    assert(errJ < 1e-12, "One-nonlocal-branch Jacobian sum is incorrect.");
    assert(branchUsed(1) == true, "Local branch was not used.");
    assert(branchUsed(2) == true, "Valid nonlocal branch 2 was not used.");
    assert(nnz(branchUsed) == 2, "Unexpected number of branches used.");

    details = struct();
    details.errD = errD;
    details.errJ = errJ;
    details.branchUsed = branchUsed;
    details.includedBranchIDs = pkt.includedBranchIDs;

end

function details = checkFullSimulationEmptyGSEqualsLocalDNN()
% Check 4.
%
% What this checks:
%   At the full simulation level, simulateLocalDNNEKF with
%   predictionResidualSource = "GS_composite" and no GS cache should produce
%   exactly the same trajectory as predictionResidualSource = "local_DNN".
%
% Why this is needed:
%   It checks that DNN_EKF_Predict_Local's GS_composite case did not change
%   Step 03 behavior when no nonlocal branches exist.
%
% Expected result:
%   xhat, augmented state, theta, and P diagonal logs are identical up to
%   numerical precision.
%
% If this fails:
%   The GS_composite prediction/Jacobian is not a correct local_DNN fallback,
%   or random stream consumption changed.

    cfgA = config_step04_GS_DNN_EKF();
    cfgA.dnn.predictionResidualSource = "local_DNN";
    cfgA.dnn.theta0_std = 0.0;
    cfgA.dnn.residualInjectionGain = 1.0;
    cfgA.truth.residualAmp = 5e-4;

    if isfield(cfgA, "gs")
        cfgA.gs.enabled = false;
    end

    rng(100);
    resLocal = simulateLocalDNNEKF(cfgA);

    cfgB = cfgA;
    cfgB.dnn.predictionResidualSource = "GS_composite";

    rng(100);
    resGSEmpty = simulateLocalDNNEKF(cfgB);

    diffXhat = maxAbsDiff(resLocal.xhat, resGSEmpty.xhat);
    diffXaug = maxAbsDiff(resLocal.xhatAug, resGSEmpty.xhatAug);
    diffTheta = maxAbsDiff(resLocal.thetaHat, resGSEmpty.thetaHat);
    diffPdiag = maxAbsDiff(resLocal.Pdiag, resGSEmpty.Pdiag);

    fprintf("max |xhat local - xhat GS_empty|     = %.3e\n", diffXhat);
    fprintf("max |xaug local - xaug GS_empty|     = %.3e\n", diffXaug);
    fprintf("max |theta local - theta GS_empty|   = %.3e\n", diffTheta);
    fprintf("max |Pdiag local - Pdiag GS_empty|   = %.3e\n", diffPdiag);

    tol = 1e-10;

    assert(diffXhat < tol, "xhat differs between local_DNN and empty GS_composite.");
    assert(diffXaug < tol, "xhatAug differs between local_DNN and empty GS_composite.");
    assert(diffTheta < tol, "thetaHat differs between local_DNN and empty GS_composite.");
    assert(diffPdiag < tol, "Pdiag differs between local_DNN and empty GS_composite.");

    details = struct();
    details.diffXhat = diffXhat;
    details.diffXaug = diffXaug;
    details.diffTheta = diffTheta;
    details.diffPdiag = diffPdiag;

end

function details = checkFullSimulationBootstrapOnlyEqualsLocalDNN()
% Check 5.
%
% What this checks:
%   simulate_GS_DNN_EKF with bootstrap upload only should match local_DNN.
%   Bootstrap makes all nonlocal branches active, but theta0_std = 0 makes
%   those nonlocal branch outputs exactly zero.
%
% Why this is needed:
%   It verifies that improvement in the real GS run is caused by later branch
%   sharing after measurement updates, not by wrapper/random-stream changes.
%
% Expected result:
%   xhat, augmented state, theta, and P diagonal logs match local_DNN exactly.
%   GS final upload count should be Nw.
%
% If this fails:
%   The GS simulation wrapper changes estimator timing/randomness, bootstrap
%   branches are not zero, or nonlocal branches are affecting covariance/mean
%   when they should not.

    cfgBase = config_step04_GS_DNN_EKF();

    cfgBase.truth.residualAmp = 5e-4;
    cfgBase.dnn.theta0_std = 0.0;
    cfgBase.dnn.residualInjectionGain = 1.0;

    seed = 100;

    cfgLocal = cfgBase;
    cfgLocal.dnn.predictionResidualSource = "local_DNN";
    cfgLocal.gs.enabled = false;

    rng(seed);
    resLocal = simulateLocalDNNEKF(cfgLocal);

    cfgGSBootstrapOnly = cfgBase;
    cfgGSBootstrapOnly.dnn.predictionResidualSource = "GS_composite";
    cfgGSBootstrapOnly.gs.enabled = true;
    cfgGSBootstrapOnly.gs.bootstrapUpload = true;
    cfgGSBootstrapOnly.gs.uploadMode = "never";
    cfgGSBootstrapOnly.gs.broadcastMode = "every_step";

    if isfield(cfgGSBootstrapOnly.gs, "useNonlocalBranchCovariance")
        cfgGSBootstrapOnly.gs.useNonlocalBranchCovariance = false;
    end

    rng(seed);
    resGSBootstrapOnly = simulate_GS_DNN_EKF(cfgGSBootstrapOnly);

    diffXhat = maxAbsDiff(resLocal.xhat, resGSBootstrapOnly.xhat);
    diffXaug = maxAbsDiff(resLocal.xhatAug, resGSBootstrapOnly.xhatAug);
    diffTheta = maxAbsDiff(resLocal.thetaHat, resGSBootstrapOnly.thetaHat);
    diffPdiag = maxAbsDiff(resLocal.Pdiag, resGSBootstrapOnly.Pdiag);

    fprintf("max |xhat local - GS bootstrap only|   = %.3e\n", diffXhat);
    fprintf("max |xaug local - GS bootstrap only|   = %.3e\n", diffXaug);
    fprintf("max |theta local - GS bootstrap only|  = %.3e\n", diffTheta);
    fprintf("max |Pdiag local - GS bootstrap only|  = %.3e\n", diffPdiag);

    fprintf("GS total uploads final = %d\n", resGSBootstrapOnly.gsNumTotalUploads(end));
    fprintf("Initial nonlocal branches used:\n");
    disp(resGSBootstrapOnly.numNonlocalBranchesUsed(1,:));
    fprintf("Final nonlocal branches used:\n");
    disp(resGSBootstrapOnly.numNonlocalBranchesUsed(end,:));

    tol = 1e-10;

    assert(diffXhat < tol, "xhat differs between local_DNN and GS bootstrap-only.");
    assert(diffXaug < tol, "xhatAug differs between local_DNN and GS bootstrap-only.");
    assert(diffTheta < tol, "thetaHat differs between local_DNN and GS bootstrap-only.");
    assert(diffPdiag < tol, "Pdiag differs between local_DNN and GS bootstrap-only.");

    assert(resGSBootstrapOnly.gsNumTotalUploads(end) == cfgBase.Nw, ...
        "Bootstrap-only GS run should have exactly Nw uploads.");

    assert(all(resGSBootstrapOnly.numNonlocalBranchesUsed(1,:) == cfgBase.Nw - 1), ...
        "Bootstrap did not activate Nw-1 nonlocal branches at k=1.");

    assert(all(resGSBootstrapOnly.numNonlocalBranchesUsed(end,:) == cfgBase.Nw - 1), ...
        "Bootstrap-only run did not keep Nw-1 nonlocal branches active.");

    details = struct();
    details.diffXhat = diffXhat;
    details.diffXaug = diffXaug;
    details.diffTheta = diffTheta;
    details.diffPdiag = diffPdiag;
    details.finalGSTotalUploads = resGSBootstrapOnly.gsNumTotalUploads(end);
    details.initialNonlocalBranchesUsed = resGSBootstrapOnly.numNonlocalBranchesUsed(1,:);
    details.finalNonlocalBranchesUsed = resGSBootstrapOnly.numNonlocalBranchesUsed(end,:);

end

function theta = makeDeterministicTheta(nTheta, scale)
% Create a deterministic nonzero theta vector without consuming randn.
%
% This keeps sanity checks independent of random stream consumption.

    theta = scale * linspace(1.0, 2.0, nTheta).';

end

function d = maxAbsDiff(a, b)
% Robust max absolute difference for arrays.

    d = max(abs(a(:) - b(:)));

end
