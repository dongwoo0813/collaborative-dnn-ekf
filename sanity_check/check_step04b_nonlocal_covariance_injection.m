function out = check_step04b_nonlocal_covariance_injection()
%{
File:
    sanity_check/check_step04b_nonlocal_covariance_injection.m

Purpose:
    Sanity checks for Step 04b nonlocal GS-branch covariance injection.

This file runs two checks:

    Check 1: OFF equivalence
        Compare:
            cfg.gs.useNonlocalBranchCovariance = false
        against:
            cfg.gs.useNonlocalBranchCovariance field removed

        Expected:
            The two runs must be identical.

        Meaning:
            The new Step 04b covariance-injection code has no effect when
            disabled.

    Check 2: ON activation
        Compare:
            cfg.gs.useNonlocalBranchCovariance = false
        against:
            cfg.gs.useNonlocalBranchCovariance = true

        Expected:
            lastNonlocalCovInjection.enabled = true
            numActiveNonlocal = Nw - 1
            traceSdNonlocal > 0
            traceQnonlocal > 0
            Pdiag changes relative to OFF case

        Meaning:
            The Step 04b nonlocal covariance term is actually being computed
            and injected into the EKF covariance prediction.

How to run:
    From project root:

        out = check_step04b_nonlocal_covariance_injection();

Assumptions:
    - config_step04_GS_DNN_EKF.m exists.
    - simulate_GS_DNN_EKF.m exists.
    - DNN_EKF_Predict_Local.m already calls
      computeNonlocalBranchCovarianceInjection.m.
    - watcher.lastNonlocalCovInjection is saved inside
      DNN_EKF_Predict_Local.m after prediction.
%}

    clearvars -except out;
    clc;

    addpath(genpath(pwd));
    rehash;

    fprintf("\n============================================================\n");
    fprintf("Step 04b sanity check: nonlocal covariance injection\n");
    fprintf("============================================================\n");

    assert(exist("config_step04_GS_DNN_EKF", "file") == 2, ...
        "Missing config_step04_GS_DNN_EKF.m.");

    assert(exist("simulate_GS_DNN_EKF", "file") == 2, ...
        "Missing simulate_GS_DNN_EKF.m.");

    assert(exist("computeNonlocalBranchCovarianceInjection", "file") == 2, ...
        "Missing computeNonlocalBranchCovarianceInjection.m.");

    assert(exist("DNN_EKF_Predict_Local", "file") == 2, ...
        "Missing DNN_EKF_Predict_Local.m.");

    out = struct();

    out.offEquivalence = runOffEquivalenceCheck();
    out.onActivation   = runOnActivationCheck();

    fprintf("\n============================================================\n");
    fprintf("All Step 04b nonlocal covariance sanity checks passed.\n");
    fprintf("============================================================\n");

end

function out = runOffEquivalenceCheck()
% Check that false and missing-field cases are exactly equivalent.

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Check 1: OFF equivalence\n");
    fprintf("------------------------------------------------------------\n");

    seed = 100;

    cfgFalse = makeBaseStep04bConfig();
    cfgFalse.gs.useNonlocalBranchCovariance = false;

    cfgNoField = cfgFalse;
    if isfield(cfgNoField, "gs") && isfield(cfgNoField.gs, "useNonlocalBranchCovariance")
        cfgNoField.gs = rmfield(cfgNoField.gs, "useNonlocalBranchCovariance");
    end

    rng(seed);
    resFalse = simulate_GS_DNN_EKF(cfgFalse);

    rng(seed);
    resNoField = simulate_GS_DNN_EKF(cfgNoField);

    diff_xhat  = maxAbsDiff(resFalse.xhat,     resNoField.xhat);
    diff_xaug  = maxAbsDiff(resFalse.xhatAug,  resNoField.xhatAug);
    diff_theta = maxAbsDiff(resFalse.thetaHat, resNoField.thetaHat);
    diff_Pdiag = maxAbsDiff(resFalse.Pdiag,    resNoField.Pdiag);

    fprintf("max |xhat false - no-field|   = %.3e\n", diff_xhat);
    fprintf("max |xaug false - no-field|   = %.3e\n", diff_xaug);
    fprintf("max |theta false - no-field|  = %.3e\n", diff_theta);
    fprintf("max |Pdiag false - no-field|  = %.3e\n", diff_Pdiag);

    tol = 1e-10;

    assert(diff_xhat < tol, ...
        "OFF equivalence failed: xhat changed.");

    assert(diff_xaug < tol, ...
        "OFF equivalence failed: xhatAug changed.");

    assert(diff_theta < tol, ...
        "OFF equivalence failed: thetaHat changed.");

    assert(diff_Pdiag < tol, ...
        "OFF equivalence failed: Pdiag changed.");

    out = struct();
    out.passed = true;
    out.diff_xhat = diff_xhat;
    out.diff_xaug = diff_xaug;
    out.diff_theta = diff_theta;
    out.diff_Pdiag = diff_Pdiag;

    fprintf("Check 1 passed: OFF gate is clean.\n");

end

function out = runOnActivationCheck()
% Check that enabling Step 04b actually injects nonlocal covariance.

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Check 2: ON activation\n");
    fprintf("------------------------------------------------------------\n");

    seed = 100;

    cfgOff = makeBaseStep04bConfig();
    cfgOff.gs.useNonlocalBranchCovariance = false;

    cfgOn = cfgOff;
    cfgOn.gs.useNonlocalBranchCovariance = true;
    cfgOn.gs.youngMode = "uniform";
    cfgOn.gs.SresNonlocal = 0.0;

    rng(seed);
    resOff = simulate_GS_DNN_EKF(cfgOff);

    rng(seed);
    resOn = simulate_GS_DNN_EKF(cfgOn);

    Nw = cfgOn.Nw;

    enabledVec = false(Nw, 1);
    numActiveVec = zeros(Nw, 1);
    traceSdVec = zeros(Nw, 1);
    traceQVec = zeros(Nw, 1);

    for i = 1:Nw

        watcher = resOn.watchersFinal(i);

        assert(isfield(watcher, "lastNonlocalCovInjection"), ...
            "ON activation failed: watcher %d has no lastNonlocalCovInjection field.", i);

        diagInfo = watcher.lastNonlocalCovInjection;

        assert(isfield(diagInfo, "enabled"), ...
            "ON activation failed: watcher %d diagInfo has no enabled field.", i);

        assert(isfield(diagInfo, "numActiveNonlocal"), ...
            "ON activation failed: watcher %d diagInfo has no numActiveNonlocal field.", i);

        assert(isfield(diagInfo, "traceSdNonlocal"), ...
            "ON activation failed: watcher %d diagInfo has no traceSdNonlocal field.", i);

        assert(isfield(diagInfo, "traceQnonlocal"), ...
            "ON activation failed: watcher %d diagInfo has no traceQnonlocal field.", i);

        enabledVec(i) = logical(diagInfo.enabled);
        numActiveVec(i) = diagInfo.numActiveNonlocal;
        traceSdVec(i) = diagInfo.traceSdNonlocal;
        traceQVec(i) = diagInfo.traceQnonlocal;

    end

    maxDiffPdiag = maxAbsDiff(resOn.Pdiag, resOff.Pdiag);
    maxDiffXhat  = maxAbsDiff(resOn.xhat,  resOff.xhat);

    fprintf("enabled flags:\n");
    disp(enabledVec.');

    fprintf("numActiveNonlocal:\n");
    disp(numActiveVec.');

    fprintf("traceSdNonlocal:\n");
    disp(traceSdVec.');

    fprintf("traceQnonlocal:\n");
    disp(traceQVec.');

    fprintf("max |Pdiag ON - OFF| = %.3e\n", maxDiffPdiag);
    fprintf("max |xhat  ON - OFF| = %.3e\n", maxDiffXhat);

    assert(all(enabledVec), ...
        "ON activation failed: not all watchers have enabled = true.");

    assert(all(numActiveVec == Nw - 1), ...
        "ON activation failed: not all watchers see Nw-1 active nonlocal branches.");

    assert(all(traceSdVec > 0), ...
        "ON activation failed: traceSdNonlocal is not positive for all watchers.");

    assert(all(traceQVec > 0), ...
        "ON activation failed: traceQnonlocal is not positive for all watchers.");

    assert(maxDiffPdiag > 0, ...
        "ON activation failed: Pdiag did not change when covariance injection was enabled.");

    out = struct();
    out.passed = true;
    out.enabledVec = enabledVec;
    out.numActiveVec = numActiveVec;
    out.traceSdVec = traceSdVec;
    out.traceQVec = traceQVec;
    out.maxDiffPdiag = maxDiffPdiag;
    out.maxDiffXhat = maxDiffXhat;

    fprintf("Check 2 passed: nonlocal covariance injection is active.\n");

end

function cfg = makeBaseStep04bConfig()
% Create a consistent Step 04 GS-composite configuration for sanity checks.

    cfg = config_step04_GS_DNN_EKF();

    cfg.truth.residualAmp = 5e-4;

    cfg.dnn.theta0_std = 0.0;
    cfg.dnn.residualInjectionGain = 1.0;
    cfg.dnn.predictionResidualSource = "GS_composite";

    cfg.gs.enabled = true;
    cfg.gs.bootstrapUpload = true;
    cfg.gs.uploadMode = "after_measurement_update";
    cfg.gs.broadcastMode = "every_step";

    if ~isfield(cfg.gs, "youngMode")
        cfg.gs.youngMode = "uniform";
    end

    if ~isfield(cfg.gs, "SresNonlocal")
        cfg.gs.SresNonlocal = 0.0;
    end

end

function d = maxAbsDiff(a, b)
% Return max absolute difference between two numeric arrays.

    d = max(abs(a(:) - b(:)));

end