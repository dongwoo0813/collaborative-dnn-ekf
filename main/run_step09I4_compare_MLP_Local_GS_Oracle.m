function out = run_step09I4_compare_MLP_Local_GS_Oracle( ...
    residualFamily, seed, dtOverride, thetaStd, useQnonlocal, verbose)
%{
File:
    main/run_step09I4_compare_MLP_Local_GS_Oracle.m

Purpose:
    Step 09-I.4 GS additive MLP full simulation.

    Compare:
        1. Local MLP DNN-EKF
        2. GS additive MLP DNN-EKF
        3. Oracle residual EKF

Current default:
    residualFamily = "feedback_sat_disturbance"
    seed           = 101
    dt             = 0.5
    thetaStd       = 7.5e-5

Why thetaStd = 7.5e-5:
    Step 09-H.5b refined sweep showed this value gave the best Local MLP
    mean tracking error and best same-input residual approximation for the
    feedback_sat_disturbance benchmark.

Notes:
    This runner uses always-available measurements first. FOV dropout should
    be tested after the always-available MLP GS trunk is confirmed.

    GS composite mode is fixed to:
        cfg.gs.compositeMode = "additive"

    This is the current main GS path. Gated/hybrid modes remain ablations.
%}

    if nargin < 1 || isempty(residualFamily)
        residualFamily = "feedback_sat_disturbance";
    end

    if nargin < 2 || isempty(seed)
        seed = 101;
    end

    if nargin < 3 || isempty(dtOverride)
        dtOverride = 0.5;
    end

    if nargin < 4 || isempty(thetaStd)
        thetaStd = 7.5e-5;
    end

    if nargin < 5 || isempty(useQnonlocal)
        useQnonlocal = true;
    end

    if nargin < 6 || isempty(verbose)
        verbose = true;
    end

    residualFamily = string(residualFamily);

    addpath(genpath(pwd));
    rehash;

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 09-I.4: Local MLP / GS additive MLP / Oracle\n");
        fprintf("============================================================\n");
        fprintf("Residual family = %s\n", residualFamily);
        fprintf("Seed            = %d\n", seed);
        fprintf("dt              = %.6g\n", dtOverride);
        fprintf("thetaStd        = %.6g\n", thetaStd);
        fprintf("Qnonlocal       = %d\n\n", logical(useQnonlocal));
    end

    % ---------------------------------------------------------------------
    % Base configuration
    % ---------------------------------------------------------------------
    cfgBase = config_step04_GS_DNN_EKF();

    cfgBase.dt = dtOverride;
    cfgBase.time = 0:cfgBase.dt:cfgBase.T;
    cfgBase.N = numel(cfgBase.time);

    cfgBase.truth.useResidual = true;
    cfgBase.truth.residualFamily = residualFamily;

    % Backward-compatible alias for older helper paths.
    if residualFamily == "simple_branchwise"
        cfgBase.truth.residualModel = "branchwise";
    else
        cfgBase.truth.residualModel = residualFamily;
    end

    cfgBase.truth.residualAmp = 5e-4;

    % Always-available first. FOV comes after GS MLP trunk is confirmed.
    cfgBase.meas.availabilityMode = "always";

    if isfield(cfgBase, "fov")
        cfgBase.fov.enabled = false;
        cfgBase.fov.guardUnimplementedMode = true;
    end

    % Use the validated block covariance prediction path.
    cfgBase.ekf.useBlockCovPrediction = true;

    % ---------------------------------------------------------------------
    % MLP branch defaults from Step 09-H.5b
    % ---------------------------------------------------------------------
    cfgBase.dnn.branchModel = "mlp_general";
    cfgBase.dnn.mlp.hiddenSizes = [12 8 6];
    cfgBase.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
    cfgBase.dnn.mlp.inputMode = "eta_phase";
    cfgBase.dnn.mlp.rScale = 1000.0;
    cfgBase.dnn.mlp.vScale = 0.1;
    cfgBase.dnn.mlp.thetaInitMode = "random_hidden_zero_output";

    cfgBase.dnn.theta0_std = 0.0;
    cfgBase.dnn.residualInjectionGain = 1.0;

    % Tuned MLP parameter mobility.
    cfgBase.dnn.Ptheta0 = thetaStd^2;
    cfgBase.dnn.thetaSigmaSS = thetaStd;

    % ---------------------------------------------------------------------
    % GS defaults
    % ---------------------------------------------------------------------
    cfgBase.gs.compositeMode = "additive";
    cfgBase.gs.nonlocalWeightMode = "none";
    cfgBase.gs.nonlocalWeight = 1.0;

    cfgBase.gs.uploadMode = "after_measurement_update";
    cfgBase.gs.broadcastMode = "every_step";

    cfgBase.gs.useNonlocalBranchCovariance = logical(useQnonlocal);

    % Gate fields are harmless for additive mode, but keep them available
    % because helper functions may expect them in shared diagnostics.
    cfgBase.gate.mode = "tight_frame_2d_rt";
    cfgBase.gate.minRange = 1e-12;

    [nThetaMLP, branchInfoMLP] = branchThetaNumel(cfgBase);

    if verbose
        fprintf("MLP nTheta     = %d\n", nThetaMLP);
        fprintf("MLP layerSizes = [%s]\n\n", ...
            sprintf("%d ", branchInfoMLP.arch.layerSizes));
    end

    % ---------------------------------------------------------------------
    % Case 1: Local MLP DNN-EKF
    % ---------------------------------------------------------------------
    cfgLocal = cfgBase;

    cfgLocal.step.name = "step09I4_local_MLP";
    cfgLocal.estimator.type = "local_DNN_EKF";
    cfgLocal.dnn.predictionResidualSource = "local_DNN";

    cfgLocal.gs.enabled = false;
    cfgLocal.gs.bootstrapUpload = false;
    cfgLocal.gs.useNonlocalBranchCovariance = false;
    cfgLocal.gs.uploadMode = "none";
    cfgLocal.gs.broadcastMode = "none";

    if verbose
        fprintf("Running Case 1: Local MLP DNN-EKF\n");
    end

    rng(seed);
    resLocal = simulateLocalDNNEKF(cfgLocal);
    assertFiniteStep09I4(resLocal, "Local MLP");

    % ---------------------------------------------------------------------
    % Case 2: GS additive MLP DNN-EKF
    % ---------------------------------------------------------------------
    cfgGS = cfgBase;

    cfgGS.step.name = "step09I4_GS_additive_MLP";
    cfgGS.estimator.type = "GS_DNN_EKF";
    cfgGS.dnn.predictionResidualSource = "GS_composite";

    cfgGS.gs.enabled = true;
    cfgGS.gs.bootstrapUpload = true;
    cfgGS.gs.uploadMode = "after_measurement_update";
    cfgGS.gs.broadcastMode = "every_step";
    cfgGS.gs.compositeMode = "additive";
    cfgGS.gs.useNonlocalBranchCovariance = logical(useQnonlocal);

    if verbose
        fprintf("Running Case 2: GS additive MLP DNN-EKF\n");
    end

    rng(seed);
    resGS = simulate_GS_DNN_EKF(cfgGS);
    assertFiniteStep09I4(resGS, "GS additive MLP");

    % ---------------------------------------------------------------------
    % Case 3: Oracle residual EKF
    % ---------------------------------------------------------------------
    cfgOracle = cfgBase;

    cfgOracle.step.name = "step09I4_oracle";
    cfgOracle.estimator.type = "oracle_residual_EKF";
    cfgOracle.dnn.predictionResidualSource = "oracle";

    % Oracle uses true residual directly. Keep fixed-feature branch here so
    % the oracle reference does not carry a 256-dimensional unused theta.
    cfgOracle.dnn.branchModel = "fixed_feature_lip";

    cfgOracle.gs.enabled = false;
    cfgOracle.gs.bootstrapUpload = false;
    cfgOracle.gs.useNonlocalBranchCovariance = false;
    cfgOracle.gs.uploadMode = "none";
    cfgOracle.gs.broadcastMode = "none";

    if verbose
        fprintf("Running Case 3: Oracle residual EKF\n");
    end

    rng(seed);
    resOracle = simulateLocalDNNEKF(cfgOracle);
    assertFiniteStep09I4(resOracle, "Oracle");

    % ---------------------------------------------------------------------
    % Metrics
    % ---------------------------------------------------------------------
    metricsLocal = evaluateEstimatorMetrics( ...
        resLocal, cfgLocal, "Local MLP + always");

    metricsGS = evaluateEstimatorMetrics( ...
        resGS, cfgGS, "GS additive MLP + always");

    metricsOracle = evaluateEstimatorMetrics( ...
        resOracle, cfgOracle, "Oracle + always");

    summaryTable = [
        metricsLocal.scalarSummary
        metricsGS.scalarSummary
        metricsOracle.scalarSummary
    ];

    localMean = metricsLocal.meanPosErr;
    gsMean = metricsGS.meanPosErr;
    oracleMean = metricsOracle.meanPosErr;

    localRms = metricsLocal.rmsPosErr;
    gsRms = metricsGS.rmsPosErr;
    oracleRms = metricsOracle.rmsPosErr;

    localP95 = metricsLocal.p95PosErr;
    gsP95 = metricsGS.p95PosErr;
    oracleP95 = metricsOracle.p95PosErr;

    gsMeanImprovementPct = percentImprovementStep09I4(localMean, gsMean);
    oracleMeanImprovementPct = percentImprovementStep09I4(localMean, oracleMean);

    gsRmsImprovementPct = percentImprovementStep09I4(localRms, gsRms);
    oracleRmsImprovementPct = percentImprovementStep09I4(localRms, oracleRms);

    gsP95ImprovementPct = percentImprovementStep09I4(localP95, gsP95);
    oracleP95ImprovementPct = percentImprovementStep09I4(localP95, oracleP95);

    localOracleGap = localMean - oracleMean;

    if isfinite(localOracleGap) && localOracleGap > 0
        gsClosedLocalOracleGapPct = 100 * (localMean - gsMean) / localOracleGap;
    else
        gsClosedLocalOracleGapPct = NaN;
    end

    compactComparisonTable = table( ...
        residualFamily, ...
        seed, ...
        dtOverride, ...
        thetaStd, ...
        logical(useQnonlocal), ...
        nThetaMLP, ...
        localMean, ...
        gsMean, ...
        oracleMean, ...
        gsMeanImprovementPct, ...
        oracleMeanImprovementPct, ...
        gsClosedLocalOracleGapPct, ...
        localRms, ...
        gsRms, ...
        oracleRms, ...
        gsRmsImprovementPct, ...
        oracleRmsImprovementPct, ...
        localP95, ...
        gsP95, ...
        oracleP95, ...
        gsP95ImprovementPct, ...
        oracleP95ImprovementPct, ...
        metricsLocal.meanNIS, ...
        metricsGS.meanNIS, ...
        metricsOracle.meanNIS, ...
        metricsGS.gsLoggedUploadDecisions, ...
        metricsGS.gsFinalTotalUploads, ...
        'VariableNames', { ...
            'residualFamily', ...
            'seed', ...
            'dt', ...
            'thetaStd', ...
            'useQnonlocal', ...
            'nThetaMLP', ...
            'localMeanPosErr_m', ...
            'gsMeanPosErr_m', ...
            'oracleMeanPosErr_m', ...
            'gsMeanImprovementPct_vsLocal', ...
            'oracleMeanImprovementPct_vsLocal', ...
            'gsClosedLocalOracleGapPct', ...
            'localRmsPosErr_m', ...
            'gsRmsPosErr_m', ...
            'oracleRmsPosErr_m', ...
            'gsRmsImprovementPct_vsLocal', ...
            'oracleRmsImprovementPct_vsLocal', ...
            'localP95PosErr_m', ...
            'gsP95PosErr_m', ...
            'oracleP95PosErr_m', ...
            'gsP95ImprovementPct_vsLocal', ...
            'oracleP95ImprovementPct_vsLocal', ...
            'localMeanNIS', ...
            'gsMeanNIS', ...
            'oracleMeanNIS', ...
            'gsLoggedUploadDecisions', ...
            'gsFinalTotalUploads'});

    % ---------------------------------------------------------------------
    % Output structure
    % ---------------------------------------------------------------------
    out = struct();

    out.residualFamily = residualFamily;
    out.seed = seed;
    out.dt = dtOverride;
    out.thetaStd = thetaStd;
    out.useQnonlocal = logical(useQnonlocal);

    out.cfgBase = cfgBase;
    out.cfgLocal = cfgLocal;
    out.cfgGS = cfgGS;
    out.cfgOracle = cfgOracle;

    out.resLocal = resLocal;
    out.resGS = resGS;
    out.resOracle = resOracle;

    out.metricsLocal = metricsLocal;
    out.metricsGS = metricsGS;
    out.metricsOracle = metricsOracle;

    out.summaryTable = summaryTable;
    out.multiMetricSummaryTable = summaryTable;
    out.compactComparisonTable = compactComparisonTable;

    out.gsMeanImprovementPct = gsMeanImprovementPct;
    out.oracleMeanImprovementPct = oracleMeanImprovementPct;
    out.gsClosedLocalOracleGapPct = gsClosedLocalOracleGapPct;

    % Reuse existing residual diagnostic, because it already compares
    % operational and same-input residual error for Local / GS / Oracle.
    out.residualDiag = run_step09d3_residual_input_mode_diagnostics(out);

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 09-I.4 multi-metric summary\n");
        fprintf("============================================================\n");
        disp(summaryTable);

        fprintf("\nCompact comparison:\n");
        disp(compactComparisonTable);

        fprintf("GS mean improvement over Local       = %.3f %%\n", ...
            gsMeanImprovementPct);
        fprintf("Oracle mean improvement over Local   = %.3f %%\n", ...
            oracleMeanImprovementPct);
        fprintf("GS closed Local-to-Oracle gap        = %.3f %%\n", ...
            gsClosedLocalOracleGapPct);

        fprintf("\nResidual diagnostic overall summary:\n");
        disp(out.residualDiag.overallSummaryTable);
    end

end

function assertFiniteStep09I4(results, caseName)
%ASSERTFINITESTEP09I4 Lightweight full-run finite-output check.
%
% This check is intentionally small. It only catches the main failure modes
% expected when moving GS sharing to MLP branches:
%   NaN/Inf state estimate
%   NaN/Inf theta estimate
%   NaN/Inf covariance diagonal
%   no finite NIS values

    caseName = string(caseName);

    if ~all(isfinite(results.xhat(:)))
        error("Step09I4:NonFiniteXhat", ...
            "%s produced non-finite entries in results.xhat.", caseName);
    end

    if isfield(results, "thetaHat")
        if ~all(isfinite(results.thetaHat(:)))
            error("Step09I4:NonFiniteThetaHat", ...
                "%s produced non-finite entries in results.thetaHat.", caseName);
        end
    end

    if isfield(results, "Pdiag")
        if ~all(isfinite(results.Pdiag(:)))
            error("Step09I4:NonFinitePdiag", ...
                "%s produced non-finite entries in results.Pdiag.", caseName);
        end
    end

    if isfield(results, "NIS")
        nisFinite = results.NIS(isfinite(results.NIS));

        if isempty(nisFinite)
            error("Step09I4:NoFiniteNIS", ...
                "%s produced no finite NIS entries.", caseName);
        end

        if any(nisFinite < -1e-10)
            error("Step09I4:NegativeNIS", ...
                "%s produced negative finite NIS entries.", caseName);
        end
    end

end

function pct = percentImprovementStep09I4(baselineValue, testValue)
%PERCENTIMPROVEMENTSTEP09I4 Positive means testValue is smaller.

    if isfinite(baselineValue) && baselineValue > 0 && isfinite(testValue)
        pct = 100 * (baselineValue - testValue) / baselineValue;
    else
        pct = NaN;
    end

end