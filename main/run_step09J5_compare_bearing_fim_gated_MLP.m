function out = run_step09J5_compare_bearing_fim_gated_MLP( ...
    residualFamily, seed, dtOverride, thetaStd, useQnonlocal, makePlots, verbose)
%{
File:
    main/run_step09J5_compare_bearing_fim_gated_MLP.m

Purpose:
    Step 09-J.5 full simulation comparison for the new direction-only
    bearing-FIM-gated GS residual fusion.

Compare:
    1. Local MLP DNN-EKF
    2. GS additive MLP DNN-EKF
    3. GS bearing_fim_gated MLP DNN-EKF
    4. Oracle residual EKF

Main question:
    Does bearing_fim_gated reduce the additive-GS branch-interference /
    wrong-direction nonlocal correction problem, especially around
    900--1200 s?

Benchmark default:
    residualFamily = "feedback_sat_disturbance"
    seed           = 101
    dt             = 0.5
    thetaStd       = 7.5e-5

Why always-available first:
    This isolates the residual-fusion effect from FOV dropout effects. FOV
    should be tested after the always-available bearing-FIM-gated GS trunk
    is confirmed.

Diagnostics:
    1. Multi-metric tracking summary.
    2. Residual operational/same-input diagnostic.
    3. Branch-contribution alignment diagnostic using

           e_L = d_local_component - d_true,
           n   = d_nonlocal_component,

       and

           ||e_L+n||^2 - ||e_L||^2
             = 2 e_L' n + ||n||^2.

Notes:
    Step 09-J.3 already made the residual mean use B_{j|m}.
    Step 09-J.4 already made Qnonlocal use the same B_{j|m}.
    This runner tests the closed-loop simulation behavior after both changes.
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

    if nargin < 6 || isempty(makePlots)
        makePlots = false;
    end

    if nargin < 7 || isempty(verbose)
        verbose = true;
    end

    residualFamily = string(residualFamily);

    addpath(genpath(pwd));
    rehash;

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 09-J.5: additive GS vs bearing-FIM-gated GS MLP\n");
        fprintf("============================================================\n");
        fprintf("Residual family = %s\n", residualFamily);
        fprintf("Seed            = %d\n", seed);
        fprintf("dt              = %.6g\n", dtOverride);
        fprintf("thetaStd        = %.6g\n", thetaStd);
        fprintf("Qnonlocal       = %d\n", logical(useQnonlocal));
        fprintf("makePlots       = %d\n\n", logical(makePlots));
    end

    % ---------------------------------------------------------------------
    % Base configuration.
    % ---------------------------------------------------------------------
    cfgBase = config_step04_GS_DNN_EKF();

    cfgBase.dt = dtOverride;
    cfgBase.time = 0:cfgBase.dt:cfgBase.T;
    cfgBase.N = numel(cfgBase.time);

    cfgBase.truth.useResidual = true;
    cfgBase.truth.residualFamily = residualFamily;

    % Backward-compatible alias for older truth helper paths.
    if residualFamily == "simple_branchwise"
        cfgBase.truth.residualModel = "branchwise";
    else
        cfgBase.truth.residualModel = residualFamily;
    end

    cfgBase.truth.residualAmp = 5e-4;

    % Always-available first. This isolates residual-fusion behavior.
    cfgBase.meas.availabilityMode = "always";

    if isfield(cfgBase, "fov")
        cfgBase.fov.enabled = false;
        cfgBase.fov.guardUnimplementedMode = true;
    end

    % Keep the validated block covariance prediction path.
    cfgBase.ekf.useBlockCovPrediction = true;

    % ---------------------------------------------------------------------
    % MLP branch setting from the current best Local MLP configuration.
    % ---------------------------------------------------------------------
    cfgBase.dnn.branchModel = "mlp_general";
    cfgBase.dnn.mlp.hiddenSizes = [6 6 6];
    cfgBase.dnn.mlp.activations = ["softplus", "softplus", "tanh"];
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
    % GS defaults shared by additive and bearing_fim_gated.
    % ---------------------------------------------------------------------
    cfgBase.gs.nonlocalWeightMode = "none";
    cfgBase.gs.nonlocalWeight = 1.0;

    cfgBase.gs.uploadMode = "after_measurement_update";
    cfgBase.gs.broadcastMode = "every_step";

    cfgBase.gs.useNonlocalBranchCovariance = logical(useQnonlocal);
    cfgBase.gs.youngMode = "uniform";

    % Bearing-FIM gate defaults. These are active only when
    % cfg.gs.compositeMode = "bearing_fim_gated".
    cfgBase.gs.fimGate.enabled = true;
    cfgBase.gs.fimGate.lambdaOmega = 0.02;
    cfgBase.gs.fimGate.epsilon = 1e-6;
    cfgBase.gs.fimGate.normalizeTrace = false;
    cfgBase.gs.fimGate.outputFrame = "inertial";

    % Old tight-frame gate fields remain for older ablation modes and shared
    % helper compatibility. They are not used by bearing_fim_gated.
    cfgBase.gate.mode = "tight_frame_2d_rt";
    cfgBase.gate.minRange = 1e-12;

    [nThetaMLP, branchInfoMLP] = branchThetaNumel(cfgBase);

    if verbose
        fprintf("MLP nTheta     = %d\n", nThetaMLP);
        fprintf("MLP layerSizes = [%s]\n\n", ...
            sprintf("%d ", branchInfoMLP.arch.layerSizes));
    end

    % ---------------------------------------------------------------------
    % Case 1: Local MLP DNN-EKF.
    % ---------------------------------------------------------------------
    cfgLocal = cfgBase;

    cfgLocal.step.name = "step09J5_local_MLP";
    cfgLocal.estimator.type = "local_DNN_EKF";
    cfgLocal.dnn.predictionResidualSource = "local_DNN";

    cfgLocal.gs.enabled = false;
    cfgLocal.gs.bootstrapUpload = false;
    cfgLocal.gs.useNonlocalBranchCovariance = false;
    cfgLocal.gs.uploadMode = "none";
    cfgLocal.gs.broadcastMode = "none";
    cfgLocal.gs.compositeMode = "additive";

    if verbose
        fprintf("Running Case 1: Local MLP DNN-EKF\n");
    end

    rng(seed);
    resLocal = simulateLocalDNNEKF(cfgLocal);
    assertFiniteStep09J5(resLocal, "Local MLP");

    % ---------------------------------------------------------------------
    % Case 2: GS additive MLP.
    % ---------------------------------------------------------------------
    cfgGSAdd = cfgBase;

    cfgGSAdd.step.name = "step09J5_GS_additive_MLP";
    cfgGSAdd.estimator.type = "GS_DNN_EKF";
    cfgGSAdd.dnn.predictionResidualSource = "GS_composite";

    cfgGSAdd.gs.enabled = true;
    cfgGSAdd.gs.bootstrapUpload = true;
    cfgGSAdd.gs.uploadMode = "after_measurement_update";
    cfgGSAdd.gs.broadcastMode = "every_step";
    cfgGSAdd.gs.compositeMode = "additive";
    cfgGSAdd.gs.useNonlocalBranchCovariance = logical(useQnonlocal);

    if verbose
        fprintf("Running Case 2: GS additive MLP DNN-EKF\n");
    end

    rng(seed);
    resGSAdd = simulate_GS_DNN_EKF(cfgGSAdd);
    assertFiniteStep09J5(resGSAdd, "GS additive MLP");

    % ---------------------------------------------------------------------
    % Case 3: GS bearing-FIM-gated MLP.
    % ---------------------------------------------------------------------
    cfgGSFIM = cfgBase;

    cfgGSFIM.step.name = "step09J5_GS_bearing_FIM_gated_MLP";
    cfgGSFIM.estimator.type = "GS_DNN_EKF";
    cfgGSFIM.dnn.predictionResidualSource = "GS_composite";

    cfgGSFIM.gs.enabled = true;
    cfgGSFIM.gs.bootstrapUpload = true;
    cfgGSFIM.gs.uploadMode = "after_measurement_update";
    cfgGSFIM.gs.broadcastMode = "every_step";
    cfgGSFIM.gs.compositeMode = "bearing_fim_gated";
    cfgGSFIM.gs.useNonlocalBranchCovariance = logical(useQnonlocal);

    if verbose
        fprintf("Running Case 3: GS bearing_fim_gated MLP DNN-EKF\n");
    end

    rng(seed);
    resGSFIM = simulate_GS_DNN_EKF(cfgGSFIM);
    assertFiniteStep09J5(resGSFIM, "GS bearing_fim_gated MLP");

    % ---------------------------------------------------------------------
    % Case 4: Oracle residual EKF.
    % ---------------------------------------------------------------------
    cfgOracle = cfgBase;

    cfgOracle.step.name = "step09J5_oracle";
    cfgOracle.estimator.type = "oracle_residual_EKF";
    cfgOracle.dnn.predictionResidualSource = "oracle";

    % Oracle uses true residual directly. Use fixed-feature branch to avoid
    % carrying an unused 256-dimensional MLP theta in the oracle reference.
    cfgOracle.dnn.branchModel = "fixed_feature_lip";

    cfgOracle.gs.enabled = false;
    cfgOracle.gs.bootstrapUpload = false;
    cfgOracle.gs.useNonlocalBranchCovariance = false;
    cfgOracle.gs.uploadMode = "none";
    cfgOracle.gs.broadcastMode = "none";
    cfgOracle.gs.compositeMode = "additive";

    if verbose
        fprintf("Running Case 4: Oracle residual EKF\n");
    end

    rng(seed);
    resOracle = simulateLocalDNNEKF(cfgOracle);
    assertFiniteStep09J5(resOracle, "Oracle");

    % ---------------------------------------------------------------------
    % Tracking metrics.
    % ---------------------------------------------------------------------
    metricsLocal = evaluateEstimatorMetrics( ...
        resLocal, cfgLocal, "Local MLP + always");

    metricsGSAdd = evaluateEstimatorMetrics( ...
        resGSAdd, cfgGSAdd, "GS additive MLP + always");

    metricsGSFIM = evaluateEstimatorMetrics( ...
        resGSFIM, cfgGSFIM, "GS bearing-FIM-gated MLP + always");

    metricsOracle = evaluateEstimatorMetrics( ...
        resOracle, cfgOracle, "Oracle + always");

    trackingSummary = [
        metricsLocal.scalarSummary
        metricsGSAdd.scalarSummary
        metricsGSFIM.scalarSummary
        metricsOracle.scalarSummary
    ];

    compactTrackingSummary = buildCompactTrackingTableStep09J5( ...
        residualFamily, seed, dtOverride, thetaStd, logical(useQnonlocal), ...
        nThetaMLP, metricsLocal, metricsGSAdd, metricsGSFIM, metricsOracle);

    % ---------------------------------------------------------------------
    % Residual operational/same-input diagnostics.
    %
    % The existing helper expects a structure with fields:
    %   resLocal, resGS, resOracle, cfgLocal, cfgGS, cfgOracle.
    % We call it twice: once with additive GS, once with FIM-gated GS.
    % ---------------------------------------------------------------------
    outAddForResidualDiag = struct();
    outAddForResidualDiag.resLocal = resLocal;
    outAddForResidualDiag.resGS = resGSAdd;
    outAddForResidualDiag.resOracle = resOracle;
    outAddForResidualDiag.cfgLocal = cfgLocal;
    outAddForResidualDiag.cfgGS = cfgGSAdd;
    outAddForResidualDiag.cfgOracle = cfgOracle;

    outFIMForResidualDiag = struct();
    outFIMForResidualDiag.resLocal = resLocal;
    outFIMForResidualDiag.resGS = resGSFIM;
    outFIMForResidualDiag.resOracle = resOracle;
    outFIMForResidualDiag.cfgLocal = cfgLocal;
    outFIMForResidualDiag.cfgGS = cfgGSFIM;
    outFIMForResidualDiag.cfgOracle = cfgOracle;

    residualDiagAdd = run_step09d3_residual_input_mode_diagnostics( ...
        outAddForResidualDiag);

    residualDiagFIM = run_step09d3_residual_input_mode_diagnostics( ...
        outFIMForResidualDiag);

    residualOverallComparison = buildResidualOverallComparisonStep09J5( ...
        residualDiagAdd, residualDiagFIM);

    residualWindowComparison = buildResidualWindowComparisonStep09J5( ...
        residualDiagAdd, residualDiagFIM);

    % ---------------------------------------------------------------------
    % Branch-contribution alignment diagnostics.
    %
    % This directly targets the known additive-GS issue:
    %   positive deltaSq / high harmful fraction around 900--1200 s.
    % ---------------------------------------------------------------------
    windowEdges = [0 800 900 1200 1400 1500 2000];

    diagAlignAdd = run_step09I5_branch_alignment_diagnostics( ...
        resGSAdd, "all", windowEdges, makePlots);

    diagAlignFIM = run_step09I5_branch_alignment_diagnostics( ...
        resGSFIM, "all", windowEdges, makePlots);

    alignmentWindowComparison = buildAlignmentWindowComparisonStep09J5( ...
        diagAlignAdd.windowSummaryTable, ...
        diagAlignFIM.windowSummaryTable);

    % ---------------------------------------------------------------------
    % Output.
    % ---------------------------------------------------------------------
    out = struct();

    out.residualFamily = residualFamily;
    out.seed = seed;
    out.dt = dtOverride;
    out.thetaStd = thetaStd;
    out.useQnonlocal = logical(useQnonlocal);
    out.nThetaMLP = nThetaMLP;

    out.cfgBase = cfgBase;
    out.cfgLocal = cfgLocal;
    out.cfgGSAdd = cfgGSAdd;
    out.cfgGSFIM = cfgGSFIM;
    out.cfgOracle = cfgOracle;

    out.resLocal = resLocal;
    out.resGSAdd = resGSAdd;
    out.resGSFIM = resGSFIM;
    out.resOracle = resOracle;

    % Aliases for compatibility with older diagnostics that expect resGS.
    out.resGS = resGSFIM;
    out.cfgGS = cfgGSFIM;

    out.metricsLocal = metricsLocal;
    out.metricsGSAdd = metricsGSAdd;
    out.metricsGSFIM = metricsGSFIM;
    out.metricsOracle = metricsOracle;

    out.trackingSummary = trackingSummary;
    out.compactTrackingSummary = compactTrackingSummary;

    out.residualDiagAdd = residualDiagAdd;
    out.residualDiagFIM = residualDiagFIM;
    out.residualOverallComparison = residualOverallComparison;
    out.residualWindowComparison = residualWindowComparison;

    out.diagAlignAdd = diagAlignAdd;
    out.diagAlignFIM = diagAlignFIM;
    out.alignmentWindowComparison = alignmentWindowComparison;

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 09-J.5 tracking summary\n");
        fprintf("============================================================\n");
        disp(trackingSummary);

        fprintf("\nCompact tracking comparison:\n");
        disp(compactTrackingSummary);

        fprintf("\nResidual overall comparison:\n");
        disp(residualOverallComparison);

        fprintf("\nResidual window comparison:\n");
        disp(residualWindowComparison);

        fprintf("\nAlignment window comparison:\n");
        disp(alignmentWindowComparison);

        printKeyWindowStep09J5(alignmentWindowComparison, "900-1200 s");
    end

end

function compactTable = buildCompactTrackingTableStep09J5( ...
    residualFamily, seed, dtOverride, thetaStd, useQnonlocal, nThetaMLP, ...
    metricsLocal, metricsGSAdd, metricsGSFIM, metricsOracle)
%BUILDCOMPACTTRACKINGTABLESTEP09J5 Build compact tracking comparison table.

    caseName = [
        "Local MLP"
        "GS additive MLP"
        "GS bearing-FIM-gated MLP"
        "Oracle"
    ];

    compositeMode = [
        "local"
        "additive"
        "bearing_fim_gated"
        "oracle"
    ];

    meanPosErr_m = [
        metricsLocal.meanPosErr
        metricsGSAdd.meanPosErr
        metricsGSFIM.meanPosErr
        metricsOracle.meanPosErr
    ];

    rmsPosErr_m = [
        metricsLocal.rmsPosErr
        metricsGSAdd.rmsPosErr
        metricsGSFIM.rmsPosErr
        metricsOracle.rmsPosErr
    ];

    medianPosErr_m = [
        metricsLocal.medianPosErr
        metricsGSAdd.medianPosErr
        metricsGSFIM.medianPosErr
        metricsOracle.medianPosErr
    ];

    p95PosErr_m = [
        metricsLocal.p95PosErr
        metricsGSAdd.p95PosErr
        metricsGSFIM.p95PosErr
        metricsOracle.p95PosErr
    ];

    finalMeanPosErr_m = [
        metricsLocal.finalMeanPosErr
        metricsGSAdd.finalMeanPosErr
        metricsGSFIM.finalMeanPosErr
        metricsOracle.finalMeanPosErr
    ];

    meanNIS = [
        metricsLocal.meanNIS
        metricsGSAdd.meanNIS
        metricsGSFIM.meanNIS
        metricsOracle.meanNIS
    ];

    availabilityRatePct = 100.0 * [
        metricsLocal.availabilityRate
        metricsGSAdd.availabilityRate
        metricsGSFIM.availabilityRate
        metricsOracle.availabilityRate
    ];

    gsFinalTotalUploads = [
        NaN
        metricsGSAdd.gsFinalTotalUploads
        metricsGSFIM.gsFinalTotalUploads
        NaN
    ];

    improvementMeanPct_vsLocal = 100.0 * ...
        (metricsLocal.meanPosErr - meanPosErr_m) ./ ...
        max(metricsLocal.meanPosErr, eps);

    improvementRmsPct_vsLocal = 100.0 * ...
        (metricsLocal.rmsPosErr - rmsPosErr_m) ./ ...
        max(metricsLocal.rmsPosErr, eps);

    improvementP95Pct_vsLocal = 100.0 * ...
        (metricsLocal.p95PosErr - p95PosErr_m) ./ ...
        max(metricsLocal.p95PosErr, eps);

    meanDeltaPct_FIM_vsAdditive = NaN(4, 1);
    rmsDeltaPct_FIM_vsAdditive = NaN(4, 1);
    p95DeltaPct_FIM_vsAdditive = NaN(4, 1);

    meanDeltaPct_FIM_vsAdditive(3) = 100.0 * ...
        (metricsGSAdd.meanPosErr - metricsGSFIM.meanPosErr) / ...
        max(metricsGSAdd.meanPosErr, eps);

    rmsDeltaPct_FIM_vsAdditive(3) = 100.0 * ...
        (metricsGSAdd.rmsPosErr - metricsGSFIM.rmsPosErr) / ...
        max(metricsGSAdd.rmsPosErr, eps);

    p95DeltaPct_FIM_vsAdditive(3) = 100.0 * ...
        (metricsGSAdd.p95PosErr - metricsGSFIM.p95PosErr) / ...
        max(metricsGSAdd.p95PosErr, eps);

    localOracleGap = metricsLocal.meanPosErr - metricsOracle.meanPosErr;

    closedLocalOracleGapPct = NaN(4, 1);

    if isfinite(localOracleGap) && localOracleGap > 0
        closedLocalOracleGapPct = 100.0 * ...
            (metricsLocal.meanPosErr - meanPosErr_m) / localOracleGap;
    end

    residualFamilyCol = repmat(string(residualFamily), 4, 1);
    seedCol = repmat(seed, 4, 1);
    dtCol = repmat(dtOverride, 4, 1);
    thetaStdCol = repmat(thetaStd, 4, 1);
    useQnonlocalCol = repmat(logical(useQnonlocal), 4, 1);
    nThetaMLPCol = repmat(nThetaMLP, 4, 1);

    compactTable = table( ...
        residualFamilyCol, ...
        seedCol, ...
        dtCol, ...
        thetaStdCol, ...
        useQnonlocalCol, ...
        nThetaMLPCol, ...
        caseName, ...
        compositeMode, ...
        meanPosErr_m, ...
        rmsPosErr_m, ...
        medianPosErr_m, ...
        p95PosErr_m, ...
        finalMeanPosErr_m, ...
        meanNIS, ...
        availabilityRatePct, ...
        gsFinalTotalUploads, ...
        improvementMeanPct_vsLocal, ...
        improvementRmsPct_vsLocal, ...
        improvementP95Pct_vsLocal, ...
        meanDeltaPct_FIM_vsAdditive, ...
        rmsDeltaPct_FIM_vsAdditive, ...
        p95DeltaPct_FIM_vsAdditive, ...
        closedLocalOracleGapPct, ...
        'VariableNames', { ...
            'residualFamily', ...
            'seed', ...
            'dt', ...
            'thetaStd', ...
            'useQnonlocal', ...
            'nThetaMLP', ...
            'caseName', ...
            'compositeMode', ...
            'meanPosErr_m', ...
            'rmsPosErr_m', ...
            'medianPosErr_m', ...
            'p95PosErr_m', ...
            'finalMeanPosErr_m', ...
            'meanNIS', ...
            'availabilityRatePct', ...
            'gsFinalTotalUploads', ...
            'improvementMeanPct_vsLocal', ...
            'improvementRmsPct_vsLocal', ...
            'improvementP95Pct_vsLocal', ...
            'meanDeltaPct_FIM_vsAdditive', ...
            'rmsDeltaPct_FIM_vsAdditive', ...
            'p95DeltaPct_FIM_vsAdditive', ...
            'closedLocalOracleGapPct'});

end

function residualOverallComparison = buildResidualOverallComparisonStep09J5( ...
    residualDiagAdd, residualDiagFIM)
%BUILDRESIDUALOVERALLCOMPARISONSTEP09J5 Compare residual errors by mode.

    rowAdd = selectResidualCaseRowStep09J5( ...
        residualDiagAdd.overallSummaryTable, "GS composite");

    rowFIM = selectResidualCaseRowStep09J5( ...
        residualDiagFIM.overallSummaryTable, "GS composite");

    caseName = [
        "GS additive"
        "GS bearing-FIM-gated"
    ];

    compositeMode = [
        "additive"
        "bearing_fim_gated"
    ];

    meanOperationalErr = [
        rowAdd.meanOperationalErr
        rowFIM.meanOperationalErr
    ];

    meanSameInputErr = [
        rowAdd.meanSameInputErr
        rowFIM.meanSameInputErr
    ];

    meanInputPenalty_OperationalMinusSameInput = [
        rowAdd.meanInputPenalty_OperationalMinusSameInput
        rowFIM.meanInputPenalty_OperationalMinusSameInput
    ];

    rmsOperationalErr = [
        rowAdd.rmsOperationalErr
        rowFIM.rmsOperationalErr
    ];

    rmsSameInputErr = [
        rowAdd.rmsSameInputErr
        rowFIM.rmsSameInputErr
    ];

    p95OperationalErr = [
        rowAdd.p95OperationalErr
        rowFIM.p95OperationalErr
    ];

    p95SameInputErr = [
        rowAdd.p95SameInputErr
        rowFIM.p95SameInputErr
    ];

    meanOperationalImprovementPct_FIM_vsAdditive = [
        NaN
        percentImprovementStep09J5(meanOperationalErr(1), meanOperationalErr(2))
    ];

    meanSameInputImprovementPct_FIM_vsAdditive = [
        NaN
        percentImprovementStep09J5(meanSameInputErr(1), meanSameInputErr(2))
    ];

    residualOverallComparison = table( ...
        caseName, ...
        compositeMode, ...
        meanOperationalErr, ...
        meanSameInputErr, ...
        meanInputPenalty_OperationalMinusSameInput, ...
        rmsOperationalErr, ...
        rmsSameInputErr, ...
        p95OperationalErr, ...
        p95SameInputErr, ...
        meanOperationalImprovementPct_FIM_vsAdditive, ...
        meanSameInputImprovementPct_FIM_vsAdditive);

end

function residualWindowComparison = buildResidualWindowComparisonStep09J5( ...
    residualDiagAdd, residualDiagFIM)
%BUILDRESIDUALWINDOWCOMPARISONSTEP09J5 Compare GS residual errors by window.
%
% Purpose:
%   Compare the residual diagnostic window table between
%
%       GS additive
%       GS bearing_fim_gated
%
%   for the "GS composite" rows.
%
% Why this helper is written defensively:
%   Some earlier residual diagnostic helpers used slightly different table
%   variable names, for example
%
%       caseName
%       meanOperationalErr
%
%   instead of
%
%       windowCaseName
%       meanOperationalErrWin.
%
%   This helper accepts either naming style so Step 09-J.5 does not fail on
%   post-processing after the expensive full simulations already ran.

    Tadd = residualDiagAdd.windowSummaryTable;
    Tfim = residualDiagFIM.windowSummaryTable;

    addCaseName = getTableVarStep09J5(Tadd, ["windowCaseName", "caseName"]);
    fimCaseName = getTableVarStep09J5(Tfim, ["windowCaseName", "caseName"]);

    TaddGS = Tadd(string(addCaseName) == "GS composite", :);
    TfimGS = Tfim(string(fimCaseName) == "GS composite", :);

    if height(TaddGS) ~= height(TfimGS)
        error("Step09J5:BadResidualWindowTables", ...
            "Additive and FIM residual window tables have different heights.");
    end

    windowNameAdd = string(getTableVarStep09J5(TaddGS, "windowName"));
    windowNameFIM = string(getTableVarStep09J5(TfimGS, "windowName"));

    if any(windowNameAdd ~= windowNameFIM)
        error("Step09J5:WindowNameMismatch", ...
            "Additive and FIM residual diagnostic windows do not match.");
    end

    windowName = windowNameAdd;
    tStart_s = getTableVarStep09J5(TaddGS, "tStart_s");
    tEnd_s = getTableVarStep09J5(TaddGS, "tEnd_s");

    add_meanOperationalErr = getTableVarStep09J5( ...
        TaddGS, ["meanOperationalErrWin", "meanOperationalErr"]);

    fim_meanOperationalErr = getTableVarStep09J5( ...
        TfimGS, ["meanOperationalErrWin", "meanOperationalErr"]);

    add_meanSameInputErr = getTableVarStep09J5( ...
        TaddGS, ["meanSameInputErrWin", "meanSameInputErr"]);

    fim_meanSameInputErr = getTableVarStep09J5( ...
        TfimGS, ["meanSameInputErrWin", "meanSameInputErr"]);

    add_rmsOperationalErr = getTableVarStep09J5( ...
        TaddGS, ["rmsOperationalErrWin", "rmsOperationalErr"]);

    fim_rmsOperationalErr = getTableVarStep09J5( ...
        TfimGS, ["rmsOperationalErrWin", "rmsOperationalErr"]);

    add_p95OperationalErr = getTableVarStep09J5( ...
        TaddGS, ["p95OperationalErrWin", "p95OperationalErr"]);

    fim_p95OperationalErr = getTableVarStep09J5( ...
        TfimGS, ["p95OperationalErrWin", "p95OperationalErr"]);

    fimImprovementOperationalPct_vsAdditive = 100.0 * ...
        (add_meanOperationalErr - fim_meanOperationalErr) ./ ...
        max(add_meanOperationalErr, eps);

    fimImprovementSameInputPct_vsAdditive = 100.0 * ...
        (add_meanSameInputErr - fim_meanSameInputErr) ./ ...
        max(add_meanSameInputErr, eps);

    fimImprovementRmsPct_vsAdditive = 100.0 * ...
        (add_rmsOperationalErr - fim_rmsOperationalErr) ./ ...
        max(add_rmsOperationalErr, eps);

    fimImprovementP95Pct_vsAdditive = 100.0 * ...
        (add_p95OperationalErr - fim_p95OperationalErr) ./ ...
        max(add_p95OperationalErr, eps);

    residualWindowComparison = table( ...
        windowName, ...
        tStart_s, ...
        tEnd_s, ...
        add_meanOperationalErr, ...
        fim_meanOperationalErr, ...
        fimImprovementOperationalPct_vsAdditive, ...
        add_meanSameInputErr, ...
        fim_meanSameInputErr, ...
        fimImprovementSameInputPct_vsAdditive, ...
        add_rmsOperationalErr, ...
        fim_rmsOperationalErr, ...
        fimImprovementRmsPct_vsAdditive, ...
        add_p95OperationalErr, ...
        fim_p95OperationalErr, ...
        fimImprovementP95Pct_vsAdditive);

end




function alignmentWindowComparison = buildAlignmentWindowComparisonStep09J5( ...
    alignAddTable, alignFIMTable)
%BUILDALIGNMENTWINDOWCOMPARISONSTEP09J5 Compare branch alignment windows.

    if height(alignAddTable) ~= height(alignFIMTable)
        error("Step09J5:BadAlignmentTables", ...
            "Additive and FIM alignment tables have different heights.");
    end

    windowName = string(alignAddTable.windowName);
    tStart_s = alignAddTable.tStart_s;
    tEnd_s = alignAddTable.tEnd_s;

    add_meanGSErrorMinusLocal = alignAddTable.meanGSErrorMinusLocal;
    fim_meanGSErrorMinusLocal = alignFIMTable.meanGSErrorMinusLocal;

    add_meanNonlocalNorm = alignAddTable.meanNonlocalNorm;
    fim_meanNonlocalNorm = alignFIMTable.meanNonlocalNorm;

    add_meanDot_eL_n = alignAddTable.meanDot_eL_n;
    fim_meanDot_eL_n = alignFIMTable.meanDot_eL_n;

    add_meanCosAlign = alignAddTable.meanCosAlign;
    fim_meanCosAlign = alignFIMTable.meanCosAlign;

    add_meanDeltaSq = alignAddTable.meanDeltaSq;
    fim_meanDeltaSq = alignFIMTable.meanDeltaSq;

    add_fracHarmfulPct = alignAddTable.fracHarmfulPct;
    fim_fracHarmfulPct = alignFIMTable.fracHarmfulPct;

    deltaSqReduction = add_meanDeltaSq - fim_meanDeltaSq;
    harmfulPctReduction = add_fracHarmfulPct - fim_fracHarmfulPct;

    alignmentWindowComparison = table( ...
        windowName, ...
        tStart_s, ...
        tEnd_s, ...
        add_meanGSErrorMinusLocal, ...
        fim_meanGSErrorMinusLocal, ...
        add_meanNonlocalNorm, ...
        fim_meanNonlocalNorm, ...
        add_meanDot_eL_n, ...
        fim_meanDot_eL_n, ...
        add_meanCosAlign, ...
        fim_meanCosAlign, ...
        add_meanDeltaSq, ...
        fim_meanDeltaSq, ...
        deltaSqReduction, ...
        add_fracHarmfulPct, ...
        fim_fracHarmfulPct, ...
        harmfulPctReduction);

end

function row = selectResidualCaseRowStep09J5(T, targetName)
%SELECTRESIDUALCASEROWSTEP09J5 Select one case row from a residual table.

    idx = string(T.caseName) == string(targetName);

    if nnz(idx) ~= 1
        error("Step09J5:ResidualCaseNotFound", ...
            "Expected exactly one residual row named %s.", string(targetName));
    end

    row = T(idx, :);

end

function pct = percentImprovementStep09J5(baselineValue, testValue)
%PERCENTIMPROVEMENTSTEP09J5 Positive means testValue is smaller.

    if isfinite(baselineValue) && baselineValue > 0 && isfinite(testValue)
        pct = 100.0 * (baselineValue - testValue) / baselineValue;
    else
        pct = NaN;
    end

end

function printKeyWindowStep09J5(alignmentWindowComparison, targetWindow)
%PRINTKEYWINDOWSTEP09J5 Print the main known problematic window.

    idx = string(alignmentWindowComparison.windowName) == string(targetWindow);

    if nnz(idx) ~= 1
        fprintf("\nKey window %s not found in alignment table.\n", string(targetWindow));
        return;
    end

    row = alignmentWindowComparison(idx, :);

    fprintf("\nKey alignment window: %s\n", string(targetWindow));
    fprintf("    additive meanDeltaSq      = %.6e\n", row.add_meanDeltaSq);
    fprintf("    FIM meanDeltaSq           = %.6e\n", row.fim_meanDeltaSq);
    fprintf("    deltaSq reduction         = %.6e\n", row.deltaSqReduction);
    fprintf("    additive harmful fraction = %.3f %%\n", row.add_fracHarmfulPct);
    fprintf("    FIM harmful fraction      = %.3f %%\n", row.fim_fracHarmfulPct);
    fprintf("    harmful pct reduction     = %.3f %% points\n", row.harmfulPctReduction);

end

function assertFiniteStep09J5(results, caseName)
%ASSERTFINITESTEP09J5 Lightweight full-run finite-output check.
%
% This is intentionally not a deep sanity check. The purpose is only to
% catch the main full-simulation failure modes after adding the new
% bearing_fim_gated path:
%   NaN/Inf state estimate,
%   NaN/Inf theta estimate,
%   NaN/Inf covariance diagonal,
%   no finite NIS values.

    caseName = string(caseName);

    if ~all(isfinite(results.xhat(:)))
        error("Step09J5:NonFiniteXhat", ...
            "%s produced non-finite entries in results.xhat.", caseName);
    end

    if isfield(results, "thetaHat")
        if ~all(isfinite(results.thetaHat(:)))
            error("Step09J5:NonFiniteThetaHat", ...
                "%s produced non-finite entries in results.thetaHat.", caseName);
        end
    end

    if isfield(results, "Pdiag")
        if ~all(isfinite(results.Pdiag(:)))
            error("Step09J5:NonFinitePdiag", ...
                "%s produced non-finite entries in results.Pdiag.", caseName);
        end
    end

    if isfield(results, "NIS")
        nisFinite = results.NIS(isfinite(results.NIS));

        if isempty(nisFinite)
            error("Step09J5:NoFiniteNIS", ...
                "%s produced no finite NIS entries.", caseName);
        end

        if any(nisFinite < -1e-10)
            error("Step09J5:NegativeNIS", ...
                "%s produced negative finite NIS entries.", caseName);
        end
    end

end



function values = getTableVarStep09J5(T, candidateNames)
%GETTABLEVARSTEP09J5 Return the first matching table variable.
%
% Inputs:
%   T              - MATLAB table.
%   candidateNames - string or string array of acceptable variable names.
%
% Output:
%   values - T.(matchedName)
%
% Purpose:
%   Keeps Step 09-J.5 compatible with older diagnostic tables whose variable
%   names may differ slightly.

candidateNames = string(candidateNames);
tableNames = string(T.Properties.VariableNames);

for k = 1:numel(candidateNames)
    name = candidateNames(k);

    if any(tableNames == name)
        values = T.(char(name));
        return;
    end
end

error("Step09J5:MissingTableVariable", ...
    "None of the candidate variables [%s] exists. Available variables are [%s].", ...
    strjoin(candidateNames, ", "), ...
    strjoin(tableNames, ", "));

end