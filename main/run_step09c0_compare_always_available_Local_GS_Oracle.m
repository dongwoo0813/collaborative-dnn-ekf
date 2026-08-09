function out = run_step09c0_compare_always_available_Local_GS_Oracle( ...
    residualFamily, seed, verbose, dtOverride, nonlocalWeightOverride, gsCompositeModeOverride)
%{
File:
    main/run_step09c0_compare_always_available_Local_GS_Oracle.m

Purpose:
    Step 09-C.0 always-available Local / GS / Oracle comparison.

    Compare Local DNN-EKF, GS composite DNN-EKF, and Oracle residual EKF
    without FOV dropout.

Why this step matters:
    The FOV experiments mix two effects:

        1. GS branch-sharing benefit.
        2. Measurement sparsity due to FOV dropout.

    This script removes the second effect by forcing measurement availability
    to "always". Therefore it gives a clean baseline for whether GS branch
    sharing helps on the selected residual family.

Default:
    residualFamily = "coupled_nonlinear"
    seed = 101
    verbose = true

Comparison cases:
    1. Local DNN, always measurement available.
    2. GS composite, always measurement available.
    3. Oracle residual, always measurement available.

Outputs:
    out.summaryTable
        Multi-metric Local / GS / Oracle table.

    out.compactComparisonTable
        Key comparison metrics and improvement percentages.

Usage:
    out09c0 = run_step09c0_compare_always_available_Local_GS_Oracle();

    out09c0 = run_step09c0_compare_always_available_Local_GS_Oracle( ...
        "coupled_nonlinear", 101, true);
    disp(out09c0.compactComparisonTable);
    disp(out09c0.multiMetricSummaryTable);

    outHybrid = run_step09c0_compare_always_available_Local_GS_Oracle( ...
        "complex_branchwise", 101, false, 0.5, [], ...
        "local_full_plus_gated_nonlocal");
%}

    if nargin < 1 || isempty(residualFamily)
        residualFamily = "coupled_nonlinear";
    end

    if nargin < 2 || isempty(seed)
        seed = 101;
    end

    if nargin < 3 || isempty(verbose)
        verbose = true;
    end

    if nargin < 4
        dtOverride = [];
    end

    if nargin < 5
        nonlocalWeightOverride = [];
    end

    if nargin < 6 || isempty(gsCompositeModeOverride)
        gsCompositeModeOverride = "additive";
    end

    gsCompositeModeOverride = string(gsCompositeModeOverride);

    switch gsCompositeModeOverride
        case {"additive", ...
                "gated_additive", ...
                "local_full_plus_gated_nonlocal"}
            % Valid modes.

        otherwise
            error("Step09C0:UnsupportedGSCompositeMode", ...
                "Unsupported gsCompositeModeOverride = %s.", gsCompositeModeOverride);
    end


    residualFamily = string(residualFamily);

    addpath(genpath(pwd));
    rehash;

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 09-C.0: Always-available Local / GS / Oracle comparison\n");
        fprintf("============================================================\n");
        fprintf("Residual family     = %s\n", residualFamily);
        fprintf("Random seed         = %d\n", seed);
        fprintf("GS composite mode   = %s\n\n", gsCompositeModeOverride);
    end

    % ---------------------------------------------------------------------
    % Base configuration
    % ---------------------------------------------------------------------
    cfgBase = config_step04_GS_DNN_EKF();

    if ~isempty(dtOverride)
        cfgBase.dt = dtOverride;
        cfgBase.time = 0:cfgBase.dt:cfgBase.T;
        cfgBase.N = numel(cfgBase.time);
    end


    % Step 09-C.0: choose residual family explicitly so the result does not
    % depend on whatever default is currently written inside the config file.
    cfgBase.truth.useResidual = true;
    cfgBase.truth.residualFamily = residualFamily;

    % Keep old field as an alias for older helper functions/scripts.
    if residualFamily == "simple_branchwise"
        cfgBase.truth.residualModel = "branchwise";
    else
        cfgBase.truth.residualModel = residualFamily;
    end

    cfgBase.truth.residualAmp = 5e-4;

    cfgBase.dnn.theta0_std = 0.0;
    cfgBase.dnn.residualInjectionGain = 1.0;

    cfgBase.ekf.useBlockCovPrediction = true;


    % Step 09-F:
    % Keep the GS composite mode explicit so that additive and gated_additive
    % runs can be compared using the same runner.
    cfgBase.gs.compositeMode = gsCompositeModeOverride;

    % Gate defaults for gated_additive mode.
    % These fields are harmless for additive mode.
    cfgBase.gate.mode = "tight_frame_2d_rt";
    cfgBase.gate.minRange = 1e-12;



    % Force always-available measurement mode.
    cfgBase.meas.availabilityMode = "always";

    if isfield(cfgBase, "fov")
        cfgBase.fov.enabled = false;
        cfgBase.fov.guardUnimplementedMode = true;
    end

    % Keep the validated GS communication policy.
    cfgBase.gs.uploadMode = "after_measurement_update";
    cfgBase.gs.broadcastMode = "every_step";

    % ---------------------------------------------------------------------
    % Case 1: Local DNN-EKF, always available
    % ---------------------------------------------------------------------
    cfgLocal = cfgBase;

    cfgLocal.step.name = "step09c0_always_local_DNN";
    cfgLocal.estimator.type = "local_DNN_EKF";
    cfgLocal.dnn.predictionResidualSource = "local_DNN";

    if isfield(cfgLocal, "gs")
        cfgLocal.gs.enabled = false;
        cfgLocal.gs.useNonlocalBranchCovariance = false;
    end

    if verbose
        fprintf("Running Case 1: Local DNN, always available\n");
    end

    rng(seed);
    resLocal = simulateLocalDNNEKF(cfgLocal);

    % ---------------------------------------------------------------------
    % Case 2: GS composite DNN-EKF, always available
    % ---------------------------------------------------------------------
    cfgGS = cfgBase;

    if ~isempty(nonlocalWeightOverride)
        cfgGS.gs.nonlocalWeightMode = "scalar";
        cfgGS.gs.nonlocalWeight = nonlocalWeightOverride;
    end

    cfgGS.step.name = "step09c0_always_GS_composite";
    cfgGS.estimator.type = "GS_DNN_EKF";
    cfgGS.dnn.predictionResidualSource = "GS_composite";

    cfgGS.gs.enabled = true;
    cfgGS.gs.bootstrapUpload = true;
    cfgGS.gs.uploadMode = "after_measurement_update";
    cfgGS.gs.broadcastMode = "every_step";
    cfgGS.gs.useNonlocalBranchCovariance = false;

    if verbose
        fprintf("Running Case 2: GS composite, always available\n");
    end

    rng(seed);
    resGS = simulate_GS_DNN_EKF(cfgGS);

    % ---------------------------------------------------------------------
    % Case 3: Oracle residual EKF, always available
    % ---------------------------------------------------------------------
    cfgOracle = cfgBase;

    cfgOracle.step.name = "step09c0_always_oracle";
    cfgOracle.estimator.type = "oracle_residual_EKF";
    cfgOracle.dnn.predictionResidualSource = "oracle";

    if isfield(cfgOracle, "gs")
        cfgOracle.gs.enabled = false;
        cfgOracle.gs.useNonlocalBranchCovariance = false;
    end

    if verbose
        fprintf("Running Case 3: Oracle residual, always available\n");
    end

    rng(seed);
    resOracle = simulateLocalDNNEKF(cfgOracle);

    % ---------------------------------------------------------------------
    % Multi-metric estimator evaluation
    % ---------------------------------------------------------------------
    metricsLocal = evaluateEstimatorMetrics( ...
        resLocal, cfgLocal, "Local DNN + always");


    gsCaseName = sprintf("%s + always", ...
        gsCompositeLabel_step09c0(gsCompositeModeOverride));

    metricsGS = evaluateEstimatorMetrics( ...
        resGS, cfgGS, gsCaseName);

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

    gsMeanImprovementPct = percentImprovement_step09c0(localMean, gsMean);
    oracleMeanImprovementPct = percentImprovement_step09c0(localMean, oracleMean);

    gsRmsImprovementPct = percentImprovement_step09c0(localRms, gsRms);
    oracleRmsImprovementPct = percentImprovement_step09c0(localRms, oracleRms);

    gsP95ImprovementPct = percentImprovement_step09c0(localP95, gsP95);
    oracleP95ImprovementPct = percentImprovement_step09c0(localP95, oracleP95);

    gapLocalOracle = localMean - oracleMean;

    if isfinite(gapLocalOracle) && gapLocalOracle > 0
        gsClosedLocalOracleGapPct = 100 * (localMean - gsMean) / gapLocalOracle;
    else
        gsClosedLocalOracleGapPct = NaN;
    end

    gsCompositeMode = gsCompositeModeOverride;

    compactComparisonTable = table( ...
        residualFamily, ...
        seed, ...
        gsCompositeMode, ...
        metricsLocal.availabilityRate * 100, ...
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
            'gsCompositeMode', ...
            'availabilityRatePercent', ...
            'localMeanPosErr_m', ...
            'gsMeanPosErr_m', ...
            'oracleMeanPosErr_m', ...
            'gsMeanImprovementPct', ...
            'oracleMeanImprovementPct', ...
            'gsClosedLocalOracleGapPct', ...
            'localRmsPosErr_m', ...
            'gsRmsPosErr_m', ...
            'oracleRmsPosErr_m', ...
            'gsRmsImprovementPct', ...
            'oracleRmsImprovementPct', ...
            'localP95PosErr_m', ...
            'gsP95PosErr_m', ...
            'oracleP95PosErr_m', ...
            'gsP95ImprovementPct', ...
            'oracleP95ImprovementPct', ...
            'localMeanNIS', ...
            'gsMeanNIS', ...
            'oracleMeanNIS', ...
            'gsLoggedUploadDecisions', ...
            'gsFinalTotalUploads'});

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 09-C.0 multi-metric summary\n");
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

        plotStep09c0Comparison_step09c0( ...
            resLocal, resGS, resOracle, ...
            metricsLocal, metricsGS, metricsOracle, ...
            cfgLocal, cfgGS, cfgOracle, ...
            residualFamily, seed, gsCompositeModeOverride);
    end

    % ---------------------------------------------------------------------
    % Output
    % ---------------------------------------------------------------------
    out = struct();

    out.residualFamily = residualFamily;
    out.seed = seed;

    out.dt = cfgBase.dt;

    out.gsCompositeMode = gsCompositeModeOverride;


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

    out.nonlocalWeight = nonlocalWeightOverride;

end

function pct = percentImprovement_step09c0(baselineValue, testValue)
%PERCENTIMPROVEMENT_STEP09C0 Positive means testValue is smaller.

    if isfinite(baselineValue) && baselineValue > 0 && isfinite(testValue)
        pct = 100 * (baselineValue - testValue) / baselineValue;
    else
        pct = NaN;
    end

end


function plotStep09c0Comparison_step09c0( ...
    resLocal, resGS, resOracle, ...
    metricsLocal, metricsGS, metricsOracle, ...
    cfgLocal, cfgGS, cfgOracle, ...
    residualFamily, seed, gsCompositeMode)
%PLOTSTEP09C0COMPARISON_STEP09C0 Plot always-available comparison.
%
% Purpose:
%     Visualize whether GS improves the estimation behavior over time,
%     not only in aggregate scalar metrics.
%
% Figures:
%     1. Mean position error over watchers.
%     2. Mean NIS over watchers.
%     3. Mean residual approximation error over watchers.
%     4. Cumulative GS uploads.

    time = resLocal.time;
    gsLabel = gsCompositeLabel_step09c0(gsCompositeMode);

    % ---------------------------------------------------------------------
    % Figure 1: Position error time history
    % ---------------------------------------------------------------------
    meanErrLocal = mean(metricsLocal.posErr, 2, "omitnan");
    meanErrGS = mean(metricsGS.posErr, 2, "omitnan");
    meanErrOracle = mean(metricsOracle.posErr, 2, "omitnan");

    figure;
    plot(time, meanErrLocal, "LineWidth", 1.5);
    hold on;
    plot(time, meanErrGS, "LineWidth", 1.5);
    plot(time, meanErrOracle, "LineWidth", 1.5);
    grid on;

    xlabel("Time [s]");
    ylabel("Mean position error over watchers [m]");
    title(sprintf( ...
        "Step 09-C.0 position error, residual = %s, seed = %d, GS = %s", ...
        residualFamily, seed, gsCompositeMode), ...
        "Interpreter", "none");

    legend( ...
        "Local DNN + always", ...
        sprintf("%s + always", gsLabel), ...
        "Oracle + always", ...
        "Location", "best");

    % ---------------------------------------------------------------------
    % Figure 2: Mean NIS time history
    % ---------------------------------------------------------------------
    meanNISLocal = meanNIS_step09c0(resLocal);
    meanNISGS = meanNIS_step09c0(resGS);
    meanNISOracle = meanNIS_step09c0(resOracle);

    figure;
    plot(time, meanNISLocal, "LineWidth", 1.3);
    hold on;
    plot(time, meanNISGS, "LineWidth", 1.3);
    plot(time, meanNISOracle, "LineWidth", 1.3);
    yline(1.0, "--", "NIS = 1 reference", "LineWidth", 1.2);
    grid on;

    xlabel("Time [s]");
    ylabel("Mean NIS over watchers");
    title(sprintf( ...
        "Step 09-C.0 NIS, residual = %s, seed = %d, GS = %s", ...
        residualFamily, seed, gsCompositeMode), ...
        "Interpreter", "none");

    legend( ...
        "Local DNN + always", ...
        sprintf("%s + always", gsLabel), ...
        "Oracle + always", ...
        "Location", "best");

    % ---------------------------------------------------------------------
    % Figure 3: Residual approximation error time history
    % ---------------------------------------------------------------------
    resErrLocal = meanResidualErrorNorm_step09c0(resLocal, cfgLocal);
    resErrGS = meanResidualErrorNorm_step09c0(resGS, cfgGS);
    resErrOracle = meanResidualErrorNorm_step09c0(resOracle, cfgOracle);

    figure;
    plot(time, resErrLocal, "LineWidth", 1.5);
    hold on;
    plot(time, resErrGS, "LineWidth", 1.5);
    plot(time, resErrOracle, "LineWidth", 1.5);
    grid on;

    xlabel("Time [s]");
    ylabel("Mean residual approximation error norm [m/s^2]");
    title(sprintf( ...
        "Step 09-C.0 residual approximation error, residual = %s, seed = %d, GS = %s", ...
        residualFamily, seed, gsCompositeMode), ...
        "Interpreter", "none");

    legend( ...
        "Local DNN + always", ...
        sprintf("%s + always", gsLabel), ...
        "Oracle + always", ...
        "Location", "best");

    % ---------------------------------------------------------------------
    % Figure 4: Cumulative GS uploads
    % ---------------------------------------------------------------------
    if isfield(resGS, "gsNumTotalUploads")
        figure;
        plot(resGS.time, resGS.gsNumTotalUploads, "LineWidth", 1.5);
        grid on;

        xlabel("Time [s]");
        ylabel("Cumulative GS uploads");
        title(sprintf( ...
            "Step 09-C.0 cumulative GS uploads, residual = %s, seed = %d, GS = %s", ...
            residualFamily, seed, gsCompositeMode), ...
            "Interpreter", "none");
    end

end

function meanNIS = meanNIS_step09c0(results)
%MEANNIS_STEP09C0 Mean NIS over watchers.

    if ~isfield(results, "NIS")
        meanNIS = NaN(numel(results.time), 1);
        return;
    end

    nis = results.NIS;

    if isfield(results, "measAvail")
        avail = logical(results.measAvail);
        nis(~avail) = NaN;
    end

    meanNIS = mean(nis, 2, "omitnan");

end

function meanErr = meanResidualErrorNorm_step09c0(results, cfg)
%MEANRESIDUALERRORNORM_STEP09C0 Mean residual approximation error.
%
% For Local/GS:
%     error = ||d_hat - d_true||
%
% For Oracle:
%     d_hat is treated as d_true, because oracle residual EKF uses the true
%     residual in prediction even if the diagnostic dnnResidual log is zero.

    dTrue = results.trueResidual;
    dHat = results.dnnResidual;

    [dim, N, Nw] = size(dHat);

    dTrue = expandTrueResidual_step09c0(dTrue, dim, N, Nw);

    residualSource = "unknown";

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "predictionResidualSource")
        residualSource = string(cfg.dnn.predictionResidualSource);
    end

    if residualSource == "oracle"
        dHat = dTrue;
    end

    errNorm = reshape(sqrt(sum((dHat - dTrue).^2, 1)), N, Nw);

    meanErr = mean(errNorm, 2, "omitnan");

end

function dTrueOut = expandTrueResidual_step09c0(dTrueIn, dim, N, Nw)
%EXPANDTRUERESIDUAL_STEP09C0 Convert true residual log to dim x N x Nw.

    sz = size(dTrueIn);

    if numel(sz) == 2
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09C0:BadTrueResidualSize", ...
                "trueResidual must be dim x N or dim x N x Nw.");
        end

        dTrueOut = reshape(dTrueIn, dim, N, 1);
        dTrueOut = repmat(dTrueOut, 1, 1, Nw);
        return;
    end

    if numel(sz) == 3
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09C0:BadTrueResidualSize", ...
                "trueResidual has incompatible first two dimensions.");
        end

        if sz(3) == Nw
            dTrueOut = dTrueIn;
        elseif sz(3) == 1
            dTrueOut = repmat(dTrueIn, 1, 1, Nw);
        else
            error("Step09C0:BadTrueResidualSize", ...
                "trueResidual third dimension must be 1 or Nw.");
        end

        return;
    end

    error("Step09C0:BadTrueResidualSize", ...
        "trueResidual must be dim x N or dim x N x Nw.");

end


function label = gsCompositeLabel_step09c0(gsCompositeMode)
% Short label for plots/tables.

gsCompositeMode = string(gsCompositeMode);

switch gsCompositeMode
    case "additive"
        label = "GS additive";

    case "gated_additive"
        label = "GS gated";

    case "local_full_plus_gated_nonlocal"
        label = "GS local-full + gated-nonlocal";

    otherwise
        label = sprintf("GS %s", gsCompositeMode);
end

end