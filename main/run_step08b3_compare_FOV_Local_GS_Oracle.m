function out = run_step08b3_compare_FOV_Local_GS_Oracle(halfAngleDeg, verbose, seed, residualFamily)
%{
File:
    main/run_step08b3_compare_FOV_Local_GS_Oracle.m

Purpose:
    Step 08-B.3 / Step 08-B.4b FOV comparison script.

    Compare Local DNN, GS composite, and Oracle residual EKF under
    FOV-based intermittent measurements using the shared multi-metric
    estimator evaluator.

Comparison cases:
    1. Local DNN + FOV
    2. GS composite + FOV
    3. Oracle residual + FOV

Default FOV scenario:
    halfAngleDeg = 40 deg

Why 40 deg:
    Step 08-B.2 FOV severity sweep showed that 40 deg gives approximately
    32 percent measurement availability. This is intermittent enough to
    matter, but not so harsh that the filter is nearly always prediction-only.

Inputs:
    halfAngleDeg
        FOV half-angle in degrees.

    verbose
        true  -> print tables and make plots.
        false -> run quietly. This is useful for Monte Carlo sweeps.

    seed
        Random seed used for all three paired cases.
        The same seed is applied to Local, GS, and Oracle so the comparison
        is paired and fair.

Outputs:
    out
        Struct containing configs, raw simulation results, evaluator metrics,
        summary tables, availability table, and comparison percentages.

Notes:
    - This script does not create a new config file.
    - It starts from config_step04_GS_DNN_EKF().
    - FOV settings are applied consistently to all three cases.
    - The same random seed is used for all cases.
    - Step 08-B.4b uses evaluateEstimatorMetrics.m directly.
%}

    if nargin < 1 || isempty(halfAngleDeg)
        halfAngleDeg = 20.0;
    end

    if nargin < 2 || isempty(verbose)
        verbose = true;
    end

    if nargin < 3 || isempty(seed)
        seed = 100;
    end
    
    if nargin < 4 || isempty(residualFamily)
        residualFamily = "simple_branchwise";
    end
    
    residualFamily = string(residualFamily);

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 08-B.3/08-B.4b: FOV Local vs GS vs Oracle multi-metric comparison\n");
        fprintf("============================================================\n\n");
    end

    addpath(genpath(pwd));
    rehash;

    % ---------------------------------------------------------------------
    % Base configuration
    % ---------------------------------------------------------------------
    cfgBase = config_step04_GS_DNN_EKF();

    cfgBase.truth.residualAmp = 5e-4;
    cfgBase.truth.residualFamily = residualFamily;
    
    cfgBase.dnn.theta0_std = 0.0;
    cfgBase.dnn.residualInjectionGain = 1.0;

    cfgBase = applyCommonFOVConfig_step08b3(cfgBase, halfAngleDeg);

    % ---------------------------------------------------------------------
    % Case 1: Local DNN-EKF under FOV
    % ---------------------------------------------------------------------
    cfgLocal = cfgBase;

    cfgLocal.step.name = "step08b3_FOV_local_DNN";
    cfgLocal.estimator.type = "local_DNN_EKF";
    cfgLocal.dnn.predictionResidualSource = "local_DNN";

    if isfield(cfgLocal, "gs")
        cfgLocal.gs.enabled = false;
    end

    if verbose
        fprintf("Running Case 1: Local DNN + FOV, halfAngle = %.3f deg, seed = %d\n", ...
            halfAngleDeg, seed);
    end

    rng(seed);
    resLocal = simulateLocalDNNEKF(cfgLocal);

    % ---------------------------------------------------------------------
    % Case 2: GS composite DNN-EKF under FOV
    % ---------------------------------------------------------------------
    cfgGS = cfgBase;

    cfgGS.step.name = "step08b3_FOV_GS_composite";
    cfgGS.estimator.type = "GS_DNN_EKF";
    cfgGS.dnn.predictionResidualSource = "GS_composite";

    cfgGS.gs.enabled = true;
    cfgGS.gs.uploadMode = "after_measurement_update";
    cfgGS.gs.broadcastMode = "every_step";
    cfgGS.gs.useNonlocalBranchCovariance = false;

    if verbose
        fprintf("Running Case 2: GS composite + FOV, halfAngle = %.3f deg, seed = %d\n", ...
            halfAngleDeg, seed);
    end

    rng(seed);
    resGS = simulate_GS_DNN_EKF(cfgGS);

    % ---------------------------------------------------------------------
    % Case 3: Oracle residual EKF under FOV
    % ---------------------------------------------------------------------
    cfgOracle = cfgBase;

    cfgOracle.step.name = "step08b3_FOV_oracle";
    cfgOracle.estimator.type = "oracle_residual_EKF";
    cfgOracle.dnn.predictionResidualSource = "oracle";

    if isfield(cfgOracle, "gs")
        cfgOracle.gs.enabled = false;
    end

    if verbose
        fprintf("Running Case 3: Oracle + FOV, halfAngle = %.3f deg, seed = %d\n", ...
            halfAngleDeg, seed);
    end

    rng(seed);
    resOracle = simulateLocalDNNEKF(cfgOracle);

    % ---------------------------------------------------------------------
    % Multi-metric estimator evaluation
    % ---------------------------------------------------------------------
    caseName = [
        "Local DNN + FOV"
        "GS composite + FOV"
        "Oracle + FOV"
    ];

    % Step 08-B.4b: use the shared evaluator so every comparison reports
    % the same metrics used later for FOV sweeps and Monte Carlo runs.
    metricsLocal  = evaluateEstimatorMetrics(resLocal,  cfgLocal,  caseName(1));
    metricsGS     = evaluateEstimatorMetrics(resGS,     cfgGS,     caseName(2));
    metricsOracle = evaluateEstimatorMetrics(resOracle, cfgOracle, caseName(3));

    localMean  = metricsLocal.meanPosErr;
    gsMean     = metricsGS.meanPosErr;
    oracleMean = metricsOracle.meanPosErr;

    gsImprovementPct = 100 * (localMean - gsMean) / localMean;
    oracleImprovementPct = 100 * (localMean - oracleMean) / localMean;

    gapLocalOracle = localMean - oracleMean;

    if gapLocalOracle > 0
        gapClosedPct = 100 * (localMean - gsMean) / gapLocalOracle;
    else
        gapClosedPct = NaN;
    end

    summaryTable = [
        metricsLocal.scalarSummary
        metricsGS.scalarSummary
        metricsOracle.scalarSummary
    ];

    % ---------------------------------------------------------------------
    % Measurement availability summary
    % ---------------------------------------------------------------------
    availLocal  = summarizeAvailability_step08b3(resLocal,  cfgBase);
    availGS     = summarizeAvailability_step08b3(resGS,     cfgBase);
    availOracle = summarizeAvailability_step08b3(resOracle, cfgBase);

    availabilityTable = table( ...
        caseName, ...
        [availLocal.availableCount; availGS.availableCount; availOracle.availableCount], ...
        [availLocal.totalCount; availGS.totalCount; availOracle.totalCount], ...
        100*[availLocal.rate; availGS.rate; availOracle.rate], ...
        'VariableNames', { ...
            'caseName', ...
            'availableCount', ...
            'totalCount', ...
            'availabilityRatePercent'});

    % ---------------------------------------------------------------------
    % Optional console output
    % ---------------------------------------------------------------------
    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 08-B.4b multi-metric FOV comparison summary\n");
        fprintf("============================================================\n");
        fprintf("FOV half-angle = %.3f deg\n", halfAngleDeg);
        fprintf("Random seed    = %d\n\n", seed);

        disp(summaryTable);

        fprintf("GS improvement over Local DNN       = %.3f %%\n", gsImprovementPct);
        fprintf("Oracle improvement over Local DNN   = %.3f %%\n", oracleImprovementPct);
        fprintf("GS closed Local-to-Oracle gap       = %.3f %%\n", gapClosedPct);

        fprintf("\nMeasurement availability summary:\n");
        disp(availabilityTable);

        if isfield(resGS, "measurementDropoutReason")
            activeRows = 2:(cfgBase.N-1);
            reasonsGS = string(resGS.measurementDropoutReason(activeRows,:));
            reasonList = unique(reasonsGS(:));

            fprintf("GS dropout reason summary:\n");
            for idx = 1:numel(reasonList)
                reason = reasonList(idx);
                fprintf("  %-18s : %d\n", reason, nnz(reasonsGS == reason));
            end
        end

        fprintf("\nGS upload summary:\n");

        % Prefer evaluator-level values so the printout matches summaryTable.
        if isfield(metricsGS, "gsFinalTotalUploads")
            fprintf("  Final GS total uploads = %.0f\n", metricsGS.gsFinalTotalUploads);
        elseif isfield(resGS, "gsNumTotalUploads")
            fprintf("  Final GS total uploads = %d\n", resGS.gsNumTotalUploads(end));
        end

        if isfield(metricsGS, "gsLoggedUploadDecisions")
            fprintf("  Logged GS upload decisions = %.0f\n", metricsGS.gsLoggedUploadDecisions);
        elseif isfield(resGS, "gsUploadDecision")
            fprintf("  Logged GS upload decisions = %d\n", nnz(logical(resGS.gsUploadDecision)));
        end
    end

    % ---------------------------------------------------------------------
    % Optional plots
    % ---------------------------------------------------------------------
    if verbose
        plotStep08b3Comparison_step08b3( ...
            resLocal, resGS, resOracle, ...
            metricsLocal, metricsGS, metricsOracle, cfgBase, halfAngleDeg);
    end

    % ---------------------------------------------------------------------
    % Output
    % ---------------------------------------------------------------------
    out = struct();

    out.halfAngleDeg = halfAngleDeg;
    out.seed = seed;
    out.residualFamily = residualFamily;

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

    % Step 08-B.4b alias: make the multi-metric table name explicit for later
    % FOV sweeps and Monte Carlo scripts while preserving the old API.
    out.multiMetricSummaryTable = summaryTable;

    out.availabilityTable = availabilityTable;

    out.gsImprovementPct = gsImprovementPct;
    out.oracleImprovementPct = oracleImprovementPct;
    out.gapClosedPct = gapClosedPct;

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 08-B.3/08-B.4b: FOV Local vs GS vs Oracle multi-metric comparison\n");
        fprintf("============================================================\n");
        fprintf("Residual family = %s\n\n", residualFamily);
    end

end

function cfg = applyCommonFOVConfig_step08b3(cfg, halfAngleDeg)
%APPLYCOMMONFOVCONFIG_STEP08B3 Apply identical FOV settings to all cases.
%
% This keeps Local, GS, and Oracle comparisons fair by changing only the
% estimator mode while preserving the same truth, measurement, and FOV setup.

    cfg.meas.availabilityMode = "fov";
    cfg.fov.enabled = true;
    cfg.fov.guardUnimplementedMode = false;

    % Use fixed inertial boresight for repeatable cone-based dropout.
    cfg.fov.boresightMode = "inertial_fixed";

    cfg.fov.boresightInertial = zeros(cfg.dim,1);
    cfg.fov.boresightInertial(1) = 1.0;

    cfg.fov.halfAngleDeg = halfAngleDeg;
    cfg.fov.halfAngleRad = deg2rad(cfg.fov.halfAngleDeg);

    % Disable range gating so dropout is caused by outside_fov only.
    cfg.fov.rhoMin = 0.0;
    cfg.fov.rhoMax = Inf;

    % Keep the validated GS communication policy.
    cfg.gs.uploadMode = "after_measurement_update";
    cfg.gs.broadcastMode = "every_step";

end

function avail = summarizeAvailability_step08b3(results, cfg)
%SUMMARIZEAVAILABILITY_STEP08B3 Summarize active update-row availability.
%
% Active rows exclude the first/last bookkeeping rows used by the simulator
% so this count matches the Step 08 FOV diagnostic convention.

    activeRows = 2:(cfg.N-1);

    measAvail = logical(results.measAvail(activeRows,:));

    avail = struct();
    avail.availableCount = nnz(measAvail);
    avail.totalCount = numel(measAvail);
    avail.rate = avail.availableCount / avail.totalCount;

end

function plotStep08b3Comparison_step08b3( ...
    resLocal, resGS, resOracle, ...
    metricsLocal, metricsGS, metricsOracle, cfg, halfAngleDeg)
%PLOTSTEP08B3COMPARISON_STEP08B3 Plot FOV comparison diagnostics.
%
% These plots are intended for interactive single-run inspection only.
% Monte Carlo scripts should call the parent function with verbose=false.

    time = resLocal.time;

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
    legend("Local DNN + FOV", "GS composite + FOV", "Oracle + FOV", ...
        "Location", "best");
    title(sprintf("Step 08-B.3/08-B.4b FOV comparison, half-angle = %.1f deg", ...
        halfAngleDeg));

    figure;
    plot(time, sum(logical(resLocal.measAvail), 2), "LineWidth", 1.2);
    hold on;
    plot(time, sum(logical(resGS.measAvail), 2), "LineWidth", 1.2);
    plot(time, sum(logical(resOracle.measAvail), 2), "LineWidth", 1.2);
    grid on;
    xlabel("Time [s]");
    ylabel("Number of watchers with measurement");
    legend("Local DNN", "GS composite", "Oracle", "Location", "best");
    title("FOV measurement availability count");

    if isfield(resGS, "gsNumTotalUploads")
        figure;
        plot(resGS.time, resGS.gsNumTotalUploads, "LineWidth", 1.5);
        grid on;
        xlabel("Time [s]");
        ylabel("Cumulative GS uploads");
        title("Step 08-B.3/08-B.4b GS cumulative uploads under FOV");
    end

    % Keep cfg as an explicit input because future plot variants may use
    % cfg.fov or cfg.gs fields for titles and annotations.
    cfg = cfg;

end