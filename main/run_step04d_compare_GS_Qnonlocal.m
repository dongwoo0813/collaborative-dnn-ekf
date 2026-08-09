function out = run_step04d_compare_GS_Qnonlocal()
%{
Function:
    run_step04d_compare_GS_Qnonlocal.m

Purpose:
    Run the Step 04d performance and consistency comparison for the
    collaborative block-structured DNN-EKF.

    This script compares four cases:

        Case 1: Local DNN-EKF
            Each watcher predicts using only its own local branch,

                d_pred,m = d_m(eta; theta_m).

        Case 2: GS composite DNN-EKF without nonlocal covariance injection
            Each watcher predicts using its local branch plus nonlocal
            ground-station branch copies,

                d_pred,m = d_m(eta; theta_m)
                         + sum_{j ~= m} d_j(eta; theta_{j|m}^{GS}),

            but treats the nonlocal branch copies as deterministic in the
            covariance prediction.

        Case 3: GS composite DNN-EKF with Qnonlocal
            The mean prediction is the same as Case 2, but the covariance
            prediction additionally injects

                Q_{X,-m} = M_m S_{d,-m} M_m^T,

            where S_{d,-m} maps nonlocal GS-branch parameter covariance into
            residual-acceleration uncertainty.

        Case 4: Oracle residual EKF
            The predictor uses the true residual model. This is an
            achievable upper-reference case, not a realizable estimator.

Outputs:
    out - structure containing configurations, raw simulation results,
          metrics, and summary table.

Main diagnostics:
    - Position-error metrics show whether GS branch sharing improves
      tracking accuracy.
    - NIS metrics show whether Qnonlocal reduces over-confidence.
    - trace(P_eta_eta) shows whether covariance inflation is being reflected
      in the physical-state covariance.
    - GS upload and branch-use diagnostics confirm that the GS branch library
      is active.

Notes:
    - All cases use the same random seed for fair comparison.
    - theta0_std is kept at zero, assuming initLocalDNNEKF.m does not call
      randn when theta0_std = 0.
    - This script does not implement event-triggered communication yet.
      It evaluates the current frequent-communication Step 04 baseline.
%}

    clearvars -except out;
    clc;
    close all;

    addpath(genpath(pwd));
    rehash;

    % ---------------------------------------------------------------------
    % Base configuration
    % ---------------------------------------------------------------------
    cfgBase = config_step04_GS_DNN_EKF();

    cfgBase.truth.residualAmp = 5e-4;
    cfgBase.dnn.theta0_std = 0.0;
    cfgBase.dnn.residualInjectionGain = 1.0;

    % Keep the block covariance predictor as the main implementation.
    cfgBase.ekf.useBlockCovPrediction = true;

    seed = 100;

    % ---------------------------------------------------------------------
    % Case 1: Local DNN-EKF baseline
    % ---------------------------------------------------------------------
    cfgLocal = cfgBase;
    cfgLocal.step.name = "step04d_local_DNN";
    cfgLocal.estimator.type = "local_DNN_EKF";
    cfgLocal.dnn.predictionResidualSource = "local_DNN";

    if isfield(cfgLocal, "gs")
        cfgLocal.gs.enabled = false;
        cfgLocal.gs.useNonlocalBranchCovariance = false;
    end

    fprintf("\n============================================================\n");
    fprintf("Running Case 1: Local DNN-EKF\n");
    fprintf("============================================================\n");

    rng(seed);
    tic;
    resLocal = simulateLocalDNNEKF(cfgLocal);
    runtimeLocal = toc;

    % ---------------------------------------------------------------------
    % Case 2: GS composite without Qnonlocal
    % ---------------------------------------------------------------------
    cfgGS = cfgBase;
    cfgGS.step.name = "step04d_GS_composite_no_Qnonlocal";
    cfgGS.estimator.type = "GS_DNN_EKF";
    cfgGS.dnn.predictionResidualSource = "GS_composite";
    cfgGS.gs.enabled = true;
    cfgGS.gs.bootstrapUpload = true;
    cfgGS.gs.uploadMode = "after_measurement_update";
    cfgGS.gs.broadcastMode = "every_step";
    cfgGS.gs.useNonlocalBranchCovariance = false;

    fprintf("\n============================================================\n");
    fprintf("Running Case 2: GS composite DNN-EKF without Qnonlocal\n");
    fprintf("============================================================\n");

    rng(seed);
    tic;
    resGS = simulate_GS_DNN_EKF(cfgGS);
    runtimeGS = toc;

    % ---------------------------------------------------------------------
    % Case 3: GS composite with Qnonlocal
    % ---------------------------------------------------------------------
    cfgGSQ = cfgBase;
    cfgGSQ.step.name = "step04d_GS_composite_with_Qnonlocal";
    cfgGSQ.estimator.type = "GS_DNN_EKF";
    cfgGSQ.dnn.predictionResidualSource = "GS_composite";
    cfgGSQ.gs.enabled = true;
    cfgGSQ.gs.bootstrapUpload = true;
    cfgGSQ.gs.uploadMode = "after_measurement_update";
    cfgGSQ.gs.broadcastMode = "every_step";
    cfgGSQ.gs.useNonlocalBranchCovariance = true;

    % First default: use the already validated uniform Young bound.
    cfgGSQ.gs.youngMode = "uniform";

    fprintf("\n============================================================\n");
    fprintf("Running Case 3: GS composite DNN-EKF with Qnonlocal\n");
    fprintf("============================================================\n");

    rng(seed);
    tic;
    resGSQ = simulate_GS_DNN_EKF(cfgGSQ);
    runtimeGSQ = toc;

    % ---------------------------------------------------------------------
    % Case 4: Oracle residual EKF
    % ---------------------------------------------------------------------
    cfgOracle = cfgBase;
    cfgOracle.step.name = "step04d_oracle_residual";
    cfgOracle.estimator.type = "oracle_residual_EKF";
    cfgOracle.dnn.predictionResidualSource = "oracle";

    if isfield(cfgOracle, "gs")
        cfgOracle.gs.enabled = false;
        cfgOracle.gs.useNonlocalBranchCovariance = false;
    end

    fprintf("\n============================================================\n");
    fprintf("Running Case 4: Oracle residual EKF\n");
    fprintf("============================================================\n");

    rng(seed);
    tic;
    resOracle = simulateLocalDNNEKF(cfgOracle);
    runtimeOracle = toc;

    % ---------------------------------------------------------------------
    % Metrics and summary table
    % ---------------------------------------------------------------------
    metLocal  = computeStep04dMetrics(resLocal,  cfgBase, "Local DNN",        runtimeLocal);
    metGS     = computeStep04dMetrics(resGS,     cfgBase, "GS composite",     runtimeGS);
    metGSQ    = computeStep04dMetrics(resGSQ,    cfgBase, "GS + Qnonlocal",  runtimeGSQ);
    metOracle = computeStep04dMetrics(resOracle, cfgBase, "Oracle",          runtimeOracle);

    summaryTable = struct2table([metLocal; metGS; metGSQ; metOracle]);

    fprintf("\n============================================================\n");
    fprintf("Step 04d GS composite and Qnonlocal comparison summary\n");
    fprintf("============================================================\n");
    disp(summaryTable);

    printStep04dInterpretation(metLocal, metGS, metGSQ, metOracle);

    % ---------------------------------------------------------------------
    % Basic assertions for GS cases
    % ---------------------------------------------------------------------
    assert(resGS.gsNumTotalUploads(1) >= cfgBase.Nw, ...
        "GS no-Q case: bootstrap upload did not initialize all GS branches.");

    assert(resGSQ.gsNumTotalUploads(1) >= cfgBase.Nw, ...
        "GS+Q case: bootstrap upload did not initialize all GS branches.");

    assert(all(resGS.numNonlocalBranchesUsed(1,:) == cfgBase.Nw - 1), ...
        "GS no-Q case: bootstrap broadcast did not activate all nonlocal branches.");

    assert(all(resGSQ.numNonlocalBranchesUsed(1,:) == cfgBase.Nw - 1), ...
        "GS+Q case: bootstrap broadcast did not activate all nonlocal branches.");

    % ---------------------------------------------------------------------
    % Plots
    % ---------------------------------------------------------------------
    plotStep04dComparison(resLocal, resGS, resGSQ, resOracle, cfgBase);

    % ---------------------------------------------------------------------
    % Output
    % ---------------------------------------------------------------------
    out = struct();

    out.cfgBase = cfgBase;
    out.cfgLocal = cfgLocal;
    out.cfgGS = cfgGS;
    out.cfgGSQ = cfgGSQ;
    out.cfgOracle = cfgOracle;

    out.resLocal = resLocal;
    out.resGS = resGS;
    out.resGSQ = resGSQ;
    out.resOracle = resOracle;

    out.metricsLocal = metLocal;
    out.metricsGS = metGS;
    out.metricsGSQ = metGSQ;
    out.metricsOracle = metOracle;

    out.summaryTable = summaryTable;

end

function metrics = computeStep04dMetrics(results, cfg, caseName, runtimeSec)
%{
Function:
    computeStep04dMetrics

Purpose:
    Compute a common metric row for Step 04d comparison cases.

Inputs:
    results    - Simulation output structure.
    cfg        - Simulation configuration.
    caseName   - Display name for the case.
    runtimeSec - Measured wall-clock runtime in seconds.

Outputs:
    metrics - Scalar structure suitable for struct2table.
%}

    pos = computePositionMetrics(results, cfg);
    nis = computeNISMetrics(results, cfg);
    ptr = computeTracePetaMetrics(results, cfg);
    gs  = computeGSDiagnostics(results);

    metrics = struct();
    metrics.caseName = string(caseName);

    metrics.meanPosErr = pos.meanPosErr;
    metrics.rmsPosErr = pos.rmsPosErr;
    metrics.finalMeanPosErr = pos.finalMeanPosErr;

    metrics.meanNIS = nis.meanNIS;
    metrics.medianNIS = nis.medianNIS;
    metrics.nis95ViolationRate = nis.nis95ViolationRate;

    metrics.meanTracePeta = ptr.meanTracePeta;
    metrics.finalMeanTracePeta = ptr.finalMeanTracePeta;

    metrics.finalGSUploads = gs.finalGSUploads;
    metrics.finalMeanNonlocalBranchesUsed = gs.finalMeanNonlocalBranchesUsed;
    metrics.finalMeanTraceQnonlocal = gs.finalMeanTraceQnonlocal;
    metrics.finalMeanTraceSdNonlocal = gs.finalMeanTraceSdNonlocal;

    metrics.runtimeSec = runtimeSec;

end

function metrics = computePositionMetrics(results, cfg)
% Compute position-error metrics from a simulation result.

    dim = cfg.dim;
    Nw = cfg.Nw;

    xhat = results.xhat;
    etaTrue = results.etaTrue;

    N = size(etaTrue, 2);

    rHat = xhat(1:dim, :, :);
    rTrue = reshape(etaTrue(1:dim, :), dim, N, 1);

    e = rHat - rTrue;
    posErr = squeeze(sqrt(sum(e.^2, 1)));

    if Nw == 1
        posErr = posErr(:);
    end

    metrics = struct();
    metrics.posErr = posErr;
    metrics.meanPosErr = mean(posErr(:), "omitnan");
    metrics.rmsPosErr = sqrt(mean(posErr(:).^2, "omitnan"));
    metrics.finalMeanPosErr = mean(posErr(end, :), "omitnan");
    metrics.finalMaxPosErr = max(posErr(end, :), [], "omitnan");

end

function metrics = computeNISMetrics(results, cfg)
% Compute NIS consistency metrics.

    metrics = struct();
    metrics.meanNIS = NaN;
    metrics.medianNIS = NaN;
    metrics.nis95ViolationRate = NaN;

    if ~isfield(results, "NIS") || isempty(results.NIS)
        return;
    end

    nisVals = results.NIS(:);
    nisVals = nisVals(isfinite(nisVals));

    if isempty(nisVals)
        return;
    end

    nis95 = getNIS95Threshold(cfg);

    metrics.meanNIS = mean(nisVals, "omitnan");
    metrics.medianNIS = median(nisVals, "omitnan");
    metrics.nis95ViolationRate = mean(nisVals > nis95);

end

function nis95 = getNIS95Threshold(cfg)
% Return the 95 percent chi-square threshold for the measurement dimension.
%
% This avoids requiring the Statistics Toolbox for chi2inv.

    if cfg.dim == 2
        % Bearing angle is scalar.
        nis95 = 3.8414588207;
    elseif cfg.dim == 3
        % Bearing measurement is [azimuth; elevation].
        nis95 = 5.9914645471;
    else
        error("Unsupported cfg.dim = %d.", cfg.dim);
    end

end

function metrics = computeTracePetaMetrics(results, cfg)
% Compute trace(P_eta_eta) metrics from logged covariance diagonals.

    metrics = struct();
    metrics.meanTracePeta = NaN;
    metrics.finalMeanTracePeta = NaN;

    if ~isfield(results, "PdiagEta") || isempty(results.PdiagEta)
        return;
    end

    PdiagEta = results.PdiagEta;

    % PdiagEta has size nEta x N x Nw.
    tracePeta = squeeze(sum(PdiagEta, 1));

    if cfg.Nw == 1
        tracePeta = tracePeta(:);
    end

    metrics.meanTracePeta = mean(tracePeta(:), "omitnan");
    metrics.finalMeanTracePeta = mean(tracePeta(end, :), "omitnan");

end

function gs = computeGSDiagnostics(results)
% Extract GS communication and final Qnonlocal diagnostics when available.

    gs = struct();
    gs.finalGSUploads = NaN;
    gs.finalMeanNonlocalBranchesUsed = NaN;
    gs.finalMeanTraceQnonlocal = NaN;
    gs.finalMeanTraceSdNonlocal = NaN;

    if isfield(results, "gsNumTotalUploads") && ~isempty(results.gsNumTotalUploads)
        gs.finalGSUploads = results.gsNumTotalUploads(end);
    end

    if isfield(results, "numNonlocalBranchesUsed") && ~isempty(results.numNonlocalBranchesUsed)
        gs.finalMeanNonlocalBranchesUsed = mean(results.numNonlocalBranchesUsed(end,:), "omitnan");
    end

    if ~isfield(results, "watchersFinal") || isempty(results.watchersFinal)
        return;
    end

    Nw = numel(results.watchersFinal);
    traceQ = NaN(Nw, 1);
    traceSd = NaN(Nw, 1);

    for i = 1:Nw
        watcher = results.watchersFinal(i);

        if ~isfield(watcher, "lastNonlocalCovInjection")
            continue;
        end

        diagInfo = watcher.lastNonlocalCovInjection;

        if isfield(diagInfo, "traceQnonlocal")
            traceQ(i) = diagInfo.traceQnonlocal;
        end

        if isfield(diagInfo, "traceSdNonlocal")
            traceSd(i) = diagInfo.traceSdNonlocal;
        end
    end

    gs.finalMeanTraceQnonlocal = mean(traceQ, "omitnan");
    gs.finalMeanTraceSdNonlocal = mean(traceSd, "omitnan");

end

function printStep04dInterpretation(metLocal, metGS, metGSQ, metOracle)
% Print the main percentage comparisons.

    localMean = metLocal.meanPosErr;
    gsMean = metGS.meanPosErr;
    gsQMean = metGSQ.meanPosErr;
    oracleMean = metOracle.meanPosErr;

    gsImprovementPct = 100 * (localMean - gsMean) / localMean;
    gsQImprovementPct = 100 * (localMean - gsQMean) / localMean;

    gapLocalOracle = localMean - oracleMean;

    if gapLocalOracle > 0
        gapClosedGS = 100 * (localMean - gsMean) / gapLocalOracle;
        gapClosedGSQ = 100 * (localMean - gsQMean) / gapLocalOracle;
    else
        gapClosedGS = NaN;
        gapClosedGSQ = NaN;
    end

    fprintf("\nPosition-error interpretation:\n");
    fprintf("  GS improvement over Local DNN        = %.3f %%\n", gsImprovementPct);
    fprintf("  GS+Q improvement over Local DNN      = %.3f %%\n", gsQImprovementPct);
    fprintf("  GS closed Local-to-Oracle gap        = %.3f %%\n", gapClosedGS);
    fprintf("  GS+Q closed Local-to-Oracle gap      = %.3f %%\n", gapClosedGSQ);

    fprintf("\nConsistency interpretation:\n");
    fprintf("  GS mean NIS                          = %.6g\n", metGS.meanNIS);
    fprintf("  GS+Q mean NIS                        = %.6g\n", metGSQ.meanNIS);
    fprintf("  GS NIS 95%% violation rate            = %.3f %%\n", 100 * metGS.nis95ViolationRate);
    fprintf("  GS+Q NIS 95%% violation rate          = %.3f %%\n", 100 * metGSQ.nis95ViolationRate);
    fprintf("  GS mean trace(P_eta_eta)             = %.6g\n", metGS.meanTracePeta);
    fprintf("  GS+Q mean trace(P_eta_eta)           = %.6g\n", metGSQ.meanTracePeta);

    fprintf("\nQnonlocal final diagnostics:\n");
    fprintf("  GS final mean trace(Qnonlocal)       = %.6g\n", metGS.finalMeanTraceQnonlocal);
    fprintf("  GS+Q final mean trace(Qnonlocal)     = %.6g\n", metGSQ.finalMeanTraceQnonlocal);
    fprintf("  GS+Q final mean trace(SdNonlocal)    = %.6g\n", metGSQ.finalMeanTraceSdNonlocal);

end

function plotStep04dComparison(resLocal, resGS, resGSQ, resOracle, cfg)
% Plot Step 04d position, consistency, covariance, and GS diagnostics.

    time = resLocal.time;

    posLocal  = computePositionMetrics(resLocal,  cfg);
    posGS     = computePositionMetrics(resGS,     cfg);
    posGSQ    = computePositionMetrics(resGSQ,    cfg);
    posOracle = computePositionMetrics(resOracle, cfg);

    meanErrLocal  = mean(posLocal.posErr,  2, "omitnan");
    meanErrGS     = mean(posGS.posErr,     2, "omitnan");
    meanErrGSQ    = mean(posGSQ.posErr,    2, "omitnan");
    meanErrOracle = mean(posOracle.posErr, 2, "omitnan");

    figure;
    plot(time, meanErrLocal, "LineWidth", 1.5);
    hold on;
    plot(time, meanErrGS, "LineWidth", 1.5);
    plot(time, meanErrGSQ, "LineWidth", 1.5);
    plot(time, meanErrOracle, "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Mean position error over watchers [m]");
    legend("Local DNN", "GS composite", "GS + Qnonlocal", "Oracle", "Location", "best");
    title("Step 04d position-error comparison");

    figure;
    plotMeanNIS(time, resLocal,  "Local DNN");
    hold on;
    plotMeanNIS(time, resGS,     "GS composite");
    plotMeanNIS(time, resGSQ,    "GS + Qnonlocal");
    plotMeanNIS(time, resOracle, "Oracle");
    grid on;
    xlabel("Time [s]");
    ylabel("Mean NIS over available measurements");
    legend("Location", "best");
    title("Step 04d NIS comparison");

    figure;
    plotTracePeta(time, resLocal,  "Local DNN");
    hold on;
    plotTracePeta(time, resGS,     "GS composite");
    plotTracePeta(time, resGSQ,    "GS + Qnonlocal");
    plotTracePeta(time, resOracle, "Oracle");
    grid on;
    xlabel("Time [s]");
    ylabel("Mean trace(P_{\eta\eta}) over watchers");
    legend("Location", "best");
    title("Step 04d physical covariance trace comparison");

    figure;
    plot(resGS.time, resGS.gsNumTotalUploads, "LineWidth", 1.5);
    hold on;
    plot(resGSQ.time, resGSQ.gsNumTotalUploads, "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Cumulative GS uploads");
    legend("GS composite", "GS + Qnonlocal", "Location", "best");
    title("Step 04d GS upload count");

    figure;
    plot(resGS.time, resGS.numNonlocalBranchesUsed, "LineWidth", 1.5);
    hold on;
    plot(resGSQ.time, resGSQ.numNonlocalBranchesUsed, "LineWidth", 1.5, "LineStyle", "--");
    grid on;
    xlabel("Time [s]");
    ylabel("Number of nonlocal branches used");
    title("Step 04d active nonlocal GS branches per watcher");

end

function plotMeanNIS(time, results, displayName)
% Plot mean NIS over watchers at each time when available.

    if ~isfield(results, "NIS") || isempty(results.NIS)
        return;
    end

    meanNIS = mean(results.NIS, 2, "omitnan");
    plot(time, meanNIS, "LineWidth", 1.5, "DisplayName", displayName);

end

function plotTracePeta(time, results, displayName)
% Plot mean trace(P_eta_eta) over watchers.

    if ~isfield(results, "PdiagEta") || isempty(results.PdiagEta)
        return;
    end

    tracePeta = squeeze(sum(results.PdiagEta, 1));
    meanTracePeta = mean(tracePeta, 2, "omitnan");
    plot(time, meanTracePeta, "LineWidth", 1.5, "DisplayName", displayName);

end
