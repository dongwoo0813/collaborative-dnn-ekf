function out = run_step04b_covariance_comparison()
%{
File:
    scripts/run_step04b_covariance_comparison.m

Purpose:
    Compare Step 04a and Step 04b GS-assisted DNN-EKF.

Cases:
    1. Local DNN-EKF
    2. GS composite without nonlocal branch covariance injection
    3. GS composite with nonlocal branch covariance injection
    4. Oracle residual EKF

Main question:
    Does adding Q_{X,k,-m} improve covariance consistency, NIS behavior, or
    RMSE relative to Step 04a?

How to run:
    out = run_step04b_covariance_comparison();
%}

    clearvars -except out;
    clc;
    close all;

    addpath(genpath(pwd));
    rehash;

    seed = 100;

    cfgBase = config_step04_GS_DNN_EKF();

    cfgBase.truth.residualAmp = 5e-4;
    cfgBase.dnn.theta0_std = 0.0;
    cfgBase.dnn.residualInjectionGain = 1.0;

    % ------------------------------------------------------------
    % Case 1: Local DNN
    % ------------------------------------------------------------
    cfgLocal = cfgBase;
    cfgLocal.dnn.predictionResidualSource = "local_DNN";
    cfgLocal.gs.enabled = false;

    fprintf("\n============================================================\n");
    fprintf("Running Case 1: Local DNN-EKF\n");
    fprintf("============================================================\n");

    rng(seed);
    resLocal = simulateLocalDNNEKF(cfgLocal);

    % ------------------------------------------------------------
    % Case 2: GS composite, Step 04a
    % ------------------------------------------------------------
    cfgGS04a = cfgBase;
    cfgGS04a.dnn.predictionResidualSource = "GS_composite";
    cfgGS04a.gs.enabled = true;
    cfgGS04a.gs.bootstrapUpload = true;
    cfgGS04a.gs.uploadMode = "after_measurement_update";
    cfgGS04a.gs.broadcastMode = "every_step";
    cfgGS04a.gs.useNonlocalBranchCovariance = false;

    fprintf("\n============================================================\n");
    fprintf("Running Case 2: GS composite, no nonlocal covariance\n");
    fprintf("============================================================\n");

    rng(seed);
    resGS04a = simulate_GS_DNN_EKF(cfgGS04a);

    % ------------------------------------------------------------
    % Case 3: GS composite, Step 04b
    % ------------------------------------------------------------
    cfgGS04b = cfgGS04a;
    cfgGS04b.gs.useNonlocalBranchCovariance = true;
    cfgGS04b.gs.youngMode = "uniform";
    cfgGS04b.gs.SresNonlocal = 0.0;

    fprintf("\n============================================================\n");
    fprintf("Running Case 3: GS composite, with nonlocal covariance\n");
    fprintf("============================================================\n");

    rng(seed);
    resGS04b = simulate_GS_DNN_EKF(cfgGS04b);

    % ------------------------------------------------------------
    % Case 4: Oracle
    % ------------------------------------------------------------
    cfgOracle = cfgBase;
    cfgOracle.dnn.predictionResidualSource = "oracle";
    cfgOracle.gs.enabled = false;

    fprintf("\n============================================================\n");
    fprintf("Running Case 4: Oracle residual EKF\n");
    fprintf("============================================================\n");

    rng(seed);
    resOracle = simulateLocalDNNEKF(cfgOracle);

    % ------------------------------------------------------------
    % Metrics
    % ------------------------------------------------------------
    metricsLocal = computeMetrics(resLocal, cfgBase);
    metricsGS04a = computeMetrics(resGS04a, cfgBase);
    metricsGS04b = computeMetrics(resGS04b, cfgBase);
    metricsOracle = computeMetrics(resOracle, cfgBase);

    caseName = [
        "Local DNN"
        "GS composite 04a"
        "GS composite 04b"
        "Oracle"
    ];

    meanPosErr = [
        metricsLocal.meanPosErr
        metricsGS04a.meanPosErr
        metricsGS04b.meanPosErr
        metricsOracle.meanPosErr
    ];

    rmsPosErr = [
        metricsLocal.rmsPosErr
        metricsGS04a.rmsPosErr
        metricsGS04b.rmsPosErr
        metricsOracle.rmsPosErr
    ];

    finalMeanPosErr = [
        metricsLocal.finalMeanPosErr
        metricsGS04a.finalMeanPosErr
        metricsGS04b.finalMeanPosErr
        metricsOracle.finalMeanPosErr
    ];

    meanNIS = [
        metricsLocal.meanNIS
        metricsGS04a.meanNIS
        metricsGS04b.meanNIS
        metricsOracle.meanNIS
    ];

    medianNIS = [
        metricsLocal.medianNIS
        metricsGS04a.medianNIS
        metricsGS04b.medianNIS
        metricsOracle.medianNIS
    ];

    summaryTable = table(caseName, meanPosErr, rmsPosErr, finalMeanPosErr, meanNIS, medianNIS);

    fprintf("\n============================================================\n");
    fprintf("Step 04b covariance comparison summary\n");
    fprintf("============================================================\n");
    disp(summaryTable);

    improvement04a = 100 * (metricsLocal.meanPosErr - metricsGS04a.meanPosErr) / metricsLocal.meanPosErr;
    improvement04b = 100 * (metricsLocal.meanPosErr - metricsGS04b.meanPosErr) / metricsLocal.meanPosErr;

    fprintf("GS 04a improvement over Local DNN = %.3f %%\n", improvement04a);
    fprintf("GS 04b improvement over Local DNN = %.3f %%\n", improvement04b);

    fprintf("\nNIS reference dimension:\n");
    if cfgBase.dim == 2
        fprintf("  Expected mean NIS roughly near 1 for consistent scalar bearing measurements.\n");
    elseif cfgBase.dim == 3
        fprintf("  Expected mean NIS roughly near 2 for consistent az/el measurements.\n");
    end

    % ------------------------------------------------------------
    % Plot
    % ------------------------------------------------------------
    plotComparison(resLocal, resGS04a, resGS04b, resOracle, cfgBase);

    % ------------------------------------------------------------
    % Output
    % ------------------------------------------------------------
    out = struct();

    out.cfgLocal = cfgLocal;
    out.cfgGS04a = cfgGS04a;
    out.cfgGS04b = cfgGS04b;
    out.cfgOracle = cfgOracle;

    out.resLocal = resLocal;
    out.resGS04a = resGS04a;
    out.resGS04b = resGS04b;
    out.resOracle = resOracle;

    out.metricsLocal = metricsLocal;
    out.metricsGS04a = metricsGS04a;
    out.metricsGS04b = metricsGS04b;
    out.metricsOracle = metricsOracle;

    out.summaryTable = summaryTable;
    out.improvement04a = improvement04a;
    out.improvement04b = improvement04b;

end

function metrics = computeMetrics(results, cfg)
% Compute position-error and NIS summary metrics.

    dim = cfg.dim;
    Nw = cfg.Nw;

    etaTrue = results.etaTrue;
    xhat = results.xhat;

    N = size(etaTrue, 2);

    rTrue = reshape(etaTrue(1:dim, :), dim, N, 1);
    rHat = xhat(1:dim, :, :);

    e = rHat - rTrue;
    posErr = squeeze(sqrt(sum(e.^2, 1)));

    if Nw == 1
        posErr = posErr(:);
    end

    nis = results.NIS(:);
    nis = nis(isfinite(nis));

    metrics = struct();

    metrics.posErr = posErr;
    metrics.meanPosErr = mean(posErr(:));
    metrics.rmsPosErr = sqrt(mean(posErr(:).^2));
    metrics.finalMeanPosErr = mean(posErr(end, :));

    if isempty(nis)
        metrics.meanNIS = NaN;
        metrics.medianNIS = NaN;
    else
        metrics.meanNIS = mean(nis);
        metrics.medianNIS = median(nis);
    end

end

function plotComparison(resLocal, resGS04a, resGS04b, resOracle, cfg)
% Plot mean position error and NIS histories.

    time = resLocal.time;

    mLocal = computeMetrics(resLocal, cfg);
    m04a = computeMetrics(resGS04a, cfg);
    m04b = computeMetrics(resGS04b, cfg);
    mOracle = computeMetrics(resOracle, cfg);

    meanErrLocal = mean(mLocal.posErr, 2);
    meanErr04a = mean(m04a.posErr, 2);
    meanErr04b = mean(m04b.posErr, 2);
    meanErrOracle = mean(mOracle.posErr, 2);

    figure;
    plot(time, meanErrLocal, "LineWidth", 1.5);
    hold on;
    plot(time, meanErr04a, "LineWidth", 1.5);
    plot(time, meanErr04b, "LineWidth", 1.5);
    plot(time, meanErrOracle, "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Mean position error over watchers [m]");
    legend("Local DNN", "GS 04a", "GS 04b", "Oracle", "Location", "best");
    title("Step 04b position-error comparison");

    figure;
    plot(time, mean(resGS04a.numNonlocalBranchesUsed, 2), "LineWidth", 1.5);
    hold on;
    plot(time, mean(resGS04b.numNonlocalBranchesUsed, 2), "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Mean active nonlocal branches");
    legend("GS 04a", "GS 04b", "Location", "best");
    title("Active nonlocal GS branches");

    figure;
    plotNISMean(time, resLocal.NIS, "Local DNN");
    hold on;
    plotNISMean(time, resGS04a.NIS, "GS 04a");
    plotNISMean(time, resGS04b.NIS, "GS 04b");
    plotNISMean(time, resOracle.NIS, "Oracle");
    grid on;
    xlabel("Time [s]");
    ylabel("Mean NIS over watchers");
    legend("Location", "best");
    title("Step 04b NIS comparison");

end

function plotNISMean(time, NIS, labelName)
% Plot mean NIS over watchers at each time.

    N = size(NIS, 1);
    meanNIS = NaN(N, 1);

    for k = 1:N
        vals = NIS(k, :);
        vals = vals(isfinite(vals));
        if ~isempty(vals)
            meanNIS(k) = mean(vals);
        end
    end

    plot(time, meanNIS, "LineWidth", 1.5, "DisplayName", labelName);

end