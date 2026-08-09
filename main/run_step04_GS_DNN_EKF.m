function out = run_step04_GS_DNN_EKF()
%{
Function:
    run_step04_GS_DNN_EKF.m

Purpose:
    Run Step 04 GS-assisted DNN-EKF comparison.

    This script compares:

        1. Local DNN-EKF:
              d_pred,m = d_m(eta; theta_m)

        2. GS composite DNN-EKF:
              d_pred,m = d_m(eta; theta_m)
                         + sum_{j ~= m} d_j(eta; theta_{j|m}^{GS})

        3. Oracle residual EKF:
              d_pred = d_true(eta,t)

    The GS composite case uses simulate_GS_DNN_EKF.m, where watchers upload
    local branch estimates to the ground station and receive nonlocal branch
    copies through GS broadcast.

Outputs:
    out - structure containing all simulation results and summary metrics.

Notes:
    - This is Step 04a.
    - Nonlocal GS branch covariance is not yet injected into EKF covariance.
    - The comparison uses the same random seed for all cases.
    - theta0_std should remain 0 for fair comparison unless intentionally
      testing random initialization.
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

    seed = 100;

    % ---------------------------------------------------------------------
    % Case 1: Local DNN-EKF baseline
    % ---------------------------------------------------------------------
    cfgLocal = cfgBase;
    cfgLocal.step.name = "step04_compare_local_DNN";
    cfgLocal.estimator.type = "local_DNN_EKF";
    cfgLocal.dnn.predictionResidualSource = "local_DNN";

    if isfield(cfgLocal, "gs")
        cfgLocal.gs.enabled = false;
    end

    fprintf("\n============================================================\n");
    fprintf("Running Case 1: Local DNN-EKF\n");
    fprintf("============================================================\n");

    rng(seed);
    resLocal = simulateLocalDNNEKF(cfgLocal);

    % ---------------------------------------------------------------------
    % Case 2: GS composite DNN-EKF
    % ---------------------------------------------------------------------
    cfgGS = cfgBase;
    cfgGS.step.name = "step04_GS_DNN_EKF";
    cfgGS.estimator.type = "GS_DNN_EKF";
    cfgGS.dnn.predictionResidualSource = "GS_composite";
    cfgGS.gs.enabled = true;
    cfgGS.gs.uploadMode = "after_measurement_update";
    cfgGS.gs.broadcastMode = "every_step";
    cfgGS.gs.useNonlocalBranchCovariance = false;

    fprintf("\n============================================================\n");
    fprintf("Running Case 2: GS composite DNN-EKF\n");
    fprintf("============================================================\n");

    rng(seed);
    resGS = simulate_GS_DNN_EKF(cfgGS);

    % ---------------------------------------------------------------------
    % Case 3: Oracle residual EKF
    % ---------------------------------------------------------------------
    cfgOracle = cfgBase;
    cfgOracle.step.name = "step04_compare_oracle";
    cfgOracle.estimator.type = "oracle_residual_EKF";
    cfgOracle.dnn.predictionResidualSource = "oracle";

    if isfield(cfgOracle, "gs")
        cfgOracle.gs.enabled = false;
    end

    fprintf("\n============================================================\n");
    fprintf("Running Case 3: Oracle residual EKF\n");
    fprintf("============================================================\n");

    rng(seed);
    resOracle = simulateLocalDNNEKF(cfgOracle);

    % ---------------------------------------------------------------------
    % Metrics
    % ---------------------------------------------------------------------
    metricsLocal  = computePositionMetrics(resLocal,  cfgBase);
    metricsGS     = computePositionMetrics(resGS,     cfgBase);
    metricsOracle = computePositionMetrics(resOracle, cfgBase);

    localMean  = metricsLocal.meanPosErr;
    gsMean     = metricsGS.meanPosErr;
    oracleMean = metricsOracle.meanPosErr;

    gsImprovementPct = 100 * (localMean - gsMean) / localMean;
    oracleImprovementPct = 100 * (localMean - oracleMean) / localMean;

    gapLocalOracle = localMean - oracleMean;
    gapGSOracle = gsMean - oracleMean;

    if gapLocalOracle > 0
        gapClosedPct = 100 * (localMean - gsMean) / gapLocalOracle;
    else
        gapClosedPct = NaN;
    end

    caseName = [
        "Local DNN"
        "GS composite"
        "Oracle"
    ];

    meanPosErr = [
        metricsLocal.meanPosErr
        metricsGS.meanPosErr
        metricsOracle.meanPosErr
    ];

    rmsPosErr = [
        metricsLocal.rmsPosErr
        metricsGS.rmsPosErr
        metricsOracle.rmsPosErr
    ];

    finalMeanPosErr = [
        metricsLocal.finalMeanPosErr
        metricsGS.finalMeanPosErr
        metricsOracle.finalMeanPosErr
    ];

    summaryTable = table(caseName, meanPosErr, rmsPosErr, finalMeanPosErr);

    fprintf("\n============================================================\n");
    fprintf("Step 04 GS composite comparison summary\n");
    fprintf("============================================================\n");
    disp(summaryTable);

    fprintf("GS improvement over Local DNN       = %.3f %%\n", gsImprovementPct);
    fprintf("Oracle improvement over Local DNN   = %.3f %%\n", oracleImprovementPct);
    fprintf("GS closed Local-to-Oracle gap       = %.3f %%\n", gapClosedPct);

    fprintf("\nGS diagnostics:\n");
    fprintf("  Final GS total uploads = %d\n", resGS.gsNumTotalUploads(end));
    fprintf("  Initial nonlocal branches used per watcher:\n");
    disp(resGS.numNonlocalBranchesUsed(1,:));
    fprintf("  Final nonlocal branches used per watcher:\n");
    disp(resGS.numNonlocalBranchesUsed(end,:));

    % ---------------------------------------------------------------------
    % Basic sanity assertions for the GS run
    % ---------------------------------------------------------------------
    assert(resGS.gsNumTotalUploads(1) >= cfgBase.Nw, ...
        "Bootstrap upload did not initialize all GS branches.");

    assert(all(resGS.numNonlocalBranchesUsed(1,:) == cfgBase.Nw - 1), ...
        "Initial bootstrap broadcast did not activate all nonlocal branches.");

    assert(all(resGS.gsValid(:,1)), ...
        "Not all GS branches are valid after bootstrap upload.");

    % ---------------------------------------------------------------------
    % Plots
    % ---------------------------------------------------------------------
    plotStep04Comparison(resLocal, resGS, resOracle, cfgBase);

    % ---------------------------------------------------------------------
    % Output
    % ---------------------------------------------------------------------
    out = struct();

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
    out.gsImprovementPct = gsImprovementPct;
    out.oracleImprovementPct = oracleImprovementPct;
    out.gapClosedPct = gapClosedPct;

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

    metrics.meanPosErr = mean(posErr(:));
    metrics.rmsPosErr = sqrt(mean(posErr(:).^2));

    metrics.perWatcherMeanPosErr = mean(posErr, 1);
    metrics.perWatcherRMSPosErr = sqrt(mean(posErr.^2, 1));

    metrics.finalMeanPosErr = mean(posErr(end, :));
    metrics.finalMaxPosErr = max(posErr(end, :));

end

function plotStep04Comparison(resLocal, resGS, resOracle, cfg)
% Plot mean position error and GS communication diagnostics.

    time = resLocal.time;

    metricsLocal = computePositionMetrics(resLocal, cfg);
    metricsGS = computePositionMetrics(resGS, cfg);
    metricsOracle = computePositionMetrics(resOracle, cfg);

    meanErrLocal = mean(metricsLocal.posErr, 2);
    meanErrGS = mean(metricsGS.posErr, 2);
    meanErrOracle = mean(metricsOracle.posErr, 2);

    figure;
    plot(time, meanErrLocal, "LineWidth", 1.5);
    hold on;
    plot(time, meanErrGS, "LineWidth", 1.5);
    plot(time, meanErrOracle, "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Mean position error over watchers");
    legend("Local DNN", "GS composite", "Oracle", "Location", "best");
    title("Step 04 position-error comparison");

    figure;
    plot(resGS.time, resGS.gsNumTotalUploads, "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Cumulative GS uploads");
    title("Step 04 GS upload count");

    figure;
    plot(resGS.time, resGS.numNonlocalBranchesUsed, "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Number of nonlocal branches used");
    title("Step 04 active nonlocal GS branches per watcher");

end