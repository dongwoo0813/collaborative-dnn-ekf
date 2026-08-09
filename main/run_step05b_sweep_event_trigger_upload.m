function out = run_step05b_sweep_event_trigger_upload()
%{
File:
    main/run_step05b_sweep_event_trigger_upload.m

Purpose:
    Sweep Step 05 event-triggered GS upload parameters.

What this script does:
    Compare frequent GS upload against event-triggered upload over a grid of:

        cfg.gs.eventDeltaThreshold
        cfg.gs.eventDwellSteps

Main question:
    How much communication can be reduced while keeping tracking performance
    close to the frequent-upload GS-composite baseline?

Outputs:
    out.summaryTable
    out.deltaThresholdList
    out.dwellStepsList
    out.resFreq
    out.cfgFreq

Notes:
    This is not a sanity check.
    This is a tuning/ablation experiment for event-triggered communication.

    Qnonlocal is kept OFF here so that the sweep isolates the communication
    trigger effect first. After this sweep is stable, we can repeat with
    Qnonlocal ON.
%}

    clearvars -except out;
    clc;
    close all;

    addpath(genpath(pwd));
    rehash;

    fprintf("\n============================================================\n");
    fprintf("Step 05-B sweep: event-triggered GS upload\n");
    fprintf("============================================================\n");

    seed = 100;

    % ---------------------------------------------------------------------
    % Sweep grid.
    % Smaller Delta threshold = easier upload.
    % Larger dwell steps      = less frequent upload.
    % ---------------------------------------------------------------------
    deltaThresholdList = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12];
    dwellStepsList     = [5, 10, 20, 50, 100];

    % ---------------------------------------------------------------------
    % Frequent-upload baseline.
    % This is the Step 04 GS-composite upper-communication baseline.
    % ---------------------------------------------------------------------
    cfgFreq = makeBaseStep05SweepConfig();
    cfgFreq.gs.uploadMode = "after_measurement_update";

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Running frequent-upload GS baseline\n");
    fprintf("------------------------------------------------------------\n");

    rng(seed);
    tic;
    resFreq = simulate_GS_DNN_EKF(cfgFreq);
    runtimeFreq = toc;

    metricsFreq = computeSweepMetrics( ...
        resFreq, cfgFreq, ...
        "frequent", NaN, NaN, ...
        NaN, NaN, runtimeFreq);

    fprintf("Frequent uploads      = %d\n", metricsFreq.finalUploads);
    fprintf("Frequent meanPosErr   = %.6f\n", metricsFreq.meanPosErr);
    fprintf("Frequent mean NIS     = %.6f\n", metricsFreq.meanNIS);

    % ---------------------------------------------------------------------
    % Event-triggered grid sweep.
    % ---------------------------------------------------------------------
    numCases = numel(deltaThresholdList) * numel(dwellStepsList);

    metrics(numCases, 1) = emptyMetricStruct();
    resultsCell = cell(numCases, 1);
    cfgCell = cell(numCases, 1);

    caseIdx = 0;

    for iDelta = 1:numel(deltaThresholdList)

        deltaThreshold = deltaThresholdList(iDelta);

        for iDwell = 1:numel(dwellStepsList)

            dwellSteps = dwellStepsList(iDwell);

            caseIdx = caseIdx + 1;

            cfgET = makeBaseStep05SweepConfig();

            % Event-triggered upload mode.
            cfgET.gs.uploadMode = "event_contribution_change";

            % Swept trigger parameters.
            cfgET.gs.eventDeltaThreshold = deltaThreshold;
            cfgET.gs.eventDwellSteps = dwellSteps;

            fprintf("\n------------------------------------------------------------\n");
            fprintf("Running event-trigger case %d/%d\n", caseIdx, numCases);
            fprintf("  Delta threshold = %.3e\n", deltaThreshold);
            fprintf("  Dwell steps     = %d\n", dwellSteps);
            fprintf("------------------------------------------------------------\n");

            rng(seed);
            tic;
            resET = simulate_GS_DNN_EKF(cfgET);
            runtimeSec = toc;

            metrics(caseIdx) = computeSweepMetrics( ...
                resET, cfgET, ...
                "event", deltaThreshold, dwellSteps, ...
                metricsFreq.finalUploads, metricsFreq.meanPosErr, runtimeSec);

            resultsCell{caseIdx} = resET;
            cfgCell{caseIdx} = cfgET;

            fprintf("  final uploads              = %d\n", metrics(caseIdx).finalUploads);
            fprintf("  upload reduction           = %.3f %%\n", metrics(caseIdx).uploadReductionPercent);
            fprintf("  mean position error         = %.6f\n", metrics(caseIdx).meanPosErr);
            fprintf("  mean position error increase= %.3f %%\n", metrics(caseIdx).meanPosErrIncreasePercent);
            fprintf("  event upload decisions      = %d\n", metrics(caseIdx).eventUploadDecisions);

        end

    end

    % ---------------------------------------------------------------------
    % Combine baseline + sweep table.
    % ---------------------------------------------------------------------
    summaryTable = [struct2table(metricsFreq); struct2table(metrics)];

    fprintf("\n============================================================\n");
    fprintf("Step 05-B event-trigger sweep summary\n");
    fprintf("============================================================\n");
    disp(summaryTable);

    % ---------------------------------------------------------------------
    % Pick simple recommended settings.
    % Criterion:
    %   among cases with mean position error increase <= 1%,
    %   choose the one with minimum uploads.
    % ---------------------------------------------------------------------
    recommended = pickRecommendedCase(summaryTable);

    fprintf("\nRecommended case under <= 1%% meanPosErr increase:\n");
    disp(recommended);

    % ---------------------------------------------------------------------
    % Plot heatmaps for quick visual inspection.
    % ---------------------------------------------------------------------
    plotStep05Sweep(summaryTable, deltaThresholdList, dwellStepsList);

    out = struct();

    out.deltaThresholdList = deltaThresholdList;
    out.dwellStepsList = dwellStepsList;

    out.cfgFreq = cfgFreq;
    out.resFreq = resFreq;
    out.metricsFreq = metricsFreq;

    out.cfgCell = cfgCell;
    out.resultsCell = resultsCell;
    out.metrics = metrics;

    out.summaryTable = summaryTable;
    out.recommended = recommended;

end

function cfg = makeBaseStep05SweepConfig()
% Shared Step 05 sweep configuration.
%
% This should match the Step 04d GS-composite comparison setting, except
% that the upload policy is varied by the sweep.

    cfg = config_step04_GS_DNN_EKF();

    % Unknown residual strength used in previous Step 04d comparisons.
    cfg.truth.residualAmp = 5e-4;

    % Fair comparison: do not consume randn for theta initialization.
    cfg.dnn.theta0_std = 0.0;

    % Use full GS composite residual model:
    % local branch + GS nonlocal branches.
    cfg.dnn.predictionResidualSource = "GS_composite";
    cfg.dnn.residualInjectionGain = 1.0;

    % GS enabled with initial branch bootstrap.
    cfg.gs.enabled = true;
    cfg.gs.bootstrapUpload = true;

    % Keep broadcast every step for Step 05-B.
    % Only upload is event-triggered.
    cfg.gs.broadcastMode = "every_step";

    % Measurement-supported event trigger.
    cfg.gs.eventRequireMeasurement = true;

    % Keep Qnonlocal OFF in this first event-trigger sweep.
    % This isolates communication-trigger effects from covariance-injection
    % effects.
    cfg.gs.useNonlocalBranchCovariance = false;

    % Main implementation remains block covariance prediction.
    if ~isfield(cfg, "ekf")
        cfg.ekf = struct();
    end
    cfg.ekf.useBlockCovPrediction = true;

end

function metrics = emptyMetricStruct()
% Table row template.
%
% One row = one simulation case.

    metrics = struct();

    metrics.caseType = "";
    metrics.deltaThreshold = NaN;
    metrics.dwellSteps = NaN;

    metrics.finalUploads = NaN;
    metrics.eventUploadDecisions = NaN;
    metrics.uploadReductionPercent = NaN;

    metrics.meanPosErr = NaN;
    metrics.rmsPosErr = NaN;
    metrics.finalMeanPosErr = NaN;
    metrics.meanPosErrIncreasePercent = NaN;

    metrics.meanNIS = NaN;
    metrics.medianNIS = NaN;
    metrics.nis95ViolationRate = NaN;

    metrics.finalMeanNonlocalBranchesUsed = NaN;

    metrics.numDeltaBelowThreshold = NaN;
    metrics.numDwellNotSatisfied = NaN;
    metrics.numEventTriggerPassed = NaN;

    metrics.meanDeltaAtUpload = NaN;
    metrics.minDeltaAtUpload = NaN;
    metrics.maxDeltaAtUpload = NaN;

    metrics.runtimeSec = NaN;

end

function metrics = computeSweepMetrics( ...
    results, cfg, caseType, deltaThreshold, dwellSteps, ...
    baselineUploads, baselineMeanPosErr, runtimeSec)
% Compute one summary row for the sweep table.

    metrics = emptyMetricStruct();

    metrics.caseType = string(caseType);
    metrics.deltaThreshold = deltaThreshold;
    metrics.dwellSteps = dwellSteps;

    % Communication metrics.
    metrics.finalUploads = getFinalUploads(results);
    metrics.eventUploadDecisions = getEventUploadDecisions(results);

    if isfinite(baselineUploads)
        metrics.uploadReductionPercent = ...
            100 * (baselineUploads - metrics.finalUploads) / baselineUploads;
    end

    % Tracking metrics.
    pos = computePositionMetrics(results, cfg);

    metrics.meanPosErr = pos.meanPosErr;
    metrics.rmsPosErr = pos.rmsPosErr;
    metrics.finalMeanPosErr = pos.finalMeanPosErr;

    if isfinite(baselineMeanPosErr)
        metrics.meanPosErrIncreasePercent = ...
            100 * (metrics.meanPosErr - baselineMeanPosErr) / baselineMeanPosErr;
    end

    % Consistency metrics.
    nis = computeNISMetrics(results, cfg);

    metrics.meanNIS = nis.meanNIS;
    metrics.medianNIS = nis.medianNIS;
    metrics.nis95ViolationRate = nis.nis95ViolationRate;

    % Final nonlocal branch availability.
    if isfield(results, "numNonlocalBranchesUsed") && ~isempty(results.numNonlocalBranchesUsed)
        metrics.finalMeanNonlocalBranchesUsed = ...
            mean(results.numNonlocalBranchesUsed(end, :), "omitnan");
    end

    % Event trigger reason counts and Delta statistics.
    reason = computeReasonMetrics(results);

    metrics.numDeltaBelowThreshold = reason.numDeltaBelowThreshold;
    metrics.numDwellNotSatisfied = reason.numDwellNotSatisfied;
    metrics.numEventTriggerPassed = reason.numEventTriggerPassed;

    metrics.meanDeltaAtUpload = reason.meanDeltaAtUpload;
    metrics.minDeltaAtUpload = reason.minDeltaAtUpload;
    metrics.maxDeltaAtUpload = reason.maxDeltaAtUpload;

    metrics.runtimeSec = runtimeSec;

end

function finalUploads = getFinalUploads(results)
% Final total GS upload count, including bootstrap uploads.

    finalUploads = NaN;

    if isfield(results, "gsNumTotalUploads") && ~isempty(results.gsNumTotalUploads)
        finalUploads = results.gsNumTotalUploads(end);
    end

end

function n = getEventUploadDecisions(results)
% Number of event upload decisions after bootstrap.
%
% Frequent-upload runs may not have gsUploadDecision, so return NaN.

    n = NaN;

    if isfield(results, "gsUploadDecision") && ~isempty(results.gsUploadDecision)
        n = nnz(logical(results.gsUploadDecision));
    end

end

function pos = computePositionMetrics(results, cfg)
% Position error metrics over all watchers and all time.

    dim = cfg.dim;

    xhat = results.xhat;
    etaTrue = results.etaTrue;

    N = size(etaTrue, 2);

    rHat = xhat(1:dim, :, :);
    rTrue = reshape(etaTrue(1:dim, :), dim, N, 1);

    posErr = squeeze(sqrt(sum((rHat - rTrue).^2, 1)));

    pos = struct();
    pos.meanPosErr = mean(posErr(:), "omitnan");
    pos.rmsPosErr = sqrt(mean(posErr(:).^2, "omitnan"));
    pos.finalMeanPosErr = mean(posErr(end, :), "omitnan");

end

function nis = computeNISMetrics(results, cfg)
% NIS summary over valid measurement updates.

    nis = struct();

    nis.meanNIS = NaN;
    nis.medianNIS = NaN;
    nis.nis95ViolationRate = NaN;

    if ~isfield(results, "NIS") || isempty(results.NIS)
        return;
    end

    nisVals = results.NIS(:);
    nisVals = nisVals(isfinite(nisVals));

    if isempty(nisVals)
        return;
    end

    nis95 = getNIS95Threshold(cfg);

    nis.meanNIS = mean(nisVals, "omitnan");
    nis.medianNIS = median(nisVals, "omitnan");
    nis.nis95ViolationRate = mean(nisVals > nis95);

end

function nis95 = getNIS95Threshold(cfg)
% Chi-square 95% threshold for angle-only measurement dimension.
%
% Current code uses:
%   dim = 2 -> one angular measurement component
%   dim = 3 -> two angular measurement components

    if cfg.dim == 2
        nis95 = 3.8414588207;
    elseif cfg.dim == 3
        nis95 = 5.9914645471;
    else
        error("Unsupported cfg.dim = %d.", cfg.dim);
    end

end

function reason = computeReasonMetrics(results)
% Count event-trigger skip/pass reasons and Delta-at-upload statistics.

    reason = struct();

    reason.numDeltaBelowThreshold = NaN;
    reason.numDwellNotSatisfied = NaN;
    reason.numEventTriggerPassed = NaN;

    reason.meanDeltaAtUpload = NaN;
    reason.minDeltaAtUpload = NaN;
    reason.maxDeltaAtUpload = NaN;

    if ~isfield(results, "gsUploadReason") || isempty(results.gsUploadReason)
        return;
    end

    reasonVals = string(results.gsUploadReason(:));

    reason.numDeltaBelowThreshold = nnz(reasonVals == "delta_below_threshold");
    reason.numDwellNotSatisfied = nnz(reasonVals == "dwell_not_satisfied");
    reason.numEventTriggerPassed = nnz(reasonVals == "event_trigger_passed");

    if isfield(results, "gsUploadDecision") && isfield(results, "gsUploadDelta")
        uploadMask = logical(results.gsUploadDecision);
        deltaAtUpload = results.gsUploadDelta(uploadMask);
        deltaAtUpload = deltaAtUpload(isfinite(deltaAtUpload));

        if ~isempty(deltaAtUpload)
            reason.meanDeltaAtUpload = mean(deltaAtUpload, "omitnan");
            reason.minDeltaAtUpload = min(deltaAtUpload);
            reason.maxDeltaAtUpload = max(deltaAtUpload);
        end
    end

end

function recommended = pickRecommendedCase(summaryTable)
% Choose a simple recommended event-trigger setting.
%
% Rule:
%   among event cases with <= 1% meanPosErr increase,
%   pick the one with the fewest final uploads.

    eventMask = summaryTable.caseType == "event";
    perfMask = summaryTable.meanPosErrIncreasePercent <= 1.0;

    validRows = summaryTable(eventMask & perfMask, :);

    if isempty(validRows)
        recommended = table();
        fprintf("\nNo event-trigger case satisfies <= 1%% meanPosErr increase.\n");
        return;
    end

    [~, idx] = min(validRows.finalUploads);
    recommended = validRows(idx, :);

end

function plotStep05Sweep(summaryTable, deltaThresholdList, dwellStepsList)
% Plot heatmaps for communication and performance tradeoff.

    eventRows = summaryTable(summaryTable.caseType == "event", :);

    uploadReductionGrid = makeGrid( ...
        eventRows, deltaThresholdList, dwellStepsList, "uploadReductionPercent");

    meanErrIncreaseGrid = makeGrid( ...
        eventRows, deltaThresholdList, dwellStepsList, "meanPosErrIncreasePercent");

    meanPosErrGrid = makeGrid( ...
        eventRows, deltaThresholdList, dwellStepsList, "meanPosErr");

    meanNISGrid = makeGrid( ...
        eventRows, deltaThresholdList, dwellStepsList, "meanNIS");

    xLabels = compose("%d", dwellStepsList);
    yLabels = compose("%.0e", deltaThresholdList);

    figure;
    imagesc(uploadReductionGrid);
    colorbar;
    xlabel("Dwell steps");
    ylabel("Delta threshold");
    title("Upload reduction [%]");
    set(gca, "XTick", 1:numel(dwellStepsList), "XTickLabel", xLabels);
    set(gca, "YTick", 1:numel(deltaThresholdList), "YTickLabel", yLabels);

    figure;
    imagesc(meanErrIncreaseGrid);
    colorbar;
    xlabel("Dwell steps");
    ylabel("Delta threshold");
    title("Mean position error increase [%]");
    set(gca, "XTick", 1:numel(dwellStepsList), "XTickLabel", xLabels);
    set(gca, "YTick", 1:numel(deltaThresholdList), "YTickLabel", yLabels);

    figure;
    imagesc(meanPosErrGrid);
    colorbar;
    xlabel("Dwell steps");
    ylabel("Delta threshold");
    title("Mean position error [m]");
    set(gca, "XTick", 1:numel(dwellStepsList), "XTickLabel", xLabels);
    set(gca, "YTick", 1:numel(deltaThresholdList), "YTickLabel", yLabels);

    figure;
    imagesc(meanNISGrid);
    colorbar;
    xlabel("Dwell steps");
    ylabel("Delta threshold");
    title("Mean NIS");
    set(gca, "XTick", 1:numel(dwellStepsList), "XTickLabel", xLabels);
    set(gca, "YTick", 1:numel(deltaThresholdList), "YTickLabel", yLabels);

end

function grid = makeGrid(eventRows, deltaThresholdList, dwellStepsList, fieldName)
% Convert long-form sweep table into a 2-D grid.
%
% Rows    = delta thresholds
% Columns = dwell steps

    grid = NaN(numel(deltaThresholdList), numel(dwellStepsList));

    for iDelta = 1:numel(deltaThresholdList)
        for iDwell = 1:numel(dwellStepsList)

            rowMask = ...
                eventRows.deltaThreshold == deltaThresholdList(iDelta) & ...
                eventRows.dwellSteps == dwellStepsList(iDwell);

            if any(rowMask)
                grid(iDelta, iDwell) = eventRows.(fieldName)(find(rowMask, 1, "first"));
            end

        end
    end

end