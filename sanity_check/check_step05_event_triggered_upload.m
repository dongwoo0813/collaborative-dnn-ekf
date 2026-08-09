function out = check_step05_event_triggered_upload()
%{
File:
    sanity_check/check_step05_event_triggered_upload.m

Purpose:
    Sanity check for Step 05 event-triggered GS upload.

What this checks:
    1. GS event-trigger mode runs without error.
    2. Event-trigger diagnostic logs are saved in the results struct.
    3. Total GS upload count is consistent with bootstrap + event uploads.
    4. Event-triggered upload uses fewer uploads than frequent upload.
    5. Every event upload satisfies measurement, dwell, and Delta conditions.

Event trigger:
    cfg.gs.uploadMode = "event_contribution_change"

    Upload condition:
        measurementSatisfied == true
        deltaContribution >= eventDeltaThreshold
        dwellSatisfied == true

Notes:
    This is not a performance test.
    It checks communication-trigger correctness only.
%}

    clearvars -except out;
    clc;

    addpath(genpath(pwd));
    rehash;

    fprintf("\n============================================================\n");
    fprintf("Step 05 sanity check: event-triggered GS upload\n");
    fprintf("============================================================\n");

    assert(exist("config_step04_GS_DNN_EKF", "file") == 2, ...
        "Missing config_step04_GS_DNN_EKF.m.");

    assert(exist("simulate_GS_DNN_EKF", "file") == 2, ...
        "Missing simulate_GS_DNN_EKF.m.");

    seed = 100;

    % ---------------------------------------------------------------------
    % Frequent-upload reference case.
    % This is the current Step 04 communication upper-baseline.
    % ---------------------------------------------------------------------
    cfgFreq = makeBaseStep05Config();
    cfgFreq.gs.uploadMode = "after_measurement_update";

    rng(seed);
    resFreq = simulate_GS_DNN_EKF(cfgFreq);

    finalUploadsFreq = resFreq.gsNumTotalUploads(end);

    % ---------------------------------------------------------------------
    % Event-triggered upload case.
    % This should reduce uploads while preserving valid GS branch exchange.
    % ---------------------------------------------------------------------
    cfgET = makeBaseStep05Config();
    cfgET.gs.uploadMode = "event_contribution_change";

    rng(seed);
    resET = simulate_GS_DNN_EKF(cfgET);

    finalUploadsET = resET.gsNumTotalUploads(end);

    % ---------------------------------------------------------------------
    % Check required Step 05 logs.
    % ---------------------------------------------------------------------
    requiredLogs = [
        "gsUploadDecision"
        "gsUploadDelta"
        "gsUploadDwellSatisfied"
        "gsUploadMeasSatisfied"
        "gsUploadReason"
    ];

    for q = 1:numel(requiredLogs)
        logName = requiredLogs(q);
        assert(isfield(resET, logName), ...
            "Missing Step 05 log results.%s.", logName);
    end

    uploadMask = logical(resET.gsUploadDecision);

    numEventUploadDecisions = nnz(uploadMask);
    expectedTotalUploads = cfgET.Nw + numEventUploadDecisions;
    % cfg.gs.bootstrapUpload = true, so the initial Nw uploads are not part
    % of gsUploadDecision. Event uploads after t0 are logged.

    fprintf("\nUpload counts:\n");
    fprintf("  Frequent final GS uploads      = %d\n", finalUploadsFreq);
    fprintf("  Event-trigger final GS uploads = %d\n", finalUploadsET);
    fprintf("  Event upload decisions         = %d\n", numEventUploadDecisions);
    fprintf("  Expected total uploads         = %d\n", expectedTotalUploads);

    assert(finalUploadsET == expectedTotalUploads, ...
        "Upload count mismatch: final uploads should equal bootstrap Nw + event decisions.");

    assert(finalUploadsET < finalUploadsFreq, ...
        "Event-triggered upload did not reduce GS uploads relative to frequent upload.");

    assert(numEventUploadDecisions > 0, ...
        "No event-triggered uploads occurred. Threshold may be too strict.");

    % ---------------------------------------------------------------------
    % Check trigger conditions at upload times.
    % ---------------------------------------------------------------------
    deltaAtUpload = resET.gsUploadDelta(uploadMask);
    measAtUpload  = resET.gsUploadMeasSatisfied(uploadMask);
    dwellAtUpload = resET.gsUploadDwellSatisfied(uploadMask);

    deltaThreshold = cfgET.gs.eventDeltaThreshold;

    minDeltaAtUpload = min(deltaAtUpload, [], "omitnan");
    numDeltaBad = nnz(deltaAtUpload < deltaThreshold);
    numMeasBad = nnz(~measAtUpload);
    numDwellBad = nnz(~dwellAtUpload);

    fprintf("\nTrigger-condition checks at upload times:\n");
    fprintf("  Delta threshold           = %.6e\n", deltaThreshold);
    fprintf("  Min Delta at upload       = %.6e\n", minDeltaAtUpload);
    fprintf("  Delta violations          = %d\n", numDeltaBad);
    fprintf("  Measurement violations    = %d\n", numMeasBad);
    fprintf("  Dwell violations          = %d\n", numDwellBad);

    assert(all(isfinite(deltaAtUpload)), ...
        "Some upload decisions have non-finite Delta values.");

    assert(numDeltaBad == 0, ...
        "Some event uploads occurred below the Delta threshold.");

    assert(numMeasBad == 0, ...
        "Some event uploads occurred without satisfying measurement condition.");

    assert(numDwellBad == 0, ...
        "Some event uploads occurred without satisfying dwell condition.");

    % ---------------------------------------------------------------------
    % Check GS branch availability.
    % Bootstrap should keep all nonlocal branches available to each watcher.
    % ---------------------------------------------------------------------
    finalNonlocalUsed = resET.numNonlocalBranchesUsed(end, :);

    fprintf("\nFinal nonlocal branches used per watcher:\n");
    disp(finalNonlocalUsed);

    assert(all(finalNonlocalUsed == cfgET.Nw - 1), ...
        "Not all watchers use Nw-1 nonlocal branches at the final step.");

    % ---------------------------------------------------------------------
    % Lightweight performance diagnostics only.
    % No pass/fail assertion on RMSE here.
    % ---------------------------------------------------------------------
    meanPosErrFreq = computeMeanPositionError(resFreq, cfgFreq);
    meanPosErrET   = computeMeanPositionError(resET, cfgET);

    fprintf("\nPosition-error diagnostics only:\n");
    fprintf("  Frequent mean position error      = %.6f\n", meanPosErrFreq);
    fprintf("  Event-trigger mean position error = %.6f\n", meanPosErrET);

    % ---------------------------------------------------------------------
    % Reason string summary.
    % Useful for quickly seeing why uploads were skipped.
    % ---------------------------------------------------------------------
    reasonVals = string(resET.gsUploadReason(:));
    reasonVals = reasonVals(reasonVals ~= "");

    [uniqueReasons, ~, idxReason] = unique(reasonVals);
    reasonCounts = accumarray(idxReason, 1);

    reasonTable = table( ...
        uniqueReasons, ...
        reasonCounts, ...
        'VariableNames', {'reason', 'count'});

    fprintf("\nEvent-trigger reason summary:\n");
    disp(reasonTable);

    out = struct();
    out.passed = true;

    out.cfgFreq = cfgFreq;
    out.cfgET = cfgET;

    out.finalUploadsFreq = finalUploadsFreq;
    out.finalUploadsET = finalUploadsET;
    out.numEventUploadDecisions = numEventUploadDecisions;
    out.expectedTotalUploads = expectedTotalUploads;

    out.minDeltaAtUpload = minDeltaAtUpload;
    out.numDeltaBad = numDeltaBad;
    out.numMeasBad = numMeasBad;
    out.numDwellBad = numDwellBad;

    out.finalNonlocalUsed = finalNonlocalUsed;

    out.meanPosErrFreq = meanPosErrFreq;
    out.meanPosErrET = meanPosErrET;

    out.reasonTable = reasonTable;

    fprintf("\n============================================================\n");
    fprintf("Step 05 event-triggered upload sanity check passed.\n");
    fprintf("============================================================\n");

end

function cfg = makeBaseStep05Config()
% Shared configuration for Step 05 event-trigger sanity check.

    cfg = config_step04_GS_DNN_EKF();

    % Same residual-learning setting used in Step 04d comparison.
    cfg.truth.residualAmp = 5e-4;

    % Fair comparison: no random theta initialization.
    cfg.dnn.theta0_std = 0.0;

    % Use GS composite residual model.
    cfg.dnn.predictionResidualSource = "GS_composite";
    cfg.dnn.residualInjectionGain = 1.0;

    % GS is enabled; all watchers upload initial branch records.
    cfg.gs.enabled = true;
    cfg.gs.bootstrapUpload = true;

    % Broadcast remains every step for this first event-trigger check.
    % Only upload is event-triggered.
    cfg.gs.broadcastMode = "every_step";

    % Step 05 event-trigger parameters.
    cfg.gs.eventDeltaThreshold = 1e-14;
    cfg.gs.eventDwellSteps = 20;
    cfg.gs.eventRequireMeasurement = true;

    % Keep Qnonlocal off here.
    % This check isolates communication-trigger logic.
    cfg.gs.useNonlocalBranchCovariance = false;

    % Keep block covariance prediction as the main implementation.
    if ~isfield(cfg, "ekf")
        cfg.ekf = struct();
    end
    cfg.ekf.useBlockCovPrediction = true;

end

function meanPosErr = computeMeanPositionError(results, cfg)
% Compute mean target position error over all watchers and time.

    dim = cfg.dim;

    xhat = results.xhat;
    etaTrue = results.etaTrue;

    N = size(etaTrue, 2);

    rHat = xhat(1:dim, :, :);
    rTrue = reshape(etaTrue(1:dim, :), dim, N, 1);

    posErr = squeeze(sqrt(sum((rHat - rTrue).^2, 1)));

    meanPosErr = mean(posErr(:), "omitnan");

end