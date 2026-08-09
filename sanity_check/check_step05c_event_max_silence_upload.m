function out = check_step05c_event_max_silence_upload()
%{
File:
    sanity_check/check_step05c_event_max_silence_upload.m

Purpose:
    Sanity check for Step 05-C:
    eventMaxSilenceSteps / maximum-silence / heartbeat GS upload.

What this checks:
    1. The Step 05-C diagnostic log exists:
           results.gsUploadMaxSilenceSatisfied

    2. Default-preservation test:
           eventMaxSilenceSteps absent
           vs
           eventMaxSilenceSteps = Inf

       These two cases should produce the same upload decisions and the
       same trajectory estimates. This verifies that the Step 05-C patch
       does not change old Step 05-B behavior by default.

    3. Heartbeat-activation test:
           eventDeltaThreshold = Inf
           eventDwellSteps = 1
           eventMaxSilenceSteps = finite

       This disables ordinary Delta-triggered uploads and forces uploads
       only through the maximum-silence condition.

    4. Upload bookkeeping:
           final GS upload count =
           bootstrap uploads + logged event upload decisions

    5. Trigger correctness:
           heartbeat uploads must satisfy:
               measurement condition
               dwell condition
               maximum-silence condition

Current assumption:
    Measurement is available for all watchers at all time.
    This check does not activate FOV mode.

Outputs:
    out - struct containing pass/fail status, diagnostic values, and
          reason-count tables.

Passing means:
    Step 05-C heartbeat logic is wired correctly, default behavior is
    preserved when eventMaxSilenceSteps = Inf, and finite maximum silence
    can force GS uploads even when the Delta trigger is disabled.

Failing means:
    The patch is incomplete or changed old behavior unexpectedly. The
    printed fail message tells which part failed.
%}

    clc;
    addpath(genpath(pwd));
    rehash;

    fprintf("\n============================================================\n");
    fprintf("Step 05-C sanity check: eventMaxSilenceSteps / heartbeat\n");
    fprintf("============================================================\n");

    out = initOutputStruct();

    try
        assert(exist("config_step04_GS_DNN_EKF", "file") == 2, ...
            "Missing config_step04_GS_DNN_EKF.m.");

        assert(exist("simulate_GS_DNN_EKF", "file") == 2, ...
            "Missing simulate_GS_DNN_EKF.m.");

        seed = 100;

        % -------------------------------------------------------------
        % Case A:
        % eventMaxSilenceSteps field absent.
        %
        % This checks that simulate_GS_DNN_EKF internally defaults to Inf
        % when the field is absent.
        % -------------------------------------------------------------
        cfgAbsent = makeBaseStep05CConfig();

        if isfield(cfgAbsent.gs, "eventMaxSilenceSteps")
            cfgAbsent.gs = rmfield(cfgAbsent.gs, "eventMaxSilenceSteps");
        end

        rng(seed);
        resAbsent = simulate_GS_DNN_EKF(cfgAbsent);

        % -------------------------------------------------------------
        % Case B:
        % eventMaxSilenceSteps explicitly Inf.
        %
        % This should preserve the absent-field/default behavior.
        % -------------------------------------------------------------
        cfgInf = makeBaseStep05CConfig();
        cfgInf.gs.eventMaxSilenceSteps = Inf;

        rng(seed);
        resInf = simulate_GS_DNN_EKF(cfgInf);

        % -------------------------------------------------------------
        % Check required Step 05-C logs.
        % -------------------------------------------------------------
        requiredLogs = [
            "gsUploadDecision"
            "gsUploadDelta"
            "gsUploadDwellSatisfied"
            "gsUploadMaxSilenceSatisfied"
            "gsUploadMeasSatisfied"
            "gsUploadReason"
        ];

        requiredOK = hasRequiredLogs(resInf, requiredLogs);

        out = addCheck(out, ...
            "Required Step 05-C logs exist", ...
            requiredOK, ...
            "results contains gsUploadMaxSilenceSatisfied and the old Step 05 upload logs.", ...
            "simulate_GS_DNN_EKF.m is missing at least one required Step 05-C result log.");

        if ~requiredOK
            out.passed = false;
            out = finalizeOutput(out);
            return;
        end

        logSizeOK = checkLogSizes(resInf, requiredLogs);

        out = addCheck(out, ...
            "Step 05-C log sizes are consistent", ...
            logSizeOK, ...
            "All upload-decision logs have the same N-by-Nw size.", ...
            "At least one upload-decision log has the wrong size.");

        % -------------------------------------------------------------
        % Default-preservation check:
        % absent field should match explicit Inf.
        % -------------------------------------------------------------
        sameUploadDecision = isequal( ...
            logical(resAbsent.gsUploadDecision), ...
            logical(resInf.gsUploadDecision));

        sameFinalUploads = ...
            resAbsent.gsNumTotalUploads(end) == resInf.gsNumTotalUploads(end);

        maxStateDiff = maxAbsDiff(resAbsent.xhatAug, resInf.xhatAug);

        numInfHeartbeatReasons = countReason(resInf, "max_silence_passed");
        numInfMaxSilenceTrue = nnz(resInf.gsUploadMaxSilenceSatisfied);

        defaultPreserved = sameUploadDecision && ...
                           sameFinalUploads && ...
                           maxStateDiff < 1e-10 && ...
                           numInfHeartbeatReasons == 0 && ...
                           numInfMaxSilenceTrue == 0;

        fprintf("\nDefault-preservation diagnostics:\n");
        fprintf("  Same upload-decision mask       = %d\n", sameUploadDecision);
        fprintf("  Same final GS upload count      = %d\n", sameFinalUploads);
        fprintf("  Max |xhatAug_absent - xhatAug_Inf| = %.3e\n", maxStateDiff);
        fprintf("  Inf-case heartbeat reason count = %d\n", numInfHeartbeatReasons);
        fprintf("  Inf-case max-silence true count = %d\n", numInfMaxSilenceTrue);

        out.default.sameUploadDecision = sameUploadDecision;
        out.default.sameFinalUploads = sameFinalUploads;
        out.default.maxStateDiff = maxStateDiff;
        out.default.numHeartbeatReasons = numInfHeartbeatReasons;
        out.default.numMaxSilenceTrue = numInfMaxSilenceTrue;

        out = addCheck(out, ...
            "Default eventMaxSilenceSteps = Inf preserves old behavior", ...
            defaultPreserved, ...
            "The heartbeat path is disabled by default and Step 05-B behavior is preserved.", ...
            "Default behavior changed. The heartbeat logic may be active even when eventMaxSilenceSteps is Inf.");

        % -------------------------------------------------------------
        % Upload bookkeeping check for explicit Inf case.
        % -------------------------------------------------------------
        numInfEventDecisions = nnz(resInf.gsUploadDecision);
        expectedInfUploads = cfgInf.Nw + numInfEventDecisions;
        finalInfUploads = resInf.gsNumTotalUploads(end);

        bookkeepingInfOK = finalInfUploads == expectedInfUploads;

        fprintf("\nDefault/Inf upload bookkeeping:\n");
        fprintf("  Bootstrap uploads            = %d\n", cfgInf.Nw);
        fprintf("  Logged event-upload decisions = %d\n", numInfEventDecisions);
        fprintf("  Expected final uploads        = %d\n", expectedInfUploads);
        fprintf("  Actual final uploads          = %d\n", finalInfUploads);

        out.default.finalUploads = finalInfUploads;
        out.default.expectedUploads = expectedInfUploads;

        out = addCheck(out, ...
            "Default/Inf upload bookkeeping is correct", ...
            bookkeepingInfOK, ...
            "Final GS upload count equals bootstrap uploads plus logged event-upload decisions.", ...
            "Upload bookkeeping is inconsistent in the default/Inf case.");

        % -------------------------------------------------------------
        % Case C:
        % Force heartbeat activation.
        %
        % Delta trigger is intentionally disabled by setting threshold = Inf.
        % Therefore, event uploads should occur only through max silence.
        % -------------------------------------------------------------
        cfgHB = makeBaseStep05CConfig();
        cfgHB.gs.eventDeltaThreshold = Inf;
        cfgHB.gs.eventDwellSteps = 1;
        cfgHB.gs.eventMaxSilenceSteps = 40;
        cfgHB.gs.eventRequireMeasurement = true;

        rng(seed);
        resHB = simulate_GS_DNN_EKF(cfgHB);

        requiredHBOK = hasRequiredLogs(resHB, requiredLogs);

        out = addCheck(out, ...
            "Heartbeat case produces required logs", ...
            requiredHBOK, ...
            "The finite-heartbeat simulation returned all Step 05-C logs.", ...
            "The finite-heartbeat simulation is missing required Step 05-C logs.");

        if ~requiredHBOK
            out.passed = false;
            out = finalizeOutput(out);
            return;
        end

        uploadMask = logical(resHB.gsUploadDecision);
        heartbeatMask = uploadMask & ...
            (string(resHB.gsUploadReason) == "max_silence_passed");

        numHeartbeatUploads = nnz(heartbeatMask);
        numTotalEventUploads = nnz(uploadMask);

        heartbeatByWatcher = squeeze(sum(heartbeatMask, 1));
        heartbeatExistsForAllWatchers = all(heartbeatByWatcher(:) > 0);

        allEventUploadsAreHeartbeat = ...
            nnz(uploadMask & ~heartbeatMask) == 0;

        heartbeatActivated = numHeartbeatUploads > 0 && ...
                             heartbeatExistsForAllWatchers && ...
                             allEventUploadsAreHeartbeat;

        fprintf("\nHeartbeat-activation diagnostics:\n");
        fprintf("  eventDeltaThreshold             = Inf\n");
        fprintf("  eventDwellSteps                 = %d\n", cfgHB.gs.eventDwellSteps);
        fprintf("  eventMaxSilenceSteps            = %d\n", cfgHB.gs.eventMaxSilenceSteps);
        fprintf("  Total logged event uploads      = %d\n", numTotalEventUploads);
        fprintf("  Heartbeat uploads               = %d\n", numHeartbeatUploads);
        fprintf("  Heartbeat uploads per watcher:\n");
        disp(heartbeatByWatcher);

        out.heartbeat.numHeartbeatUploads = numHeartbeatUploads;
        out.heartbeat.numTotalEventUploads = numTotalEventUploads;
        out.heartbeat.heartbeatByWatcher = heartbeatByWatcher;

        out = addCheck(out, ...
            "Finite eventMaxSilenceSteps activates heartbeat uploads", ...
            heartbeatActivated, ...
            "When Delta trigger is disabled, uploads are still forced by maximum silence.", ...
            "Heartbeat did not activate correctly, or non-heartbeat uploads occurred even though Delta trigger was disabled.");

        % -------------------------------------------------------------
        % Check heartbeat upload conditions.
        % -------------------------------------------------------------
        if numHeartbeatUploads > 0
            measAtHB = resHB.gsUploadMeasSatisfied(heartbeatMask);
            dwellAtHB = resHB.gsUploadDwellSatisfied(heartbeatMask);
            maxSilenceAtHB = resHB.gsUploadMaxSilenceSatisfied(heartbeatMask);
            deltaAtHB = resHB.gsUploadDelta(heartbeatMask);
        else
            measAtHB = false;
            dwellAtHB = false;
            maxSilenceAtHB = false;
            deltaAtHB = NaN;
        end

        numMeasBadHB = nnz(~measAtHB);
        numDwellBadHB = nnz(~dwellAtHB);
        numMaxSilenceBadHB = nnz(~maxSilenceAtHB);
        numNonFiniteDeltaHB = nnz(~isfinite(deltaAtHB));

        heartbeatConditionOK = numHeartbeatUploads > 0 && ...
                               numMeasBadHB == 0 && ...
                               numDwellBadHB == 0 && ...
                               numMaxSilenceBadHB == 0 && ...
                               numNonFiniteDeltaHB == 0;

        fprintf("\nHeartbeat trigger-condition checks:\n");
        fprintf("  Measurement violations      = %d\n", numMeasBadHB);
        fprintf("  Dwell violations            = %d\n", numDwellBadHB);
        fprintf("  Max-silence violations      = %d\n", numMaxSilenceBadHB);
        fprintf("  Nonfinite Delta values      = %d\n", numNonFiniteDeltaHB);

        out.heartbeat.numMeasBad = numMeasBadHB;
        out.heartbeat.numDwellBad = numDwellBadHB;
        out.heartbeat.numMaxSilenceBad = numMaxSilenceBadHB;
        out.heartbeat.numNonFiniteDelta = numNonFiniteDeltaHB;

        out = addCheck(out, ...
            "Heartbeat uploads satisfy measurement, dwell, and max-silence conditions", ...
            heartbeatConditionOK, ...
            "Every heartbeat upload is conditionally valid.", ...
            "At least one heartbeat upload violated measurement, dwell, or max-silence logic.");

        % -------------------------------------------------------------
        % Check that upload intervals do not exceed eventMaxSilenceSteps
        % between accepted uploads.
        % -------------------------------------------------------------
        [gapOK, maxObservedGap] = checkMaxUploadGap( ...
            resHB.gsUploadDecision, cfgHB.gs.eventMaxSilenceSteps);

        fprintf("\nHeartbeat interval check:\n");
        fprintf("  Max observed inter-upload gap = %d steps\n", maxObservedGap);
        fprintf("  Allowed max silence           = %d steps\n", cfgHB.gs.eventMaxSilenceSteps);

        out.heartbeat.maxObservedGap = maxObservedGap;

        out = addCheck(out, ...
            "Inter-upload gap is bounded by eventMaxSilenceSteps", ...
            gapOK, ...
            "No watcher remains silent longer than the configured maximum silence interval between uploads.", ...
            "At least one watcher exceeded the maximum allowed silence interval between uploads.");


        % -------------------------------------------------------------
        % Check current simplifying assumption:
        % all measurements are available during EKF update steps.
        %
        % Note:
        %     Row 1 of results.measAvail corresponds to the initialization row.
        %     No measurement update is performed at this row, so it may remain false
        %     even when the simulation is in the always-available measurement mode.
        %     Therefore, this check intentionally ignores row 1 and verifies rows
        %     2:end only.
        % -------------------------------------------------------------
        measAlwaysAvailable = false;
        numUnavailableMeasUpdateRows = NaN;
        
        if isfield(resHB, "measAvail")
            measAvailLog = logical(resHB.measAvail);
        
            if size(measAvailLog, 1) >= 2
                measAvailUpdateRows = measAvailLog(2:end, :);
            else
                measAvailUpdateRows = measAvailLog;
            end
        
            numUnavailableMeasUpdateRows = nnz(~measAvailUpdateRows);
            measAlwaysAvailable = numUnavailableMeasUpdateRows == 0;
        end
        
        fprintf("\nMeasurement-availability assumption check:\n");
        fprintf("  Ignored initialization row      = true\n");
        fprintf("  Unavailable update-step entries = %d\n", numUnavailableMeasUpdateRows);
        
        out.heartbeat.numUnavailableMeasUpdateRows = numUnavailableMeasUpdateRows;
        
        out = addCheck(out, ...
            "Current always-available measurement assumption is preserved", ...
            measAlwaysAvailable, ...
            "All EKF update-step measurement availability entries are true; FOV/dropout behavior is not active.", ...
            "At least one EKF update-step measurement availability entry is false.");

        % -------------------------------------------------------------
        % Upload bookkeeping check for heartbeat case.
        % -------------------------------------------------------------
        expectedHBUploads = cfgHB.Nw + nnz(resHB.gsUploadDecision);
        finalHBUploads = resHB.gsNumTotalUploads(end);

        bookkeepingHBOK = finalHBUploads == expectedHBUploads;

        fprintf("\nHeartbeat upload bookkeeping:\n");
        fprintf("  Bootstrap uploads            = %d\n", cfgHB.Nw);
        fprintf("  Logged event-upload decisions = %d\n", nnz(resHB.gsUploadDecision));
        fprintf("  Expected final uploads        = %d\n", expectedHBUploads);
        fprintf("  Actual final uploads          = %d\n", finalHBUploads);

        out.heartbeat.finalUploads = finalHBUploads;
        out.heartbeat.expectedUploads = expectedHBUploads;

        out = addCheck(out, ...
            "Heartbeat upload bookkeeping is correct", ...
            bookkeepingHBOK, ...
            "Final GS upload count equals bootstrap uploads plus logged heartbeat decisions.", ...
            "Upload bookkeeping is inconsistent in the heartbeat case.");

        % -------------------------------------------------------------
        % Reason summaries.
        % -------------------------------------------------------------
        out.reasonTableInf = buildReasonTable(resInf.gsUploadReason);
        out.reasonTableHeartbeat = buildReasonTable(resHB.gsUploadReason);

        fprintf("\nReason summary: explicit Inf case\n");
        disp(out.reasonTableInf);

        fprintf("\nReason summary: finite heartbeat case\n");
        disp(out.reasonTableHeartbeat);

        out.cfgInf = cfgInf;
        out.cfgHeartbeat = cfgHB;
        out.resInf = resInf;
        out.resHeartbeat = resHB;

        out = finalizeOutput(out);

    catch ME
        out.passed = false;
        out.error = ME;

        fprintf("\n[FAIL] Step 05-C sanity check terminated with an error.\n");
        fprintf("Error message:\n");
        fprintf("  %s\n", ME.message);

        fprintf("\nMost likely causes:\n");
        fprintf("  1. simulate_GS_DNN_EKF.m was not patched correctly.\n");
        fprintf("  2. decision.maxSilenceSatisfied was not added to initUploadDecision().\n");
        fprintf("  3. results.gsUploadMaxSilenceSatisfied was not added to results.\n");
        fprintf("  4. eventMaxSilenceSteps logic was added but not logged.\n");

        fprintf("\n============================================================\n");
        fprintf("Step 05-C sanity check FAILED.\n");
        fprintf("============================================================\n");
    end

end

function cfg = makeBaseStep05CConfig()
%{
Function:
    makeBaseStep05CConfig

Purpose:
    Create a short-horizon Step 05-C test configuration.

Notes:
    This is a logic/sanity test, not a paper-level performance run.
    The shorter horizon keeps the check faster while still allowing many
    heartbeat cycles when eventMaxSilenceSteps is finite.
%}

    cfg = config_step04_GS_DNN_EKF();

    % Shorten this sanity check for faster debugging.
    cfg.T = 300;
    cfg.time = 0:cfg.dt:cfg.T;
    cfg.N = numel(cfg.time);

    % Same residual-learning setup used in Step 04/05 checks.
    cfg.truth.residualAmp = 5e-4;

    % Fair comparison: no random theta initialization.
    cfg.dnn.theta0_std = 0.0;

    % Use GS-composite residual prediction.
    cfg.dnn.predictionResidualSource = "GS_composite";
    cfg.dnn.residualInjectionGain = 1.0;

    % GS communication enabled.
    cfg.gs.enabled = true;
    cfg.gs.bootstrapUpload = true;
    cfg.gs.uploadMode = "event_contribution_change";
    cfg.gs.broadcastMode = "every_step";

    % Standard Step 05 event-trigger settings.
    cfg.gs.eventDeltaThreshold = 1e-14;
    cfg.gs.eventDwellSteps = 20;
    cfg.gs.eventRequireMeasurement = true;

    % Keep Qnonlocal off so this check isolates communication logic.
    cfg.gs.useNonlocalBranchCovariance = false;

    if ~isfield(cfg, "ekf")
        cfg.ekf = struct();
    end
    cfg.ekf.useBlockCovPrediction = true;

end

function out = initOutputStruct()
% Initialize output struct.

    out = struct();
    out.passed = false;
    out.checks = struct( ...
        "name", {}, ...
        "passed", {}, ...
        "passMeaning", {}, ...
        "failMeaning", {});

end

function out = addCheck(out, name, passed, passMeaning, failMeaning)
% Add one pass/fail check and print its interpretation.

    idx = numel(out.checks) + 1;

    out.checks(idx).name = string(name);
    out.checks(idx).passed = logical(passed);
    out.checks(idx).passMeaning = string(passMeaning);
    out.checks(idx).failMeaning = string(failMeaning);

    if passed
        status = "PASS";
        meaning = passMeaning;
    else
        status = "FAIL";
        meaning = failMeaning;
    end

    fprintf("\n[%s] %s\n", char(status), name);
    fprintf("  Meaning: %s\n", meaning);

end

function out = finalizeOutput(out)
% Finalize output status and print final verdict.

    if isempty(out.checks)
        out.passed = false;
        out.summaryTable = table();
    else
        names = string({out.checks.name})';
        passed = logical([out.checks.passed])';

        out.summaryTable = table(names, passed, ...
            'VariableNames', {'checkName', 'passed'});

        out.passed = all(passed);
    end

    fprintf("\n============================================================\n");

    if out.passed
        fprintf("Step 05-C sanity check PASSED.\n");
        fprintf("Meaning:\n");
        fprintf("  1. Default eventMaxSilenceSteps = Inf preserves old behavior.\n");
        fprintf("  2. Finite eventMaxSilenceSteps forces heartbeat uploads.\n");
        fprintf("  3. Heartbeat uploads satisfy measurement, dwell, and max-silence logic.\n");
        fprintf("  4. Upload bookkeeping is consistent.\n");
    else
        fprintf("Step 05-C sanity check FAILED.\n");
        fprintf("Meaning:\n");
        fprintf("  At least one Step 05-C condition is not implemented correctly.\n");
        fprintf("  See the failed check above before moving to the next patch.\n");
    end

    fprintf("============================================================\n");

end

function tf = hasRequiredLogs(results, requiredLogs)
% Return true if all required log fields are present.

    tf = true;

    for i = 1:numel(requiredLogs)
        fieldName = char(requiredLogs(i));
        tf = tf && isfield(results, fieldName);
    end

end

function tf = checkLogSizes(results, requiredLogs)
% Check that all required logs have the same size as gsUploadDecision.

    tf = true;

    if ~isfield(results, "gsUploadDecision")
        tf = false;
        return;
    end

    baseSize = size(results.gsUploadDecision);

    for i = 1:numel(requiredLogs)
        fieldName = char(requiredLogs(i));

        if ~isfield(results, fieldName)
            tf = false;
            return;
        end

        thisSize = size(results.(fieldName));

        if ~isequal(thisSize, baseSize)
            tf = false;
            return;
        end
    end

end

function d = maxAbsDiff(A, B)
% Maximum absolute difference between two numeric arrays, ignoring NaNs.

    if ~isequal(size(A), size(B))
        d = Inf;
        return;
    end

    diffVal = abs(A(:) - B(:));
    diffVal = diffVal(isfinite(diffVal));

    if isempty(diffVal)
        d = 0;
    else
        d = max(diffVal);
    end

end

function n = countReason(results, reasonString)
% Count occurrences of a reason string in results.gsUploadReason.

    if ~isfield(results, "gsUploadReason")
        n = NaN;
        return;
    end

    reasons = string(results.gsUploadReason(:));
    n = nnz(reasons == string(reasonString));

end

function [tf, maxObservedGap] = checkMaxUploadGap(uploadDecision, maxSilenceSteps)
%{
Function:
    checkMaxUploadGap

Purpose:
    Check that the gap between accepted uploads for each watcher does not
    exceed eventMaxSilenceSteps.

Notes:
    The bootstrap upload occurs at row 1 but is not included in
    gsUploadDecision. Therefore row 1 is manually inserted as the initial
    upload time for each watcher.
%}

    uploadDecision = logical(uploadDecision);

    [~, Nw] = size(uploadDecision);

    tf = true;
    maxObservedGap = 0;

    for i = 1:Nw
        uploadRows = find(uploadDecision(:, i));

        % Include bootstrap upload at initial row.
        uploadRows = [1; uploadRows(:)];

        if numel(uploadRows) < 2
            tf = false;
            continue;
        end

        gaps = diff(uploadRows);

        if ~isempty(gaps)
            maxObservedGap = max(maxObservedGap, max(gaps));
        end

        if any(gaps > maxSilenceSteps)
            tf = false;
        end
    end

end

function tbl = buildReasonTable(reasonArray)
% Build compact reason-count table.

    reasons = string(reasonArray(:));
    reasons = reasons(reasons ~= "");

    if isempty(reasons)
        tbl = table(strings(0,1), zeros(0,1), ...
            'VariableNames', {'reason', 'count'});
        return;
    end

    [uniqueReasons, ~, idx] = unique(reasons);
    counts = accumarray(idx, 1);

    tbl = table(uniqueReasons, counts, ...
        'VariableNames', {'reason', 'count'});

end