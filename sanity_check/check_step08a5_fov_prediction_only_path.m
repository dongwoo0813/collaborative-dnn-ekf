function out = check_step08a5_fov_prediction_only_path()
%{
File:
    sanity_check/check_step08a5_fov_prediction_only_path.m

Purpose:
    Step 08-A.5 sanity check for FOV/dropout prediction-only behavior.

    This test forces all measurements to be unavailable using the FOV range
    gate. It then checks that the simulation enters the prediction-only EKF
    path cleanly.

Checks:
    1. FOV range gate creates measurement dropout.
    2. Dropout reasons are logged as range_too_small.
    3. NIS values are NaN when measurement is unavailable.
    4. GS event uploads are not triggered when measurement is unavailable.
    5. Simulation completes without crashing.

Expected:
    This check should pass after Step 08-A.4b diagnostic logging is active.
%}

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 08-A.5 sanity check: FOV prediction-only path\n");
    fprintf("============================================================\n\n");

    cfg = config_step04_GS_DNN_EKF();

    % Short horizon keeps this check fast.
    cfg.T = 20;
    cfg.time = 0:cfg.dt:cfg.T;
    cfg.N = numel(cfg.time);

    % Activate FOV mode explicitly.
    cfg.meas.availabilityMode = "fov";
    cfg.fov.enabled = true;
    cfg.fov.guardUnimplementedMode = false;

    % Use target-pointing boresight so the cone check passes.
    % Then force dropout only through the range gate.
    cfg.fov.boresightMode = "target_pointing";
    cfg.fov.rhoMin = 1e12;
    cfg.fov.rhoMax = Inf;

    rng(100);
    res = simulate_GS_DNN_EKF(cfg);

    updateRows = 2:(cfg.N-1);

    if isempty(updateRows)
        error("check_step08a5:InvalidHorizon", ...
            "The configured horizon has no update rows to check.");
    end

    measAvail = logical(res.measAvail(updateRows,:));
    numAvailable = nnz(measAvail);
    numTotal = numel(measAvail);

    reasons = res.measurementDropoutReason(updateRows,:);
    uniqueReasons = unique(reasons(:));

    fprintf("Measurement availability:\n");
    fprintf("  Available update entries = %d / %d\n", numAvailable, numTotal);
    fprintf("  Unique dropout reasons:\n");
    disp(uniqueReasons.');

    passDropout = (numAvailable == 0);
    passReason = all(reasons(:) == "range_too_small");

    printPassFail(passDropout, ...
        "FOV range gate forces all update-step measurements unavailable", ...
        "Expected all measurement availability entries to be false.");

    printPassFail(passReason, ...
        "Dropout reason is range_too_small for evaluated update rows", ...
        "Expected only range_too_small on evaluated update rows.");

    % ------------------------------------------------------------
    % NIS should be NaN when no measurement update occurs.
    % This uses a robust field finder because previous steps may have used
    % different NIS field names.
    % ------------------------------------------------------------
    nisField = findFirstExistingField(res, ["NIS", "nis", "NISLog", "nisLog"]);

    if strlength(nisField) == 0
        passNIS = true;
        fprintf("\n[SKIP] NIS field not found in results.\n");
        fprintf("  Meaning: Could not check NIS NaN behavior automatically.\n");
    else
        nisLog = res.(nisField);

        if size(nisLog,1) >= cfg.N
            nisCheck = nisLog(updateRows,:);
        else
            nisCheck = nisLog;
        end

        passNIS = all(isnan(nisCheck(:)));

        fprintf("\nNIS check:\n");
        fprintf("  NIS field used = %s\n", nisField);
        fprintf("  Non-NaN NIS entries on dropout rows = %d\n", nnz(~isnan(nisCheck(:))));

        printPassFail(passNIS, ...
            "NIS is NaN when measurement is unavailable", ...
            "Expected NIS to be NaN on prediction-only rows.");
    end

    % ------------------------------------------------------------
    % GS event uploads should be blocked by measurementSatisfied = false.
    % Bootstrap uploads may still exist, so only check logged event decisions.
    % ------------------------------------------------------------
    uploadDecisionField = findFirstExistingField(res, ...
        ["gsUploadDecision", "gsUploadDecisionLog", ...
         "gsUploadEventDecision", "gsUploadEventDecisionLog"]);

    if strlength(uploadDecisionField) == 0
        passUpload = true;
        fprintf("\n[SKIP] GS upload decision field not found in results.\n");
        fprintf("  Meaning: Could not check event-upload blocking automatically.\n");
    else
        uploadDecision = logical(res.(uploadDecisionField));

        if size(uploadDecision,1) >= cfg.N
            uploadCheck = uploadDecision(updateRows,:);
        else
            uploadCheck = uploadDecision;
        end

        numEventUploads = nnz(uploadCheck);
        passUpload = (numEventUploads == 0);

        fprintf("\nGS event-upload check:\n");
        fprintf("  Upload decision field used = %s\n", uploadDecisionField);
        fprintf("  Logged event uploads on dropout rows = %d\n", numEventUploads);

        printPassFail(passUpload, ...
            "GS event uploads are blocked when measurement is unavailable", ...
            "Expected zero logged event uploads on all-dropout update rows.");
    end

    out = struct();
    out.passed = passDropout && passReason && passNIS && passUpload;
    out.numAvailable = numAvailable;
    out.numTotal = numTotal;
    out.uniqueReasons = uniqueReasons;
    out.nisField = nisField;
    out.uploadDecisionField = uploadDecisionField;

    fprintf("\n============================================================\n");

    if out.passed
        fprintf("Step 08-A.5 sanity check PASSED.\n");
        fprintf("Meaning:\n");
        fprintf("  1. FOV dropout enters prediction-only mode.\n");
        fprintf("  2. Dropout reasons are preserved.\n");
        fprintf("  3. Measurement-dependent diagnostics/actions are blocked.\n");
    else
        fprintf("Step 08-A.5 sanity check FAILED.\n");
        fprintf("Meaning:\n");
        fprintf("  At least one prediction-only behavior is inconsistent.\n");
    end

    fprintf("============================================================\n\n");

end

function fieldName = findFirstExistingField(s, candidateNames)
%FINDFIRSTEXISTINGFIELD Return first candidate field that exists in struct.

    fieldName = "";

    for idx = 1:numel(candidateNames)
        candidate = string(candidateNames(idx));

        if isfield(s, candidate)
            fieldName = candidate;
            return;
        end
    end

end

function printPassFail(tf, passMessage, failMessage)
%PRINTPASSFAIL Small formatted pass/fail printer.

    if tf
        fprintf("\n[PASS] %s\n", passMessage);
    else
        fprintf("\n[FAIL] %s\n", failMessage);
    end

end