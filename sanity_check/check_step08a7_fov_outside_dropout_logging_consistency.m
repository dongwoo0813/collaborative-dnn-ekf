function out = check_step08a7_fov_outside_dropout_logging_consistency()
%{
File:
    sanity_check/check_step08a7_fov_outside_dropout_logging_consistency.m

Purpose:
    Step 08-A.7 sanity check for simulation-level outside-FOV dropout and
    measurement-log consistency.

    This check activates FOV mode with an inertial-fixed camera boresight
    and a narrow FOV half-angle. The goal is to create real cone-based
    dropout, i.e., dropoutReason = "outside_fov", not artificial range-gate
    dropout.

Checks:
    1. The simulation produces at least one outside_fov dropout.
    2. The simulation produces at least one available measurement.
    3. res.measAvail and res.measurementDropoutReason are consistent:
           measAvail = true  -> reason = "available"
           measAvail = false -> reason ~= "available"
    4. No evaluated update row remains "not_evaluated" or "uninitialized".

Expected result after logs are correctly aligned:
    outside_fov entries > 0
    available entries   > 0
    measAvail true      == reason=="available"
    mismatch counts     == 0
    out.passed          == 1

If this check fails with exactly one mismatch:
    The likely cause is an off-by-one or branch-specific logging mismatch
    between res.measAvail(k,i) and res.measurementDropoutReason(k,i).
%}

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 08-A.7 sanity check: outside-FOV dropout log consistency\n");
    fprintf("============================================================\n\n");

    cfg = config_step04_GS_DNN_EKF();

    % Short horizon keeps the test fast.
    cfg.T = 20;
    cfg.time = 0:cfg.dt:cfg.T;
    cfg.N = numel(cfg.time);

    % Explicit FOV activation.
    cfg.meas.availabilityMode = "fov";
    cfg.fov.enabled = true;
    cfg.fov.guardUnimplementedMode = false;

    % Use a fixed inertial boresight and a narrow FOV cone so that some
    % watcher-target geometries fall outside the cone.
    cfg.fov.boresightMode = "inertial_fixed";
    cfg.fov.boresightInertial = zeros(cfg.dim,1);
    cfg.fov.boresightInertial(1) = 1.0;

    cfg.fov.halfAngleDeg = 1.0;
    cfg.fov.halfAngleRad = deg2rad(cfg.fov.halfAngleDeg);

    % Disable range gating so dropout is caused by the cone check.
    cfg.fov.rhoMin = 0.0;
    cfg.fov.rhoMax = Inf;

    rng(100);
    res = simulate_GS_DNN_EKF(cfg);

    % The first row is usually initialization. The final row may be terminal
    % storage with no measurement evaluation. Only evaluate active update rows.
    updateRows = 2:(cfg.N-1);

    if isempty(updateRows)
        error("check_step08a7:InvalidHorizon", ...
            "The configured horizon has no active update rows.");
    end

    measAvail = logical(res.measAvail(updateRows,:));
    reasons = string(res.measurementDropoutReason(updateRows,:));

    numTotal = numel(measAvail);
    numMeasAvailable = nnz(measAvail);

    numReasonAvailable = nnz(reasons == "available");
    numOutsideFOV = nnz(reasons == "outside_fov");
    numRangeTooSmall = nnz(reasons == "range_too_small");
    numRangeTooLarge = nnz(reasons == "range_too_large");
    numNotEvaluated = nnz(reasons == "not_evaluated");
    numUninitialized = nnz(reasons == "uninitialized");

    uniqueReasons = unique(reasons(:));

    fprintf("FOV scenario:\n");
    fprintf("  cfg.fov.boresightMode = %s\n", string(cfg.fov.boresightMode));
    fprintf("  cfg.fov.halfAngleDeg  = %.3f\n", cfg.fov.halfAngleDeg);
    fprintf("  cfg.fov.rhoMin        = %.3f\n", cfg.fov.rhoMin);
    fprintf("  cfg.fov.rhoMax        = %.3f\n\n", cfg.fov.rhoMax);

    fprintf("Reason summary on active update rows:\n");
    fprintf("  Total evaluated entries      = %d\n", numTotal);
    fprintf("  measAvail true entries       = %d\n", numMeasAvailable);
    fprintf("  reason == available entries  = %d\n", numReasonAvailable);
    fprintf("  outside_fov entries          = %d\n", numOutsideFOV);
    fprintf("  range_too_small entries      = %d\n", numRangeTooSmall);
    fprintf("  range_too_large entries      = %d\n", numRangeTooLarge);
    fprintf("  not_evaluated entries        = %d\n", numNotEvaluated);
    fprintf("  uninitialized entries        = %d\n", numUninitialized);
    fprintf("  Unique reasons:\n");
    disp(uniqueReasons.');

    % ------------------------------------------------------------
    % Core consistency checks.
    % ------------------------------------------------------------
    passOutsideFOVExists = numOutsideFOV > 0;
    passAvailableExists = numMeasAvailable > 0;

    mismatchAvailButReasonNotAvailable = measAvail & reasons ~= "available";
    mismatchUnavailableButReasonAvailable = ~measAvail & reasons == "available";

    numMismatch1 = nnz(mismatchAvailButReasonNotAvailable);
    numMismatch2 = nnz(mismatchUnavailableButReasonAvailable);

    passConsistency = (numMismatch1 == 0) && (numMismatch2 == 0);
    passNoUnevaluatedOnUpdateRows = (numNotEvaluated == 0) && (numUninitialized == 0);

    printPassFail(passOutsideFOVExists, ...
        "Simulation produces cone-based outside_fov dropout", ...
        "Expected at least one outside_fov dropout entry.");

    printPassFail(passAvailableExists, ...
        "Simulation also has at least one available measurement", ...
        "Expected at least one available measurement entry.");

    fprintf("\nLog consistency:\n");
    fprintf("  avail=true  but reason~=available mismatches = %d\n", numMismatch1);
    fprintf("  avail=false but reason=available mismatches  = %d\n", numMismatch2);

    printPassFail(passConsistency, ...
        "measAvail and measurementDropoutReason are mutually consistent", ...
        "measAvail and measurementDropoutReason are inconsistent.");

    printPassFail(passNoUnevaluatedOnUpdateRows, ...
        "No active update row is not_evaluated/uninitialized", ...
        "Found not_evaluated or uninitialized on active update rows.");

    % ------------------------------------------------------------
    % Detailed mismatch report.
    % ------------------------------------------------------------
    if ~passConsistency
        fprintf("\nDetailed mismatch report:\n");

        [row1Local, col1] = find(mismatchAvailButReasonNotAvailable);
        [row2Local, col2] = find(mismatchUnavailableButReasonAvailable);

        row1Global = updateRows(row1Local);
        row2Global = updateRows(row2Local);

        if ~isempty(row1Global)
            fprintf("\n  Case A: measAvail=true but reason~=available\n");
            fprintf("  Global rows:\n");
            disp(row1Global(:).');
            fprintf("  Watchers:\n");
            disp(col1(:).');
            fprintf("  Reasons:\n");
            disp(reasons(mismatchAvailButReasonNotAvailable).');
        end

        if ~isempty(row2Global)
            fprintf("\n  Case B: measAvail=false but reason=available\n");
            fprintf("  Global rows:\n");
            disp(row2Global(:).');
            fprintf("  Watchers:\n");
            disp(col2(:).');
        end
    end

    out = struct();
    out.passed = passOutsideFOVExists && passAvailableExists && ...
                 passConsistency && passNoUnevaluatedOnUpdateRows;

    out.numTotal = numTotal;
    out.numMeasAvailable = numMeasAvailable;
    out.numReasonAvailable = numReasonAvailable;
    out.numOutsideFOV = numOutsideFOV;
    out.numRangeTooSmall = numRangeTooSmall;
    out.numRangeTooLarge = numRangeTooLarge;
    out.numNotEvaluated = numNotEvaluated;
    out.numUninitialized = numUninitialized;

    out.numMismatchAvailButReasonNotAvailable = numMismatch1;
    out.numMismatchUnavailableButReasonAvailable = numMismatch2;
    out.uniqueReasons = uniqueReasons;

    fprintf("\nExpected result after correct logging:\n");
    fprintf("  outside_fov entries > 0\n");
    fprintf("  available entries   > 0\n");
    fprintf("  mismatch counts     = 0\n");
    fprintf("  not_evaluated on active update rows = 0\n");
    fprintf("  out.passed          = 1\n");

    fprintf("\n============================================================\n");

    if out.passed
        fprintf("Step 08-A.7 sanity check PASSED.\n");
        fprintf("Meaning:\n");
        fprintf("  1. Real FOV cone dropout is active.\n");
        fprintf("  2. FOV dropout reasons are logged.\n");
        fprintf("  3. measAvail and measurementDropoutReason are aligned.\n");
    else
        fprintf("Step 08-A.7 sanity check FAILED.\n");
        fprintf("Meaning:\n");
        fprintf("  FOV cone dropout may exist, but diagnostic logs are not fully consistent.\n");
    end

    fprintf("============================================================\n\n");

end

function printPassFail(tf, passMessage, failMessage)
%PRINTPASSFAIL Small formatted pass/fail printer.

    if tf
        fprintf("\n[PASS] %s\n", passMessage);
    else
        fprintf("\n[FAIL] %s\n", failMessage);
    end

end