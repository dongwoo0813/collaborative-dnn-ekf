function out = run_step09f4_compare_additive_vs_gated_diagnostics( ...
    residualFamily, seed, dtOverride)
%{
File:
    main/run_step09f4_compare_additive_vs_gated_diagnostics.m

Purpose:
    Step 09-F.4 diagnostic comparison between

        1) GS additive
        2) GS gated_additive

    under the same always-available Local / GS / Oracle setup.

Why this step matters:
    The first gated_additive tracking result showed that gated_additive
    almost recovered Local performance, while additive still improved over
    Local.

    This file checks whether that happens because the gated composite
    residual itself is a worse approximation of the true residual, or
    because of estimator/covariance/trajectory interaction.

What it runs:
    For the same residual family, seed, and dt:

        outAdd  = run_step09c0_compare_always_available_Local_GS_Oracle(..., "additive")
        outGate = run_step09c0_compare_always_available_Local_GS_Oracle(..., "gated_additive")

    Then it runs Step 09-D.3 residual input-mode diagnostics on both cases.

Key residual errors:
    Operational residual error:

        || d_hat(eta_hat) - d_true(eta_true) ||

    Same-input residual error:

        || d_hat(eta_true) - d_true(eta_true) ||

Interpretation:
    If gated_additive has worse same-input residual error, then the gated
    composite model is not approximating the residual as well.

    If same-input residual error is similar but operational residual error
    is worse, then the issue is more likely related to EKF interaction,
    state-estimate trajectory, covariance propagation, or learning dynamics.

Usage:
    out09f4 = run_step09f4_compare_additive_vs_gated_diagnostics( ...
        "complex_branchwise", 101, 0.5);

    disp(out09f4.trackingSummary);
    disp(out09f4.residualOverallSummary);
%}

    if nargin < 1 || isempty(residualFamily)
        residualFamily = "complex_branchwise";
    end

    if nargin < 2 || isempty(seed)
        seed = 101;
    end

    if nargin < 3 || isempty(dtOverride)
        dtOverride = 0.5;
    end

    residualFamily = string(residualFamily);

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-F.4: Additive vs gated_additive diagnostics\n");
    fprintf("============================================================\n");
    fprintf("Residual family = %s\n", residualFamily);
    fprintf("Seed            = %d\n", seed);
    fprintf("dt              = %.6g\n", dtOverride);

    % ---------------------------------------------------------------------
    % Run paired tracking comparisons
    % ---------------------------------------------------------------------
    outAdd = run_step09c0_compare_always_available_Local_GS_Oracle( ...
        residualFamily, seed, false, dtOverride, [], "additive");

    outGate = run_step09c0_compare_always_available_Local_GS_Oracle( ...
        residualFamily, seed, false, dtOverride, [], "gated_additive");

    trackingSummary = [
        outAdd.compactComparisonTable
        outGate.compactComparisonTable
    ];

    fprintf("\nTracking summary:\n");
    disp(trackingSummary(:, [
        "residualFamily", ...
        "seed", ...
        "gsCompositeMode", ...
        "localMeanPosErr_m", ...
        "gsMeanPosErr_m", ...
        "oracleMeanPosErr_m", ...
        "gsMeanImprovementPct", ...
        "gsRmsImprovementPct", ...
        "gsP95ImprovementPct", ...
        "localMeanNIS", ...
        "gsMeanNIS"]));

    % ---------------------------------------------------------------------
    % Run residual input-mode diagnostics
    % ---------------------------------------------------------------------
    diagAdd = run_step09d3_residual_input_mode_diagnostics(outAdd);
    diagGate = run_step09d3_residual_input_mode_diagnostics(outGate);

    residualOverallSummary = buildOverallResidualSummary_step09f4( ...
        diagAdd, diagGate);

    residualWindowSummary = buildWindowResidualSummary_step09f4( ...
        diagAdd, diagGate);

    fprintf("\nResidual overall summary:\n");
    disp(residualOverallSummary);

    fprintf("\nResidual window summary:\n");
    disp(residualWindowSummary);

    % ---------------------------------------------------------------------
    % Pack output
    % ---------------------------------------------------------------------
    out = struct();

    out.residualFamily = residualFamily;
    out.seed = seed;
    out.dt = dtOverride;

    out.outAdd = outAdd;
    out.outGate = outGate;

    out.diagAdd = diagAdd;
    out.diagGate = diagGate;

    out.trackingSummary = trackingSummary;
    out.residualOverallSummary = residualOverallSummary;
    out.residualWindowSummary = residualWindowSummary;

end

function residualOverallSummary = buildOverallResidualSummary_step09f4( ...
    diagAdd, diagGate)
% Build a compact residual summary for Local, GS-additive, GS-gated, Oracle.

    Tadd = diagAdd.overallSummaryTable;
    Tgate = diagGate.overallSummaryTable;

    localRow = Tadd(Tadd.caseName == "Local DNN", :);
    gsAddRow = Tadd(Tadd.caseName == "GS composite", :);
    gsGateRow = Tgate(Tgate.caseName == "GS composite", :);
    oracleRow = Tadd(Tadd.caseName == "Oracle", :);

    compositeMode = [
        "local"
        "additive"
        "gated_additive"
        "oracle"
    ];

    caseName = [
        "Local DNN"
        "GS additive"
        "GS gated_additive"
        "Oracle"
    ];

    sourceRows = [
        localRow
        gsAddRow
        gsGateRow
        oracleRow
    ];

    residualOverallSummary = table( ...
        caseName, ...
        compositeMode, ...
        sourceRows.meanOperationalErr, ...
        sourceRows.meanSameInputErr, ...
        sourceRows.meanInputPenalty_OperationalMinusSameInput, ...
        sourceRows.rmsOperationalErr, ...
        sourceRows.rmsSameInputErr, ...
        sourceRows.p95OperationalErr, ...
        sourceRows.p95SameInputErr, ...
        'VariableNames', { ...
            'caseName', ...
            'compositeMode', ...
            'meanOperationalErr', ...
            'meanSameInputErr', ...
            'meanInputPenalty_OperationalMinusSameInput', ...
            'rmsOperationalErr', ...
            'rmsSameInputErr', ...
            'p95OperationalErr', ...
            'p95SameInputErr'});

end

function residualWindowSummary = buildWindowResidualSummary_step09f4( ...
    diagAdd, diagGate)
% Build windowed residual summary for GS additive vs GS gated_additive.

    Tadd = diagAdd.windowSummaryTable;
    Tgate = diagGate.windowSummaryTable;

    gsAdd = Tadd(Tadd.caseName == "GS composite", :);
    gsGate = Tgate(Tgate.caseName == "GS composite", :);

    nAdd = height(gsAdd);
    nGate = height(gsGate);

    if nAdd ~= nGate
        error("Step09F4:WindowTableSizeMismatch", ...
            "GS additive and GS gated window tables have different heights.");
    end

    compositeMode = [
        repmat("additive", nAdd, 1)
        repmat("gated_additive", nGate, 1)
    ];

    sourceRows = [
        gsAdd
        gsGate
    ];

    residualWindowSummary = table( ...
        compositeMode, ...
        sourceRows.windowName, ...
        sourceRows.tStart_s, ...
        sourceRows.tEnd_s, ...
        sourceRows.meanOperationalErr, ...
        sourceRows.meanSameInputErr, ...
        sourceRows.meanInputPenalty_OperationalMinusSameInput, ...
        sourceRows.rmsOperationalErr, ...
        sourceRows.rmsSameInputErr, ...
        sourceRows.p95OperationalErr, ...
        sourceRows.p95SameInputErr, ...
        'VariableNames', { ...
            'compositeMode', ...
            'windowName', ...
            'tStart_s', ...
            'tEnd_s', ...
            'meanOperationalErr', ...
            'meanSameInputErr', ...
            'meanInputPenalty_OperationalMinusSameInput', ...
            'rmsOperationalErr', ...
            'rmsSameInputErr', ...
            'p95OperationalErr', ...
            'p95SameInputErr'});

end