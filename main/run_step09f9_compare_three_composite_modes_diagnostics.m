function out = run_step09f9_compare_three_composite_modes_diagnostics( ...
    residualFamily, seed, dtOverride)
%{
File:
    main/run_step09f9_compare_three_composite_modes_diagnostics.m

Purpose:
    Step 09-F.9 diagnostic comparison among three GS composite modes:

        1) additive
        2) gated_additive
        3) local_full_plus_gated_nonlocal

Why this step matters:
    Step 09-F.8 showed:

        additive                         best mean/RMS tracking
        local_full_plus_gated_nonlocal   intermediate tracking
        gated_additive                   close to Local

    However, previous diagnostics showed that residual approximation quality
    and tracking quality do not always rank the same way.

    This script compares tracking and residual approximation for all three
    modes under the same residual family, seed, and dt.

Key residual errors:
    Operational residual error:

        || d_hat(eta_hat) - d_true(eta_true) ||

    Same-input residual error:

        || d_hat(eta_true) - d_true(eta_true) ||

Usage:
    out09f9 = run_step09f9_compare_three_composite_modes_diagnostics( ...
        "complex_branchwise", 101, 0.5);

    disp(out09f9.trackingSummary);
    disp(out09f9.residualOverallSummary);
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

    modeList = [
        "additive"
        "gated_additive"
        "local_full_plus_gated_nonlocal"
    ];

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-F.9: Three GS composite mode diagnostics\n");
    fprintf("============================================================\n");
    fprintf("Residual family = %s\n", residualFamily);
    fprintf("Seed            = %d\n", seed);
    fprintf("dt              = %.6g\n\n", dtOverride);

    nMode = numel(modeList);

    outRuns = cell(nMode, 1);
    outDiag = cell(nMode, 1);

    trackingSummary = table();

    for iMode = 1:nMode

        mode_i = modeList(iMode);

        fprintf("Running mode: %s\n", mode_i);

        outRuns{iMode} = run_step09c0_compare_always_available_Local_GS_Oracle( ...
            residualFamily, seed, false, dtOverride, [], mode_i);

        trackingSummary = [
            trackingSummary
            outRuns{iMode}.compactComparisonTable
        ];

        outDiag{iMode} = run_step09d3_residual_input_mode_diagnostics( ...
            outRuns{iMode});

    end

    residualOverallSummary = buildOverallResidualSummary_step09f9( ...
        modeList, outDiag);

    residualWindowSummary = buildWindowResidualSummary_step09f9( ...
        modeList, outDiag);

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

    fprintf("\nResidual overall summary:\n");
    disp(residualOverallSummary);

    fprintf("\nResidual window summary:\n");
    disp(residualWindowSummary);

    out = struct();

    out.residualFamily = residualFamily;
    out.seed = seed;
    out.dt = dtOverride;

    out.modeList = modeList;
    out.outRuns = outRuns;
    out.outDiag = outDiag;

    out.outAdd = outRuns{1};
    out.outGate = outRuns{2};
    out.outHybrid = outRuns{3};

    out.diagAdd = outDiag{1};
    out.diagGate = outDiag{2};
    out.diagHybrid = outDiag{3};

    out.trackingSummary = trackingSummary;
    out.residualOverallSummary = residualOverallSummary;
    out.residualWindowSummary = residualWindowSummary;

end

function residualOverallSummary = buildOverallResidualSummary_step09f9( ...
    modeList, outDiag)
% Build compact overall residual table for Local, each GS mode, and Oracle.

    nMode = numel(modeList);

    % Local and Oracle are identical across these paired runs, so use the
    % first diagnostic run for those rows.
    T0 = outDiag{1}.overallSummaryTable;

    localRow = selectCaseRow_step09f9(T0, "local");
    oracleRow = selectCaseRow_step09f9(T0, "oracle");

    caseName = strings(nMode + 2, 1);
    compositeMode = strings(nMode + 2, 1);

    meanOperationalErr = NaN(nMode + 2, 1);
    meanSameInputErr = NaN(nMode + 2, 1);
    meanInputPenalty_OperationalMinusSameInput = NaN(nMode + 2, 1);
    rmsOperationalErr = NaN(nMode + 2, 1);
    rmsSameInputErr = NaN(nMode + 2, 1);
    p95OperationalErr = NaN(nMode + 2, 1);
    p95SameInputErr = NaN(nMode + 2, 1);

    % Local row.
    caseName(1) = "Local DNN";
    compositeMode(1) = "local";

    [meanOperationalErr(1), ...
     meanSameInputErr(1), ...
     meanInputPenalty_OperationalMinusSameInput(1), ...
     rmsOperationalErr(1), ...
     rmsSameInputErr(1), ...
     p95OperationalErr(1), ...
     p95SameInputErr(1)] = extractResidualScalars_step09f9(localRow);

    % GS rows.
    for iMode = 1:nMode

        T = outDiag{iMode}.overallSummaryTable;
        gsRow = selectCaseRow_step09f9(T, "gs");

        iRow = iMode + 1;

        caseName(iRow) = "GS " + modeList(iMode);
        compositeMode(iRow) = modeList(iMode);

        [meanOperationalErr(iRow), ...
         meanSameInputErr(iRow), ...
         meanInputPenalty_OperationalMinusSameInput(iRow), ...
         rmsOperationalErr(iRow), ...
         rmsSameInputErr(iRow), ...
         p95OperationalErr(iRow), ...
         p95SameInputErr(iRow)] = extractResidualScalars_step09f9(gsRow);

    end

    % Oracle row.
    iOracle = nMode + 2;

    caseName(iOracle) = "Oracle";
    compositeMode(iOracle) = "oracle";

    [meanOperationalErr(iOracle), ...
     meanSameInputErr(iOracle), ...
     meanInputPenalty_OperationalMinusSameInput(iOracle), ...
     rmsOperationalErr(iOracle), ...
     rmsSameInputErr(iOracle), ...
     p95OperationalErr(iOracle), ...
     p95SameInputErr(iOracle)] = extractResidualScalars_step09f9(oracleRow);

    residualOverallSummary = table( ...
        caseName, ...
        compositeMode, ...
        meanOperationalErr, ...
        meanSameInputErr, ...
        meanInputPenalty_OperationalMinusSameInput, ...
        rmsOperationalErr, ...
        rmsSameInputErr, ...
        p95OperationalErr, ...
        p95SameInputErr);

end

function residualWindowSummary = buildWindowResidualSummary_step09f9( ...
    modeList, outDiag)
% Build GS-only windowed residual summary for all composite modes.

    Tall = table();

    for iMode = 1:numel(modeList)

        T = outDiag{iMode}.windowSummaryTable;

        gsMask = isGSCase_step09f9(T.caseName);
        Tgs = T(gsMask, :);

        compositeMode = repmat(modeList(iMode), height(Tgs), 1);

        Tsmall = table( ...
            compositeMode, ...
            Tgs.windowName, ...
            Tgs.tStart_s, ...
            Tgs.tEnd_s, ...
            Tgs.meanOperationalErr, ...
            Tgs.meanSameInputErr, ...
            Tgs.meanInputPenalty_OperationalMinusSameInput, ...
            Tgs.rmsOperationalErr, ...
            Tgs.rmsSameInputErr, ...
            Tgs.p95OperationalErr, ...
            Tgs.p95SameInputErr, ...
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

        Tall = [
            Tall
            Tsmall
        ];

    end

    residualWindowSummary = Tall;

end

function row = selectCaseRow_step09f9(T, whichCase)
% Robust row selector because diagnostic labels may change slightly after
% plot/label cleanup patches.

    whichCase = string(whichCase);
    caseNames = string(T.caseName);

    switch whichCase

        case "local"
            mask = contains(lower(caseNames), "local");

        case "oracle"
            mask = contains(lower(caseNames), "oracle");

        case "gs"
            mask = isGSCase_step09f9(caseNames);

        otherwise
            error("Step09F9:UnsupportedCaseSelector", ...
                "Unsupported selector: %s.", whichCase);

    end

    if nnz(mask) ~= 1
        error("Step09F9:CaseRowSelectionFailed", ...
            "Expected exactly one %s row, found %d.", whichCase, nnz(mask));
    end

    row = T(mask, :);

end

function mask = isGSCase_step09f9(caseNames)
% Identify the GS row while excluding Local and Oracle.

    caseNames = string(caseNames);
    lowerNames = lower(caseNames);

    mask = contains(lowerNames, "gs") | contains(lowerNames, "composite");

    mask = mask & ~contains(lowerNames, "local");
    mask = mask & ~contains(lowerNames, "oracle");

end

function [meanOperationalErr, ...
          meanSameInputErr, ...
          meanInputPenalty, ...
          rmsOperationalErr, ...
          rmsSameInputErr, ...
          p95OperationalErr, ...
          p95SameInputErr] = extractResidualScalars_step09f9(row)

    meanOperationalErr = row.meanOperationalErr;
    meanSameInputErr = row.meanSameInputErr;
    meanInputPenalty = row.meanInputPenalty_OperationalMinusSameInput;
    rmsOperationalErr = row.rmsOperationalErr;
    rmsSameInputErr = row.rmsSameInputErr;
    p95OperationalErr = row.p95OperationalErr;
    p95SameInputErr = row.p95SameInputErr;

end