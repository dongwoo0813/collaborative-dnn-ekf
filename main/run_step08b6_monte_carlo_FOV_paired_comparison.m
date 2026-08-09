function out = run_step08b6_monte_carlo_FOV_paired_comparison(halfAngleDeg, seedList, keepRawOutputs)
%{
File:
    main/run_step08b6_monte_carlo_FOV_paired_comparison.m

Purpose:
    Step 08-B.6 Monte Carlo paired comparison for the FOV Local / GS /
    Oracle estimator cases.

    This script repeatedly calls:
        run_step08b3_compare_FOV_Local_GS_Oracle(halfAngleDeg, false, seed)

    For each seed, Local DNN, GS composite, and Oracle are run with the same
    random seed so that the comparison is paired and fair.

Default:
    halfAngleDeg = 90
        Start with the mild-dropout case because Step 08-B.5 showed a clear
        GS/Oracle benefit there.

    seedList = 101:103
        Small default for a quick first Monte Carlo smoke run.

Outputs:
    out.perSeedTable
        One row per seed with Local / GS / Oracle metrics.

    out.aggregateTable
        Monte Carlo summary of paired improvement metrics.

    out.rawOutputCell
        Optional raw Step 08-B.3/08-B.4b outputs if keepRawOutputs = true.

Usage:
    out08b6 = run_step08b6_monte_carlo_FOV_paired_comparison();

    out08b6 = run_step08b6_monte_carlo_FOV_paired_comparison(90, 101:110);

    out08b6 = run_step08b6_monte_carlo_FOV_paired_comparison(40, 101:105);
%}

    if nargin < 1 || isempty(halfAngleDeg)
        halfAngleDeg = 90.0;
    end

    if nargin < 2 || isempty(seedList)
        seedList = 101:103;
    end

    if nargin < 3 || isempty(keepRawOutputs)
        keepRawOutputs = false;
    end

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 08-B.6: Monte Carlo paired FOV comparison\n");
    fprintf("============================================================\n");
    fprintf("FOV half-angle = %.3f deg\n", halfAngleDeg);
    fprintf("Number of seeds = %d\n\n", numel(seedList));

    addpath(genpath(pwd));
    rehash;

    seedList = seedList(:);
    nSeed = numel(seedList);

    if keepRawOutputs
        rawOutputCell = cell(nSeed, 1);
    else
        rawOutputCell = {};
    end

    halfAngleDegCol = repmat(halfAngleDeg, nSeed, 1);
    availabilityRatePercent = NaN(nSeed, 1);

    localMeanPosErr_m  = NaN(nSeed, 1);
    gsMeanPosErr_m     = NaN(nSeed, 1);
    oracleMeanPosErr_m = NaN(nSeed, 1);

    localRmsPosErr_m  = NaN(nSeed, 1);
    gsRmsPosErr_m     = NaN(nSeed, 1);
    oracleRmsPosErr_m = NaN(nSeed, 1);

    localMedianPosErr_m  = NaN(nSeed, 1);
    gsMedianPosErr_m     = NaN(nSeed, 1);
    oracleMedianPosErr_m = NaN(nSeed, 1);

    localP95PosErr_m  = NaN(nSeed, 1);
    gsP95PosErr_m     = NaN(nSeed, 1);
    oracleP95PosErr_m = NaN(nSeed, 1);

    localDropoutMeanPosErr_m  = NaN(nSeed, 1);
    gsDropoutMeanPosErr_m     = NaN(nSeed, 1);
    oracleDropoutMeanPosErr_m = NaN(nSeed, 1);

    localFinalMeanPosErr_m  = NaN(nSeed, 1);
    gsFinalMeanPosErr_m     = NaN(nSeed, 1);
    oracleFinalMeanPosErr_m = NaN(nSeed, 1);

    gsLoggedUploadDecisions = NaN(nSeed, 1);
    gsFinalTotalUploads = NaN(nSeed, 1);

    gsMeanImprovementPct = NaN(nSeed, 1);
    oracleMeanImprovementPct = NaN(nSeed, 1);

    gsRmsImprovementPct = NaN(nSeed, 1);
    oracleRmsImprovementPct = NaN(nSeed, 1);

    gsP95ImprovementPct = NaN(nSeed, 1);
    oracleP95ImprovementPct = NaN(nSeed, 1);

    gsDropoutMeanImprovementPct = NaN(nSeed, 1);
    oracleDropoutMeanImprovementPct = NaN(nSeed, 1);

    gsClosedLocalOracleGapPct = NaN(nSeed, 1);

    for iseed = 1:nSeed

        seed = seedList(iseed);

        fprintf("Running MC seed %d / %d: seed = %d\n", iseed, nSeed, seed);

        % Run the paired Local / GS / Oracle comparison quietly.
        outCase = run_step08b3_compare_FOV_Local_GS_Oracle( ...
            halfAngleDeg, false, seed);

        if keepRawOutputs
            rawOutputCell{iseed} = outCase;
        end

        T = outCase.multiMetricSummaryTable;

        rowLocal  = findCaseRow_step08b6(T, "Local DNN + FOV");
        rowGS     = findCaseRow_step08b6(T, "GS composite + FOV");
        rowOracle = findCaseRow_step08b6(T, "Oracle + FOV");

        availabilityRatePercent(iseed) = readTableValue_step08b6(T, rowGS, "availabilityRatePercent");

        localMeanPosErr_m(iseed)  = readTableValue_step08b6(T, rowLocal,  "meanPosErr_m");
        gsMeanPosErr_m(iseed)     = readTableValue_step08b6(T, rowGS,     "meanPosErr_m");
        oracleMeanPosErr_m(iseed) = readTableValue_step08b6(T, rowOracle, "meanPosErr_m");

        localRmsPosErr_m(iseed)  = readTableValue_step08b6(T, rowLocal,  "rmsPosErr_m");
        gsRmsPosErr_m(iseed)     = readTableValue_step08b6(T, rowGS,     "rmsPosErr_m");
        oracleRmsPosErr_m(iseed) = readTableValue_step08b6(T, rowOracle, "rmsPosErr_m");

        localMedianPosErr_m(iseed)  = readTableValue_step08b6(T, rowLocal,  "medianPosErr_m");
        gsMedianPosErr_m(iseed)     = readTableValue_step08b6(T, rowGS,     "medianPosErr_m");
        oracleMedianPosErr_m(iseed) = readTableValue_step08b6(T, rowOracle, "medianPosErr_m");

        localP95PosErr_m(iseed)  = readTableValue_step08b6(T, rowLocal,  "p95PosErr_m");
        gsP95PosErr_m(iseed)     = readTableValue_step08b6(T, rowGS,     "p95PosErr_m");
        oracleP95PosErr_m(iseed) = readTableValue_step08b6(T, rowOracle, "p95PosErr_m");

        localDropoutMeanPosErr_m(iseed)  = readTableValue_step08b6(T, rowLocal,  "meanPosErr_dropout_m");
        gsDropoutMeanPosErr_m(iseed)     = readTableValue_step08b6(T, rowGS,     "meanPosErr_dropout_m");
        oracleDropoutMeanPosErr_m(iseed) = readTableValue_step08b6(T, rowOracle, "meanPosErr_dropout_m");

        localFinalMeanPosErr_m(iseed)  = readTableValue_step08b6(T, rowLocal,  "finalMeanPosErr_m");
        gsFinalMeanPosErr_m(iseed)     = readTableValue_step08b6(T, rowGS,     "finalMeanPosErr_m");
        oracleFinalMeanPosErr_m(iseed) = readTableValue_step08b6(T, rowOracle, "finalMeanPosErr_m");

        gsLoggedUploadDecisions(iseed) = readTableValue_step08b6(T, rowGS, "gsLoggedUploadDecisions");
        gsFinalTotalUploads(iseed)     = readTableValue_step08b6(T, rowGS, "gsFinalTotalUploads");

        gsMeanImprovementPct(iseed) = percentImprovement_step08b6( ...
            localMeanPosErr_m(iseed), gsMeanPosErr_m(iseed));

        oracleMeanImprovementPct(iseed) = percentImprovement_step08b6( ...
            localMeanPosErr_m(iseed), oracleMeanPosErr_m(iseed));

        gsRmsImprovementPct(iseed) = percentImprovement_step08b6( ...
            localRmsPosErr_m(iseed), gsRmsPosErr_m(iseed));

        oracleRmsImprovementPct(iseed) = percentImprovement_step08b6( ...
            localRmsPosErr_m(iseed), oracleRmsPosErr_m(iseed));

        gsP95ImprovementPct(iseed) = percentImprovement_step08b6( ...
            localP95PosErr_m(iseed), gsP95PosErr_m(iseed));

        oracleP95ImprovementPct(iseed) = percentImprovement_step08b6( ...
            localP95PosErr_m(iseed), oracleP95PosErr_m(iseed));

        gsDropoutMeanImprovementPct(iseed) = percentImprovement_step08b6( ...
            localDropoutMeanPosErr_m(iseed), gsDropoutMeanPosErr_m(iseed));

        oracleDropoutMeanImprovementPct(iseed) = percentImprovement_step08b6( ...
            localDropoutMeanPosErr_m(iseed), oracleDropoutMeanPosErr_m(iseed));

        localOracleGap = localMeanPosErr_m(iseed) - oracleMeanPosErr_m(iseed);

        % Gap closure is meaningful only when Oracle improves over Local.
        if localOracleGap > 0
            gsClosedLocalOracleGapPct(iseed) = ...
                100 * (localMeanPosErr_m(iseed) - gsMeanPosErr_m(iseed)) / localOracleGap;
        end

    end

    perSeedTable = table( ...
        seedList, ...
        halfAngleDegCol, ...
        availabilityRatePercent, ...
        localMeanPosErr_m, ...
        gsMeanPosErr_m, ...
        oracleMeanPosErr_m, ...
        gsMeanImprovementPct, ...
        oracleMeanImprovementPct, ...
        localRmsPosErr_m, ...
        gsRmsPosErr_m, ...
        oracleRmsPosErr_m, ...
        gsRmsImprovementPct, ...
        oracleRmsImprovementPct, ...
        localMedianPosErr_m, ...
        gsMedianPosErr_m, ...
        oracleMedianPosErr_m, ...
        localP95PosErr_m, ...
        gsP95PosErr_m, ...
        oracleP95PosErr_m, ...
        gsP95ImprovementPct, ...
        oracleP95ImprovementPct, ...
        localDropoutMeanPosErr_m, ...
        gsDropoutMeanPosErr_m, ...
        oracleDropoutMeanPosErr_m, ...
        gsDropoutMeanImprovementPct, ...
        oracleDropoutMeanImprovementPct, ...
        localFinalMeanPosErr_m, ...
        gsFinalMeanPosErr_m, ...
        oracleFinalMeanPosErr_m, ...
        gsClosedLocalOracleGapPct, ...
        gsLoggedUploadDecisions, ...
        gsFinalTotalUploads, ...
        'VariableNames', { ...
            'seed', ...
            'halfAngleDeg', ...
            'availabilityRatePercent', ...
            'localMeanPosErr_m', ...
            'gsMeanPosErr_m', ...
            'oracleMeanPosErr_m', ...
            'gsMeanImprovementPct', ...
            'oracleMeanImprovementPct', ...
            'localRmsPosErr_m', ...
            'gsRmsPosErr_m', ...
            'oracleRmsPosErr_m', ...
            'gsRmsImprovementPct', ...
            'oracleRmsImprovementPct', ...
            'localMedianPosErr_m', ...
            'gsMedianPosErr_m', ...
            'oracleMedianPosErr_m', ...
            'localP95PosErr_m', ...
            'gsP95PosErr_m', ...
            'oracleP95PosErr_m', ...
            'gsP95ImprovementPct', ...
            'oracleP95ImprovementPct', ...
            'localDropoutMeanPosErr_m', ...
            'gsDropoutMeanPosErr_m', ...
            'oracleDropoutMeanPosErr_m', ...
            'gsDropoutMeanImprovementPct', ...
            'oracleDropoutMeanImprovementPct', ...
            'localFinalMeanPosErr_m', ...
            'gsFinalMeanPosErr_m', ...
            'oracleFinalMeanPosErr_m', ...
            'gsClosedLocalOracleGapPct', ...
            'gsLoggedUploadDecisions', ...
            'gsFinalTotalUploads'});

    metricName = [
        "gsMeanImprovementPct"
        "oracleMeanImprovementPct"
        "gsRmsImprovementPct"
        "oracleRmsImprovementPct"
        "gsP95ImprovementPct"
        "oracleP95ImprovementPct"
        "gsDropoutMeanImprovementPct"
        "oracleDropoutMeanImprovementPct"
        "gsClosedLocalOracleGapPct"
    ];

    metricData = [
        gsMeanImprovementPct, ...
        oracleMeanImprovementPct, ...
        gsRmsImprovementPct, ...
        oracleRmsImprovementPct, ...
        gsP95ImprovementPct, ...
        oracleP95ImprovementPct, ...
        gsDropoutMeanImprovementPct, ...
        oracleDropoutMeanImprovementPct, ...
        gsClosedLocalOracleGapPct];

    aggregateTable = summarizeMetricColumns_step08b6(metricName, metricData);

    fprintf("\n============================================================\n");
    fprintf("Step 08-B.6 per-seed paired comparison table\n");
    fprintf("============================================================\n");
    disp(perSeedTable);

    fprintf("\n============================================================\n");
    fprintf("Step 08-B.6 aggregate paired-improvement summary\n");
    fprintf("============================================================\n");
    disp(aggregateTable);

    out = struct();
    out.halfAngleDeg = halfAngleDeg;
    out.seedList = seedList;
    out.perSeedTable = perSeedTable;
    out.aggregateTable = aggregateTable;
    out.keepRawOutputs = keepRawOutputs;
    out.rawOutputCell = rawOutputCell;

    fprintf("\n============================================================\n");
    fprintf("Step 08-B.6 Monte Carlo paired comparison complete.\n");
    fprintf("============================================================\n\n");

end

function row = findCaseRow_step08b6(T, caseLabel)
%FINDCASEROW_STEP08B6 Find one estimator-case row in a metric table.

    caseName = string(T.caseName);
    row = find(caseName == caseLabel, 1, 'first');

    if isempty(row)
        error("step08b6:MissingCaseRow", ...
            "Could not find caseName = %s in the metric table.", caseLabel);
    end

end

function value = readTableValue_step08b6(T, row, varName)
%READTABLEVALUE_STEP08B6 Read a scalar table value with NaN fallback.
%
% The NaN fallback keeps the MC script robust if a future evaluator version
% omits a diagnostic that is not available for every estimator case.

    if ismember(varName, string(T.Properties.VariableNames))
        value = T.(varName)(row);
    else
        value = NaN;
    end

end

function pct = percentImprovement_step08b6(baselineValue, testValue)
%PERCENTIMPROVEMENT_STEP08B6 Positive value means testValue is better.
%
% For error metrics, improvement is defined as:
%     100 * (baseline - test) / baseline.

    if isfinite(baselineValue) && baselineValue > 0 && isfinite(testValue)
        pct = 100 * (baselineValue - testValue) / baselineValue;
    else
        pct = NaN;
    end

end

function aggregateTable = summarizeMetricColumns_step08b6(metricName, metricData)
%SUMMARIZEMETRICCOLUMNS_STEP08B6 Summarize MC paired-improvement metrics.
%
% positiveRatePercent tells how often the corresponding improvement metric
% was positive across seeds. For GS metrics, positive means GS beat Local.
% For Oracle metrics, positive means Oracle beat Local.

    nMetric = numel(metricName);

    nFinite = NaN(nMetric, 1);
    meanValue = NaN(nMetric, 1);
    stdValue = NaN(nMetric, 1);
    medianValue = NaN(nMetric, 1);
    minValue = NaN(nMetric, 1);
    maxValue = NaN(nMetric, 1);
    positiveCount = NaN(nMetric, 1);
    positiveRatePercent = NaN(nMetric, 1);

    for im = 1:nMetric

        x = metricData(:, im);
        x = x(isfinite(x));

        nFinite(im) = numel(x);

        if isempty(x)
            continue;
        end

        meanValue(im) = mean(x);
        stdValue(im) = std(x);
        medianValue(im) = median(x);
        minValue(im) = min(x);
        maxValue(im) = max(x);

        positiveCount(im) = nnz(x > 0);
        positiveRatePercent(im) = 100 * positiveCount(im) / nFinite(im);

    end

    aggregateTable = table( ...
        metricName, ...
        nFinite, ...
        meanValue, ...
        stdValue, ...
        medianValue, ...
        minValue, ...
        maxValue, ...
        positiveCount, ...
        positiveRatePercent, ...
        'VariableNames', { ...
            'metricName', ...
            'nFinite', ...
            'meanValue', ...
            'stdValue', ...
            'medianValue', ...
            'minValue', ...
            'maxValue', ...
            'positiveCount', ...
            'positiveRatePercent'});

end