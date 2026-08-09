function out = run_step09b5_compare_residual_families_single_seed(halfAngleDeg, seed, residualFamilyList)
%{
File:
    main/run_step09b5_compare_residual_families_single_seed.m

Purpose:
    Step 09-B.5 single-seed residual-family comparison.

    Compare the validated simple residual benchmark against the new
    coupled_nonlinear benchmark using the same FOV angle and random seed.

    Each residual family calls:
        run_step08b3_compare_FOV_Local_GS_Oracle( ...
            halfAngleDeg, false, seed, residualFamily)

Default:
    halfAngleDeg = 90
    seed = 101
    residualFamilyList = ["simple_branchwise"; "coupled_nonlinear"]

Outputs:
    out.longMetricTable
        Local / GS / Oracle metric rows for every residual family.

    out.compactComparisonTable
        One row per residual family with key comparison metrics.

Usage:
    out09b5 = run_step09b5_compare_residual_families_single_seed();

    out09b5 = run_step09b5_compare_residual_families_single_seed(90, 101);
%}

    if nargin < 1 || isempty(halfAngleDeg)
        halfAngleDeg = 90.0;
    end

    if nargin < 2 || isempty(seed)
        seed = 101;
    end

    if nargin < 3 || isempty(residualFamilyList)
        residualFamilyList = [
            "simple_branchwise"
            "coupled_nonlinear"
        ];
    end

    residualFamilyList = string(residualFamilyList(:));
    nFamily = numel(residualFamilyList);

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-B.5: Residual-family single-seed comparison\n");
    fprintf("============================================================\n");
    fprintf("FOV half-angle = %.3f deg\n", halfAngleDeg);
    fprintf("Random seed    = %d\n\n", seed);

    addpath(genpath(pwd));
    rehash;

    outCaseCell = cell(nFamily, 1);
    metricTableCell = cell(nFamily, 1);

    familyName = residualFamilyList;
    availabilityRatePercent = NaN(nFamily, 1);

    localMeanPosErr_m = NaN(nFamily, 1);
    gsMeanPosErr_m = NaN(nFamily, 1);
    oracleMeanPosErr_m = NaN(nFamily, 1);

    localRmsPosErr_m = NaN(nFamily, 1);
    gsRmsPosErr_m = NaN(nFamily, 1);
    oracleRmsPosErr_m = NaN(nFamily, 1);

    localP95PosErr_m = NaN(nFamily, 1);
    gsP95PosErr_m = NaN(nFamily, 1);
    oracleP95PosErr_m = NaN(nFamily, 1);

    localMeanNIS = NaN(nFamily, 1);
    gsMeanNIS = NaN(nFamily, 1);
    oracleMeanNIS = NaN(nFamily, 1);

    gsMeanImprovementPct = NaN(nFamily, 1);
    oracleMeanImprovementPct = NaN(nFamily, 1);

    gsRmsImprovementPct = NaN(nFamily, 1);
    oracleRmsImprovementPct = NaN(nFamily, 1);

    gsP95ImprovementPct = NaN(nFamily, 1);
    oracleP95ImprovementPct = NaN(nFamily, 1);

    gsFinalTotalUploads = NaN(nFamily, 1);

    for iFamily = 1:nFamily

        residualFamily = residualFamilyList(iFamily);

        fprintf("Running residual family %d / %d: %s\n", ...
            iFamily, nFamily, residualFamily);

        outCase = run_step08b3_compare_FOV_Local_GS_Oracle( ...
            halfAngleDeg, false, seed, residualFamily);

        outCaseCell{iFamily} = outCase;

        T = outCase.multiMetricSummaryTable;

        residualFamilyCol = repmat(residualFamily, height(T), 1);
        T = addvars(T, residualFamilyCol, ...
            'Before', 1, ...
            'NewVariableNames', 'residualFamily');

        metricTableCell{iFamily} = T;

        rowLocal = find(string(T.caseName) == "Local DNN + FOV", 1);
        rowGS = find(string(T.caseName) == "GS composite + FOV", 1);
        rowOracle = find(string(T.caseName) == "Oracle + FOV", 1);

        availabilityRatePercent(iFamily) = T.availabilityRatePercent(rowGS);

        localMeanPosErr_m(iFamily) = T.meanPosErr_m(rowLocal);
        gsMeanPosErr_m(iFamily) = T.meanPosErr_m(rowGS);
        oracleMeanPosErr_m(iFamily) = T.meanPosErr_m(rowOracle);

        localRmsPosErr_m(iFamily) = T.rmsPosErr_m(rowLocal);
        gsRmsPosErr_m(iFamily) = T.rmsPosErr_m(rowGS);
        oracleRmsPosErr_m(iFamily) = T.rmsPosErr_m(rowOracle);

        localP95PosErr_m(iFamily) = T.p95PosErr_m(rowLocal);
        gsP95PosErr_m(iFamily) = T.p95PosErr_m(rowGS);
        oracleP95PosErr_m(iFamily) = T.p95PosErr_m(rowOracle);

        localMeanNIS(iFamily) = T.meanNIS(rowLocal);
        gsMeanNIS(iFamily) = T.meanNIS(rowGS);
        oracleMeanNIS(iFamily) = T.meanNIS(rowOracle);

        gsFinalTotalUploads(iFamily) = T.gsFinalTotalUploads(rowGS);

        gsMeanImprovementPct(iFamily) = percentImprovement_step09b5( ...
            localMeanPosErr_m(iFamily), gsMeanPosErr_m(iFamily));

        oracleMeanImprovementPct(iFamily) = percentImprovement_step09b5( ...
            localMeanPosErr_m(iFamily), oracleMeanPosErr_m(iFamily));

        gsRmsImprovementPct(iFamily) = percentImprovement_step09b5( ...
            localRmsPosErr_m(iFamily), gsRmsPosErr_m(iFamily));

        oracleRmsImprovementPct(iFamily) = percentImprovement_step09b5( ...
            localRmsPosErr_m(iFamily), oracleRmsPosErr_m(iFamily));

        gsP95ImprovementPct(iFamily) = percentImprovement_step09b5( ...
            localP95PosErr_m(iFamily), gsP95PosErr_m(iFamily));

        oracleP95ImprovementPct(iFamily) = percentImprovement_step09b5( ...
            localP95PosErr_m(iFamily), oracleP95PosErr_m(iFamily));

    end

    longMetricTable = vertcat(metricTableCell{:});

    compactComparisonTable = table( ...
        familyName, ...
        repmat(halfAngleDeg, nFamily, 1), ...
        repmat(seed, nFamily, 1), ...
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
        localP95PosErr_m, ...
        gsP95PosErr_m, ...
        oracleP95PosErr_m, ...
        gsP95ImprovementPct, ...
        oracleP95ImprovementPct, ...
        localMeanNIS, ...
        gsMeanNIS, ...
        oracleMeanNIS, ...
        gsFinalTotalUploads, ...
        'VariableNames', { ...
            'residualFamily', ...
            'halfAngleDeg', ...
            'seed', ...
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
            'localP95PosErr_m', ...
            'gsP95PosErr_m', ...
            'oracleP95PosErr_m', ...
            'gsP95ImprovementPct', ...
            'oracleP95ImprovementPct', ...
            'localMeanNIS', ...
            'gsMeanNIS', ...
            'oracleMeanNIS', ...
            'gsFinalTotalUploads'});

    fprintf("\n============================================================\n");
    fprintf("Step 09-B.5 compact residual-family comparison\n");
    fprintf("============================================================\n");
    disp(compactComparisonTable);

    out = struct();
    out.halfAngleDeg = halfAngleDeg;
    out.seed = seed;
    out.residualFamilyList = residualFamilyList;
    out.outCaseCell = outCaseCell;
    out.longMetricTable = longMetricTable;
    out.compactComparisonTable = compactComparisonTable;

    fprintf("\n============================================================\n");
    fprintf("Step 09-B.5 residual-family comparison complete.\n");
    fprintf("============================================================\n\n");

end

function pct = percentImprovement_step09b5(baselineValue, testValue)
%PERCENTIMPROVEMENT_STEP09B5 Positive means testValue has lower error.

    if isfinite(baselineValue) && baselineValue > 0 && isfinite(testValue)
        pct = 100 * (baselineValue - testValue) / baselineValue;
    else
        pct = NaN;
    end

end