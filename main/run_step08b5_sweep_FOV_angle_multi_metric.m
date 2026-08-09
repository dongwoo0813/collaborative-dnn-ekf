function out = run_step08b5_sweep_FOV_angle_multi_metric(halfAngleListDeg, residualFamily, seed)
%{
File:
    main/run_step08b5_sweep_FOV_angle_multi_metric.m

Purpose:
    Step 08-B.5 multi-metric FOV half-angle comparison sweep.

    This script repeatedly calls the validated Step 08-B.3/08-B.4b
    Local-vs-GS-vs-Oracle FOV comparison and collects the multi-metric
    estimator tables into one sweep-level result.

Default sweep:
    halfAngleListDeg = [20 40 90]

Cases inside each FOV angle:
    1. Local DNN + FOV
    2. GS composite + FOV
    3. Oracle + FOV

Outputs:
    out.multiMetricTable
        Long-format table with all case metrics for every FOV angle.

    out.comparisonSummaryTable
        Compact angle-by-angle comparison of Local, GS, and Oracle.

Usage:
    out08b5 = run_step08b5_sweep_FOV_angle_multi_metric();

    or

    out08b5 = run_step08b5_sweep_FOV_angle_multi_metric([20 40 90]);
%}

    if nargin < 1 || isempty(halfAngleListDeg)
        halfAngleListDeg = [20 40 90];
    end

    if nargin < 2 || isempty(residualFamily)
        residualFamily = "simple_branchwise";
    end
    
    if nargin < 3 || isempty(seed)
        seed = 100;
    end
    
    residualFamily = string(residualFamily);

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 08-B.5: Multi-metric FOV angle comparison sweep\n");
    fprintf("============================================================\n\n");

    addpath(genpath(pwd));
    rehash;

    halfAngleListDeg = halfAngleListDeg(:);
    nAngle = numel(halfAngleListDeg);

    caseOutputCell = cell(nAngle, 1);
    metricTableCell = cell(nAngle, 1);

    availabilityRatePercent = NaN(nAngle, 1);

    localMeanPosErr_m  = NaN(nAngle, 1);
    gsMeanPosErr_m     = NaN(nAngle, 1);
    oracleMeanPosErr_m = NaN(nAngle, 1);

    localRmsPosErr_m  = NaN(nAngle, 1);
    gsRmsPosErr_m     = NaN(nAngle, 1);
    oracleRmsPosErr_m = NaN(nAngle, 1);

    localP95PosErr_m  = NaN(nAngle, 1);
    gsP95PosErr_m     = NaN(nAngle, 1);
    oracleP95PosErr_m = NaN(nAngle, 1);

    localDropoutMeanPosErr_m  = NaN(nAngle, 1);
    gsDropoutMeanPosErr_m     = NaN(nAngle, 1);
    oracleDropoutMeanPosErr_m = NaN(nAngle, 1);

    gsMeanImprovementPct = NaN(nAngle, 1);
    oracleMeanImprovementPct = NaN(nAngle, 1);
    gsClosedLocalOracleGapPct = NaN(nAngle, 1);

    gsFinalTotalUploads = NaN(nAngle, 1);
    gsLoggedUploadDecisions = NaN(nAngle, 1);

    for ia = 1:nAngle

        halfAngleDeg = halfAngleListDeg(ia);

        fprintf("\n------------------------------------------------------------\n");
        fprintf("Step 08-B.5 sweep case %d / %d: halfAngle = %.3f deg\n", ...
            ia, nAngle, halfAngleDeg);
        fprintf("------------------------------------------------------------\n");

        % Reuse the validated Step 08-B.3/08-B.4b comparison pipeline.
        outCase = run_step08b3_compare_FOV_Local_GS_Oracle( ...
            halfAngleDeg, false, seed, residualFamily);

        caseOutputCell{ia} = outCase;

        T = outCase.multiMetricSummaryTable;

        % Store the FOV angle directly in the long-format metric table.
        halfAngleCol = repmat(halfAngleDeg, height(T), 1);
        T = addvars(T, halfAngleCol, ...
            'Before', 1, ...
            'NewVariableNames', 'halfAngleDeg');

        metricTableCell{ia} = T;

        rowLocal  = findCaseRow_step08b5(T, "Local DNN + FOV");
        rowGS     = findCaseRow_step08b5(T, "GS composite + FOV");
        rowOracle = findCaseRow_step08b5(T, "Oracle + FOV");

        availabilityRatePercent(ia) = readTableValue_step08b5(T, rowGS, "availabilityRatePercent");

        localMeanPosErr_m(ia)  = readTableValue_step08b5(T, rowLocal,  "meanPosErr_m");
        gsMeanPosErr_m(ia)     = readTableValue_step08b5(T, rowGS,     "meanPosErr_m");
        oracleMeanPosErr_m(ia) = readTableValue_step08b5(T, rowOracle, "meanPosErr_m");

        localRmsPosErr_m(ia)  = readTableValue_step08b5(T, rowLocal,  "rmsPosErr_m");
        gsRmsPosErr_m(ia)     = readTableValue_step08b5(T, rowGS,     "rmsPosErr_m");
        oracleRmsPosErr_m(ia) = readTableValue_step08b5(T, rowOracle, "rmsPosErr_m");

        localP95PosErr_m(ia)  = readTableValue_step08b5(T, rowLocal,  "p95PosErr_m");
        gsP95PosErr_m(ia)     = readTableValue_step08b5(T, rowGS,     "p95PosErr_m");
        oracleP95PosErr_m(ia) = readTableValue_step08b5(T, rowOracle, "p95PosErr_m");

        localDropoutMeanPosErr_m(ia)  = readTableValue_step08b5(T, rowLocal,  "meanPosErr_dropout_m");
        gsDropoutMeanPosErr_m(ia)     = readTableValue_step08b5(T, rowGS,     "meanPosErr_dropout_m");
        oracleDropoutMeanPosErr_m(ia) = readTableValue_step08b5(T, rowOracle, "meanPosErr_dropout_m");

        gsLoggedUploadDecisions(ia) = readTableValue_step08b5(T, rowGS, "gsLoggedUploadDecisions");
        gsFinalTotalUploads(ia)     = readTableValue_step08b5(T, rowGS, "gsFinalTotalUploads");

        if localMeanPosErr_m(ia) > 0
            gsMeanImprovementPct(ia) = ...
                100 * (localMeanPosErr_m(ia) - gsMeanPosErr_m(ia)) / localMeanPosErr_m(ia);

            oracleMeanImprovementPct(ia) = ...
                100 * (localMeanPosErr_m(ia) - oracleMeanPosErr_m(ia)) / localMeanPosErr_m(ia);
        end

        % Gap closure is meaningful only when Oracle improves over Local.
        localOracleGap = localMeanPosErr_m(ia) - oracleMeanPosErr_m(ia);

        if localOracleGap > 0
            gsClosedLocalOracleGapPct(ia) = ...
                100 * (localMeanPosErr_m(ia) - gsMeanPosErr_m(ia)) / localOracleGap;
        end

    end

    multiMetricTable = vertcat(metricTableCell{:});

    comparisonSummaryTable = table( ...
        halfAngleListDeg, ...
        availabilityRatePercent, ...
        localMeanPosErr_m, ...
        gsMeanPosErr_m, ...
        oracleMeanPosErr_m, ...
        gsMeanImprovementPct, ...
        oracleMeanImprovementPct, ...
        gsClosedLocalOracleGapPct, ...
        localRmsPosErr_m, ...
        gsRmsPosErr_m, ...
        oracleRmsPosErr_m, ...
        localP95PosErr_m, ...
        gsP95PosErr_m, ...
        oracleP95PosErr_m, ...
        localDropoutMeanPosErr_m, ...
        gsDropoutMeanPosErr_m, ...
        oracleDropoutMeanPosErr_m, ...
        gsLoggedUploadDecisions, ...
        gsFinalTotalUploads, ...
        'VariableNames', { ...
            'halfAngleDeg', ...
            'availabilityRatePercent', ...
            'localMeanPosErr_m', ...
            'gsMeanPosErr_m', ...
            'oracleMeanPosErr_m', ...
            'gsMeanImprovementPct', ...
            'oracleMeanImprovementPct', ...
            'gsClosedLocalOracleGapPct', ...
            'localRmsPosErr_m', ...
            'gsRmsPosErr_m', ...
            'oracleRmsPosErr_m', ...
            'localP95PosErr_m', ...
            'gsP95PosErr_m', ...
            'oracleP95PosErr_m', ...
            'localDropoutMeanPosErr_m', ...
            'gsDropoutMeanPosErr_m', ...
            'oracleDropoutMeanPosErr_m', ...
            'gsLoggedUploadDecisions', ...
            'gsFinalTotalUploads'});

    fprintf("\n============================================================\n");
    fprintf("Step 08-B.5 compact comparison summary\n");
    fprintf("============================================================\n");
    disp(comparisonSummaryTable);

    fprintf("\nStep 08-B.5 long-format multi-metric table\n");
    disp(multiMetricTable);

    out = struct();
    out.halfAngleListDeg = halfAngleListDeg;
    out.caseOutputCell = caseOutputCell;
    out.multiMetricTable = multiMetricTable;
    out.comparisonSummaryTable = comparisonSummaryTable;
    out.residualFamily = residualFamily;
    out.seed = seed;

    fprintf("\n============================================================\n");
    fprintf("Step 08-B.5 multi-metric FOV angle sweep complete.\n");
    fprintf("============================================================\n\n");

end

function row = findCaseRow_step08b5(T, caseLabel)
%FINDCASEROW_STEP08B5 Find the row associated with a named estimator case.

    caseName = string(T.caseName);
    row = find(caseName == caseLabel, 1, 'first');

    if isempty(row)
        error("step08b5:MissingCaseRow", ...
            "Could not find caseName = %s in the multi-metric table.", caseLabel);
    end

end

function value = readTableValue_step08b5(T, row, varName)
%READTABLEVALUE_STEP08B5 Read a scalar table entry with a NaN fallback.
%
% The NaN fallback keeps the sweep robust if a later evaluator version omits
% a diagnostic that is not available for every estimator case.

    if ismember(varName, string(T.Properties.VariableNames))
        value = T.(varName)(row);
    else
        value = NaN;
    end

end