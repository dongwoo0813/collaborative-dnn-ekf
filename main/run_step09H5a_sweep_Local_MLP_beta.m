function sweepOut = run_step09H5a_sweep_Local_MLP_beta( ...
    residualFamily, seed, dtOverride, betaList)
%{
File:
    main/run_step09H5a_sweep_Local_MLP_beta.m

Purpose:
    Step 09-H.5a Local MLP tuning diagnostic.

    Sweep only cfg.dnn.residualInjectionGain for the MLP Local case.

Why this is the first MLP tuning sweep:
    Step 09-H.4 showed that mlp_general runs to completion, but its learned
    residual is slightly worse than fixed_feature_lip on the first benchmark.

    Before changing Ptheta0, thetaSigmaSS, architecture, or covariance
    matching, this script checks whether the MLP residual is being injected
    too aggressively into the physical prediction model.

Cases:
    Each beta run calls:

        run_step09H4_compare_Local_fixed_MLP_Oracle(..., mlpOverride)

    where only the MLP case receives the beta override.

Outputs:
    sweepOut.betaTable
        Main comparison table indexed by MLP residualInjectionGain.

    sweepOut.runs
        Cell array of full Step 09-H.4 outputs for each beta.
%}

    if nargin < 1 || isempty(residualFamily)
        residualFamily = "feedback_sat_disturbance";
    end

    if nargin < 2 || isempty(seed)
        seed = 101;
    end

    if nargin < 3 || isempty(dtOverride)
        dtOverride = 0.5;
    end

    if nargin < 4 || isempty(betaList)
        betaList = [0.25 0.5 0.75 1.0];
    end

    residualFamily = string(residualFamily);
    betaList = betaList(:);

    addpath(genpath(pwd));
    rehash;

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.5a: Local MLP beta sweep\n");
    fprintf("============================================================\n");
    fprintf("Residual family = %s\n", residualFamily);
    fprintf("Seed            = %d\n", seed);
    fprintf("dt              = %.6g\n", dtOverride);
    fprintf("betaList        = [%s]\n\n", sprintf("%.3g ", betaList));

    nCase = numel(betaList);

    runs = cell(nCase, 1);

    mlpMean = NaN(nCase, 1);
    mlpRms = NaN(nCase, 1);
    mlpMedian = NaN(nCase, 1);
    mlpP95 = NaN(nCase, 1);
    mlpFinal = NaN(nCase, 1);
    mlpMeanNIS = NaN(nCase, 1);

    fixedMean = NaN(nCase, 1);
    oracleMean = NaN(nCase, 1);

    mlpMeanResidualErr = NaN(nCase, 1);
    mlpRmsResidualErr = NaN(nCase, 1);
    mlpMeanSameInputResidualErr = NaN(nCase, 1);
    mlpRmsSameInputResidualErr = NaN(nCase, 1);

    for k = 1:nCase

        beta = betaList(k);

        fprintf("Running MLP beta = %.6g\n", beta);

        mlpOverride = struct();

        % Only the MLP Local case is changed by this override.
        % Fixed-feature and Oracle remain the same reference cases.
        mlpOverride.dnn.residualInjectionGain = beta;

        outK = run_step09H4_compare_Local_fixed_MLP_Oracle( ...
            residualFamily, seed, dtOverride, false, mlpOverride);

        runs{k} = outK;

        fixedMean(k) = outK.metricsFixed.meanPosErr;
        oracleMean(k) = outK.metricsOracle.meanPosErr;

        mlpMean(k) = outK.metricsMLP.meanPosErr;
        mlpRms(k) = outK.metricsMLP.rmsPosErr;
        mlpMedian(k) = outK.metricsMLP.medianPosErr;
        mlpP95(k) = outK.metricsMLP.p95PosErr;
        mlpFinal(k) = outK.metricsMLP.finalMeanPosErr;
        mlpMeanNIS(k) = outK.metricsMLP.meanNIS;

        % Row 2 of residualSummaryTable is Local MLP.
        mlpMeanResidualErr(k) = ...
            outK.residualSummaryTable.meanOperationalResidualErr(2);

        mlpRmsResidualErr(k) = ...
            outK.residualSummaryTable.rmsOperationalResidualErr(2);

        mlpMeanSameInputResidualErr(k) = ...
            outK.residualSummaryTable.meanSameInputResidualErr(2);

        mlpRmsSameInputResidualErr(k) = ...
            outK.residualSummaryTable.rmsSameInputResidualErr(2);

    end

    mlpMeanImprovementVsFixedPct = 100 * (fixedMean - mlpMean) ./ fixedMean;
    oracleMeanImprovementVsFixedPct = 100 * (fixedMean - oracleMean) ./ fixedMean;

    betaTable = table( ...
        betaList, ...
        fixedMean, ...
        mlpMean, ...
        oracleMean, ...
        mlpMeanImprovementVsFixedPct, ...
        oracleMeanImprovementVsFixedPct, ...
        mlpRms, ...
        mlpMedian, ...
        mlpP95, ...
        mlpFinal, ...
        mlpMeanNIS, ...
        mlpMeanResidualErr, ...
        mlpRmsResidualErr, ...
        mlpMeanSameInputResidualErr, ...
        mlpRmsSameInputResidualErr, ...
        'VariableNames', { ...
            'betaDNN_MLP', ...
            'fixedMeanPosErr_m', ...
            'mlpMeanPosErr_m', ...
            'oracleMeanPosErr_m', ...
            'mlpMeanImprovementPct_vsFixed', ...
            'oracleMeanImprovementPct_vsFixed', ...
            'mlpRmsPosErr_m', ...
            'mlpMedianPosErr_m', ...
            'mlpP95PosErr_m', ...
            'mlpFinalMeanPosErr_m', ...
            'mlpMeanNIS', ...
            'mlpMeanOperationalResidualErr', ...
            'mlpRmsOperationalResidualErr', ...
            'mlpMeanSameInputResidualErr', ...
            'mlpRmsSameInputResidualErr'});

    [~, bestIdxMean] = min(mlpMean);
    [~, bestIdxResidual] = min(mlpMeanSameInputResidualErr);

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.5a beta sweep summary\n");
    fprintf("============================================================\n");
    disp(betaTable);

    fprintf("Best beta by MLP mean position error      = %.6g\n", ...
        betaList(bestIdxMean));
    fprintf("Best beta by MLP same-input residual error = %.6g\n", ...
        betaList(bestIdxResidual));

    sweepOut = struct();
    sweepOut.residualFamily = residualFamily;
    sweepOut.seed = seed;
    sweepOut.dt = dtOverride;
    sweepOut.betaList = betaList;
    sweepOut.betaTable = betaTable;
    sweepOut.runs = runs;
    sweepOut.bestBetaByMeanPosition = betaList(bestIdxMean);
    sweepOut.bestBetaBySameInputResidual = betaList(bestIdxResidual);

end