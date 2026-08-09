function sweepOut = run_step09H5b_sweep_Local_MLP_thetaStd( ...
    residualFamily, seed, dtOverride, thetaStdList)
%{
File:
    main/run_step09H5b_sweep_Local_MLP_thetaStd.m

Purpose:
    Step 09-H.5b Local MLP theta-mobility sweep.

    Sweep the MLP parameter covariance scale while keeping:
        cfg.dnn.residualInjectionGain = 1.0

Why this comes after Step 09-H.5a:
    The beta sweep showed that full DNN residual injection is best.
    Therefore, the next likely bottleneck is not over-injection, but
    insufficient MLP parameter mobility.

Swept fields:
    cfg.dnn.Ptheta0      = thetaStd^2
    cfg.dnn.thetaSigmaSS = thetaStd

Notes:
    For thetaDynamics = "FOGM", qTheta is not used.
    The FOGM process noise is determined by thetaSigmaSS and thetaTau.

Recommended first run:
    thetaStdList = [2e-5 5e-5 1e-4 2e-4]
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

    if nargin < 4 || isempty(thetaStdList)
        thetaStdList = [2e-5 5e-5 1e-4 2e-4];
    end

    residualFamily = string(residualFamily);
    thetaStdList = thetaStdList(:);

    addpath(genpath(pwd));
    rehash;

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.5b: Local MLP theta mobility sweep\n");
    fprintf("============================================================\n");
    fprintf("Residual family = %s\n", residualFamily);
    fprintf("Seed            = %d\n", seed);
    fprintf("dt              = %.6g\n", dtOverride);
    fprintf("thetaStdList    = [%s]\n\n", sprintf("%.3g ", thetaStdList));

    nCase = numel(thetaStdList);
    runs = cell(nCase, 1);

    fixedMean = NaN(nCase, 1);
    mlpMean = NaN(nCase, 1);
    oracleMean = NaN(nCase, 1);

    mlpRms = NaN(nCase, 1);
    mlpMedian = NaN(nCase, 1);
    mlpP95 = NaN(nCase, 1);
    mlpFinal = NaN(nCase, 1);
    mlpMeanNIS = NaN(nCase, 1);

    mlpMeanResidualErr = NaN(nCase, 1);
    mlpRmsResidualErr = NaN(nCase, 1);
    mlpMeanSameInputResidualErr = NaN(nCase, 1);
    mlpRmsSameInputResidualErr = NaN(nCase, 1);

    mlpMeanGammaTheta = NaN(nCase, 1);
    mlpFinalGammaTheta = NaN(nCase, 1);
    mlpMeanCmRatio = NaN(nCase, 1);

    for k = 1:nCase

        thetaStd = thetaStdList(k);

        fprintf("Running MLP thetaStd = %.6g\n", thetaStd);

        mlpOverride = struct();

        % Keep beta fixed at the best value from Step 09-H.5a.
        mlpOverride.dnn.residualInjectionGain = 1.0;

        % Initial uncertainty of MLP parameters.
        % Larger Ptheta0 lets the EKF accept larger theta corrections early.
        mlpOverride.dnn.Ptheta0 = thetaStd^2;

        % FOGM steady-state parameter standard deviation.
        % Larger thetaSigmaSS allows more ongoing parameter mobility.
        mlpOverride.dnn.thetaSigmaSS = thetaStd;

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

        % Row 2 is Local MLP in the Step 09-H.4 residual summary.
        mlpMeanResidualErr(k) = ...
            outK.residualSummaryTable.meanOperationalResidualErr(2);

        mlpRmsResidualErr(k) = ...
            outK.residualSummaryTable.rmsOperationalResidualErr(2);

        mlpMeanSameInputResidualErr(k) = ...
            outK.residualSummaryTable.meanSameInputResidualErr(2);

        mlpRmsSameInputResidualErr(k) = ...
            outK.residualSummaryTable.rmsSameInputResidualErr(2);

        [mlpMeanGammaTheta(k), mlpFinalGammaTheta(k), mlpMeanCmRatio(k)] = ...
            thetaAdaptationStatsStep09H5b(outK.resMLP);

    end

    mlpMeanImprovementVsFixedPct = 100 * (fixedMean - mlpMean) ./ fixedMean;
    oracleMeanImprovementVsFixedPct = 100 * (fixedMean - oracleMean) ./ fixedMean;

    thetaTable = table( ...
        thetaStdList, ...
        thetaStdList.^2, ...
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
        mlpMeanGammaTheta, ...
        mlpFinalGammaTheta, ...
        mlpMeanCmRatio, ...
        'VariableNames', { ...
            'thetaStd', ...
            'Ptheta0', ...
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
            'mlpRmsSameInputResidualErr', ...
            'mlpMeanGammaTheta', ...
            'mlpFinalGammaTheta', ...
            'mlpMeanCmRatio'});

    [~, bestIdxMean] = min(mlpMean);
    [~, bestIdxResidual] = min(mlpMeanSameInputResidualErr);

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.5b theta mobility sweep summary\n");
    fprintf("============================================================\n");
    disp(thetaTable);

    fprintf("Best thetaStd by MLP mean position error       = %.6g\n", ...
        thetaStdList(bestIdxMean));
    fprintf("Best thetaStd by MLP same-input residual error = %.6g\n", ...
        thetaStdList(bestIdxResidual));

    sweepOut = struct();
    sweepOut.residualFamily = residualFamily;
    sweepOut.seed = seed;
    sweepOut.dt = dtOverride;
    sweepOut.thetaStdList = thetaStdList;
    sweepOut.thetaTable = thetaTable;
    sweepOut.runs = runs;
    sweepOut.bestThetaStdByMeanPosition = thetaStdList(bestIdxMean);
    sweepOut.bestThetaStdBySameInputResidual = thetaStdList(bestIdxResidual);

end

function [meanGamma, finalGamma, meanRatio] = thetaAdaptationStatsStep09H5b(results)
%THETAADAPTATIONSTATSSTEP09H5B Summarize covariance-matching logs.
%
% gammaTheta:
%   adaptive multiplier on Qtheta_base.
%
% cmRatio:
%   covariance matching ratio trace(S_empirical)/trace(S_model).
%
% These are diagnostic only. They help decide whether the MLP parameter
% process noise is being actively inflated or clamped.

    meanGamma = NaN;
    finalGamma = NaN;
    meanRatio = NaN;

    if isfield(results, "gammaTheta")
        gammaAll = results.gammaTheta(:);
        meanGamma = mean(gammaAll, "omitnan");

        gammaFinal = results.gammaTheta(end, :);
        finalGamma = mean(gammaFinal, "omitnan");
    end

    if isfield(results, "cmRatio")
        ratioAll = results.cmRatio(:);
        meanRatio = mean(ratioAll, "omitnan");
    end

end