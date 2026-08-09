function out = run_step08b7_residual_alignment_diagnostics(halfAngleDeg, seed, verbose, out08b3)
%{
File:
    main/run_step08b7_residual_alignment_diagnostics.m

Purpose:
    Step 08-B.7 residual alignment diagnostics.

    This script compares the logged DNN residual estimates against the
    logged true unknown residual acceleration for:

        1. Local DNN + FOV
        2. GS composite + FOV
        3. Oracle + FOV

    The goal is to determine whether an estimator is improving because its
    learned residual direction/magnitude is actually closer to the unknown
    truth residual, or because of other filtering effects.

Inputs:
    halfAngleDeg
        FOV half-angle in degrees.

    seed
        Random seed passed into the Step 08-B.3/08-B.4b comparison.

    verbose
        true  -> print tables and make plots.
        false -> compute quietly.

    out08b3
        Optional existing output from:
            run_step08b3_compare_FOV_Local_GS_Oracle(...)
        If omitted, this script runs Step 08-B.3/08-B.4b internally.

Outputs:
    out
        Struct containing the original Step 08-B.3/08-B.4b output,
        residual alignment metrics, summary tables, and comparison table.

Main diagnostics:
    residual error norm:
        || d_hat - d_true ||

    relative residual error:
        || d_hat - d_true || / || d_true ||

    cosine alignment:
        <d_hat, d_true> / (||d_hat|| ||d_true||)

    projection ratio:
        <d_hat, d_true> / ||d_true||^2

Notes:
    - This is post-processing only.
    - It does not change the estimator.
    - It uses already logged fields:
          results.trueResidual
          results.dnnResidual
          results.measAvail
%}

    if nargin < 1 || isempty(halfAngleDeg)
        halfAngleDeg = 90.0;
    end

    if nargin < 2 || isempty(seed)
        seed = 100;
    end

    if nargin < 3 || isempty(verbose)
        verbose = true;
    end

    if nargin < 4
        out08b3 = [];
    end

    addpath(genpath(pwd));
    rehash;

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 08-B.7: Residual alignment diagnostics\n");
        fprintf("============================================================\n");
        fprintf("FOV half-angle = %.3f deg\n", halfAngleDeg);
        fprintf("Random seed    = %d\n\n", seed);
    end

    % ---------------------------------------------------------------------
    % Reuse existing Step 08-B.3/08-B.4b result if provided.
    % ---------------------------------------------------------------------
    if isempty(out08b3)
        out08b3 = run_step08b3_compare_FOV_Local_GS_Oracle( ...
            halfAngleDeg, false, seed);
    end

    % ---------------------------------------------------------------------
    % Evaluate residual alignment for each estimator case.
    % ---------------------------------------------------------------------
    metricsLocal = evaluateResidualAlignmentCase_step08b7( ...
        out08b3.resLocal, out08b3.cfgLocal, "Local DNN + FOV");

    metricsGS = evaluateResidualAlignmentCase_step08b7( ...
        out08b3.resGS, out08b3.cfgGS, "GS composite + FOV");

    metricsOracle = evaluateResidualAlignmentCase_step08b7( ...
        out08b3.resOracle, out08b3.cfgOracle, "Oracle + FOV");

    residualAlignmentSummaryTable = [
        metricsLocal.scalarSummary
        metricsGS.scalarSummary
        metricsOracle.scalarSummary
    ];

    comparisonTable = makeResidualComparisonTable_step08b7( ...
        halfAngleDeg, seed, metricsLocal, metricsGS, metricsOracle);

    % ---------------------------------------------------------------------
    % Optional output
    % ---------------------------------------------------------------------
    if verbose
        fprintf("Residual alignment summary:\n");
        disp(residualAlignmentSummaryTable);

        fprintf("Residual alignment comparison:\n");
        disp(comparisonTable);

        plotResidualAlignment_step08b7( ...
            out08b3.resLocal, out08b3.resGS, out08b3.resOracle, ...
            metricsLocal, metricsGS, metricsOracle, halfAngleDeg, seed);

        fprintf("\n============================================================\n");
        fprintf("Step 08-B.7 residual alignment diagnostics complete.\n");
        fprintf("============================================================\n\n");
    end

    % ---------------------------------------------------------------------
    % Output
    % ---------------------------------------------------------------------
    out = struct();

    out.halfAngleDeg = halfAngleDeg;
    out.seed = seed;

    out.out08b3 = out08b3;

    out.metricsLocal = metricsLocal;
    out.metricsGS = metricsGS;
    out.metricsOracle = metricsOracle;

    out.residualAlignmentSummaryTable = residualAlignmentSummaryTable;
    out.comparisonTable = comparisonTable;

end

function metrics = evaluateResidualAlignmentCase_step08b7(res, cfg, caseName)
%EVALUATERESIDUALALIGNMENTCASE_STEP08B7 Evaluate one estimator case.
%
% This function compares dnnResidual against trueResidual over all watchers
% and separates available-measurement rows from dropout rows.

    caseName = string(caseName);

    if ~isfield(res, "trueResidual")
        error("step08b7:MissingTrueResidual", ...
            "Result structure is missing res.trueResidual.");
    end

    if ~isfield(res, "dnnResidual")
        error("step08b7:MissingDNNResidual", ...
            "Result structure is missing res.dnnResidual.");
    end

    [arrays, activeRows] = computeResidualAlignmentArrays_step08b7(res, cfg);

    metrics = struct();

    metrics.caseName = caseName;
    metrics.activeRows = activeRows;

    metrics.residualErrNorm = arrays.residualErrNorm;
    metrics.relativeResidualErr = arrays.relativeResidualErr;
    metrics.cosineAlignment = arrays.cosineAlignment;
    metrics.projectionRatio = arrays.projectionRatio;
    metrics.trueResidualNorm = arrays.trueResidualNorm;
    metrics.dnnResidualNorm = arrays.dnnResidualNorm;

    allMask = true(numel(activeRows), size(arrays.residualErrNorm, 2));

    if isfield(res, "measAvail")
        measAvail = logical(res.measAvail);
        measAvail = alignRowsCols_step08b7(measAvail, size(arrays.residualErrNorm, 1), ...
            size(arrays.residualErrNorm, 2));

        availMask = measAvail(activeRows,:);
        dropoutMask = ~availMask;
    else
        availMask = false(size(allMask));
        dropoutMask = false(size(allMask));
    end

    % Overall residual error metrics.
    metrics.meanResErrNorm = maskedMean_step08b7(arrays.residualErrNorm, activeRows, allMask);
    metrics.rmsResErrNorm = maskedRMS_step08b7(arrays.residualErrNorm, activeRows, allMask);
    metrics.medianResErrNorm = maskedMedian_step08b7(arrays.residualErrNorm, activeRows, allMask);
    metrics.p95ResErrNorm = maskedPercentile_step08b7(arrays.residualErrNorm, activeRows, allMask, 95);
    metrics.maxResErrNorm = maskedMax_step08b7(arrays.residualErrNorm, activeRows, allMask);

    % Relative error and alignment metrics.
    metrics.meanRelativeResErr = maskedMean_step08b7(arrays.relativeResidualErr, activeRows, allMask);
    metrics.medianRelativeResErr = maskedMedian_step08b7(arrays.relativeResidualErr, activeRows, allMask);

    metrics.meanCosineAlignment = maskedMean_step08b7(arrays.cosineAlignment, activeRows, allMask);
    metrics.medianCosineAlignment = maskedMedian_step08b7(arrays.cosineAlignment, activeRows, allMask);
    metrics.p10CosineAlignment = maskedPercentile_step08b7(arrays.cosineAlignment, activeRows, allMask, 10);

    metrics.meanProjectionRatio = maskedMean_step08b7(arrays.projectionRatio, activeRows, allMask);
    metrics.medianProjectionRatio = maskedMedian_step08b7(arrays.projectionRatio, activeRows, allMask);

    % Available/dropout split.
    metrics.meanResErrNorm_available = maskedMean_step08b7( ...
        arrays.residualErrNorm, activeRows, availMask);

    metrics.meanResErrNorm_dropout = maskedMean_step08b7( ...
        arrays.residualErrNorm, activeRows, dropoutMask);

    metrics.meanCosineAlignment_available = maskedMean_step08b7( ...
        arrays.cosineAlignment, activeRows, availMask);

    metrics.meanCosineAlignment_dropout = maskedMean_step08b7( ...
        arrays.cosineAlignment, activeRows, dropoutMask);

    metrics.meanRelativeResErr_available = maskedMean_step08b7( ...
        arrays.relativeResidualErr, activeRows, availMask);

    metrics.meanRelativeResErr_dropout = maskedMean_step08b7( ...
        arrays.relativeResidualErr, activeRows, dropoutMask);

    metrics.availableCount = nnz(availMask);
    metrics.dropoutCount = nnz(dropoutMask);
    metrics.availabilityRatePercent = 100 * metrics.availableCount / ...
        max(metrics.availableCount + metrics.dropoutCount, 1);

    metrics.scalarSummary = table( ...
        caseName, ...
        metrics.meanResErrNorm, ...
        metrics.rmsResErrNorm, ...
        metrics.medianResErrNorm, ...
        metrics.p95ResErrNorm, ...
        metrics.maxResErrNorm, ...
        metrics.meanRelativeResErr, ...
        metrics.medianRelativeResErr, ...
        metrics.meanCosineAlignment, ...
        metrics.medianCosineAlignment, ...
        metrics.p10CosineAlignment, ...
        metrics.meanProjectionRatio, ...
        metrics.meanResErrNorm_available, ...
        metrics.meanResErrNorm_dropout, ...
        metrics.meanCosineAlignment_available, ...
        metrics.meanCosineAlignment_dropout, ...
        metrics.availabilityRatePercent, ...
        'VariableNames', { ...
            'caseName', ...
            'meanResErrNorm_mps2', ...
            'rmsResErrNorm_mps2', ...
            'medianResErrNorm_mps2', ...
            'p95ResErrNorm_mps2', ...
            'maxResErrNorm_mps2', ...
            'meanRelativeResErr', ...
            'medianRelativeResErr', ...
            'meanCosineAlignment', ...
            'medianCosineAlignment', ...
            'p10CosineAlignment', ...
            'meanProjectionRatio', ...
            'meanResErrNorm_available_mps2', ...
            'meanResErrNorm_dropout_mps2', ...
            'meanCosineAlignment_available', ...
            'meanCosineAlignment_dropout', ...
            'availabilityRatePercent'});

end

function [arrays, activeRows] = computeResidualAlignmentArrays_step08b7(res, cfg)
%COMPUTERESIDUALALIGNMENTARRAYS_STEP08B7 Build residual alignment arrays.
%
% Array convention:
%     each diagnostic is N x Nw.

    dim = cfg.dim;
    
    aTrue = res.trueResidual;
    
    % For Local/GS cases, dnnResidual is the learned/composite residual log.
    % For Oracle cases, the estimator uses the true residual source, but some
    % simulator logs may leave dnnResidual as zero or non-oracle diagnostic data.
    % In that case, use trueResidual as the residual estimate for this diagnostic.
    
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "predictionResidualSource") && ...
            string(cfg.dnn.predictionResidualSource) == "oracle"
        dHat = aTrue;
    else
        dHat = res.dnnResidual;
    end
    
    if ismatrix(dHat)
        % Oracle residual logs may be dim x N instead of dim x N x Nw.
        % Replicate across watchers so the diagnostic dimensions match Local/GS.
        if isfield(cfg, "Nw")
            NwCfg = cfg.Nw;
        else
            NwCfg = 1;
        end
    
        dHat = reshape(dHat, size(dHat,1), size(dHat,2), 1);
        dHat = repmat(dHat, 1, 1, NwCfg);
    end

    N = min(size(aTrue,2), size(dHat,2));
    Nw = size(dHat,3);

    aTrue = aTrue(1:dim, 1:N);
    dHat = dHat(1:dim, 1:N, 1:Nw);

    residualErrNorm = NaN(N, Nw);
    relativeResidualErr = NaN(N, Nw);
    cosineAlignment = NaN(N, Nw);
    projectionRatio = NaN(N, Nw);
    trueResidualNorm = NaN(N, Nw);
    dnnResidualNorm = NaN(N, Nw);

    tiny = 1e-14;

    for iw = 1:Nw

        dHat_i = dHat(:,:,iw);
        err_i = dHat_i - aTrue;

        trueNorm_i = sqrt(sum(aTrue.^2, 1)).';
        dHatNorm_i = sqrt(sum(dHat_i.^2, 1)).';
        errNorm_i = sqrt(sum(err_i.^2, 1)).';

        dot_i = sum(dHat_i .* aTrue, 1).';

        denomCos = trueNorm_i .* dHatNorm_i;
        denomProj = trueNorm_i.^2;

        cos_i = dot_i ./ max(denomCos, tiny);
        proj_i = dot_i ./ max(denomProj, tiny);
        relErr_i = errNorm_i ./ max(trueNorm_i, tiny);

        % When the true residual is essentially zero, direction and relative
        % error are not meaningful.
        zeroTruth = trueNorm_i < tiny;

        cos_i(zeroTruth) = NaN;
        proj_i(zeroTruth) = NaN;
        relErr_i(zeroTruth) = NaN;

        residualErrNorm(:,iw) = errNorm_i;
        relativeResidualErr(:,iw) = relErr_i;
        cosineAlignment(:,iw) = cos_i;
        projectionRatio(:,iw) = proj_i;
        trueResidualNorm(:,iw) = trueNorm_i;
        dnnResidualNorm(:,iw) = dHatNorm_i;

    end

    if N >= 3
        activeRows = 2:(N-1);
    else
        activeRows = 1:N;
    end

    arrays = struct();

    arrays.residualErrNorm = residualErrNorm;
    arrays.relativeResidualErr = relativeResidualErr;
    arrays.cosineAlignment = cosineAlignment;
    arrays.projectionRatio = projectionRatio;
    arrays.trueResidualNorm = trueResidualNorm;
    arrays.dnnResidualNorm = dnnResidualNorm;

end

function comparisonTable = makeResidualComparisonTable_step08b7( ...
    halfAngleDeg, seed, metricsLocal, metricsGS, metricsOracle)
%MAKERESIDUALCOMPARISONTABLE_STEP08B7 Compare GS/Oracle against Local.
%
% Positive error-improvement values mean lower residual error than Local.
% Positive cosine gain means better directional alignment than Local.

    gsResErrImprovementPct = percentImprovement_step08b7( ...
        metricsLocal.meanResErrNorm, metricsGS.meanResErrNorm);

    oracleResErrImprovementPct = percentImprovement_step08b7( ...
        metricsLocal.meanResErrNorm, metricsOracle.meanResErrNorm);

    gsRelativeErrImprovementPct = percentImprovement_step08b7( ...
        metricsLocal.meanRelativeResErr, metricsGS.meanRelativeResErr);

    oracleRelativeErrImprovementPct = percentImprovement_step08b7( ...
        metricsLocal.meanRelativeResErr, metricsOracle.meanRelativeResErr);

    gsDropoutResErrImprovementPct = percentImprovement_step08b7( ...
        metricsLocal.meanResErrNorm_dropout, metricsGS.meanResErrNorm_dropout);

    oracleDropoutResErrImprovementPct = percentImprovement_step08b7( ...
        metricsLocal.meanResErrNorm_dropout, metricsOracle.meanResErrNorm_dropout);

    gsMeanCosineGain = metricsGS.meanCosineAlignment - ...
        metricsLocal.meanCosineAlignment;

    oracleMeanCosineGain = metricsOracle.meanCosineAlignment - ...
        metricsLocal.meanCosineAlignment;

    gsDropoutCosineGain = metricsGS.meanCosineAlignment_dropout - ...
        metricsLocal.meanCosineAlignment_dropout;

    oracleDropoutCosineGain = metricsOracle.meanCosineAlignment_dropout - ...
        metricsLocal.meanCosineAlignment_dropout;

    comparisonTable = table( ...
        halfAngleDeg, ...
        seed, ...
        gsResErrImprovementPct, ...
        oracleResErrImprovementPct, ...
        gsRelativeErrImprovementPct, ...
        oracleRelativeErrImprovementPct, ...
        gsDropoutResErrImprovementPct, ...
        oracleDropoutResErrImprovementPct, ...
        gsMeanCosineGain, ...
        oracleMeanCosineGain, ...
        gsDropoutCosineGain, ...
        oracleDropoutCosineGain, ...
        'VariableNames', { ...
            'halfAngleDeg', ...
            'seed', ...
            'gsResErrImprovementPct', ...
            'oracleResErrImprovementPct', ...
            'gsRelativeErrImprovementPct', ...
            'oracleRelativeErrImprovementPct', ...
            'gsDropoutResErrImprovementPct', ...
            'oracleDropoutResErrImprovementPct', ...
            'gsMeanCosineGain', ...
            'oracleMeanCosineGain', ...
            'gsDropoutCosineGain', ...
            'oracleDropoutCosineGain'});

end

function plotResidualAlignment_step08b7( ...
    resLocal, resGS, resOracle, ...
    metricsLocal, metricsGS, metricsOracle, halfAngleDeg, seed)
%PLOTRESIDUALALIGNMENT_STEP08B7 Plot residual alignment diagnostics.

    time = resLocal.time(:);

    N = min([ ...
        numel(time), ...
        size(metricsLocal.residualErrNorm,1), ...
        size(metricsGS.residualErrNorm,1), ...
        size(metricsOracle.residualErrNorm,1)]);

    time = time(1:N);

    meanErrLocal = mean(metricsLocal.residualErrNorm(1:N,:), 2, "omitnan");
    meanErrGS = mean(metricsGS.residualErrNorm(1:N,:), 2, "omitnan");
    meanErrOracle = mean(metricsOracle.residualErrNorm(1:N,:), 2, "omitnan");

    meanCosLocal = mean(metricsLocal.cosineAlignment(1:N,:), 2, "omitnan");
    meanCosGS = mean(metricsGS.cosineAlignment(1:N,:), 2, "omitnan");
    meanCosOracle = mean(metricsOracle.cosineAlignment(1:N,:), 2, "omitnan");

    meanTrueNorm = mean(metricsLocal.trueResidualNorm(1:N,:), 2, "omitnan");
    meanDNNNormLocal = mean(metricsLocal.dnnResidualNorm(1:N,:), 2, "omitnan");
    meanDNNNormGS = mean(metricsGS.dnnResidualNorm(1:N,:), 2, "omitnan");
    meanDNNNormOracle = mean(metricsOracle.dnnResidualNorm(1:N,:), 2, "omitnan");

    figure;
    plot(time, meanErrLocal, "LineWidth", 1.5);
    hold on;
    plot(time, meanErrGS, "LineWidth", 1.5);
    plot(time, meanErrOracle, "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Mean residual error norm [m/s^2]");
    legend("Local DNN", "GS composite", "Oracle", "Location", "best");
    title(sprintf("Step 08-B.7 residual error, FOV %.1f deg, seed %d", ...
        halfAngleDeg, seed));

    figure;
    plot(time, meanCosLocal, "LineWidth", 1.5);
    hold on;
    plot(time, meanCosGS, "LineWidth", 1.5);
    plot(time, meanCosOracle, "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Mean cosine alignment");
    ylim([-1.05, 1.05]);
    legend("Local DNN", "GS composite", "Oracle", "Location", "best");
    title(sprintf("Step 08-B.7 residual direction alignment, FOV %.1f deg, seed %d", ...
        halfAngleDeg, seed));

    figure;
    plot(time, meanTrueNorm, "LineWidth", 1.5);
    hold on;
    plot(time, meanDNNNormLocal, "LineWidth", 1.2);
    plot(time, meanDNNNormGS, "LineWidth", 1.2);
    plot(time, meanDNNNormOracle, "LineWidth", 1.2);
    grid on;
    xlabel("Time [s]");
    ylabel("Mean residual norm [m/s^2]");
    legend("True residual", "Local DNN", "GS composite", "Oracle", ...
        "Location", "best");
    title(sprintf("Step 08-B.7 residual magnitude comparison, FOV %.1f deg, seed %d", ...
        halfAngleDeg, seed));

    % Keep inputs explicit in the function signature for future plot variants.
    resGS = resGS;
    resOracle = resOracle;

end

function X = alignRowsCols_step08b7(X, N, Nw)
%ALIGNROWSCOLS_STEP08B7 Trim/pad a 2-D diagnostic array to N x Nw.

    X = X(1:min(size(X,1),N), :);

    if size(X,1) < N
        X(end+1:N,:) = false;
    end

    X = X(:,1:min(size(X,2),Nw));

    if size(X,2) < Nw
        X(:,end+1:Nw) = false;
    end

end

function y = maskedMean_step08b7(X, activeRows, mask)
%MASKEDMEAN_STEP08B7 Mean of X(activeRows,:) over true mask entries.

    values = maskedValues_step08b7(X, activeRows, mask);
    y = mean(values, "omitnan");

end

function y = maskedRMS_step08b7(X, activeRows, mask)
%MASKEDRMS_STEP08B7 RMS of X(activeRows,:) over true mask entries.

    values = maskedValues_step08b7(X, activeRows, mask);
    y = sqrt(mean(values.^2, "omitnan"));

end

function y = maskedMedian_step08b7(X, activeRows, mask)
%MASKEDMEDIAN_STEP08B7 Median of X(activeRows,:) over true mask entries.

    values = maskedValues_step08b7(X, activeRows, mask);
    y = median(values, "omitnan");

end

function y = maskedMax_step08b7(X, activeRows, mask)
%MASKEDMAX_STEP08B7 Max of X(activeRows,:) over true mask entries.

    values = maskedValues_step08b7(X, activeRows, mask);

    if isempty(values)
        y = NaN;
    else
        y = max(values, [], "omitnan");
    end

end

function y = maskedPercentile_step08b7(X, activeRows, mask, pct)
%MASKEDPERCENTILE_STEP08B7 Percentile of X(activeRows,:) over mask entries.

    values = maskedValues_step08b7(X, activeRows, mask);
    y = percentileLocal_step08b7(values, pct);

end

function values = maskedValues_step08b7(X, activeRows, mask)
%MASKEDVALUES_STEP08B7 Extract finite-compatible masked values.

    Xactive = X(activeRows,:);

    if isempty(mask)
        values = Xactive(:);
        return;
    end

    if ~isequal(size(Xactive), size(mask))
        error("step08b7:MaskSizeMismatch", ...
            "Mask size does not match active diagnostic array size.");
    end

    values = Xactive(mask);
    values = values(:);

end

function pct = percentImprovement_step08b7(baselineValue, testValue)
%PERCENTIMPROVEMENT_STEP08B7 Positive value means testValue is better.
%
% For residual error metrics:
%     pct = 100 * (baseline - test) / baseline

    if isfinite(baselineValue) && baselineValue > 0 && isfinite(testValue)
        pct = 100 * (baselineValue - testValue) / baselineValue;
    else
        pct = NaN;
    end

end

function y = percentileLocal_step08b7(x, pct)
%PERCENTILELOCAL_STEP08B7 Percentile without requiring Statistics Toolbox.

    x = x(:);
    x = x(isfinite(x));

    if isempty(x)
        y = NaN;
        return;
    end

    x = sort(x);

    if isscalar(x)
        y = x;
        return;
    end

    pct = max(0, min(100, pct));

    pos = 1 + (numel(x)-1) * pct / 100;
    lo = floor(pos);
    hi = ceil(pos);

    if lo == hi
        y = x(lo);
    else
        alpha = pos - lo;
        y = (1-alpha)*x(lo) + alpha*x(hi);
    end

end