function out = run_step09c0a_adaptive_Qtheta_diagnostics(out09c0)
%{
File:
    main/run_step09c0a_adaptive_Qtheta_diagnostics.m

Purpose:
    Step 09-C.0a adaptive Qtheta covariance-matching diagnostic.

    Check whether the time-varying adaptive DNN parameter process-noise
    scale gammaTheta is related to the interval where the GS composite
    residual approximation error becomes worse than the Local DNN residual
    approximation error.

Why this step matters:
    In Step 09-C.0, GS improves mean/RMS/median position error, but the
    residual approximation error can become worse than Local DNN during
    an intermediate time interval.

    This diagnostic compares:
        - residual approximation error,
        - position error,
        - adaptive gammaTheta,
        - covariance matching ratio cmRatio,

    for Local DNN and GS composite estimators.

Input:
    out09c0
        Output from:
            run_step09c0_compare_always_available_Local_GS_Oracle(...)

Output:
    out.windowSummaryTable
        Time-window summary of residual error, tracking error, gammaTheta,
        and covariance matching ratio.

Usage:
    out09c0a = run_step09c0a_adaptive_Qtheta_diagnostics(out09c0);
%}

    if nargin < 1 || isempty(out09c0)
        out09c0 = run_step09c0_compare_always_available_Local_GS_Oracle( ...
            "coupled_nonlinear", 101, false);
    end

    resLocal = out09c0.resLocal;
    resGS = out09c0.resGS;
    cfgLocal = out09c0.cfgLocal;
    cfgGS = out09c0.cfgGS;

    time = resLocal.time(:);

    % ---------------------------------------------------------------------
    % Mean tracking error over watchers
    % ---------------------------------------------------------------------
    % Prefer the already-computed position error from evaluateEstimatorMetrics.
    % This avoids depending on a specific result field name such as etaHat.
    if isfield(out09c0, "metricsLocal") && isfield(out09c0.metricsLocal, "posErr")
        posErrLocal = out09c0.metricsLocal.posErr;
    else
        posErrLocal = computePosErr_step09c0a(resLocal, cfgLocal);
    end

    if isfield(out09c0, "metricsGS") && isfield(out09c0.metricsGS, "posErr")
        posErrGS = out09c0.metricsGS.posErr;
    else
        posErrGS = computePosErr_step09c0a(resGS, cfgGS);
    end

    meanPosErrLocal = mean(posErrLocal, 2, "omitnan");
    meanPosErrGS = mean(posErrGS, 2, "omitnan");
    meanPosErrDiffLocalMinusGS = meanPosErrLocal - meanPosErrGS;

    % ---------------------------------------------------------------------
    % Mean residual approximation error over watchers
    % ---------------------------------------------------------------------
    meanResErrLocal = computeMeanResidualErr_step09c0a(resLocal, cfgLocal);
    meanResErrGS = computeMeanResidualErr_step09c0a(resGS, cfgGS);
    meanResErrDiffLocalMinusGS = meanResErrLocal - meanResErrGS;

    % ---------------------------------------------------------------------
    % Adaptive Qtheta logs
    % ---------------------------------------------------------------------
    gammaLocal = getLogMatrix_step09c0a(resLocal, "gammaTheta", numel(time));
    gammaGS = getLogMatrix_step09c0a(resGS, "gammaTheta", numel(time));

    cmRatioLocal = getLogMatrix_step09c0a(resLocal, "cmRatio", numel(time));
    cmRatioGS = getLogMatrix_step09c0a(resGS, "cmRatio", numel(time));

    meanGammaLocal = mean(gammaLocal, 2, "omitnan");
    meanGammaGS = mean(gammaGS, 2, "omitnan");

    meanCMRatioLocal = mean(cmRatioLocal, 2, "omitnan");
    meanCMRatioGS = mean(cmRatioGS, 2, "omitnan");

    % ---------------------------------------------------------------------
    % Window summary
    % ---------------------------------------------------------------------
    windowStart_s = [0; 800; 900; 1200; 1500];
    windowEnd_s   = [800; 900; 1200; 1500; 2000];

    nWin = numel(windowStart_s);

    windowName = strings(nWin, 1);

    meanLocalResErr = NaN(nWin, 1);
    meanGSResErr = NaN(nWin, 1);
    meanResErrDiff = NaN(nWin, 1);

    meanLocalPosErr = NaN(nWin, 1);
    meanGSPosErr = NaN(nWin, 1);
    meanPosErrDiff = NaN(nWin, 1);

    meanLocalGamma = NaN(nWin, 1);
    meanGSGamma = NaN(nWin, 1);
    maxLocalGamma = NaN(nWin, 1);
    maxGSGamma = NaN(nWin, 1);

    meanLocalCMRatio = NaN(nWin, 1);
    meanGSCMRatio = NaN(nWin, 1);
    maxLocalCMRatio = NaN(nWin, 1);
    maxGSCMRatio = NaN(nWin, 1);

    for iw = 1:nWin

        t0 = windowStart_s(iw);
        t1 = windowEnd_s(iw);

        windowName(iw) = sprintf("%g-%g s", t0, t1);

        idx = time >= t0 & time < t1;

        meanLocalResErr(iw) = mean(meanResErrLocal(idx), "omitnan");
        meanGSResErr(iw) = mean(meanResErrGS(idx), "omitnan");
        meanResErrDiff(iw) = mean(meanResErrDiffLocalMinusGS(idx), "omitnan");

        meanLocalPosErr(iw) = mean(meanPosErrLocal(idx), "omitnan");
        meanGSPosErr(iw) = mean(meanPosErrGS(idx), "omitnan");
        meanPosErrDiff(iw) = mean(meanPosErrDiffLocalMinusGS(idx), "omitnan");

        meanLocalGamma(iw) = mean(meanGammaLocal(idx), "omitnan");
        meanGSGamma(iw) = mean(meanGammaGS(idx), "omitnan");
        maxLocalGamma(iw) = max(meanGammaLocal(idx), [], "omitnan");
        maxGSGamma(iw) = max(meanGammaGS(idx), [], "omitnan");

        meanLocalCMRatio(iw) = mean(meanCMRatioLocal(idx), "omitnan");
        meanGSCMRatio(iw) = mean(meanCMRatioGS(idx), "omitnan");
        maxLocalCMRatio(iw) = max(meanCMRatioLocal(idx), [], "omitnan");
        maxGSCMRatio(iw) = max(meanCMRatioGS(idx), [], "omitnan");

    end

    windowSummaryTable = table( ...
        windowName, ...
        windowStart_s, ...
        windowEnd_s, ...
        meanLocalResErr, ...
        meanGSResErr, ...
        meanResErrDiff, ...
        meanLocalPosErr, ...
        meanGSPosErr, ...
        meanPosErrDiff, ...
        meanLocalGamma, ...
        meanGSGamma, ...
        maxLocalGamma, ...
        maxGSGamma, ...
        meanLocalCMRatio, ...
        meanGSCMRatio, ...
        maxLocalCMRatio, ...
        maxGSCMRatio, ...
        'VariableNames', { ...
            'windowName', ...
            'windowStart_s', ...
            'windowEnd_s', ...
            'meanLocalResErr', ...
            'meanGSResErr', ...
            'meanResErrDiff_LocalMinusGS', ...
            'meanLocalPosErr', ...
            'meanGSPosErr', ...
            'meanPosErrDiff_LocalMinusGS', ...
            'meanLocalGammaTheta', ...
            'meanGSGammaTheta', ...
            'maxLocalGammaTheta', ...
            'maxGSGammaTheta', ...
            'meanLocalCMRatio', ...
            'meanGSCMRatio', ...
            'maxLocalCMRatio', ...
            'maxGSCMRatio'});

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-C.0a: Adaptive Qtheta diagnostic\n");
    fprintf("============================================================\n");
    disp(windowSummaryTable);

    % ---------------------------------------------------------------------
    % Figure 1: gammaTheta
    % ---------------------------------------------------------------------
    figure;
    plot(time, meanGammaLocal, "LineWidth", 1.5);
    hold on;
    plot(time, meanGammaGS, "LineWidth", 1.5);
    grid on;

    xlabel("Time [s]");
    ylabel("Mean gammaTheta over watchers");
    title("Step 09-C.0a adaptive Qtheta scale");
    legend("Local DNN", "GS composite", "Location", "best");

    % ---------------------------------------------------------------------
    % Figure 2: covariance matching ratio
    % ---------------------------------------------------------------------
    figure;
    plot(time, meanCMRatioLocal, "LineWidth", 1.5);
    hold on;
    plot(time, meanCMRatioGS, "LineWidth", 1.5);
    yline(1.0, "--", "ratio = 1", "LineWidth", 1.2);
    grid on;

    xlabel("Time [s]");
    ylabel("Mean trace(Semp) / trace(Smodel)");
    title("Step 09-C.0a covariance matching ratio");
    legend("Local DNN", "GS composite", "ratio = 1", "Location", "best");

    % ---------------------------------------------------------------------
    % Figure 3: residual error difference
    % ---------------------------------------------------------------------
    figure;
    plot(time, meanResErrDiffLocalMinusGS, "LineWidth", 1.5);
    yline(0.0, "--", "Local = GS", "LineWidth", 1.2);
    grid on;

    xlabel("Time [s]");
    ylabel("Residual error difference [m/s^2]");
    title("Step 09-C.0a residual error difference: Local - GS");
    legend("Positive means GS residual error is smaller", "Location", "best");

    % ---------------------------------------------------------------------
    % Figure 4: position error difference
    % ---------------------------------------------------------------------
    figure;
    plot(time, meanPosErrDiffLocalMinusGS, "LineWidth", 1.5);
    yline(0.0, "--", "Local = GS", "LineWidth", 1.2);
    grid on;

    xlabel("Time [s]");
    ylabel("Position error difference [m]");
    title("Step 09-C.0a position error difference: Local - GS");
    legend("Positive means GS position error is smaller", "Location", "best");

    out = struct();
    out.out09c0 = out09c0;
    out.time = time;
    out.windowSummaryTable = windowSummaryTable;

    out.meanPosErrLocal = meanPosErrLocal;
    out.meanPosErrGS = meanPosErrGS;
    out.meanPosErrDiffLocalMinusGS = meanPosErrDiffLocalMinusGS;

    out.meanResErrLocal = meanResErrLocal;
    out.meanResErrGS = meanResErrGS;
    out.meanResErrDiffLocalMinusGS = meanResErrDiffLocalMinusGS;

    out.meanGammaLocal = meanGammaLocal;
    out.meanGammaGS = meanGammaGS;
    out.meanCMRatioLocal = meanCMRatioLocal;
    out.meanCMRatioGS = meanCMRatioGS;

end

function posErr = computePosErr_step09c0a(results, cfg)
%COMPUTEPOSERR_STEP09C0A Position error norm for each watcher.
%
% This fallback supports several possible result-field conventions.

dim = cfg.dim;

if ~isfield(results, "etaTrue")
    error("Step09C0a:MissingEtaTrue", ...
        "results must contain etaTrue, or posErr must be provided by metrics.");
end

etaTrue = results.etaTrue;
rTrue = etaTrue(1:dim, :);

if isfield(results, "etaHat")
    etaHat = results.etaHat;

elseif isfield(results, "xhat")
    etaHat = results.xhat;

elseif isfield(results, "xhatLog")
    etaHat = results.xhatLog;

elseif isfield(results, "xaugHat")
    etaHat = results.xaugHat;

else
    error("Step09C0a:MissingEstimateLog", ...
        "Could not find etaHat/xhat/xhatLog/xaugHat in results.");
end

if ndims(etaHat) == 2
    etaHat = reshape(etaHat, size(etaHat,1), size(etaHat,2), 1);
end

[~, N, Nw] = size(etaHat);

posErr = NaN(N, Nw);

for iw = 1:Nw
    rHat = etaHat(1:dim, :, iw);
    posErr(:, iw) = sqrt(sum((rHat - rTrue).^2, 1)).';
end

end


function meanErr = computeMeanResidualErr_step09c0a(results, cfg)
%COMPUTEMEANRESIDUALERR_STEP09C0A Mean residual approximation error.

    dHat = results.dnnResidual;
    dTrue = results.trueResidual;

    [dim, N, Nw] = size(dHat);

    dTrue = expandTrueResidual_step09c0a(dTrue, dim, N, Nw);

    residualSource = "unknown";

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "predictionResidualSource")
        residualSource = string(cfg.dnn.predictionResidualSource);
    end

    if residualSource == "oracle"
        dHat = dTrue;
    end

    errNorm = reshape(sqrt(sum((dHat - dTrue).^2, 1)), N, Nw);

    meanErr = mean(errNorm, 2, "omitnan");

end

function dTrueOut = expandTrueResidual_step09c0a(dTrueIn, dim, N, Nw)
%EXPANDTRUERESIDUAL_STEP09C0A Convert true residual log to dim x N x Nw.

    sz = size(dTrueIn);

    if numel(sz) == 2
        dTrueOut = reshape(dTrueIn, dim, N, 1);
        dTrueOut = repmat(dTrueOut, 1, 1, Nw);
        return;
    end

    if numel(sz) == 3
        if sz(3) == Nw
            dTrueOut = dTrueIn;
        elseif sz(3) == 1
            dTrueOut = repmat(dTrueIn, 1, 1, Nw);
        else
            error("Step09C0a:BadTrueResidualSize", ...
                "trueResidual third dimension must be 1 or Nw.");
        end
        return;
    end

    error("Step09C0a:BadTrueResidualSize", ...
        "trueResidual must be dim x N or dim x N x Nw.");

end

function X = getLogMatrix_step09c0a(results, fieldName, N)
%GETLOGMATRIX_STEP09C0A Read an N x Nw diagnostic log safely.

    if ~isfield(results, fieldName)
        X = NaN(N, 1);
        return;
    end

    X = results.(fieldName);

    if isvector(X)
        X = X(:);
    end

    if size(X, 1) ~= N && size(X, 2) == N
        X = X.';
    end

    if size(X, 1) ~= N
        error("Step09C0a:BadLogSize", ...
            "%s must have N rows or N columns.", fieldName);
    end

end