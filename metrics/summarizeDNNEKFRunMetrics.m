function summary = summarizeDNNEKFRunMetrics(results, cfg, caseName, runtimeSec)
%{
Function:
    summarizeDNNEKFRunMetrics

Purpose:
    Compute a compact, case-agnostic metric summary for the current
    collaborative DNN-EKF simulation family.

    This helper is intended for Step 04d performance comparison scripts.
    It works with the result structures produced by:

        1. simulatePhysicalEKF
        2. simulateLocalDNNEKF
        3. simulate_GS_DNN_EKF

    The function focuses on quantities that are common across cases:

        - target position / velocity tracking errors,
        - measurement availability,
        - NIS consistency diagnostics when available,
        - physical covariance trace,
        - DNN residual approximation error when available,
        - GS communication/cache diagnostics when available.

Inputs:
    results    - Simulation result structure.
                 Required fields:
                     results.time
                     results.etaTrue
                     results.xhat

                 Optional fields used when present:
                     results.Pdiag
                     results.PdiagEta
                     results.PdiagTheta
                     results.NIS
                     results.measAvail
                     results.dnnResidual
                     results.trueResidual
                     results.numNonlocalBranchesUsed
                     results.gsNumTotalUploads
                     results.gsNumIncluded
                     results.gsValid
                     results.traceQnonlocal

    cfg        - Simulation configuration structure.
                 Required fields:
                     cfg.dim
                     cfg.Nw

    caseName   - Optional string/char label for this run.

    runtimeSec - Optional wall-clock runtime in seconds.

Outputs:
    summary    - Structure with scalar and per-watcher metrics.
                 The field summary.tableRow is a one-row MATLAB table that
                 can be vertically concatenated across cases.

Main equations:
    For watcher i,

        e_{r,i}(k) = \hat r_i(k) - r(k),
        e_{v,i}(k) = \hat v_i(k) - v(k),

        RMSE_{r,i}
            = sqrt( mean_k ||e_{r,i}(k)||^2 ),

        RMSE_{v,i}
            = sqrt( mean_k ||e_{v,i}(k)||^2 ).

    If NIS is available,

        epsilon_i(k) = nu_i(k)' S_i(k)^{-1} nu_i(k).

    The 95-percent violation rate is computed using the chi-square 95%
    threshold for the bearing measurement dimension.

Notes:
    - Missing optional diagnostics are returned as NaN, not as errors.
    - This function does not plot. Plotting should be handled by a later
      Step 04d comparison script.
    - If results.traceQnonlocal is not logged yet, meanTraceQnonlocal is
      reported as NaN. A later small patch can add this log to
      simulate_GS_DNN_EKF.
%}

    if nargin < 3 || strlength(string(caseName)) == 0
        caseName = "unnamed_case";
    end

    if nargin < 4
        runtimeSec = NaN;
    end

    dim = cfg.dim;
    Nw  = cfg.Nw;

    time = results.time(:);
    N = numel(time);

    etaTrue = results.etaTrue;
    xhat = results.xhat;

    if size(etaTrue, 1) < 2*dim
        error("results.etaTrue must contain at least 2*cfg.dim rows.");
    end

    if size(xhat, 1) < 2*dim
        error("results.xhat must contain at least 2*cfg.dim rows.");
    end

    if size(xhat, 2) ~= N
        error("results.xhat and results.time have inconsistent lengths.");
    end

    if size(xhat, 3) ~= Nw
        error("results.xhat third dimension must match cfg.Nw.");
    end

    % ------------------------------------------------------------------
    % Tracking error histories
    % ------------------------------------------------------------------
    rTrue = etaTrue(1:dim, :);
    vTrue = etaTrue(dim+1:2*dim, :);

    posErr = NaN(N, Nw);
    velErr = NaN(N, Nw);

    for i = 1:Nw
        rHat = xhat(1:dim, :, i);
        vHat = xhat(dim+1:2*dim, :, i);

        er = rHat - rTrue;
        ev = vHat - vTrue;

        posErr(:, i) = sqrt(sum(er.^2, 1)).';
        velErr(:, i) = sqrt(sum(ev.^2, 1)).';
    end

    posRMSE = sqrt(localMeanFinite(posErr.^2, 1)).';
    velRMSE = sqrt(localMeanFinite(velErr.^2, 1)).';

    % ------------------------------------------------------------------
    % Measurement availability
    % ------------------------------------------------------------------
    if isfield(results, "measAvail")
        measAvail = logical(results.measAvail);
        measAvailFraction = localMeanFinite(double(measAvail(:)), 1);
        measAvailByWatcher = localMeanFinite(double(measAvail), 1).';
        numMeasurements = nnz(measAvail);
    else
        measAvailFraction = NaN;
        measAvailByWatcher = NaN(Nw, 1);
        numMeasurements = NaN;
    end

    % ------------------------------------------------------------------
    % NIS diagnostics
    % ------------------------------------------------------------------
    if cfg.dim == 2
        nz = 1;
    elseif cfg.dim == 3
        nz = 2;
    else
        nz = NaN;
    end

    chi2Threshold95 = localChi2Inv95(nz);

    if isfield(results, "NIS")
        nisAll = results.NIS(:);
        nisAll = nisAll(isfinite(nisAll));

        meanNIS = localMeanFinite(nisAll, 1);
        medianNIS = localMedianFinite(nisAll);

        if isempty(nisAll) || isnan(chi2Threshold95)
            nis95ViolationRate = NaN;
        else
            nis95ViolationRate = mean(nisAll > chi2Threshold95);
        end
    else
        meanNIS = NaN;
        medianNIS = NaN;
        nis95ViolationRate = NaN;
    end

    % ------------------------------------------------------------------
    % Covariance trace diagnostics
    % ------------------------------------------------------------------
    tracePeta = NaN(N, Nw);
    tracePtheta = NaN(N, Nw);

    if isfield(results, "PdiagEta")
        tracePeta = squeeze(sum(results.PdiagEta, 1));
        tracePeta = localForceTimeWatcherMatrix(tracePeta, N, Nw);
    elseif isfield(results, "Pdiag")
        pdiag = results.Pdiag;
        nP = size(pdiag, 1);
        nEta = min(2*dim, nP);
        tracePeta = squeeze(sum(pdiag(1:nEta, :, :), 1));
        tracePeta = localForceTimeWatcherMatrix(tracePeta, N, Nw);
    end

    if isfield(results, "PdiagTheta")
        tracePtheta = squeeze(sum(results.PdiagTheta, 1));
        tracePtheta = localForceTimeWatcherMatrix(tracePtheta, N, Nw);
    end

    % ------------------------------------------------------------------
    % DNN residual approximation diagnostics
    % ------------------------------------------------------------------
    if isfield(results, "dnnResidual") && isfield(results, "trueResidual")
        dnnResidual = results.dnnResidual;
        trueResidual = results.trueResidual;

        residualErr = NaN(N, Nw);

        for i = 1:Nw
            ed = dnnResidual(:, :, i) - trueResidual;
            residualErr(:, i) = sqrt(sum(ed.^2, 1)).';
        end

        meanDNNResidualErr = localMeanFinite(residualErr(:), 1);
        rmsDNNResidualErr = sqrt(localMeanFinite(residualErr(:).^2, 1));
    else
        residualErr = NaN(N, Nw);
        meanDNNResidualErr = NaN;
        rmsDNNResidualErr = NaN;
    end

    % ------------------------------------------------------------------
    % GS / nonlocal branch diagnostics
    % ------------------------------------------------------------------
    if isfield(results, "numNonlocalBranchesUsed")
        meanNonlocalBranchesUsed = localMeanFinite(results.numNonlocalBranchesUsed(:), 1);
        finalMeanNonlocalBranchesUsed = localMeanFinite(results.numNonlocalBranchesUsed(end, :), 1);
    else
        meanNonlocalBranchesUsed = NaN;
        finalMeanNonlocalBranchesUsed = NaN;
    end

    if isfield(results, "gsNumTotalUploads")
        totalGSUploads = results.gsNumTotalUploads(end);
        bootstrapGSUploads = results.gsNumTotalUploads(1);
        postBootstrapGSUploads = totalGSUploads - bootstrapGSUploads;
    else
        totalGSUploads = NaN;
        bootstrapGSUploads = NaN;
        postBootstrapGSUploads = NaN;
    end

    if isfield(results, "gsNumIncluded")
        meanGSIncludedBranches = localMeanFinite(results.gsNumIncluded(:), 1);
    else
        meanGSIncludedBranches = NaN;
    end

    if isfield(results, "gsValid")
        numValidBranchesTime = sum(results.gsValid, 1);
        meanGSValidBranches = localMeanFinite(numValidBranchesTime(:), 1);
        finalGSValidBranches = numValidBranchesTime(end);
    else
        meanGSValidBranches = NaN;
        finalGSValidBranches = NaN;
    end

    if isfield(results, "traceQnonlocal")
        traceQnonlocal = results.traceQnonlocal;
        meanTraceQnonlocal = localMeanFinite(traceQnonlocal(:), 1);
        maxTraceQnonlocal = max(traceQnonlocal(:), [], "omitnan");
    else
        traceQnonlocal = NaN(N, Nw);
        meanTraceQnonlocal = NaN;
        maxTraceQnonlocal = NaN;
    end

    % ------------------------------------------------------------------
    % Output structure
    % ------------------------------------------------------------------
    summary = struct();

    summary.caseName = string(caseName);
    summary.runtimeSec = runtimeSec;
    summary.N = N;
    summary.Nw = Nw;
    summary.dim = dim;
    summary.nz = nz;

    summary.time = time;

    summary.posErr = posErr;
    summary.velErr = velErr;
    summary.posRMSE = posRMSE;
    summary.velRMSE = velRMSE;

    summary.meanPosRMSE = localMeanFinite(posRMSE, 1);
    summary.meanVelRMSE = localMeanFinite(velRMSE, 1);
    summary.rmsPosErrAll = sqrt(localMeanFinite(posErr(:).^2, 1));
    summary.rmsVelErrAll = sqrt(localMeanFinite(velErr(:).^2, 1));
    summary.finalMeanPosErr = localMeanFinite(posErr(end, :), 1);
    summary.finalMeanVelErr = localMeanFinite(velErr(end, :), 1);

    summary.measAvailFraction = measAvailFraction;
    summary.measAvailByWatcher = measAvailByWatcher;
    summary.numMeasurements = numMeasurements;

    summary.meanNIS = meanNIS;
    summary.medianNIS = medianNIS;
    summary.chi2Threshold95 = chi2Threshold95;
    summary.nis95ViolationRate = nis95ViolationRate;

    summary.tracePeta = tracePeta;
    summary.tracePtheta = tracePtheta;
    summary.meanTracePeta = localMeanFinite(tracePeta(:), 1);
    summary.finalMeanTracePeta = localMeanFinite(tracePeta(end, :), 1);
    summary.meanTracePtheta = localMeanFinite(tracePtheta(:), 1);
    summary.finalMeanTracePtheta = localMeanFinite(tracePtheta(end, :), 1);

    summary.residualErr = residualErr;
    summary.meanDNNResidualErr = meanDNNResidualErr;
    summary.rmsDNNResidualErr = rmsDNNResidualErr;

    summary.meanNonlocalBranchesUsed = meanNonlocalBranchesUsed;
    summary.finalMeanNonlocalBranchesUsed = finalMeanNonlocalBranchesUsed;

    summary.totalGSUploads = totalGSUploads;
    summary.bootstrapGSUploads = bootstrapGSUploads;
    summary.postBootstrapGSUploads = postBootstrapGSUploads;
    summary.meanGSIncludedBranches = meanGSIncludedBranches;
    summary.meanGSValidBranches = meanGSValidBranches;
    summary.finalGSValidBranches = finalGSValidBranches;

    summary.traceQnonlocal = traceQnonlocal;
    summary.meanTraceQnonlocal = meanTraceQnonlocal;
    summary.maxTraceQnonlocal = maxTraceQnonlocal;

    summary.tableRow = table( ...
        summary.caseName, ...
        summary.meanPosRMSE, ...
        summary.meanVelRMSE, ...
        summary.finalMeanPosErr, ...
        summary.meanNIS, ...
        summary.medianNIS, ...
        summary.nis95ViolationRate, ...
        summary.meanTracePeta, ...
        summary.meanTracePtheta, ...
        summary.meanDNNResidualErr, ...
        summary.meanNonlocalBranchesUsed, ...
        summary.postBootstrapGSUploads, ...
        summary.meanTraceQnonlocal, ...
        summary.runtimeSec, ...
        'VariableNames', { ...
            'Case', ...
            'MeanPosRMSE', ...
            'MeanVelRMSE', ...
            'FinalMeanPosErr', ...
            'MeanNIS', ...
            'MedianNIS', ...
            'NIS95ViolationRate', ...
            'MeanTracePeta', ...
            'MeanTracePtheta', ...
            'MeanDNNResidualErr', ...
            'MeanNonlocalBranchesUsed', ...
            'PostBootstrapGSUploads', ...
            'MeanTraceQnonlocal', ...
            'RuntimeSec'});

end

function y = localMeanFinite(x, dim)
% localMeanFinite
%
% Purpose:
%     Compute mean over finite values only. Supports vector and matrix
%     inputs without relying on newer MATLAB omitnan behavior.

    if nargin < 2
        dim = 1;
    end

    finiteMask = isfinite(x);
    xWork = x;
    xWork(~finiteMask) = 0;

    count = sum(finiteMask, dim);
    total = sum(xWork, dim);

    y = total ./ count;
    y(count == 0) = NaN;

end

function y = localMedianFinite(x)
% localMedianFinite
%
% Purpose:
%     Median over finite values. Returns NaN if no finite values exist.

    x = x(isfinite(x));

    if isempty(x)
        y = NaN;
    else
        y = median(x);
    end

end

function x = localForceTimeWatcherMatrix(x, N, Nw)
% localForceTimeWatcherMatrix
%
% Purpose:
%     Convert squeezed trace arrays into an N-by-Nw matrix.

    if isscalar(x)
        x = repmat(x, N, Nw);
        return;
    end

    if isvector(x)
        if Nw == 1 && numel(x) == N
            x = x(:);
        elseif N == 1 && numel(x) == Nw
            x = reshape(x, 1, Nw);
        elseif numel(x) == N*Nw
            x = reshape(x, N, Nw);
        else
            error("Could not reshape trace diagnostic into N-by-Nw matrix.");
        end
        return;
    end

    if ~isequal(size(x), [N, Nw])
        if isequal(size(x), [Nw, N])
            x = x.';
        else
            error("Trace diagnostic has incompatible size.");
        end
    end

end

function q95 = localChi2Inv95(nz)
% localChi2Inv95
%
% Purpose:
%     Return chi-square 95% thresholds for the measurement dimensions used
%     in the current bearing-only simulations, without requiring the
%     Statistics and Machine Learning Toolbox.

    if nz == 1
        q95 = 3.841458820694124;
    elseif nz == 2
        q95 = 5.991464547107982;
    elseif nz == 3
        q95 = 7.814727903251179;
    else
        q95 = NaN;
    end

end
