function metrics = evaluateEstimatorMetrics(res, cfg, caseName)
%{
File:
    analysis/evaluateEstimatorMetrics.m

Purpose:
    Compute multi-metric estimator performance diagnostics.

    This helper goes beyond a single RMSE number. It evaluates:
        1. Position-error magnitude
        2. FOV / measurement availability
        3. Error during available vs dropout rows
        4. NIS availability/consistency diagnostics
        5. GS communication count diagnostics

Inputs:
    res
        Simulation result structure.

    cfg
        Simulation configuration structure.

    caseName
        Optional string label for the estimator case.

Outputs:
    metrics
        Struct containing scalar metrics, per-watcher metrics, and tables.

Notes:
    This is a post-processing helper. It does not change the simulation.
%}

    if nargin < 3 || isempty(caseName)
        caseName = "unnamed_case";
    end

    caseName = string(caseName);

    dim = cfg.dim;
    N = numel(cfg.time);

    % ---------------------------------------------------------------------
    % Position error array
    % ---------------------------------------------------------------------
    [posErr, timeVec] = computePositionErrorArray(res, cfg);

    % posErr is N x Nw after normalization.
    posErrAll = posErr(:);

    metrics = struct();
    metrics.caseName = caseName;
    metrics.time = timeVec;
    metrics.posErr = posErr;

    % ---------------------------------------------------------------------
    % Global position-error magnitude metrics
    % ---------------------------------------------------------------------
    metrics.meanPosErr = mean(posErrAll, "omitnan");
    metrics.rmsPosErr = sqrt(mean(posErrAll.^2, "omitnan"));
    metrics.medianPosErr = median(posErrAll, "omitnan");
    metrics.p95PosErr = percentileLocal(posErrAll, 95);
    metrics.maxPosErr = max(posErrAll, [], "omitnan");
    metrics.finalMeanPosErr = mean(posErr(end,:), "omitnan");
    metrics.finalMaxPosErr = max(posErr(end,:), [], "omitnan");

    metrics.perWatcherMeanPosErr = mean(posErr, 1, "omitnan");
    metrics.perWatcherRMSPosErr = sqrt(mean(posErr.^2, 1, "omitnan"));
    metrics.perWatcherP95PosErr = applyPercentileByColumn(posErr, 95);
    metrics.perWatcherMaxPosErr = max(posErr, [], 1, "omitnan");

    % ---------------------------------------------------------------------
    % Measurement availability and dropout metrics
    % ---------------------------------------------------------------------
    activeRows = getActiveRows(cfg, posErr);

    metrics.activeRows = activeRows;

    if isfield(res, "measAvail")
        measAvail = logical(res.measAvail);

        [measAvailActive, posErrActive] = alignMatrixToActiveRows( ...
            measAvail, posErr, activeRows);

        metrics.availableCount = nnz(measAvailActive);
        metrics.totalAvailabilityEntries = numel(measAvailActive);
        metrics.availabilityRate = metrics.availableCount / ...
            max(metrics.totalAvailabilityEntries, 1);

        metrics.dropoutCount = metrics.totalAvailabilityEntries - ...
            metrics.availableCount;
        metrics.dropoutRate = 1.0 - metrics.availabilityRate;

        errAvailable = posErrActive(measAvailActive);
        errDropout = posErrActive(~measAvailActive);

        metrics.meanPosErr_available = mean(errAvailable, "omitnan");
        metrics.rmsPosErr_available = sqrt(mean(errAvailable.^2, "omitnan"));
        metrics.p95PosErr_available = percentileLocal(errAvailable, 95);
        metrics.maxPosErr_available = max(errAvailable, [], "omitnan");

        metrics.meanPosErr_dropout = mean(errDropout, "omitnan");
        metrics.rmsPosErr_dropout = sqrt(mean(errDropout.^2, "omitnan"));
        metrics.p95PosErr_dropout = percentileLocal(errDropout, 95);
        metrics.maxPosErr_dropout = max(errDropout, [], "omitnan");
    else
        metrics.availableCount = NaN;
        metrics.totalAvailabilityEntries = NaN;
        metrics.availabilityRate = NaN;
        metrics.dropoutCount = NaN;
        metrics.dropoutRate = NaN;

        metrics.meanPosErr_available = NaN;
        metrics.rmsPosErr_available = NaN;
        metrics.p95PosErr_available = NaN;
        metrics.maxPosErr_available = NaN;

        metrics.meanPosErr_dropout = NaN;
        metrics.rmsPosErr_dropout = NaN;
        metrics.p95PosErr_dropout = NaN;
        metrics.maxPosErr_dropout = NaN;
    end

    % ---------------------------------------------------------------------
    % Dropout reason table
    % ---------------------------------------------------------------------
    if isfield(res, "measurementDropoutReason")
        reasons = string(res.measurementDropoutReason);
        reasonsActive = reasons(activeRows,:);

        metrics.dropoutReasonTable = makeReasonCountTable(reasonsActive);
    else
        metrics.dropoutReasonTable = table();
    end

    % ---------------------------------------------------------------------
    % NIS diagnostics
    % ---------------------------------------------------------------------
    nisField = findFirstExistingField(res, ["NIS", "nis", "NISLog", "nisLog"]);

    metrics.nisField = nisField;

    if strlength(nisField) > 0
        nis = res.(nisField);
        nisActive = selectActiveRowsIfPossible(nis, activeRows);

        nisAll = nisActive(:);
        nisFinite = nisAll(isfinite(nisAll));

        metrics.nisFiniteCount = numel(nisFinite);
        metrics.nisNaNCount = nnz(isnan(nisAll));
        metrics.meanNIS = mean(nisFinite, "omitnan");
        metrics.medianNIS = median(nisFinite, "omitnan");
        metrics.p95NIS = percentileLocal(nisFinite, 95);
        metrics.maxNIS = max(nisFinite, [], "omitnan");

        if isfield(res, "measAvail")
            measAvail = logical(res.measAvail);
            measAvailActive = selectActiveRowsIfPossible(measAvail, activeRows);

            if isequal(size(measAvailActive), size(nisActive))
                metrics.nisFiniteOnAvailableCount = nnz( ...
                    isfinite(nisActive) & measAvailActive);
                metrics.nisFiniteOnDropoutCount = nnz( ...
                    isfinite(nisActive) & ~measAvailActive);
            else
                metrics.nisFiniteOnAvailableCount = NaN;
                metrics.nisFiniteOnDropoutCount = NaN;
            end
        else
            metrics.nisFiniteOnAvailableCount = NaN;
            metrics.nisFiniteOnDropoutCount = NaN;
        end
    else
        metrics.nisFiniteCount = NaN;
        metrics.nisNaNCount = NaN;
        metrics.meanNIS = NaN;
        metrics.medianNIS = NaN;
        metrics.p95NIS = NaN;
        metrics.maxNIS = NaN;
        metrics.nisFiniteOnAvailableCount = NaN;
        metrics.nisFiniteOnDropoutCount = NaN;
    end

    % ---------------------------------------------------------------------
    % GS communication metrics
    % ---------------------------------------------------------------------
    if isfield(res, "gsUploadDecision")
        uploadDecision = logical(res.gsUploadDecision);
        uploadDecisionActive = selectActiveRowsIfPossible(uploadDecision, activeRows);

        metrics.gsLoggedUploadDecisions = nnz(uploadDecisionActive);
    else
        metrics.gsLoggedUploadDecisions = NaN;
    end

    if isfield(res, "gsNumTotalUploads")
        metrics.gsFinalTotalUploads = res.gsNumTotalUploads(end);
    elseif isfield(res, "gsUploadCount")
        metrics.gsFinalTotalUploads = res.gsUploadCount(end);
    else
        metrics.gsFinalTotalUploads = NaN;
    end

    if isfinite(metrics.availableCount) && metrics.availableCount > 0
        metrics.gsUploadsPerAvailableMeasurement = ...
            metrics.gsLoggedUploadDecisions / metrics.availableCount;
    else
        metrics.gsUploadsPerAvailableMeasurement = NaN;
    end

    % ---------------------------------------------------------------------
    % Compact scalar summary table
    % ---------------------------------------------------------------------
    metrics.scalarSummary = table( ...
        caseName, ...
        metrics.meanPosErr, ...
        metrics.rmsPosErr, ...
        metrics.medianPosErr, ...
        metrics.p95PosErr, ...
        metrics.maxPosErr, ...
        metrics.finalMeanPosErr, ...
        100*metrics.availabilityRate, ...
        metrics.meanPosErr_available, ...
        metrics.meanPosErr_dropout, ...
        metrics.p95PosErr_dropout, ...
        metrics.nisFiniteCount, ...
        metrics.meanNIS, ...
        metrics.gsLoggedUploadDecisions, ...
        metrics.gsFinalTotalUploads, ...
        'VariableNames', { ...
            'caseName', ...
            'meanPosErr_m', ...
            'rmsPosErr_m', ...
            'medianPosErr_m', ...
            'p95PosErr_m', ...
            'maxPosErr_m', ...
            'finalMeanPosErr_m', ...
            'availabilityRatePercent', ...
            'meanPosErr_available_m', ...
            'meanPosErr_dropout_m', ...
            'p95PosErr_dropout_m', ...
            'nisFiniteCount', ...
            'meanNIS', ...
            'gsLoggedUploadDecisions', ...
            'gsFinalTotalUploads'});

end

function [posErr, timeVec] = computePositionErrorArray(res, cfg)
%COMPUTEPOSITIONERRORARRAY Return position error as N x Nw.

    dim = cfg.dim;

    if isfield(res, "etaTrue")
        etaTrue = res.etaTrue;
    elseif isfield(res, "xTrue")
        etaTrue = res.xTrue;
    else
        error("evaluateEstimatorMetrics:MissingTruth", ...
            "Could not find etaTrue or xTrue in results.");
    end

    etaTrue = orientStateTimeArray(etaTrue, 2*dim);

    if isfield(res, "xhat")
        xhat = res.xhat;
    elseif isfield(res, "xhatAug")
        xhat = res.xhatAug;
    elseif isfield(res, "etaHat")
        xhat = res.etaHat;
    else
        error("evaluateEstimatorMetrics:MissingEstimate", ...
            "Could not find xhat, xhatAug, or etaHat in results.");
    end

    rTrue = etaTrue(1:dim,:);
    NtTruth = size(rTrue, 2);

    if ndims(xhat) == 2
        xhat = orientStateTimeArray(xhat, dim);
        rHat = xhat(1:dim,:);

        NtCommon = min(size(rHat,2), NtTruth);
        posErr = vecnorm(rHat(:,1:NtCommon) - rTrue(:,1:NtCommon), 2, 1).';

    elseif ndims(xhat) == 3
        xhat = orientStateTimeWatcherArray(xhat, dim);

        rHat = xhat(1:dim,:,:);

        NtCommon = min(size(rHat,2), NtTruth);
        Nw = size(rHat,3);

        posErr = NaN(NtCommon, Nw);

        for iw = 1:Nw
            rHat_i = rHat(:,1:NtCommon,iw);
            rTrue_i = rTrue(:,1:NtCommon);

            posErr(:,iw) = vecnorm(rHat_i - rTrue_i, 2, 1).';
        end
    else
        error("evaluateEstimatorMetrics:UnsupportedEstimateShape", ...
            "Unsupported estimate array dimension: ndims(xhat) = %d.", ndims(xhat));
    end

    if isfield(res, "time")
        timeVec = res.time(:);
    elseif isfield(cfg, "time")
        timeVec = cfg.time(:);
    else
        timeVec = (0:size(posErr,1)-1).';
    end

    NtCommon = min(numel(timeVec), size(posErr,1));
    posErr = posErr(1:NtCommon,:);
    timeVec = timeVec(1:NtCommon);

end

function X = orientStateTimeArray(X, minStateDim)
%ORIENTSTATETIMEARRAY Convert 2D state-time array to state x time.

    if ndims(X) ~= 2
        error("evaluateEstimatorMetrics:Expected2DArray", ...
            "Expected a 2D state-time array.");
    end

    nRow = size(X,1);
    nCol = size(X,2);

    if nRow < minStateDim && nCol >= minStateDim
        X = X.';
    end

end

function X = orientStateTimeWatcherArray(X, minStateDim)
%ORIENTSTATETIMEWATCHERARRAY Convert 3D estimate to state x time x watcher.

    if ndims(X) ~= 3
        error("evaluateEstimatorMetrics:Expected3DArray", ...
            "Expected a 3D state-time-watcher array.");
    end

    n1 = size(X,1);
    n2 = size(X,2);

    if n1 >= minStateDim
        return;
    end

    if n2 >= minStateDim
        X = permute(X, [2 1 3]);
        return;
    end

    error("evaluateEstimatorMetrics:Invalid3DEstimateShape", ...
        "Could not infer state dimension from estimate size [%s].", ...
        num2str(size(X)));

end

function activeRows = getActiveRows(cfg, posErr)
%GETACTIVEROWS Return rows that correspond to measurement/evaluation updates.

    N = size(posErr, 1);

    if isfield(cfg, "N")
        Ncfg = cfg.N;
        activeRows = 2:(Ncfg-1);
        activeRows = activeRows(activeRows <= N);
    else
        activeRows = 2:(N-1);
    end

    if isempty(activeRows)
        activeRows = 1:N;
    end

end

function [Aactive, Pactive] = alignMatrixToActiveRows(A, P, activeRows)
%ALIGNMATRIXTOACTIVEROWS Align diagnostic matrix A and posErr P on active rows.

    Aactive = selectActiveRowsIfPossible(A, activeRows);
    Pactive = selectActiveRowsIfPossible(P, activeRows);

    % If A has one watcher dimension and P has one or more, expand if needed.
    if size(Aactive,2) == 1 && size(Pactive,2) > 1
        Aactive = repmat(Aactive, 1, size(Pactive,2));
    end

    % If dimensions still differ, crop to common size.
    nRow = min(size(Aactive,1), size(Pactive,1));
    nCol = min(size(Aactive,2), size(Pactive,2));

    Aactive = Aactive(1:nRow,1:nCol);
    Pactive = Pactive(1:nRow,1:nCol);

end

function Xactive = selectActiveRowsIfPossible(X, activeRows)
%SELECTACTIVEROWSIFPOSSIBLE Select active rows from a 2D array when possible.

    if isempty(X)
        Xactive = X;
        return;
    end

    if ndims(X) ~= 2
        Xactive = X;
        return;
    end

    if size(X,1) >= max(activeRows)
        Xactive = X(activeRows,:);
    else
        Xactive = X;
    end

end

function fieldName = findFirstExistingField(s, candidateNames)
%FINDFIRSTEXISTINGFIELD Return first candidate field that exists.

    fieldName = "";

    for idx = 1:numel(candidateNames)
        candidate = string(candidateNames(idx));

        if isfield(s, candidate)
            fieldName = candidate;
            return;
        end
    end

end

function reasonTable = makeReasonCountTable(reasons)
%MAKEREASONCOUNTTABLE Count string-valued dropout reasons.

    reasons = string(reasons(:));
    reasonList = unique(reasons);

    count = NaN(numel(reasonList),1);

    for idx = 1:numel(reasonList)
        count(idx) = nnz(reasons == reasonList(idx));
    end

    reasonTable = table(reasonList, count, ...
        'VariableNames', {'reason', 'count'});

end

function y = percentileLocal(x, pct)
%PERCENTILELOCAL Compute percentile without requiring Statistics Toolbox.

    x = x(:);
    x = x(isfinite(x));

    if isempty(x)
        y = NaN;
        return;
    end

    x = sort(x);
    n = numel(x);

    if n == 1
        y = x;
        return;
    end

    q = pct / 100;
    idx = 1 + q*(n - 1);

    idxLow = floor(idx);
    idxHigh = ceil(idx);

    if idxLow == idxHigh
        y = x(idxLow);
    else
        w = idx - idxLow;
        y = (1-w)*x(idxLow) + w*x(idxHigh);
    end

end

function y = applyPercentileByColumn(X, pct)
%APPLYPERCENTILEBYCOLUMN Compute percentile for each column.

    y = NaN(1, size(X,2));

    for idx = 1:size(X,2)
        y(idx) = percentileLocal(X(:,idx), pct);
    end

end