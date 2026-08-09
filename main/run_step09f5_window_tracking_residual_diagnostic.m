function out = run_step09f5_window_tracking_residual_diagnostic(out09f4)
%{
File:
    main/run_step09f5_window_tracking_residual_diagnostic.m

Purpose:
    Step 09-F.5 diagnostic.

    Compare additive and gated_additive GS results over the same time
    windows using both:

        1) tracking error
        2) residual approximation error

Why this step matters:
    Step 09-F.4 showed an important mismatch:

        gated_additive has better overall residual approximation,
        but worse overall tracking performance.

    This file checks whether that mismatch is caused by time-window effects.
    In particular, it checks whether additive is better during early or
    transient windows where the EKF trajectory/covariance history is more
    sensitive.

Input:
    out09f4
        Output from:

            run_step09f4_compare_additive_vs_gated_diagnostics(...)

Output:
    out.windowComparisonTable

Interpretation:
    If gated_additive has lower residual error in late windows but worse
    tracking in early windows, then the issue is likely estimator dynamics
    / correction authority / covariance history, not simply residual
    approximation quality.
%}

    if nargin < 1 || isempty(out09f4)
        out09f4 = run_step09f4_compare_additive_vs_gated_diagnostics( ...
            "complex_branchwise", 101, 0.5);
    end

    requiredFields = [
        "outAdd"
        "outGate"
        "residualWindowSummary"
    ];

    for iField = 1:numel(requiredFields)
        if ~isfield(out09f4, requiredFields(iField))
            error("Step09F5:MissingField", ...
                "out09f4 is missing required field: %s.", requiredFields(iField));
        end
    end

    TaddTrack = buildTrackingWindowTable_step09f5( ...
        out09f4.outAdd.metricsLocal, ...
        out09f4.outAdd.metricsGS, ...
        "additive");

    TgateTrack = buildTrackingWindowTable_step09f5( ...
        out09f4.outGate.metricsLocal, ...
        out09f4.outGate.metricsGS, ...
        "gated_additive");

    Ttrack = [
        TaddTrack
        TgateTrack
    ];

    Tres = out09f4.residualWindowSummary;

    % Keep only the columns needed for side-by-side interpretation.
    TresSmall = Tres(:, [
        "compositeMode", ...
        "windowName", ...
        "meanOperationalErr", ...
        "meanSameInputErr", ...
        "rmsOperationalErr", ...
        "rmsSameInputErr", ...
        "p95OperationalErr", ...
        "p95SameInputErr"]);

    windowComparisonTable = outerjoin( ...
        Ttrack, ...
        TresSmall, ...
        "Keys", ["compositeMode", "windowName"], ...
        "MergeKeys", true, ...
        "Type", "left");

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-F.5: Window tracking/residual diagnostic\n");
    fprintf("============================================================\n\n");

    disp(windowComparisonTable(:, [
        "compositeMode", ...
        "windowName", ...
        "meanPosErr_m", ...
        "rmsPosErr_m", ...
        "p95PosErr_m", ...
        "gsMeanImprovementPct_window", ...
        "meanOperationalErr", ...
        "meanSameInputErr", ...
        "rmsOperationalErr", ...
        "rmsSameInputErr"]));

    out = struct();
    out.out09f4 = out09f4;
    out.windowComparisonTable = windowComparisonTable;

end

function T = buildTrackingWindowTable_step09f5(metricsLocal, metricsGS, compositeMode)
% Build windowed tracking metrics for one GS composite mode.

    time = metricsGS.time(:);

    posLocal = metricsLocal.posErr;
    posGS = metricsGS.posErr;

    if size(posLocal, 1) ~= numel(time)
        error("Step09F5:LocalTimeSizeMismatch", ...
            "metricsLocal.posErr and metricsGS.time have inconsistent sizes.");
    end

    if size(posGS, 1) ~= numel(time)
        error("Step09F5:GSTimeSizeMismatch", ...
            "metricsGS.posErr and metricsGS.time have inconsistent sizes.");
    end

    windowName = [
        "0-800 s"
        "800-900 s"
        "900-1200 s"
        "1200-1500 s"
        "1500-2000 s"
    ];

    tStart_s = [
        0
        800
        900
        1200
        1500
    ];

    tEnd_s = [
        800
        900
        1200
        1500
        2000
    ];

    nW = numel(windowName);

    meanPosErr_m = NaN(nW, 1);
    rmsPosErr_m = NaN(nW, 1);
    p95PosErr_m = NaN(nW, 1);

    localMeanPosErr_m = NaN(nW, 1);
    gsMeanImprovementPct_window = NaN(nW, 1);

    for iW = 1:nW

        mask = time >= tStart_s(iW) & time < tEnd_s(iW);

        gsVals = posGS(mask, :);
        localVals = posLocal(mask, :);

        gsVals = gsVals(:);
        localVals = localVals(:);

        meanPosErr_m(iW) = mean(gsVals, "omitnan");
        rmsPosErr_m(iW) = sqrt(mean(gsVals.^2, "omitnan"));
        p95PosErr_m(iW) = percentileLocal_step09f5(gsVals, 95);

        localMeanPosErr_m(iW) = mean(localVals, "omitnan");

        if isfinite(localMeanPosErr_m(iW)) && localMeanPosErr_m(iW) > 0
            gsMeanImprovementPct_window(iW) = ...
                100 * (localMeanPosErr_m(iW) - meanPosErr_m(iW)) ...
                / localMeanPosErr_m(iW);
        end

    end

    T = table( ...
        repmat(string(compositeMode), nW, 1), ...
        windowName, ...
        tStart_s, ...
        tEnd_s, ...
        localMeanPosErr_m, ...
        meanPosErr_m, ...
        rmsPosErr_m, ...
        p95PosErr_m, ...
        gsMeanImprovementPct_window, ...
        'VariableNames', { ...
            'compositeMode', ...
            'windowName', ...
            'tStart_s', ...
            'tEnd_s', ...
            'localMeanPosErr_m', ...
            'meanPosErr_m', ...
            'rmsPosErr_m', ...
            'p95PosErr_m', ...
            'gsMeanImprovementPct_window'});

end

function p = percentileLocal_step09f5(x, pct)
% Toolbox-free percentile helper.

    x = x(:);
    x = x(isfinite(x));

    if isempty(x)
        p = NaN;
        return;
    end

    x = sort(x);

    if isscalar(x)
        p = x(1);
        return;
    end

    q = pct / 100;
    idx = 1 + q * (numel(x) - 1);

    idxLo = floor(idx);
    idxHi = ceil(idx);

    if idxLo == idxHi
        p = x(idxLo);
    else
        w = idx - idxLo;
        p = (1 - w) * x(idxLo) + w * x(idxHi);
    end

end