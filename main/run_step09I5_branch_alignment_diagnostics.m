function diagOut = run_step09I5_branch_alignment_diagnostics( ...
    outOrResults, watcherID, windowEdges, makePlots)
%{
File:
    main/run_step09I5_branch_alignment_diagnostics.m

Purpose:
    Step 09-I.5 branch-contribution alignment diagnostic.

    Diagnose why GS additive residual can improve overall residual
    approximation but still spike in selected time intervals.

Main diagnostic:
    For the GS run, decompose

        d_GS = d_local_component + d_nonlocal_sum

    and define

        e_L = d_local_component - d_true
        n   = d_nonlocal_sum

    Then

        ||e_L + n||^2 - ||e_L||^2
        = 2 e_L' n + ||n||^2.

    If this value is positive, the nonlocal GS contribution increases the
    residual error relative to the GS-local component.

Important:
    This diagnostic uses the component logs already stored in resGS:
        results.dnnResidualLocalComponent
        results.dnnResidualNonlocalComponent
        results.dnnResidualBranchContrib

    These component logs are evaluated at etaHat, i.e. operational input.
    In Step 09-I.4, operational-vs-same-input penalty was negligible, so
    this is sufficient for diagnosing the 800--1400 s spike.

Inputs:
    outOrResults:
        Either the full output from run_step09I4_compare_MLP_Local_GS_Oracle,
        or directly a GS results struct.

    watcherID:
        Numeric watcher ID, e.g. 1, or "all".
        Default: "all".

    windowEdges:
        Time window edges for summary.
        Default: [0 800 900 1200 1400 1500 2000].

    makePlots:
        true/false.
        Default: true.

Outputs:
    diagOut.windowSummaryTable
        Window-level diagnostic summary.

    diagOut.branchSummaryTable
        Branch-level contribution summary over the 800--1400 s interval.

    diagOut.timeSeries
        Time-series arrays for further plotting.
%}

    if nargin < 2 || isempty(watcherID)
        watcherID = "all";
    end

    if nargin < 3 || isempty(windowEdges)
        windowEdges = [0 800 900 1200 1400 1500 2000];
    end

    if nargin < 4 || isempty(makePlots)
        makePlots = true;
    end

    % ---------------------------------------------------------------------
    % Accept either the Step 09-I.4 output struct or a raw GS results struct.
    % ---------------------------------------------------------------------
    if isfield(outOrResults, "resGS")
        resGS = outOrResults.resGS;
    else
        resGS = outOrResults;
    end

    requiredFields = [
        "time"
        "trueResidual"
        "dnnResidual"
        "dnnResidualLocalComponent"
        "dnnResidualNonlocalComponent"
        "dnnResidualBranchContrib"
    ];

    for k = 1:numel(requiredFields)
        fieldName = requiredFields(k);

        if ~isfield(resGS, fieldName)
            error("Step09I5:MissingField", ...
                "resGS.%s is required for this diagnostic.", fieldName);
        end
    end

    time = resGS.time(:).';
    [dim, N, Nw] = size(resGS.dnnResidual);

    selectedWatchers = resolveWatcherSelectionStep09I5(watcherID, Nw);

    if ischar(watcherID) || isstring(watcherID)
        watcherLabel = string(watcherID);
    else
        watcherLabel = "watcher_" + strjoin(string(selectedWatchers), "_");
    end

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-I.5: branch contribution alignment diagnostic\n");
    fprintf("============================================================\n");
    fprintf("Selected watchers = %s\n", mat2str(selectedWatchers));
    fprintf("N                 = %d\n", N);
    fprintf("Nw                = %d\n", Nw);
    fprintf("dim               = %d\n\n", dim);

    % ---------------------------------------------------------------------
    % Expand true residual to dim x N x Nw.
    % ---------------------------------------------------------------------
    dTrue = expandTrueResidualStep09I5(resGS.trueResidual, dim, N, Nw);

    % Component logs from the GS run.
    dLocalComp = resGS.dnnResidualLocalComponent;
    dNonlocal = resGS.dnnResidualNonlocalComponent;

    % Use the sum of logged components for exact decomposition.
    dGSFromComponents = dLocalComp + dNonlocal;

    % Compare with logged dnnResidual for a lightweight consistency check.
    compMismatch = norm(dGSFromComponents(:) - resGS.dnnResidual(:)) / ...
        max(1.0e-14, norm(resGS.dnnResidual(:)));

    fprintf("Relative mismatch: local + nonlocal vs dnnResidual = %.3e\n\n", ...
        compMismatch);

    if compMismatch > 1.0e-10
        warning("Step09I5:ComponentMismatch", ...
            "Component sum does not exactly match resGS.dnnResidual.");
    end

    % ---------------------------------------------------------------------
    % Alignment quantities.
    % ---------------------------------------------------------------------
    eLocal = dLocalComp - dTrue;
    eGS = dGSFromComponents - dTrue;
    nNonlocal = dNonlocal;

    localErrNorm = vecNormOverDimStep09I5(eLocal);
    gsErrNorm = vecNormOverDimStep09I5(eGS);
    nonlocalNorm = vecNormOverDimStep09I5(nNonlocal);

    dot_eL_n = squeeze(sum(eLocal .* nNonlocal, 1));

    localNorm2 = localErrNorm.^2;
    gsNorm2 = gsErrNorm.^2;
    nNorm2 = nonlocalNorm.^2;

    % Theoretical identity:
    %   gsNorm2 - localNorm2 = 2 e_L' n + ||n||^2
    deltaSq = 2.0 * dot_eL_n + nNorm2;
    actualDeltaSq = gsNorm2 - localNorm2;

    identityErr = norm(deltaSq(:) - actualDeltaSq(:)) / ...
        max(1.0e-14, norm(actualDeltaSq(:)));

    fprintf("Relative identity error = %.3e\n", identityErr);
    fprintf("Positive deltaSq means GS nonlocal increases residual error ");
    fprintf("relative to GS-local component.\n\n");

    cosAlign = dot_eL_n ./ max(1.0e-14, localErrNorm .* nonlocalNorm);

    harmfulFlag = deltaSq > 0;

    % ---------------------------------------------------------------------
    % Window summary.
    % ---------------------------------------------------------------------
    windowSummaryTable = buildWindowSummaryStep09I5( ...
        time, selectedWatchers, windowEdges, ...
        localErrNorm, gsErrNorm, nonlocalNorm, ...
        dot_eL_n, cosAlign, deltaSq, harmfulFlag);

    disp(windowSummaryTable);

    % ---------------------------------------------------------------------
    % Branch-level summary over the main problematic interval.
    % ---------------------------------------------------------------------
    branchSummaryTable = buildBranchSummaryStep09I5( ...
        resGS, dTrue, selectedWatchers, [800 1400]);

    fprintf("\nBranch contribution summary over 800--1400 s:\n");
    disp(branchSummaryTable);

    % ---------------------------------------------------------------------
    % Plots.
    % ---------------------------------------------------------------------
    if makePlots
        plotAlignmentDiagnosticsStep09I5( ...
            time, selectedWatchers, watcherLabel, ...
            localErrNorm, gsErrNorm, nonlocalNorm, ...
            dot_eL_n, cosAlign, deltaSq, harmfulFlag);

        plotBranchContributionNormsStep09I5( ...
            time, resGS.dnnResidualBranchContrib, selectedWatchers, watcherLabel);
    end

    % ---------------------------------------------------------------------
    % Output.
    % ---------------------------------------------------------------------
    diagOut = struct();

    diagOut.selectedWatchers = selectedWatchers;
    diagOut.windowEdges = windowEdges;
    diagOut.componentMismatch = compMismatch;
    diagOut.identityErr = identityErr;

    diagOut.windowSummaryTable = windowSummaryTable;
    diagOut.branchSummaryTable = branchSummaryTable;

    diagOut.timeSeries = struct();
    diagOut.timeSeries.time = time;
    diagOut.timeSeries.localErrNorm = localErrNorm;
    diagOut.timeSeries.gsErrNorm = gsErrNorm;
    diagOut.timeSeries.nonlocalNorm = nonlocalNorm;
    diagOut.timeSeries.dot_eL_n = dot_eL_n;
    diagOut.timeSeries.cosAlign = cosAlign;
    diagOut.timeSeries.deltaSq = deltaSq;
    diagOut.timeSeries.harmfulFlag = harmfulFlag;

end

function selectedWatchers = resolveWatcherSelectionStep09I5(watcherID, Nw)
%RESOLVEWATCHERSELECTIONSTEP09I5 Convert watcherID input to numeric indices.

    if ischar(watcherID) || isstring(watcherID)
        watcherID = string(watcherID);

        if watcherID == "all"
            selectedWatchers = 1:Nw;
        else
            error("Step09I5:BadWatcherID", ...
                "watcherID string must be ""all"".");
        end

    else
        selectedWatchers = watcherID(:).';

        if any(selectedWatchers < 1) || any(selectedWatchers > Nw)
            error("Step09I5:BadWatcherID", ...
                "watcherID must be between 1 and Nw.");
        end
    end

end

function dTrueOut = expandTrueResidualStep09I5(dTrueIn, dim, N, Nw)
%EXPANDTRUERESIDUALSTEP09I5 Convert true residual to dim x N x Nw.

    sz = size(dTrueIn);

    if numel(sz) == 2
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09I5:BadTrueResidualSize", ...
                "trueResidual must be dim x N or dim x N x Nw.");
        end

        dTrueOut = repmat(reshape(dTrueIn, dim, N, 1), 1, 1, Nw);
        return;
    end

    if numel(sz) == 3
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09I5:BadTrueResidualSize", ...
                "trueResidual has incompatible first two dimensions.");
        end

        if sz(3) == 1
            dTrueOut = repmat(dTrueIn, 1, 1, Nw);
        elseif sz(3) == Nw
            dTrueOut = dTrueIn;
        else
            error("Step09I5:BadTrueResidualSize", ...
                "trueResidual third dimension must be 1 or Nw.");
        end

        return;
    end

    error("Step09I5:BadTrueResidualSize", ...
        "trueResidual must be dim x N or dim x N x Nw.");

end

function nrm = vecNormOverDimStep09I5(x)
%VECNORMOVERDIMSTEP09I5 Compute norm over first dimension of dim x N x Nw.

    nrm = squeeze(sqrt(sum(x.^2, 1)));

end

function tbl = buildWindowSummaryStep09I5( ...
    time, selectedWatchers, windowEdges, ...
    localErrNorm, gsErrNorm, nonlocalNorm, ...
    dot_eL_n, cosAlign, deltaSq, harmfulFlag)
%BUILDWINDOWSUMMARYSTEP09I5 Summarize alignment metrics by time windows.

    nWindow = numel(windowEdges) - 1;

    windowName = strings(nWindow, 1);
    tStart_s = NaN(nWindow, 1);
    tEnd_s = NaN(nWindow, 1);

    meanLocalComponentErr = NaN(nWindow, 1);
    meanGSErr = NaN(nWindow, 1);
    meanGSErrorMinusLocal = NaN(nWindow, 1);
    meanNonlocalNorm = NaN(nWindow, 1);

    meanDot_eL_n = NaN(nWindow, 1);
    meanCosAlign = NaN(nWindow, 1);
    meanDeltaSq = NaN(nWindow, 1);
    fracHarmfulPct = NaN(nWindow, 1);

    for k = 1:nWindow

        t0 = windowEdges(k);
        t1 = windowEdges(k+1);

        idxT = (time >= t0) & (time < t1);

        if k == nWindow
            idxT = (time >= t0) & (time <= t1);
        end

        windowName(k) = sprintf("%.0f-%.0f s", t0, t1);
        tStart_s(k) = t0;
        tEnd_s(k) = t1;

        localVals = localErrNorm(idxT, selectedWatchers);
        gsVals = gsErrNorm(idxT, selectedWatchers);
        nVals = nonlocalNorm(idxT, selectedWatchers);

        dotVals = dot_eL_n(idxT, selectedWatchers);
        cosVals = cosAlign(idxT, selectedWatchers);
        deltaVals = deltaSq(idxT, selectedWatchers);
        harmfulVals = harmfulFlag(idxT, selectedWatchers);

        meanLocalComponentErr(k) = mean(localVals(:), "omitnan");
        meanGSErr(k) = mean(gsVals(:), "omitnan");
        meanGSErrorMinusLocal(k) = mean(gsVals(:) - localVals(:), "omitnan");
        meanNonlocalNorm(k) = mean(nVals(:), "omitnan");

        meanDot_eL_n(k) = mean(dotVals(:), "omitnan");
        meanCosAlign(k) = mean(cosVals(:), "omitnan");
        meanDeltaSq(k) = mean(deltaVals(:), "omitnan");
        fracHarmfulPct(k) = 100.0 * mean(harmfulVals(:), "omitnan");

    end

    tbl = table( ...
        windowName, ...
        tStart_s, ...
        tEnd_s, ...
        meanLocalComponentErr, ...
        meanGSErr, ...
        meanGSErrorMinusLocal, ...
        meanNonlocalNorm, ...
        meanDot_eL_n, ...
        meanCosAlign, ...
        meanDeltaSq, ...
        fracHarmfulPct);

end

function tbl = buildBranchSummaryStep09I5( ...
    resGS, dTrue, selectedWatchers, mainWindow)
%BUILDBRANCHSUMMARYSTEP09I5 Branch-level contribution alignment summary.
%
% This helps identify which branch contribution is large or misaligned in
% the problematic interval.

    time = resGS.time(:).';

    branchContrib = resGS.dnnResidualBranchContrib;
    [dim, Nbranch, ~, Nw] = size(branchContrib);

    idxT = (time >= mainWindow(1)) & (time <= mainWindow(2));

    sourceBranchID = [];
    watcherID = [];
    meanContributionNorm = [];
    meanDotWithLocalError = [];
    meanCosWithLocalError = [];
    fracPositiveDotPct = [];

    dLocalComp = resGS.dnnResidualLocalComponent;
    eLocal = dLocalComp - dTrue;

    for iWatcher = selectedWatchers

        eLocal_i = eLocal(:, idxT, iWatcher);

        for jBranch = 1:Nbranch

            if jBranch == iWatcher
                % Skip the local branch in this branch-level nonlocal table.
                continue;
            end

            c_j = squeeze(branchContrib(:, jBranch, idxT, iWatcher));

            if dim == 1
                c_j = reshape(c_j, 1, []);
            end

            e_i = squeeze(eLocal_i);

            if dim == 1
                e_i = reshape(e_i, 1, []);
            end

            cNorm = sqrt(sum(c_j.^2, 1));
            eNorm = sqrt(sum(e_i.^2, 1));
            dotVals = sum(e_i .* c_j, 1);

            cosVals = dotVals ./ max(1.0e-14, eNorm .* cNorm);

            sourceBranchID(end+1, 1) = jBranch; %#ok<AGROW>
            watcherID(end+1, 1) = iWatcher; %#ok<AGROW>
            meanContributionNorm(end+1, 1) = mean(cNorm(:), "omitnan"); %#ok<AGROW>
            meanDotWithLocalError(end+1, 1) = mean(dotVals(:), "omitnan"); %#ok<AGROW>
            meanCosWithLocalError(end+1, 1) = mean(cosVals(:), "omitnan"); %#ok<AGROW>
            fracPositiveDotPct(end+1, 1) = 100.0 * mean(dotVals(:) > 0, "omitnan"); %#ok<AGROW>

        end
    end

    tbl = table( ...
        watcherID, ...
        sourceBranchID, ...
        meanContributionNorm, ...
        meanDotWithLocalError, ...
        meanCosWithLocalError, ...
        fracPositiveDotPct);

end

function plotAlignmentDiagnosticsStep09I5( ...
    time, selectedWatchers, watcherLabel, ...
    localErrNorm, gsErrNorm, nonlocalNorm, ...
    dot_eL_n, cosAlign, deltaSq, harmfulFlag)
%PLOTALIGNMENTDIAGNOSTICSSTEP09I5 Plot aggregate alignment diagnostics.

    localMean = mean(localErrNorm(:, selectedWatchers), 2, "omitnan");
    gsMean = mean(gsErrNorm(:, selectedWatchers), 2, "omitnan");
    nonlocalMean = mean(nonlocalNorm(:, selectedWatchers), 2, "omitnan");

    dotMean = mean(dot_eL_n(:, selectedWatchers), 2, "omitnan");
    cosMean = mean(cosAlign(:, selectedWatchers), 2, "omitnan");
    deltaMean = mean(deltaSq(:, selectedWatchers), 2, "omitnan");
    harmfulPct = 100.0 * mean(harmfulFlag(:, selectedWatchers), 2, "omitnan");

    figure("Name", "Step 09-I.5 GS branch alignment diagnostic");

    tiledlayout(4, 1);

    nexttile;
    plot(time, localMean, "LineWidth", 1.2);
    hold on;
    plot(time, gsMean, "LineWidth", 1.2);
    plot(time, nonlocalMean, "LineWidth", 1.0);
    xline(800, "--");
    xline(1400, "--");
    grid on;
    xlabel("Time [s]");
    ylabel("Residual norm");
    title("Residual error and nonlocal contribution norm: " + watcherLabel);
    legend("GS-local component error", "GS composite error", ...
        "nonlocal sum norm", "Location", "best");

    nexttile;
    plot(time, dotMean, "LineWidth", 1.2);
    xline(800, "--");
    xline(1400, "--");
    yline(0, "-");
    grid on;
    xlabel("Time [s]");
    ylabel("mean e_L^T n");
    title("Alignment dot product: positive means nonlocal tends to align with local error");

    nexttile;
    plot(time, cosMean, "LineWidth", 1.2);
    xline(800, "--");
    xline(1400, "--");
    yline(0, "-");
    grid on;
    xlabel("Time [s]");
    ylabel("mean cos alignment");
    title("Cosine alignment between local error and nonlocal contribution");

    nexttile;
    plot(time, deltaMean, "LineWidth", 1.2);
    hold on;
    plot(time, harmfulPct / 100.0 * max(abs(deltaMean), [], "omitnan"), ...
        "LineWidth", 1.0);
    xline(800, "--");
    xline(1400, "--");
    yline(0, "-");
    grid on;
    xlabel("Time [s]");
    ylabel("mean deltaSq");
    title("deltaSq = 2 e_L^T n + ||n||^2; positive means GS worse than GS-local");
    legend("mean deltaSq", "scaled harmful fraction", "Location", "best");

end

function plotBranchContributionNormsStep09I5( ...
    time, branchContrib, selectedWatchers, watcherLabel)
%PLOTBRANCHCONTRIBUTIONNORMSSTEP09I5 Plot branch contribution magnitudes.

    [~, Nbranch, ~, ~] = size(branchContrib);

    figure("Name", "Step 09-I.5 branch contribution norms");

    hold on;

    for jBranch = 1:Nbranch

        c_j = squeeze(branchContrib(:, jBranch, :, selectedWatchers));

        if ndims(c_j) == 2
            cNorm = sqrt(sum(c_j.^2, 1)).';
        else
            cNorm = squeeze(sqrt(sum(c_j.^2, 1)));
            cNorm = mean(cNorm, 2, "omitnan");
        end

        plot(time, cNorm, "LineWidth", 1.0);

    end

    xline(800, "--");
    xline(1400, "--");

    grid on;
    xlabel("Time [s]");
    ylabel("Mean branch contribution norm");
    title("Branch contribution norms: " + watcherLabel);
    legend(compose("branch %d", 1:Nbranch), "Location", "best");

end