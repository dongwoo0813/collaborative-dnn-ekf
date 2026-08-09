function [figSummary, figTime] = plot_step09J5_additive_vs_fim_same_figure(out09j5)
%{
File:
    main/plot_step09J5_additive_vs_fim_same_figure.m

Purpose:
    Plot GS additive MLP and GS bearing-FIM-gated MLP results from the
    Step 09-J.5 output structure.

Inputs:
    out09j5 - Output structure from

        out09j5 = run_step09J5_compare_bearing_fim_gated_MLP(...);

Required fields:
    out09j5.compactTrackingSummary
    out09j5.residualWindowComparison
    out09j5.alignmentWindowComparison
    out09j5.residualDiagAdd
    out09j5.residualDiagFIM

Outputs:
    figSummary - Summary comparison figure:
                    residual window bar plot,
                    harmful fraction,
                    meanDeltaSq,
                    overall tracking metrics.

    figTime    - Time-history comparison figure:
                    additive vs bearing-FIM-gated residual approximation
                    over time.

What this function compares:
    1. GS additive MLP
    2. GS bearing-FIM-gated MLP

Notes:
    This is a visualization helper only. It does not rerun simulation and
    does not modify estimator / GS logic.
%}

    if nargin < 1 || isempty(out09j5)
        error("plot_step09J5_additive_vs_fim_same_figure:MissingInput", ...
            "Input out09j5 is required.");
    end

    requiredFields = [
        "compactTrackingSummary"
        "residualWindowComparison"
        "alignmentWindowComparison"
        "residualDiagAdd"
        "residualDiagFIM"
    ];

    for k = 1:numel(requiredFields)
        f = requiredFields(k);

        if ~isfield(out09j5, f)
            error("plot_step09J5_additive_vs_fim_same_figure:MissingField", ...
                "out09j5.%s is missing.", f);
        end
    end

    Ttrack = out09j5.compactTrackingSummary;
    Tres   = out09j5.residualWindowComparison;
    Talign = out09j5.alignmentWindowComparison;

    % ---------------------------------------------------------------------
    % Separate x-axes because residual and alignment diagnostics may use
    % different window partitions.
    % ---------------------------------------------------------------------
    residualWindowLabels = string(Tres.windowName);
    xRes = categorical(residualWindowLabels);
    xRes = reordercats(xRes, cellstr(residualWindowLabels));

    alignmentWindowLabels = string(Talign.windowName);
    xAlign = categorical(alignmentWindowLabels);
    xAlign = reordercats(xAlign, cellstr(alignmentWindowLabels));

    % ---------------------------------------------------------------------
    % Window-level residual metrics.
    % ---------------------------------------------------------------------
    residualMean = [
        Tres.add_meanOperationalErr(:), ...
        Tres.fim_meanOperationalErr(:)
    ];

    % ---------------------------------------------------------------------
    % Window-level alignment metrics.
    % ---------------------------------------------------------------------
    harmfulFrac = [
        Talign.add_fracHarmfulPct(:), ...
        Talign.fim_fracHarmfulPct(:)
    ];

    meanDeltaSq = [
        Talign.add_meanDeltaSq(:), ...
        Talign.fim_meanDeltaSq(:)
    ];

    % ---------------------------------------------------------------------
    % Overall tracking metrics.
    % ---------------------------------------------------------------------
    idxAdd = string(Ttrack.compositeMode) == "additive";
    idxFim = string(Ttrack.compositeMode) == "bearing_fim_gated";

    if nnz(idxAdd) ~= 1 || nnz(idxFim) ~= 1
        error("plot_step09J5_additive_vs_fim_same_figure:MissingTrackingRows", ...
            "Could not find exactly one additive row and one bearing_fim_gated row.");
    end

    trackingMetricLabels = ["Mean", "RMS", "p95", "Final"];
    xTrack = categorical(trackingMetricLabels);
    xTrack = reordercats(xTrack, cellstr(trackingMetricLabels));

    trackingVals = [
        Ttrack.meanPosErr_m(idxAdd),       Ttrack.meanPosErr_m(idxFim);
        Ttrack.rmsPosErr_m(idxAdd),        Ttrack.rmsPosErr_m(idxFim);
        Ttrack.p95PosErr_m(idxAdd),        Ttrack.p95PosErr_m(idxFim);
        Ttrack.finalMeanPosErr_m(idxAdd),  Ttrack.finalMeanPosErr_m(idxFim)
    ];

    % =====================================================================
    % Figure 1: compact summary comparison.
    % =====================================================================
    figSummary = figure( ...
        "Name", "Step 09-J.5 Additive vs Bearing-FIM-Gated Summary", ...
        "Color", "w");

    tiledlayout(figSummary, 2, 2, ...
        "TileSpacing", "compact", ...
        "Padding", "compact");

    % 1. Residual approximation error by residual diagnostic window.
    nexttile;
    bar(xRes, residualMean);
    grid on;
    ylabel("Mean residual error");
    title("Residual approximation by window");
    legend(["GS additive", "GS FIM-gated"], "Location", "best");
    xtickangle(30);

    % 2. Harmful fraction by alignment diagnostic window.
    nexttile;
    bar(xAlign, harmfulFrac);
    grid on;
    ylabel("Harmful fraction [%]");
    title("Wrong-direction / harmful nonlocal fraction");
    legend(["GS additive", "GS FIM-gated"], "Location", "best");
    xtickangle(30);

    % 3. meanDeltaSq by alignment diagnostic window.
    %
    % Positive meanDeltaSq:
    %   nonlocal contribution worsens residual approximation relative to the
    %   local component.
    %
    % Negative meanDeltaSq:
    %   nonlocal contribution improves residual approximation.
    nexttile;
    bar(xAlign, meanDeltaSq);
    grid on;
    yline(0, "--");
    ylabel("meanDeltaSq");
    title("Mean \Delta = ||e_L+n||^2 - ||e_L||^2");
    legend(["GS additive", "GS FIM-gated"], "Location", "best");
    xtickangle(30);

    % 4. Overall tracking metrics.
    nexttile;
    bar(xTrack, trackingVals);
    grid on;
    ylabel("Position error [m]");
    title("Overall tracking metrics");
    legend(["GS additive", "GS FIM-gated"], "Location", "best");

    sgtitle(sprintf( ...
        "Step 09-J.5: Additive vs Bearing-FIM-Gated, residual=%s, seed=%d, dt=%.3g", ...
        string(out09j5.residualFamily), out09j5.seed, out09j5.dt));

    % =====================================================================
    % Figure 2: time-history residual approximation comparison.
    % =====================================================================
    [tAdd, addOperational, addSameInput] = getGSResidualTimeHistoryStep09J5( ...
        out09j5.residualDiagAdd);

    [tFim, fimOperational, fimSameInput] = getGSResidualTimeHistoryStep09J5( ...
        out09j5.residualDiagFIM);

    assertCompatibleTimeHistoryStep09J5(tAdd, tFim);

    % Positive means FIM is worse. Negative means FIM is better.
    diffOperational_FIMminusAdd = fimOperational - addOperational;
    diffSameInput_FIMminusAdd = fimSameInput - addSameInput;

    figTime = figure( ...
        "Name", "Step 09-J.5 Additive vs Bearing-FIM-Gated Residual Time History", ...
        "Color", "w");

    tiledlayout(figTime, 2, 1, ...
        "TileSpacing", "compact", ...
        "Padding", "compact");

    % ---------------------------------------------------------------------
    % Top panel: raw residual approximation error time histories.
    % ---------------------------------------------------------------------
    nexttile;
    plot(tAdd, addOperational, "LineWidth", 1.2);
    hold on;
    plot(tFim, fimOperational, "LineWidth", 1.2);
    plot(tAdd, addSameInput, "--", "LineWidth", 1.1);
    plot(tFim, fimSameInput, "--", "LineWidth", 1.1);

    addStep09J5WindowMarkers();

    grid on;
    xlabel("Time [s]");
    ylabel("Residual acceleration error norm");
    title("Residual approximation time history");
    legend( ...
        "Additive: operational", ...
        "FIM-gated: operational", ...
        "Additive: same-input", ...
        "FIM-gated: same-input", ...
        "Location", "best");

    % ---------------------------------------------------------------------
    % Bottom panel: direct FIM-minus-additive difference.
    %
    % Interpretation:
    %   < 0 means FIM-gated has smaller residual error than additive.
    %   > 0 means FIM-gated has larger residual error than additive.
    % ---------------------------------------------------------------------
    nexttile;
    plot(tAdd, diffOperational_FIMminusAdd, "LineWidth", 1.2);
    hold on;
    plot(tAdd, diffSameInput_FIMminusAdd, "--", "LineWidth", 1.1);
    yline(0.0, "--", "LineWidth", 1.1);

    addStep09J5WindowMarkers();

    grid on;
    xlabel("Time [s]");
    ylabel("FIM-gated error - additive error");
    title("Negative means bearing-FIM-gated is better");
    legend( ...
        "Operational difference", ...
        "Same-input difference", ...
        "zero", ...
        "Location", "best");

    sgtitle(sprintf( ...
        "Residual approximation time history: additive vs bearing-FIM-gated, residual=%s, seed=%d", ...
        string(out09j5.residualFamily), out09j5.seed));

end

function [time, meanOperational, meanSameInput] = getGSResidualTimeHistoryStep09J5(residualDiag)
%GETGSRESIDUALTIMEHISTORYSTEP09J5 Extract GS composite residual histories.
%
% The residual diagnostic generated by run_step09d3_residual_input_mode_diagnostics
% stores the case order in:
%
%   residualDiag.overallSummaryTable.caseName
%   residualDiag.operationalErrCell
%   residualDiag.sameInputErrCell
%   residualDiag.timeCell
%
% This helper selects the "GS composite" row and averages over watchers.

    requiredFields = [
        "overallSummaryTable"
        "operationalErrCell"
        "sameInputErrCell"
        "timeCell"
    ];

    for k = 1:numel(requiredFields)
        f = requiredFields(k);

        if ~isfield(residualDiag, f)
            error("plot_step09J5_additive_vs_fim_same_figure:BadResidualDiag", ...
                "residualDiag.%s is missing.", f);
        end
    end

    T = residualDiag.overallSummaryTable;

    if ~ismember("caseName", string(T.Properties.VariableNames))
        error("plot_step09J5_additive_vs_fim_same_figure:MissingCaseName", ...
            "residualDiag.overallSummaryTable.caseName is missing.");
    end

    idx = find(string(T.caseName) == "GS composite");

    if numel(idx) ~= 1
        error("plot_step09J5_additive_vs_fim_same_figure:MissingGSComposite", ...
            "Could not find exactly one GS composite row in residual diagnostic.");
    end

    time = residualDiag.timeCell{idx};
    operationalErr = residualDiag.operationalErrCell{idx};
    sameInputErr = residualDiag.sameInputErrCell{idx};

    time = time(:);

    % operationalErr and sameInputErr are N-by-Nw.
    meanOperational = mean(operationalErr, 2, "omitnan");
    meanSameInput = mean(sameInputErr, 2, "omitnan");

    meanOperational = meanOperational(:);
    meanSameInput = meanSameInput(:);

    if numel(time) ~= numel(meanOperational) || numel(time) ~= numel(meanSameInput)
        error("plot_step09J5_additive_vs_fim_same_figure:BadTimeHistorySize", ...
            "Residual time history length mismatch.");
    end

end

function assertCompatibleTimeHistoryStep09J5(tAdd, tFim)
%ASSERTCOMPATIBLETIMEHISTORYSTEP09J5 Check additive/FIM time axes match.

    if numel(tAdd) ~= numel(tFim)
        error("plot_step09J5_additive_vs_fim_same_figure:TimeLengthMismatch", ...
            "Additive and FIM-gated time histories have different lengths.");
    end

    timeErr = norm(tAdd(:) - tFim(:), inf);

    if timeErr > 1e-12
        error("plot_step09J5_additive_vs_fim_same_figure:TimeMismatch", ...
            "Additive and FIM-gated time vectors do not match. Max error = %.3e.", ...
            timeErr);
    end

end

function addStep09J5WindowMarkers()
%ADDSTEP09J5WINDOWMARKERS Add light vertical markers for known windows.
%
% These are diagnostic window boundaries used in Step 09-J.5. They make it
% easier to visually connect the time-history plot to the window-summary
% bar plots.

    windowEdges = [800, 900, 1200, 1400, 1500];

    yl = ylim;

    for k = 1:numel(windowEdges)
        xline(windowEdges(k), ":", "LineWidth", 0.9);
    end

    ylim(yl);

end