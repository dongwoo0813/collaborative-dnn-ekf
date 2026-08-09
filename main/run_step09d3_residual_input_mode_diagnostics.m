function out = run_step09d3_residual_input_mode_diagnostics(outCase, windowTable)
%{
File:
    main/run_step09d3_residual_input_mode_diagnostics.m

Purpose:
    Step 09-D.3 residual input-mode diagnostic.

    Compare two residual-acceleration error definitions:

        1) Operational residual error:
           || d_hat(eta_hat) - d_true(eta_true) ||

        2) Same-input residual error:
           || d_hat(eta_true) - d_true(eta_true) ||

Why this step matters:
    The operational error is what the EKF actually uses during prediction.
    The same-input error is closer to pure residual-function learning quality.

    If GS improves same-input error, then GS is learning the residual
    function better.

    If GS improves operational/tracking error but not same-input error,
    then the benefit is coming from state-estimator interaction, correction
    direction, covariance effects, or trajectory-dependent behavior rather
    than pointwise residual-function accuracy.

Required result fields:
    results.dnnResidual
        DNN residual evaluated at eta_hat during the simulation.

    results.dnnResidualAtTrueEta
        DNN residual evaluated at eta_true using the same learned parameters.

    results.trueResidual
        True residual acceleration evaluated along the truth trajectory.

Usage:
    out09d3 = run_step09d3_residual_input_mode_diagnostics(out09d2);
    disp(out09d3.overallSummaryTable);
    disp(out09d3.windowSummaryTable);
%}

    if nargin < 1 || isempty(outCase)
        error("Step09D3:MissingInput", ...
            "Pass an output structure such as out09d2.");
    end

    if nargin < 2 || isempty(windowTable)
        windowTable = table( ...
            ["0-800 s"; "800-900 s"; "900-1200 s"; "1200-1500 s"; "1500-2000 s"], ...
            [0; 800; 900; 1200; 1500], ...
            [800; 900; 1200; 1500; 2000], ...
            'VariableNames', {'windowName', 'tStart_s', 'tEnd_s'});
    end

    caseName = [
        "Local DNN"
        "GS composite"
        "Oracle"
    ];

    resCell = {
        outCase.resLocal
        outCase.resGS
        outCase.resOracle
    };

    cfgCell = {
        outCase.cfgLocal
        outCase.cfgGS
        outCase.cfgOracle
    };

    nCase = numel(caseName);

    operationalErrCell = cell(nCase, 1);
    sameInputErrCell = cell(nCase, 1);
    timeCell = cell(nCase, 1);

    meanOperationalErr = NaN(nCase, 1);
    meanSameInputErr = NaN(nCase, 1);
    meanInputPenalty = NaN(nCase, 1);

    rmsOperationalErr = NaN(nCase, 1);
    rmsSameInputErr = NaN(nCase, 1);

    p95OperationalErr = NaN(nCase, 1);
    p95SameInputErr = NaN(nCase, 1);

    for ic = 1:nCase

        [operationalErr, sameInputErr, time] = computeResidualErrors_step09d3( ...
            resCell{ic}, cfgCell{ic});

        operationalErrCell{ic} = operationalErr;
        sameInputErrCell{ic} = sameInputErr;
        timeCell{ic} = time;

        opVec = operationalErr(:);
        sameVec = sameInputErr(:);

        meanOperationalErr(ic) = mean(opVec, "omitnan");
        meanSameInputErr(ic) = mean(sameVec, "omitnan");
        meanInputPenalty(ic) = mean(opVec - sameVec, "omitnan");

        rmsOperationalErr(ic) = sqrt(mean(opVec.^2, "omitnan"));
        rmsSameInputErr(ic) = sqrt(mean(sameVec.^2, "omitnan"));

        p95OperationalErr(ic) = percentile_step09d3(opVec, 95);
        p95SameInputErr(ic) = percentile_step09d3(sameVec, 95);

    end

    overallSummaryTable = table( ...
        caseName, ...
        meanOperationalErr, ...
        meanSameInputErr, ...
        meanInputPenalty, ...
        rmsOperationalErr, ...
        rmsSameInputErr, ...
        p95OperationalErr, ...
        p95SameInputErr, ...
        'VariableNames', { ...
            'caseName', ...
            'meanOperationalErr', ...
            'meanSameInputErr', ...
            'meanInputPenalty_OperationalMinusSameInput', ...
            'rmsOperationalErr', ...
            'rmsSameInputErr', ...
            'p95OperationalErr', ...
            'p95SameInputErr'});

    % ---------------------------------------------------------------------
    % Window summary
    % ---------------------------------------------------------------------
    nWin = height(windowTable);
    nRow = nCase * nWin;

    windowCaseName = strings(nRow, 1);
    windowName = strings(nRow, 1);
    tStart_s = NaN(nRow, 1);
    tEnd_s = NaN(nRow, 1);

    meanOperationalErrWin = NaN(nRow, 1);
    meanSameInputErrWin = NaN(nRow, 1);
    meanInputPenaltyWin = NaN(nRow, 1);

    rmsOperationalErrWin = NaN(nRow, 1);
    rmsSameInputErrWin = NaN(nRow, 1);

    p95OperationalErrWin = NaN(nRow, 1);
    p95SameInputErrWin = NaN(nRow, 1);

    row = 0;

    for ic = 1:nCase

        time = timeCell{ic};
        operationalErr = operationalErrCell{ic};
        sameInputErr = sameInputErrCell{ic};

        for iw = 1:nWin

            row = row + 1;

            idx = time >= windowTable.tStart_s(iw) & ...
                  time <  windowTable.tEnd_s(iw);

            opVec = operationalErr(idx, :);
            sameVec = sameInputErr(idx, :);

            opVec = opVec(:);
            sameVec = sameVec(:);

            windowCaseName(row) = caseName(ic);
            windowName(row) = string(windowTable.windowName(iw));
            tStart_s(row) = windowTable.tStart_s(iw);
            tEnd_s(row) = windowTable.tEnd_s(iw);

            meanOperationalErrWin(row) = mean(opVec, "omitnan");
            meanSameInputErrWin(row) = mean(sameVec, "omitnan");
            meanInputPenaltyWin(row) = mean(opVec - sameVec, "omitnan");

            rmsOperationalErrWin(row) = sqrt(mean(opVec.^2, "omitnan"));
            rmsSameInputErrWin(row) = sqrt(mean(sameVec.^2, "omitnan"));

            p95OperationalErrWin(row) = percentile_step09d3(opVec, 95);
            p95SameInputErrWin(row) = percentile_step09d3(sameVec, 95);

        end

    end

    windowSummaryTable = table( ...
        windowCaseName, ...
        windowName, ...
        tStart_s, ...
        tEnd_s, ...
        meanOperationalErrWin, ...
        meanSameInputErrWin, ...
        meanInputPenaltyWin, ...
        rmsOperationalErrWin, ...
        rmsSameInputErrWin, ...
        p95OperationalErrWin, ...
        p95SameInputErrWin, ...
        'VariableNames', { ...
            'caseName', ...
            'windowName', ...
            'tStart_s', ...
            'tEnd_s', ...
            'meanOperationalErr', ...
            'meanSameInputErr', ...
            'meanInputPenalty_OperationalMinusSameInput', ...
            'rmsOperationalErr', ...
            'rmsSameInputErr', ...
            'p95OperationalErr', ...
            'p95SameInputErr'});

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-D.3: Residual input-mode diagnostics\n");
    fprintf("============================================================\n");
    disp(overallSummaryTable);

    fprintf("\nWindow summary:\n");
    disp(windowSummaryTable);

    % ---------------------------------------------------------------------
    % Plot mean residual error over watchers
    % ---------------------------------------------------------------------
    time = timeCell{1};

    meanLocalOperational = mean(operationalErrCell{1}, 2, "omitnan");
    meanLocalSameInput = mean(sameInputErrCell{1}, 2, "omitnan");

    meanGSOperational = mean(operationalErrCell{2}, 2, "omitnan");
    meanGSSameInput = mean(sameInputErrCell{2}, 2, "omitnan");

    figure;
    plot(time, meanLocalOperational, "LineWidth", 1.3);
    hold on;
    plot(time, meanLocalSameInput, "LineWidth", 1.3);
    plot(time, meanGSOperational, "LineWidth", 1.3);
    plot(time, meanGSSameInput, "LineWidth", 1.3);
    grid on;

    xlabel("Time [s]");
    ylabel("Residual acceleration error norm");
    title("Step 09-D.3 residual error: operational vs same-input");
    legend( ...
        "Local: dHat(etaHat) vs dTrue(etaTrue)", ...
        "Local: dHat(etaTrue) vs dTrue(etaTrue)", ...
        "GS: dHat(etaHat) vs dTrue(etaTrue)", ...
        "GS: dHat(etaTrue) vs dTrue(etaTrue)", ...
        "Location", "best");

    figure;
    plot(time, meanLocalOperational - meanLocalSameInput, "LineWidth", 1.3);
    hold on;
    plot(time, meanGSOperational - meanGSSameInput, "LineWidth", 1.3);
    yline(0.0, "--", "LineWidth", 1.1);
    grid on;

    xlabel("Time [s]");
    ylabel("Operational error - same-input error");
    title("Step 09-D.3 input-state penalty in residual evaluation");
    legend("Local DNN", "GS composite", "zero", "Location", "best");

    out = struct();
    out.outCase = outCase;
    out.windowTable = windowTable;
    out.overallSummaryTable = overallSummaryTable;
    out.windowSummaryTable = windowSummaryTable;

    out.operationalErrCell = operationalErrCell;
    out.sameInputErrCell = sameInputErrCell;
    out.timeCell = timeCell;

end

function [operationalErr, sameInputErr, time] = computeResidualErrors_step09d3(results, cfg)
%COMPUTERESIDUALERRORS_STEP09D3 Compute operational and same-input errors.

    if ~isfield(results, "time")
        error("Step09D3:MissingTime", "results.time is missing.");
    end

    if ~isfield(results, "dnnResidual")
        error("Step09D3:MissingDNNResidual", ...
            "results.dnnResidual is missing.");
    end

    if ~isfield(results, "trueResidual")
        error("Step09D3:MissingTrueResidual", ...
            "results.trueResidual is missing.");
    end

    time = results.time(:);

    dOperational = results.dnnResidual;
    [dim, N, Nw] = size(dOperational);

    dTrue = expandTrueResidual_step09d3(results.trueResidual, dim, N, Nw);

    residualSource = "unknown";

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "predictionResidualSource")
        residualSource = string(cfg.dnn.predictionResidualSource);
    end

    if residualSource == "oracle"
        dOperational = dTrue;
        dSameInput = dTrue;
    else
        if ~isfield(results, "dnnResidualAtTrueEta")
            error("Step09D3:MissingSameInputLog", ...
                "results.dnnResidualAtTrueEta is missing. Add Step 09-D.3 logging and rerun the simulation.");
        end

        dSameInput = results.dnnResidualAtTrueEta;

        if ~isequal(size(dSameInput), size(dOperational))
            error("Step09D3:BadSameInputLogSize", ...
                "dnnResidualAtTrueEta must have the same size as dnnResidual.");
        end
    end

    operationalErr = reshape(sqrt(sum((dOperational - dTrue).^2, 1)), N, Nw);
    sameInputErr = reshape(sqrt(sum((dSameInput - dTrue).^2, 1)), N, Nw);

end

function dTrueOut = expandTrueResidual_step09d3(dTrueIn, dim, N, Nw)
%EXPANDTRUERESIDUAL_STEP09D3 Convert true residual to dim x N x Nw.

    sz = size(dTrueIn);

    if numel(sz) == 2
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09D3:BadTrueResidualSize", ...
                "trueResidual must be dim x N or dim x N x Nw.");
        end

        dTrueOut = reshape(dTrueIn, dim, N, 1);
        dTrueOut = repmat(dTrueOut, 1, 1, Nw);
        return;
    end

    if numel(sz) == 3
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09D3:BadTrueResidualSize", ...
                "trueResidual first two dimensions are incompatible.");
        end

        if sz(3) == Nw
            dTrueOut = dTrueIn;
        elseif sz(3) == 1
            dTrueOut = repmat(dTrueIn, 1, 1, Nw);
        else
            error("Step09D3:BadTrueResidualSize", ...
                "trueResidual third dimension must be 1 or Nw.");
        end

        return;
    end

    error("Step09D3:BadTrueResidualSize", ...
        "trueResidual must be dim x N or dim x N x Nw.");

end

function p = percentile_step09d3(x, pct)
%PERCENTILE_STEP09D3 Small percentile helper without requiring prctile.

    x = x(:);
    x = x(isfinite(x));

    if isempty(x)
        p = NaN;
        return;
    end

    x = sort(x);
    q = pct / 100;

    idx = 1 + q * (numel(x) - 1);
    idxLow = floor(idx);
    idxHigh = ceil(idx);

    if idxLow == idxHigh
        p = x(idxLow);
    else
        w = idx - idxLow;
        p = (1 - w) * x(idxLow) + w * x(idxHigh);
    end

end