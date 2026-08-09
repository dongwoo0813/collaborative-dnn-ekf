function diag = run_step09J5c_residual_truth_magnitude(out09j5, resultField, windowEdges, makePlots)
%{
File:
    main/run_step09J5c_residual_truth_magnitude.m

Purpose:
    Plot and summarize the magnitude of the true residual acceleration used
    in Step 09-J.5.

Motivation:
    We want to check whether the residual approximation error is genuinely
    small relative to the true residual, or whether the error only appears
    small because the true residual itself is very small.

Inputs:
    out09j5     - Output from run_step09J5_compare_bearing_fim_gated_MLP(...)

    resultField - Optional result field used to extract truth trajectory.
                  Default: "resGSFIM".
                  Other useful options:
                      "resGSAdd"
                      "resLocal"
                      "resOracle"

    windowEdges - Optional diagnostic window edges in seconds.
                  Default: [0 800 900 1200 1400 1500 2000].

    makePlots   - Optional logical.
                  Default: true.

Outputs:
    diag - Structure containing:
              time
              dTrue
              trueResidualNorm
              additive/FIM residual approximation errors, when available
              summaryTable
              windowSummaryTable
              figure handle

What this diagnostic checks:
    1. Magnitude of true residual:
           ||d_true(t)||

    2. Approximation error compared with true residual:
           ||d_hat_add(t) - d_true(t)|| / ||d_true(t)||

           ||d_hat_fim(t) - d_true(t)|| / ||d_true(t)||

    3. Whether residual approximation error reduction is meaningful or only
       caused by the residual truth itself being small.

Important:
    This helper first tries to find an already-logged true residual in the
    simulation output. If not found, it tries to recompute the true residual
    from the truth trajectory using common residual-truth function names.

    If your project uses a different residual truth function name, add it to
    candidateFunctionNames inside evaluateTrueResidualFromEta_step09j5c().
%}

    if nargin < 1 || isempty(out09j5)
        error("run_step09J5c_residual_truth_magnitude:MissingInput", ...
            "Input out09j5 is required.");
    end

    if nargin < 2 || isempty(resultField)
        resultField = "resGSFIM";
    end

    if nargin < 3 || isempty(windowEdges)
        windowEdges = [0 800 900 1200 1400 1500 2000];
    end

    if nargin < 4 || isempty(makePlots)
        makePlots = true;
    end

    resultField = string(resultField);

    if ~isfield(out09j5, resultField)
        error("run_step09J5c_residual_truth_magnitude:MissingResultField", ...
            "out09j5.%s is missing.", resultField);
    end

    res = out09j5.(resultField);
    time = extractTime_step09j5c(res);

    residualFamily = "unknown";
    if isfield(out09j5, "residualFamily")
        residualFamily = string(out09j5.residualFamily);
    end

    cfg = extractCfg_step09j5c(out09j5, res);

    % ---------------------------------------------------------------------
    % Extract or recompute true residual acceleration.
    % ---------------------------------------------------------------------
    [dTrue, truthSource] = extractOrComputeTrueResidual_step09j5c( ...
        out09j5, res, time, cfg, residualFamily);

    trueResidualNorm = vecnorm(dTrue, 2, 2);

    % ---------------------------------------------------------------------
    % Pull residual approximation errors from existing Step 09-D.3 residual
    % diagnostics, when available.
    % ---------------------------------------------------------------------
    hasAddDiag = isfield(out09j5, "residualDiagAdd");
    hasFIMDiag = isfield(out09j5, "residualDiagFIM");

    addOperationalErr = NaN(size(time));
    addSameInputErr = NaN(size(time));
    fimOperationalErr = NaN(size(time));
    fimSameInputErr = NaN(size(time));

    if hasAddDiag
        [tAdd, addOperationalErr, addSameInputErr] = ...
            getGSResidualApproximationError_step09j5c(out09j5.residualDiagAdd);
        assertSameTime_step09j5c(time, tAdd, "additive residual diagnostic");
    end

    if hasFIMDiag
        [tFim, fimOperationalErr, fimSameInputErr] = ...
            getGSResidualApproximationError_step09j5c(out09j5.residualDiagFIM);
        assertSameTime_step09j5c(time, tFim, "FIM residual diagnostic");
    end

    % ---------------------------------------------------------------------
    % Relative errors.
    %
    % We use a small denominator floor so the ratio does not explode when
    % d_true is nearly zero. The chosen floor is reported.
    % ---------------------------------------------------------------------
    medianTruthNorm = medianOmitNaN_step09j5c(trueResidualNorm);
    denomFloor = max(1e-12, 1e-3 * medianTruthNorm);
    denom = max(trueResidualNorm, denomFloor);

    addOperationalRelErr = addOperationalErr ./ denom;
    addSameInputRelErr = addSameInputErr ./ denom;

    fimOperationalRelErr = fimOperationalErr ./ denom;
    fimSameInputRelErr = fimSameInputErr ./ denom;

    % ---------------------------------------------------------------------
    % Summary tables.
    % ---------------------------------------------------------------------
    summaryTable = buildSignalSummaryTable_step09j5c( ...
        trueResidualNorm, ...
        addOperationalErr, ...
        fimOperationalErr, ...
        addOperationalRelErr, ...
        fimOperationalRelErr);

    windowSummaryTable = buildResidualTruthWindowSummary_step09j5c( ...
        time, ...
        windowEdges, ...
        trueResidualNorm, ...
        addOperationalErr, ...
        fimOperationalErr, ...
        addOperationalRelErr, ...
        fimOperationalRelErr);

    % ---------------------------------------------------------------------
    % Output.
    % ---------------------------------------------------------------------
    diag = struct();
    diag.time = time;
    diag.dTrue = dTrue;
    diag.truthSource = truthSource;
    diag.trueResidualNorm = trueResidualNorm;
    diag.denomFloor = denomFloor;

    diag.addOperationalErr = addOperationalErr;
    diag.addSameInputErr = addSameInputErr;
    diag.fimOperationalErr = fimOperationalErr;
    diag.fimSameInputErr = fimSameInputErr;

    diag.addOperationalRelErr = addOperationalRelErr;
    diag.addSameInputRelErr = addSameInputRelErr;
    diag.fimOperationalRelErr = fimOperationalRelErr;
    diag.fimSameInputRelErr = fimSameInputRelErr;

    diag.summaryTable = summaryTable;
    diag.windowSummaryTable = windowSummaryTable;
    diag.figTruthMagnitude = [];

    if makePlots
        diag.figTruthMagnitude = plotResidualTruthMagnitude_step09j5c( ...
            diag, residualFamily, resultField);
    end

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-J.5c: residual truth magnitude diagnostic\n");
    fprintf("============================================================\n");
    fprintf("Residual family       = %s\n", residualFamily);
    fprintf("Result field          = %s\n", resultField);
    fprintf("True residual source  = %s\n", truthSource);
    fprintf("Denominator floor     = %.6e\n", denomFloor);
    fprintf("\nSummary:\n");
    disp(summaryTable);

    fprintf("\nWindow summary:\n");
    disp(windowSummaryTable);

end

function time = extractTime_step09j5c(res)
%EXTRACTTIME_STEP09J5C Extract simulation time vector.

    if isfield(res, "time")
        time = res.time(:);
        return;
    end

    if isfield(res, "t")
        time = res.t(:);
        return;
    end

    error("run_step09J5c_residual_truth_magnitude:MissingTime", ...
        "Could not find res.time or res.t.");
end

function cfg = extractCfg_step09j5c(out09j5, res)
%EXTRACTCFG_STEP09J5C Try to recover simulation configuration.

    cfg = [];

    candidateFieldsOut = [
        "cfgGSFIM"
        "cfgFIM"
        "cfgGSAdd"
        "cfgAdd"
        "cfgLocal"
        "cfg"
    ];

    for k = 1:numel(candidateFieldsOut)
        f = candidateFieldsOut(k);
        if isfield(out09j5, f)
            cfg = out09j5.(f);
            return;
        end
    end

    if isfield(res, "cfg")
        cfg = res.cfg;
        return;
    end
end

function [dTrue, truthSource] = extractOrComputeTrueResidual_step09j5c( ...
    out09j5, res, time, cfg, residualFamily)
%EXTRACTORCOMPUTETRUERESIDUAL_STEP09J5C Extract or recompute d_true(t).

    [dTrue, truthSource, ok] = tryExtractLoggedTrueResidual_step09j5c(res, time);

    if ok
        return;
    end

    [etaTrue, etaSource] = extractEtaTrue_step09j5c(res, time, cfg);

    [dTrue, evalSource] = evaluateTrueResidualFromEta_step09j5c( ...
        etaTrue, time, cfg, residualFamily);

    truthSource = sprintf("%s using %s", evalSource, etaSource);

    if isempty(dTrue)
        error("run_step09J5c_residual_truth_magnitude:FailedTruthResidual", ...
            "Could not extract or recompute true residual.");
    end

    dTrue = orientSignalWithTime_step09j5c(dTrue, time, "dTrue");
end

function [dTrue, truthSource, ok] = tryExtractLoggedTrueResidual_step09j5c(res, time)
%TRYEXTRACTLOGGEDTRUERESIDUAL_STEP09J5C Search common logged truth fields.

    ok = false;
    dTrue = [];
    truthSource = "";

    candidateFields = [
        "dTrue"
        "dTruth"
        "trueResidual"
        "truthResidual"
        "residualTruth"
        "aResidualTrue"
        "aTruthResidual"
        "trueResidualAcc"
        "truthResidualAcc"
        "residualTruthAcc"
        "dTrueLog"
        "trueResidualLog"
        "truthResidualLog"
        "residualTruthLog"
    ];

    for k = 1:numel(candidateFields)
        f = candidateFields(k);

        if isfield(res, f)
            candidate = res.(f);
            candidate = reducePossibleWatcherDimension_step09j5c(candidate);
            candidate = orientSignalWithTime_step09j5c(candidate, time, f);

            if ~isempty(candidate)
                dTrue = candidate;
                truthSource = sprintf("logged field res.%s", f);
                ok = true;
                return;
            end
        end
    end

    if isfield(res, "truth") && isstruct(res.truth)
        for k = 1:numel(candidateFields)
            f = candidateFields(k);

            if isfield(res.truth, f)
                candidate = res.truth.(f);
                candidate = reducePossibleWatcherDimension_step09j5c(candidate);
                candidate = orientSignalWithTime_step09j5c(candidate, time, "res.truth." + f);

                if ~isempty(candidate)
                    dTrue = candidate;
                    truthSource = sprintf("logged field res.truth.%s", f);
                    ok = true;
                    return;
                end
            end
        end
    end
end

function X = reducePossibleWatcherDimension_step09j5c(X)
%REDUCEPOSSIBLEWATCHERDIMENSION_STEP09J5C Convert N x dim x Nw to N x dim.

    if isnumeric(X) && ndims(X) == 3
        % If the third dimension is watcher/source index, average over it.
        % For true residual, all watchers should see the same truth. If not,
        % this still gives a useful magnitude diagnostic.
        X = mean(X, 3, "omitnan");
    end
end

function [etaTrue, etaSource] = extractEtaTrue_step09j5c(res, time, cfg)
%EXTRACTETATRUE_STEP09J5C Try to recover eta_true = [r; v].

    dim = inferDim_step09j5c(cfg, res);

    candidateFields = [
        "etaTrue"
        "truthEta"
        "etaTruth"
        "xEtaTrue"
        "xTrueEta"
        "trueEta"
    ];

    for k = 1:numel(candidateFields)
        f = candidateFields(k);

        if isfield(res, f)
            etaTrue = orientSignalWithTime_step09j5c(res.(f), time, f);
            etaSource = sprintf("res.%s", f);
            return;
        end
    end

    if isfield(res, "truth") && isstruct(res.truth)
        for k = 1:numel(candidateFields)
            f = candidateFields(k);

            if isfield(res.truth, f)
                etaTrue = orientSignalWithTime_step09j5c(res.truth.(f), time, "res.truth." + f);
                etaSource = sprintf("res.truth.%s", f);
                return;
            end
        end
    end

    % Try position/velocity pairs.
    rFields = [
        "rTrue"
        "truthR"
        "rTruth"
        "truePosition"
        "trueTargetPos"
        "targetTruePos"
    ];

    vFields = [
        "vTrue"
        "truthV"
        "vTruth"
        "trueVelocity"
        "trueTargetVel"
        "targetTrueVel"
    ];

    for ir = 1:numel(rFields)
        rf = rFields(ir);

        if ~isfield(res, rf)
            continue;
        end

        rTrue = orientSignalWithTime_step09j5c(res.(rf), time, rf);

        for iv = 1:numel(vFields)
            vf = vFields(iv);

            if isfield(res, vf)
                vTrue = orientSignalWithTime_step09j5c(res.(vf), time, vf);
                etaTrue = [rTrue(:,1:dim), vTrue(:,1:dim)];
                etaSource = sprintf("res.%s + res.%s", rf, vf);
                return;
            end
        end
    end

    % Try full truth state. Assume first [r; v] block.
    fullStateFields = [
        "xTrue"
        "truthState"
        "xTruth"
        "stateTrue"
        "trueState"
    ];

    for k = 1:numel(fullStateFields)
        f = fullStateFields(k);

        if isfield(res, f)
            X = orientSignalWithTime_step09j5c(res.(f), time, f);

            if size(X, 2) >= 2*dim
                etaTrue = X(:, 1:(2*dim));
                etaSource = sprintf("first 2*dim columns of res.%s", f);
                return;
            end
        end
    end

    if isfield(res, "truth") && isstruct(res.truth)
        for k = 1:numel(fullStateFields)
            f = fullStateFields(k);

            if isfield(res.truth, f)
                X = orientSignalWithTime_step09j5c(res.truth.(f), time, "res.truth." + f);

                if size(X, 2) >= 2*dim
                    etaTrue = X(:, 1:(2*dim));
                    etaSource = sprintf("first 2*dim columns of res.truth.%s", f);
                    return;
                end
            end
        end
    end

    error("run_step09J5c_residual_truth_magnitude:MissingEtaTrue", ...
        "Could not find eta_true or truth position/velocity in the result structure.");
end

function dim = inferDim_step09j5c(cfg, res)
%INFERDIM_STEP09J5C Infer physical dimension.

    dim = 2;

    if ~isempty(cfg) && isfield(cfg, "dim")
        dim = cfg.dim;
        return;
    end

    if isfield(res, "dim")
        dim = res.dim;
        return;
    end
end

function X = orientSignalWithTime_step09j5c(X, time, fieldName)
%ORIENTSIGNALWITHTIME_STEP09J5C Ensure signal is N x nDim.

    X = double(X);
    N = numel(time);

    if isempty(X)
        return;
    end

    if isvector(X)
        X = X(:);
        return;
    end

    if size(X, 1) == N
        return;
    end

    if size(X, 2) == N
        X = X.';
        return;
    end

    error("run_step09J5c_residual_truth_magnitude:BadSignalSize", ...
        "Field %s has size [%s], which does not match time length N=%d.", ...
        string(fieldName), num2str(size(X)), N);
end

function [dTrue, evalSource] = evaluateTrueResidualFromEta_step09j5c( ...
    etaTrue, time, cfg, residualFamily)
%EVALUATETRUERESIDUALFROMETA_STEP09J5C Recompute true residual from eta_true.
%
% Add your project-specific true residual function name to this list if the
% auto-detection does not find it.

    candidateFunctionNames = [
        "evaluateTruthResidualModel"
        "evaluateResidualTruthModel"
        "evaluateTruthResidual"
        "evaluateTrueResidual"
        "computeTruthResidual"
        "computeResidualTruth"
        "truthResidualModel"
        "trueResidualModel"
        "trueResidualAcceleration"
        "truthResidualAcceleration"
        "evaluateTrueResidualAcceleration"
        "evaluateResidualTruthAcceleration"
        "feedback_sat_disturbance"
        "feedbackSatDisturbanceResidual"
    ];

    N = numel(time);
    dTrue = [];
    evalSource = "";

    for k = 1:numel(candidateFunctionNames)
        fname = candidateFunctionNames(k);

        if exist(fname, "file") ~= 2 && exist(fname, "builtin") ~= 5
            continue;
        end

        fun = str2func(fname);

        [y0, signatureID, ok] = tryEvaluateOneResidual_step09j5c( ...
            fun, etaTrue(1,:).', time(1), cfg, residualFamily);

        if ~ok
            continue;
        end

        y0 = y0(:);
        dimOut = numel(y0);
        dCandidate = NaN(N, dimOut);
        dCandidate(1, :) = y0.';

        success = true;

        for i = 2:N
            try
                yi = callResidualWithSignature_step09j5c( ...
                    fun, signatureID, etaTrue(i,:).', time(i), cfg, residualFamily);
                dCandidate(i, :) = yi(:).';
            catch
                success = false;
                break;
            end
        end

        if success
            dTrue = dCandidate;
            evalSource = sprintf("%s(), signature %d", fname, signatureID);
            return;
        end
    end

    error("run_step09J5c_residual_truth_magnitude:NoTruthResidualFunction", ...
        ["Could not find a compatible true residual function. ", ...
         "Add your function name to candidateFunctionNames in ", ...
         "evaluateTrueResidualFromEta_step09j5c()."]);
end

function [y, signatureID, ok] = tryEvaluateOneResidual_step09j5c( ...
    fun, eta, t, cfg, residualFamily)
%TRYEVALUATEONERESIDUAL_STEP09J5C Try common residual function signatures.

    ok = false;
    y = [];
    signatureID = NaN;

    for s = 1:10
        try
            yTest = callResidualWithSignature_step09j5c( ...
                fun, s, eta, t, cfg, residualFamily);

            if isnumeric(yTest) && ~isempty(yTest)
                y = yTest(:);
                signatureID = s;
                ok = true;
                return;
            end
        catch
        end
    end
end

function y = callResidualWithSignature_step09j5c( ...
    fun, signatureID, eta, t, cfg, residualFamily)
%CALLRESIDUALWITHSIGNATURE_STEP09J5C Common residual function signatures.

    switch signatureID
        case 1
            y = fun(eta, t, cfg);
        case 2
            y = fun(t, eta, cfg);
        case 3
            y = fun(eta, t, residualFamily, cfg);
        case 4
            y = fun(residualFamily, eta, t, cfg);
        case 5
            y = fun(eta, cfg, t);
        case 6
            y = fun(eta, t, residualFamily);
        case 7
            y = fun(eta, t);
        case 8
            y = fun(t, eta);
        case 9
            y = fun(eta, cfg);
        case 10
            y = fun(eta);
        otherwise
            error("Unknown residual signature.");
    end
end

function [time, operationalErr, sameInputErr] = ...
    getGSResidualApproximationError_step09j5c(residualDiag)
%GETGSRESIDUALAPPROXIMATIONERROR_STEP09J5C Extract GS composite errors.

    requiredFields = [
        "overallSummaryTable"
        "operationalErrCell"
        "sameInputErrCell"
        "timeCell"
    ];

    for k = 1:numel(requiredFields)
        f = requiredFields(k);

        if ~isfield(residualDiag, f)
            error("run_step09J5c_residual_truth_magnitude:BadResidualDiag", ...
                "residualDiag.%s is missing.", f);
        end
    end

    T = residualDiag.overallSummaryTable;

    if ~ismember("caseName", string(T.Properties.VariableNames))
        error("run_step09J5c_residual_truth_magnitude:MissingCaseName", ...
            "residualDiag.overallSummaryTable.caseName is missing.");
    end

    idx = find(string(T.caseName) == "GS composite");

    if numel(idx) ~= 1
        error("run_step09J5c_residual_truth_magnitude:MissingGSComposite", ...
            "Could not find exactly one GS composite row.");
    end

    time = residualDiag.timeCell{idx};
    operationalRaw = residualDiag.operationalErrCell{idx};
    sameInputRaw = residualDiag.sameInputErrCell{idx};

    time = time(:);

    % Raw diagnostic is usually N x Nw. Average across watchers.
    operationalErr = mean(operationalRaw, 2, "omitnan");
    sameInputErr = mean(sameInputRaw, 2, "omitnan");

    operationalErr = operationalErr(:);
    sameInputErr = sameInputErr(:);
end

function assertSameTime_step09j5c(tRef, tOther, label)
%ASSERTSAMETIME_STEP09J5C Verify time histories are compatible.

    if numel(tRef) ~= numel(tOther)
        error("run_step09J5c_residual_truth_magnitude:TimeLengthMismatch", ...
            "Time length mismatch for %s.", label);
    end

    err = norm(tRef(:) - tOther(:), inf);

    if err > 1e-12
        error("run_step09J5c_residual_truth_magnitude:TimeMismatch", ...
            "Time vector mismatch for %s. Max error = %.3e.", label, err);
    end
end

function T = buildSignalSummaryTable_step09j5c( ...
    trueNorm, addErr, fimErr, addRelErr, fimRelErr)
%BUILDSIGNALSUMMARYTABLE_STEP09J5C Build overall signal summary.

    signalName = [
        "||d_true||"
        "additive residual error"
        "FIM residual error"
        "additive error / ||d_true||"
        "FIM error / ||d_true||"
    ];

    signals = {
        trueNorm
        addErr
        fimErr
        addRelErr
        fimRelErr
    };

    meanValue = NaN(numel(signals), 1);
    rmsValue = NaN(numel(signals), 1);
    medianValue = NaN(numel(signals), 1);
    p95Value = NaN(numel(signals), 1);
    maxValue = NaN(numel(signals), 1);

    for i = 1:numel(signals)
        x = signals{i};
        meanValue(i) = mean(x, "omitnan");
        rmsValue(i) = sqrt(mean(x.^2, "omitnan"));
        medianValue(i) = medianOmitNaN_step09j5c(x);
        p95Value(i) = percentileOmitNaN_step09j5c(x, 95);
        maxValue(i) = max(x, [], "omitnan");
    end

    T = table(signalName, meanValue, rmsValue, medianValue, p95Value, maxValue);
end

function T = buildResidualTruthWindowSummary_step09j5c( ...
    time, windowEdges, trueNorm, addErr, fimErr, addRelErr, fimRelErr)
%BUILDRESIDUALTRUTHWINDOWSUMMARY_STEP09J5C Window summary.

    nWin = numel(windowEdges) - 1;

    windowName = strings(nWin, 1);
    tStart_s = zeros(nWin, 1);
    tEnd_s = zeros(nWin, 1);

    meanTrueNorm = NaN(nWin, 1);
    p95TrueNorm = NaN(nWin, 1);

    meanAddErr = NaN(nWin, 1);
    meanFIMErr = NaN(nWin, 1);

    meanAddErrOverTruth = NaN(nWin, 1);
    meanFIMErrOverTruth = NaN(nWin, 1);

    addErrPctOfMeanTruth = NaN(nWin, 1);
    fimErrPctOfMeanTruth = NaN(nWin, 1);

    for w = 1:nWin
        t0 = windowEdges(w);
        t1 = windowEdges(w+1);

        idx = time >= t0 & time < t1;

        if w == nWin
            idx = time >= t0 & time <= t1;
        end

        windowName(w) = sprintf("%g-%g s", t0, t1);
        tStart_s(w) = t0;
        tEnd_s(w) = t1;

        meanTrueNorm(w) = mean(trueNorm(idx), "omitnan");
        p95TrueNorm(w) = percentileOmitNaN_step09j5c(trueNorm(idx), 95);

        meanAddErr(w) = mean(addErr(idx), "omitnan");
        meanFIMErr(w) = mean(fimErr(idx), "omitnan");

        meanAddErrOverTruth(w) = mean(addRelErr(idx), "omitnan");
        meanFIMErrOverTruth(w) = mean(fimRelErr(idx), "omitnan");

        addErrPctOfMeanTruth(w) = 100 * meanAddErr(w) / max(meanTrueNorm(w), eps);
        fimErrPctOfMeanTruth(w) = 100 * meanFIMErr(w) / max(meanTrueNorm(w), eps);
    end

    T = table( ...
        windowName, ...
        tStart_s, ...
        tEnd_s, ...
        meanTrueNorm, ...
        p95TrueNorm, ...
        meanAddErr, ...
        meanFIMErr, ...
        meanAddErrOverTruth, ...
        meanFIMErrOverTruth, ...
        addErrPctOfMeanTruth, ...
        fimErrPctOfMeanTruth);
end

function fig = plotResidualTruthMagnitude_step09j5c(diag, residualFamily, resultField)
%PLOTRESIDUALTRUTHMAGNITUDE_STEP09J5C Plot truth and relative errors.

    time = diag.time;
    dTrue = diag.dTrue;

    fig = figure( ...
        "Name", "Step 09-J.5c Residual Truth Magnitude", ...
        "Color", "w");

    tiledlayout(fig, 3, 1, ...
        "TileSpacing", "compact", ...
        "Padding", "compact");

    % ---------------------------------------------------------------------
    % 1. True residual components and norm.
    % ---------------------------------------------------------------------
    nexttile;
    plot(time, dTrue, "LineWidth", 1.0);
    hold on;
    plot(time, diag.trueResidualNorm, "k", "LineWidth", 1.3);
    addWindowMarkers_step09j5c();
    grid on;
    xlabel("Time [s]");
    ylabel("True residual acceleration");
    title("True residual components and norm");
    legendEntries = compose("d_{true,%d}", 1:size(dTrue,2));
    legend([legendEntries, "||d_{true}||"], "Location", "best");

    % ---------------------------------------------------------------------
    % 2. Residual truth magnitude vs approximation error.
    % ---------------------------------------------------------------------
    nexttile;
    plot(time, diag.trueResidualNorm, "LineWidth", 1.3);
    hold on;
    plot(time, diag.addOperationalErr, "LineWidth", 1.1);
    plot(time, diag.fimOperationalErr, "LineWidth", 1.1);
    addWindowMarkers_step09j5c();
    grid on;
    xlabel("Time [s]");
    ylabel("Magnitude");
    title("Truth magnitude vs approximation error");
    legend( ...
        "||d_{true}||", ...
        "Additive: ||d_hat - d_true||", ...
        "FIM-gated: ||d_hat - d_true||", ...
        "Location", "best");

    % ---------------------------------------------------------------------
    % 3. Approximation error normalized by true residual magnitude.
    % ---------------------------------------------------------------------
    nexttile;
    plot(time, diag.addOperationalRelErr, "LineWidth", 1.1);
    hold on;
    plot(time, diag.fimOperationalRelErr, "LineWidth", 1.1);
    yline(1.0, "--", "LineWidth", 1.0);
    addWindowMarkers_step09j5c();
    grid on;
    xlabel("Time [s]");
    ylabel("Error / ||d_{true}||");
    title(sprintf("Relative residual approximation error, denom floor = %.2e", ...
        diag.denomFloor));
    legend( ...
        "Additive", ...
        "FIM-gated", ...
        "error = truth magnitude", ...
        "Location", "best");

    sgtitle(sprintf( ...
        "Step 09-J.5c residual truth magnitude, residual=%s, result=%s", ...
        string(residualFamily), string(resultField)));
end

function addWindowMarkers_step09j5c()
%ADDWINDOWMARKERS_STEP09J5C Add standard Step 09-J window markers.

    edges = [800 900 1200 1400 1500];

    yl = ylim;

    for k = 1:numel(edges)
        xline(edges(k), ":", "LineWidth", 0.8);
    end

    ylim(yl);
end

function m = medianOmitNaN_step09j5c(x)
%MEDIANOMITNAN_STEP09J5C Median without relying on toolbox behavior.

    x = x(:);
    x = x(isfinite(x));

    if isempty(x)
        m = NaN;
    else
        m = median(x);
    end
end

function p = percentileOmitNaN_step09j5c(x, q)
%PERCENTILEOMITNAN_STEP09J5C Simple percentile without prctile dependency.

    x = x(:);
    x = x(isfinite(x));

    if isempty(x)
        p = NaN;
        return;
    end

    x = sort(x);
    n = numel(x);

    if n == 1
        p = x;
        return;
    end

    pos = 1 + (q/100) * (n - 1);
    lo = floor(pos);
    hi = ceil(pos);

    if lo == hi
        p = x(lo);
    else
        alpha = pos - lo;
        p = (1 - alpha) * x(lo) + alpha * x(hi);
    end
end