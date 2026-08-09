function diag = run_step09J6_estimate_norm_cosine_diagnostic(out09j5, windowEdges, makePlots)
%{
File:
    main/run_step09J6_estimate_norm_cosine_diagnostic.m

Purpose:
    Compare additive and bearing-FIM-gated residual estimates by magnitude
    and direction. The diagnostic uses the operational residual logged at
    each watcher's estimated state, not the same-input truth-state log.

Inputs:
    out09j5     - Output from run_step09J5_compare_bearing_fim_gated_MLP.
    windowEdges - Optional window edges [s].
    makePlots   - Optional logical flag. Default: true.

Outputs:
    diag.dTrue                  dim x N truth residual history
    diag.dHatAdd/FIM            dim x N x Nw operational estimates
    diag.meanNorm*              N x 1 watcher-mean norm histories
    diag.meanCosine*            N x 1 watcher-mean cosine histories
    diag.windowSummaryTable     window-level magnitude/direction summary

Notes:
    normRatio = ||d_hat|| / max(||d_true||, truthNormFloor).
    cosine is NaN if either vector norm is below its reported floor.
%}

    if nargin < 1 || isempty(out09j5)
        error("run_step09J6_estimate_norm_cosine_diagnostic:MissingInput", ...
            "out09j5 is required.");
    end
    if nargin < 2 || isempty(windowEdges)
        windowEdges = [0 800 900 1200 1400 1500 2000];
    end
    if nargin < 3 || isempty(makePlots)
        makePlots = true;
    end

    required = ["resGSAdd", "resGSFIM"];
    for k = 1:numel(required)
        if ~isfield(out09j5, required(k))
            error("run_step09J6_estimate_norm_cosine_diagnostic:MissingResult", ...
                "out09j5.%s is missing.", required(k));
        end
    end

    add = out09j5.resGSAdd;
    fim = out09j5.resGSFIM;
    weightedLabel = "bearing-FIM-gated";
    if isfield(out09j5,"cfgGSFIM") && ...
            string(out09j5.cfgGSFIM.gs.compositeMode)== ...
            "output_information_fusion"
        weightedLabel = "output-information-fusion";
    elseif isfield(out09j5,"cfgGSFIM") && ...
            string(out09j5.cfgGSFIM.gs.compositeMode)== ...
            "fim_weighted_additive"
        weightedLabel = "FIM-weighted-additive";
    end
    requireResultFields(add, "additive");
    requireResultFields(fim, "bearing-FIM-gated");

    time = add.time(:);
    assertSameTime(time, fim.time(:));
    dTrue = orientTruth(add.trueResidual, numel(time));
    dTrueFim = orientTruth(fim.trueResidual, numel(time));
    if norm(dTrue(:) - dTrueFim(:), inf) > 1e-12
        error("run_step09J6_estimate_norm_cosine_diagnostic:TruthMismatch", ...
            "Additive and FIM runs do not share the same truth residual.");
    end

    dHatAdd = orientEstimate(add.dnnResidual, size(dTrue,1), numel(time));
    dHatFIM = orientEstimate(fim.dnnResidual, size(dTrue,1), numel(time));
    if size(dHatAdd,3) ~= size(dHatFIM,3)
        error("run_step09J6_estimate_norm_cosine_diagnostic:WatcherMismatch", ...
            "Additive and FIM runs have different watcher counts.");
    end

    trueNorm = vecnorm(dTrue, 2, 1).';
    truthNormFloor = max(1e-12, 1e-3 * median(trueNorm, "omitnan"));
    estimateNormFloor = truthNormFloor;
    [normAdd, ratioAdd, cosineAdd] = metricsByWatcher( ...
        dHatAdd, dTrue, truthNormFloor, estimateNormFloor);
    [normFIM, ratioFIM, cosineFIM] = metricsByWatcher( ...
        dHatFIM, dTrue, truthNormFloor, estimateNormFloor);

    diag = struct();
    diag.time = time;
    diag.dTrue = dTrue;
    diag.dHatAdd = dHatAdd;
    diag.dHatFIM = dHatFIM;
    diag.trueNorm = trueNorm;
    diag.normAdd = normAdd;
    diag.normFIM = normFIM;
    diag.normRatioAdd = ratioAdd;
    diag.normRatioFIM = ratioFIM;
    diag.cosineAdd = cosineAdd;
    diag.cosineFIM = cosineFIM;
    diag.meanNormAdd = mean(normAdd, 2, "omitnan");
    diag.meanNormFIM = mean(normFIM, 2, "omitnan");
    diag.meanCosineAdd = mean(cosineAdd, 2, "omitnan");
    diag.meanCosineFIM = mean(cosineFIM, 2, "omitnan");
    diag.truthNormFloor = truthNormFloor;
    diag.estimateNormFloor = estimateNormFloor;
    diag.windowSummaryTable = buildWindowTable( ...
        time, windowEdges, trueNorm, ratioAdd, ratioFIM, cosineAdd, cosineFIM);
    diag.figure = [];

    if makePlots
        diag.figure = figure("Name", "Step 09-J.6 estimate norm and cosine", ...
            "Color", "w");
        tiledlayout(2,1, "TileSpacing", "compact");
        nexttile;
        plot(time, trueNorm, "k", "LineWidth", 1.4); hold on;
        plot(time, diag.meanNormAdd, "LineWidth", 1.1);
        plot(time, diag.meanNormFIM, "LineWidth", 1.1);
        grid on; ylabel("residual norm [m/s^2]");
        legend("||d_{true}||", "mean ||d_{add}||", ...
            "mean ||d_{weighted}||", ...
            "Location", "best");
        title("Operational residual magnitude");
        nexttile;
        plot(time, diag.meanCosineAdd, "LineWidth", 1.1); hold on;
        plot(time, diag.meanCosineFIM, "LineWidth", 1.1);
        yline(0, "k:"); ylim([-1.05 1.05]); grid on;
        xlabel("time [s]"); ylabel("cosine with d_{true}");
        legend("additive", weightedLabel, "Location", "best");
        title("Operational residual direction alignment");
    end

    fprintf("\nStep 09-J.6 estimate norm and cosine diagnostic\n");
    fprintf("truth/estimate norm floors = %.6e / %.6e m/s^2\n", ...
        truthNormFloor, estimateNormFloor);
    disp(diag.windowSummaryTable);
end

function requireResultFields(res, label)
    fields = ["time", "trueResidual", "dnnResidual"];
    for k = 1:numel(fields)
        if ~isfield(res, fields(k))
            error("run_step09J6_estimate_norm_cosine_diagnostic:MissingField", ...
                "%s result is missing %s.", label, fields(k));
        end
    end
end

function assertSameTime(a, b)
    if numel(a) ~= numel(b) || norm(a-b, inf) > 1e-12
        error("run_step09J6_estimate_norm_cosine_diagnostic:TimeMismatch", ...
            "Additive and FIM time vectors differ.");
    end
end

function X = orientTruth(X, N)
    X = double(X);
    if size(X,2) == N
        return;
    elseif size(X,1) == N
        X = X.';
    else
        error("run_step09J6_estimate_norm_cosine_diagnostic:BadTruthSize", ...
            "trueResidual does not match the time dimension.");
    end
end

function X = orientEstimate(X, dim, N)
    X = double(X);
    if ndims(X) ~= 3
        error("run_step09J6_estimate_norm_cosine_diagnostic:BadEstimateRank", ...
            "dnnResidual must be dim-by-N-by-Nw.");
    end
    if size(X,1) == dim && size(X,2) == N
        return;
    elseif size(X,1) == N && size(X,2) == dim
        X = permute(X, [2 1 3]);
    else
        error("run_step09J6_estimate_norm_cosine_diagnostic:BadEstimateSize", ...
            "dnnResidual dimensions do not match truth/time.");
    end
end

function [hatNorm, ratio, cosine] = metricsByWatcher( ...
    dHat, dTrue, truthFloor, estimateFloor)
    Nw = size(dHat,3);
    truthNorm = vecnorm(dTrue, 2, 1).';
    hatNorm = zeros(size(dHat,2), Nw);
    cosine = NaN(size(hatNorm));
    for i = 1:Nw
        Di = dHat(:,:,i);
        hatNorm(:,i) = vecnorm(Di, 2, 1).';
        dotProduct = sum(Di .* dTrue, 1).';
        valid = truthNorm > truthFloor & hatNorm(:,i) > estimateFloor;
        cosine(valid,i) = dotProduct(valid) ./ ...
            (truthNorm(valid) .* hatNorm(valid,i));
        cosine(valid,i) = max(-1, min(1, cosine(valid,i)));
    end
    ratio = hatNorm ./ max(truthNorm, truthFloor);
end

function T = buildWindowTable(time, edges, trueNorm, addRatio, fimRatio, addCos, fimCos)
    nWin = numel(edges)-1;
    window = strings(nWin,1);
    meanTrueNorm = NaN(nWin,1);
    meanNormRatioAdd = NaN(nWin,1);
    meanNormRatioFIM = NaN(nWin,1);
    meanCosineAdd = NaN(nWin,1);
    meanCosineFIM = NaN(nWin,1);
    validCosineFractionAdd = NaN(nWin,1);
    validCosineFractionFIM = NaN(nWin,1);
    for w = 1:nWin
        if w < nWin
            idx = time >= edges(w) & time < edges(w+1);
        else
            idx = time >= edges(w) & time <= edges(w+1);
        end
        window(w) = sprintf("%g-%g s", edges(w), edges(w+1));
        meanTrueNorm(w) = mean(trueNorm(idx), "omitnan");
        meanNormRatioAdd(w) = mean(addRatio(idx,:), "all", "omitnan");
        meanNormRatioFIM(w) = mean(fimRatio(idx,:), "all", "omitnan");
        meanCosineAdd(w) = mean(addCos(idx,:), "all", "omitnan");
        meanCosineFIM(w) = mean(fimCos(idx,:), "all", "omitnan");
        validCosineFractionAdd(w) = nnz(isfinite(addCos(idx,:))) / numel(addCos(idx,:));
        validCosineFractionFIM(w) = nnz(isfinite(fimCos(idx,:))) / numel(fimCos(idx,:));
    end
    T = table(window, meanTrueNorm, meanNormRatioAdd, meanNormRatioFIM, ...
        meanCosineAdd, meanCosineFIM, validCosineFractionAdd, validCosineFractionFIM);
end
