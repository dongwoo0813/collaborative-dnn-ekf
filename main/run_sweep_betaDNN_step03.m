%{
Script:
    run_sweep_betaDNN_step03.m

Purpose:
    Sweep the DNN residual injection gain betaDNN for Step 03 local DNN-EKF.

    The Step 03 prediction acceleration is

        a_pred = a_nom + betaDNN * d_i(eta;theta_i).

    This script compares several betaDNN values against the Step 02
    physical EKF baseline under the same random seed.

Inputs:
    None directly.

Outputs:
    sweep - structure containing RMSE and DNN diagnostics for each betaDNN.

Important:
    For a fair comparison, this script forces

        cfg03.dnn.theta0_std = 0.0

    so that theta_hat(0) starts exactly at zero.

    This assumes initLocalDNNEKF.m has already been fixed so that it does
    not call randn when cfg.dnn.theta0_std == 0.
%}

clear; clc; close all;

% -------------------------------------------------------------------------
% Add project root and all subfolders to MATLAB path
% -------------------------------------------------------------------------
thisFile = mfilename("fullpath");
mainDir = fileparts(thisFile);
projectRoot = fileparts(mainDir);

addpath(genpath(projectRoot));
rehash;

% -------------------------------------------------------------------------
% Sweep settings
% -------------------------------------------------------------------------
rngSeed = 1;

% -------------------------------------------------------------------------
% Test residual amplitude
% -------------------------------------------------------------------------
% residualAmp = 1e-4 was too weak: oracle improvement was only about 1%.
% residualAmp = 5e-4 gives about 18% oracle improvement, so it is a more
% meaningful case for testing whether local DNN-EKF can close part of the
% physical-oracle gap.
testResidualAmp = 5e-4;


betaList = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00];
nBeta = numel(betaList);

% -------------------------------------------------------------------------
% Step 02 baseline: residual truth + physical EKF
% -------------------------------------------------------------------------
rng(rngSeed);

cfg02 = config_step02_residual_physical_EKF();

cfg02.truth.useResidual = true;
cfg02.truth.residualModel = "branchwise";
cfg02.truth.residualAmp = testResidualAmp;

results02 = simulatePhysicalEKF(cfg02);
metrics02 = computeSweepPhysicalRMSE(results02, cfg02);

baselineMeanPosRMSE = metrics02.meanPosRMSE;
baselineMeanVelRMSE = metrics02.meanVelRMSE;

% -------------------------------------------------------------------------
% Allocate sweep logs
% -------------------------------------------------------------------------
meanPosRMSE = NaN(nBeta,1);
meanVelRMSE = NaN(nBeta,1);

posRMSE = NaN(cfg02.Nw, nBeta);
velRMSE = NaN(cfg02.Nw, nBeta);

thetaNormFinal = NaN(cfg02.Nw, nBeta);
residualRMSE = NaN(cfg02.Nw, nBeta);
gammaFinal = NaN(cfg02.Nw, nBeta);

% Store results only for the best beta later if desired.
resultsCell = cell(nBeta,1);
cfgCell = cell(nBeta,1);

% -------------------------------------------------------------------------
% Run Step 03 for each betaDNN
% -------------------------------------------------------------------------
for b = 1:nBeta

    betaDNN = betaList(b);

    rng(rngSeed);

    cfg03 = config_step03_local_DNN_EKF();
    
    % Use the same residual scenario as the Step 02 baseline.
    cfg03.truth.useResidual = true;
    cfg03.truth.residualModel = "branchwise";
    cfg03.truth.residualAmp = testResidualAmp;
    
    % Force fair initialization.
    cfg03.dnn.theta0_std = 0.0;
    
    % Local DNN residual prediction, not oracle.
    cfg03.dnn.useResidualInPrediction = true;
    cfg03.dnn.predictionResidualSource = "local_DNN";
    cfg03.dnn.residualInjectionGain = betaDNN;


    fprintf("    cfg03.truth.residualAmp = %.3e\n", cfg03.truth.residualAmp);
    
    if isfield(cfg03.dnn, "predictionResidualSource")
        fprintf("    predictionResidualSource = %s\n", cfg03.dnn.predictionResidualSource);
    else
        fprintf("    predictionResidualSource = local_DNN default\n");
    end

    fprintf("\n============================================================\n");
    fprintf(" Running Step 03 local DNN-EKF with betaDNN = %.3f\n", betaDNN);
    fprintf("============================================================\n");

    results03 = simulateLocalDNNEKF(cfg03);

    metrics03 = computeSweepPhysicalRMSE(results03, cfg03);
    dnnStats = computeSweepDNNDiagnostics(results03, cfg03);

    meanPosRMSE(b) = metrics03.meanPosRMSE;
    meanVelRMSE(b) = metrics03.meanVelRMSE;

    posRMSE(:,b) = metrics03.posRMSE;
    velRMSE(:,b) = metrics03.velRMSE;

    thetaNormFinal(:,b) = dnnStats.thetaNormFinal;
    residualRMSE(:,b) = dnnStats.residualRMSE;
    gammaFinal(:,b) = dnnStats.gammaFinal;

    resultsCell{b} = results03;
    cfgCell{b} = cfg03;

end

% -------------------------------------------------------------------------
% Build output structure
% -------------------------------------------------------------------------
sweep = struct();

sweep.rngSeed = rngSeed;
sweep.betaList = betaList(:);

sweep.baseline.meanPosRMSE = baselineMeanPosRMSE;
sweep.baseline.meanVelRMSE = baselineMeanVelRMSE;
sweep.baseline.posRMSE = metrics02.posRMSE;
sweep.baseline.velRMSE = metrics02.velRMSE;

sweep.meanPosRMSE = meanPosRMSE;
sweep.meanVelRMSE = meanVelRMSE;
sweep.posRMSE = posRMSE;
sweep.velRMSE = velRMSE;

sweep.thetaNormFinal = thetaNormFinal;
sweep.residualRMSE = residualRMSE;
sweep.gammaFinal = gammaFinal;

sweep.resultsCell = resultsCell;
sweep.cfgCell = cfgCell;

% -------------------------------------------------------------------------
% Print summary table
% -------------------------------------------------------------------------
[bestPosRMSE, bestIdx] = min(meanPosRMSE);
bestBeta = betaList(bestIdx);



fprintf("\n");
fprintf("============================================================\n");
fprintf(" betaDNN Sweep Summary\n");
fprintf("============================================================\n");
fprintf("Random seed: %d\n", rngSeed);
fprintf("Test residualAmp: %.3e\n", testResidualAmp);
fprintf("Baseline Step 02 mean position RMSE: %.6f\n", baselineMeanPosRMSE);
fprintf("Baseline Step 02 mean velocity RMSE: %.6f\n", baselineMeanVelRMSE);
fprintf("\n");



fprintf("%10s %16s %16s %16s %16s\n", ...
    "betaDNN", "MeanPosRMSE", "PosRatio", "MeanVelRMSE", "VelRatio");

for b = 1:nBeta

    posRatio = safeRatioLocal(meanPosRMSE(b), baselineMeanPosRMSE);
    velRatio = safeRatioLocal(meanVelRMSE(b), baselineMeanVelRMSE);

    fprintf("%10.3f %16.6f %16.6f %16.6f %16.6f\n", ...
        betaList(b), meanPosRMSE(b), posRatio, meanVelRMSE(b), velRatio);

end

fprintf("\nBest betaDNN by mean position RMSE: %.3f\n", bestBeta);
fprintf("Best mean position RMSE: %.6f\n", bestPosRMSE);
fprintf("Best ratio against Step 02: %.6f\n", ...
    safeRatioLocal(bestPosRMSE, baselineMeanPosRMSE));
fprintf("============================================================\n\n");

% -------------------------------------------------------------------------
% Sanity check
% -------------------------------------------------------------------------
idxBetaZero = find(abs(betaList) < 1e-12, 1, "first");

if ~isempty(idxBetaZero)
    ratioZero = safeRatioLocal(meanPosRMSE(idxBetaZero), baselineMeanPosRMSE);

    if abs(ratioZero - 1) > 1e-8
        warning("betaDNN = 0 does not match Step 02 baseline. Check random stream or betaDNN implementation.");
    else
        fprintf("Sanity check passed: betaDNN = 0 matches Step 02 baseline.\n\n");
    end
end

% -------------------------------------------------------------------------
% Plots
% -------------------------------------------------------------------------

figure;
hold on;
grid on;

plot(betaList, meanPosRMSE, "-o", "LineWidth", 1.5, ...
    "DisplayName", "Step 03 local DNN-EKF");

yline(baselineMeanPosRMSE, "--", "Step 02 baseline", ...
    "LineWidth", 1.2, "DisplayName", "Step 02 baseline");

xlabel("\beta_{DNN}");
ylabel("Mean Position RMSE");
title("Mean Position RMSE vs DNN Residual Injection Gain");
legend("Location", "best");

figure;
hold on;
grid on;

plot(betaList, meanVelRMSE, "-o", "LineWidth", 1.5, ...
    "DisplayName", "Step 03 local DNN-EKF");

yline(baselineMeanVelRMSE, "--", "Step 02 baseline", ...
    "LineWidth", 1.2, "DisplayName", "Step 02 baseline");

xlabel("\beta_{DNN}");
ylabel("Mean Velocity RMSE");
title("Mean Velocity RMSE vs DNN Residual Injection Gain");
legend("Location", "best");

figure;
hold on;
grid on;

for i = 1:cfg02.Nw
    plot(betaList, posRMSE(i,:), "-o", "LineWidth", 1.2, ...
        "DisplayName", "Watcher " + string(i));
end

xlabel("\beta_{DNN}");
ylabel("Position RMSE");
title("Per-Watcher Position RMSE vs DNN Residual Injection Gain");
legend("Location", "best");

figure;
hold on;
grid on;

for i = 1:cfg02.Nw
    plot(betaList, thetaNormFinal(i,:), "-o", "LineWidth", 1.2, ...
        "DisplayName", "Watcher " + string(i));
end

xlabel("\beta_{DNN}");
ylabel("Final ||\theta_i||");
title("Final Local DNN Parameter Norm vs DNN Residual Injection Gain");
legend("Location", "best");

figure;
hold on;
grid on;

for i = 1:cfg02.Nw
    plot(betaList, residualRMSE(i,:), "-o", "LineWidth", 1.2, ...
        "DisplayName", "Watcher " + string(i));
end

xlabel("\beta_{DNN}");
ylabel("Residual RMSE");
title("DNN Residual Learning Diagnostic vs DNN Residual Injection Gain");
legend("Location", "best");

% -------------------------------------------------------------------------
% Local helper functions
% -------------------------------------------------------------------------

function metrics = computeSweepPhysicalRMSE(results, cfg)
%{
Function:
    computeSweepPhysicalRMSE

Purpose:
    Compute physical target tracking RMSE without printing command-window
    diagnostics.

Inputs:
    results - simulation results structure
    cfg     - simulation configuration

Outputs:
    metrics - structure with posRMSE, velRMSE, meanPosRMSE, meanVelRMSE
%}

    dim = cfg.dim;
    nEta = 2 * dim;
    Nw = cfg.Nw;

    if ~isfield(results, "etaTrue")
        error("results.etaTrue does not exist.");
    end

    if ~isfield(results, "xhat")
        error("results.xhat does not exist.");
    end

    etaTrue = results.etaTrue;
    xhat = results.xhat;

    % Expected shape:
    %   etaTrue: nEta x N
    %   xhat:    nEta x N x Nw
    %
    % Add a small amount of defensive shape handling.
    if size(etaTrue,1) ~= nEta && size(etaTrue,2) == nEta
        etaTrue = etaTrue.';
    end

    if size(xhat,1) ~= nEta && size(xhat,2) == nEta
        xhat = permute(xhat, [2,1,3]);
    end

    if size(etaTrue,1) ~= nEta
        error("Unexpected results.etaTrue shape.");
    end

    if size(xhat,1) ~= nEta
        error("Unexpected results.xhat shape.");
    end

    N = min(size(etaTrue,2), size(xhat,2));

    posRMSE = NaN(Nw,1);
    velRMSE = NaN(Nw,1);

    for i = 1:Nw

        err = xhat(:,1:N,i) - etaTrue(:,1:N);

        posErr = err(1:dim,:);
        velErr = err(dim+1:2*dim,:);

        posRMSE(i) = sqrt(mean(sum(posErr.^2, 1)));
        velRMSE(i) = sqrt(mean(sum(velErr.^2, 1)));

    end

    metrics.posRMSE = posRMSE;
    metrics.velRMSE = velRMSE;
    metrics.meanPosRMSE = mean(posRMSE);
    metrics.meanVelRMSE = mean(velRMSE);

end

function dnnStats = computeSweepDNNDiagnostics(results, cfg)
%{
Function:
    computeSweepDNNDiagnostics

Purpose:
    Compute simple DNN diagnostics without plotting.

Outputs:
    dnnStats.thetaNormFinal
    dnnStats.residualRMSE
    dnnStats.gammaFinal
%}

    dim = cfg.dim;
    Nw = cfg.Nw;

    thetaNormFinal = NaN(Nw,1);
    residualRMSE = NaN(Nw,1);
    gammaFinal = NaN(Nw,1);

    % ---------------------------------------------------------------------
    % Final theta norm
    % ---------------------------------------------------------------------
    if isfield(results, "thetaHat")

        thetaHat = results.thetaHat;

        for i = 1:Nw

            if ndims(thetaHat) == 3 && size(thetaHat,3) >= i
                % Expected shape: nTheta x N x Nw
                theta_i = thetaHat(:,end,i);
                thetaNormFinal(i) = norm(theta_i);

            elseif ndims(thetaHat) == 2 && Nw == 1
                theta_i = thetaHat(:,end);
                thetaNormFinal(i) = norm(theta_i);
            end

        end

    end

    % ---------------------------------------------------------------------
    % DNN residual RMSE against true residual
    % ---------------------------------------------------------------------
    if isfield(results, "dnnResidual") && isfield(results, "trueResidual")

        dnnResidual = results.dnnResidual;
        trueResidual = results.trueResidual;

        if size(trueResidual,1) ~= dim && size(trueResidual,2) == dim
            trueResidual = trueResidual.';
        end

        for i = 1:Nw

            if ndims(dnnResidual) == 3 && size(dnnResidual,3) >= i

                % Expected shape: dim x N x Nw
                dHat_i = dnnResidual(:,:,i);

                if size(dHat_i,1) ~= dim && size(dHat_i,2) == dim
                    dHat_i = dHat_i.';
                end

                N = min(size(dHat_i,2), size(trueResidual,2));

                err = dHat_i(:,1:N) - trueResidual(:,1:N);

                residualRMSE(i) = sqrt(mean(sum(err.^2, 1)));

            end

        end

    end

    % ---------------------------------------------------------------------
    % Final gammaTheta
    % ---------------------------------------------------------------------
    if isfield(results, "gammaTheta")

        for i = 1:Nw
            gammaFinal(i) = lastFiniteLocal(results.gammaTheta(:,i));
        end

    end

    dnnStats.thetaNormFinal = thetaNormFinal;
    dnnStats.residualRMSE = residualRMSE;
    dnnStats.gammaFinal = gammaFinal;

end

function value = lastFiniteLocal(x)
% Return the last finite value in a vector.

    x = x(:);
    idx = find(isfinite(x), 1, "last");

    if isempty(idx)
        value = NaN;
    else
        value = x(idx);
    end

end

function r = safeRatioLocal(a, b)
% Compute a/b while avoiding division by zero.

    if abs(b) < eps
        r = NaN;
    else
        r = a / b;
    end

end