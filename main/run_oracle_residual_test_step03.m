%{
Script:
    run_oracle_residual_test_step03.m

Purpose:
    Run an oracle residual prediction test.

    This script compares:

    1) Step 02 physical EKF:
        Truth:
            dot v = a_nom + d_true

        Estimator:
            dot v_hat = a_nom

    2) Step 03 oracle residual EKF:
        Truth:
            dot v = a_nom + d_true

        Estimator:
            dot v_hat = a_nom + betaOracle * d_true(t, eta_hat)

    The oracle residual is not a realizable estimator because it uses the
    hidden truth residual function inside the filter prediction.

    The purpose is to answer:

        If the residual were known perfectly, would tracking improve?

Inputs:
    None directly.

Outputs:
    oracleSweep - structure containing oracle residual test results.

Notes:
    - This script uses simulateLocalDNNEKF.m because the oracle residual
      source switch was added inside DNN_EKF_Predict_Local.m.
    - For fair comparison, theta initialization is forced to zero.
    - betaOracle = 0 should match the Step 02 physical EKF baseline.
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
% Settings
% -------------------------------------------------------------------------
rngSeed = 1;

% betaOracle = 0 should recover physical EKF prediction.
% betaOracle = 1 means full oracle residual compensation.
betaList = [0.0, 0.25, 0.50, 0.75, 1.00];
nBeta = numel(betaList);

% -------------------------------------------------------------------------
% Step 02 baseline: residual truth + physical EKF
% -------------------------------------------------------------------------
rng(rngSeed);

cfg02 = config_step02_residual_physical_EKF();

results02 = simulatePhysicalEKF(cfg02);
metrics02 = computeOraclePhysicalRMSE(results02, cfg02);

baselineMeanPosRMSE = metrics02.meanPosRMSE;
baselineMeanVelRMSE = metrics02.meanVelRMSE;

% -------------------------------------------------------------------------
% Allocate logs
% -------------------------------------------------------------------------
meanPosRMSE = NaN(nBeta,1);
meanVelRMSE = NaN(nBeta,1);

posRMSE = NaN(cfg02.Nw, nBeta);
velRMSE = NaN(cfg02.Nw, nBeta);

resultsCell = cell(nBeta,1);
cfgCell = cell(nBeta,1);

% -------------------------------------------------------------------------
% Run oracle residual tests
% -------------------------------------------------------------------------
for b = 1:nBeta

    betaOracle = betaList(b);

    rng(rngSeed);

    cfgOracle = config_step03_local_DNN_EKF();

    % Fair initialization.
    %
    % With the initLocalDNNEKF.m fix, theta0_std = 0 prevents randn from
    % being called for theta initialization.
    cfgOracle.dnn.theta0_std = 0.0;

    % Use oracle residual in prediction.
    cfgOracle.dnn.useResidualInPrediction = true;
    cfgOracle.dnn.predictionResidualSource = "oracle";
    cfgOracle.dnn.residualInjectionGain = betaOracle;

    % The oracle residual does not depend on theta. Adaptive Qtheta is
    % irrelevant for this test, so disable it for cleaner diagnostics.
    cfgOracle.dnn.adaptQThetaEnabled = false;

    fprintf("\n============================================================\n");
    fprintf(" Running oracle residual EKF with betaOracle = %.3f\n", betaOracle);
    fprintf("============================================================\n");

    resultsOracle = simulateLocalDNNEKF(cfgOracle);

    metricsOracle = computeOraclePhysicalRMSE(resultsOracle, cfgOracle);

    meanPosRMSE(b) = metricsOracle.meanPosRMSE;
    meanVelRMSE(b) = metricsOracle.meanVelRMSE;

    posRMSE(:,b) = metricsOracle.posRMSE;
    velRMSE(:,b) = metricsOracle.velRMSE;

    resultsCell{b} = resultsOracle;
    cfgCell{b} = cfgOracle;

end

% -------------------------------------------------------------------------
% Build output structure
% -------------------------------------------------------------------------
oracleSweep = struct();

oracleSweep.rngSeed = rngSeed;
oracleSweep.betaList = betaList(:);

oracleSweep.baseline.meanPosRMSE = baselineMeanPosRMSE;
oracleSweep.baseline.meanVelRMSE = baselineMeanVelRMSE;
oracleSweep.baseline.posRMSE = metrics02.posRMSE;
oracleSweep.baseline.velRMSE = metrics02.velRMSE;

oracleSweep.meanPosRMSE = meanPosRMSE;
oracleSweep.meanVelRMSE = meanVelRMSE;
oracleSweep.posRMSE = posRMSE;
oracleSweep.velRMSE = velRMSE;

oracleSweep.resultsCell = resultsCell;
oracleSweep.cfgCell = cfgCell;

% -------------------------------------------------------------------------
% Print summary
% -------------------------------------------------------------------------
[bestPosRMSE, bestIdx] = min(meanPosRMSE);
bestBeta = betaList(bestIdx);

fprintf("\n");
fprintf("============================================================\n");
fprintf(" Oracle Residual Prediction Test Summary\n");
fprintf("============================================================\n");
fprintf("Random seed: %d\n", rngSeed);
fprintf("Baseline Step 02 mean position RMSE: %.6f\n", baselineMeanPosRMSE);
fprintf("Baseline Step 02 mean velocity RMSE: %.6f\n", baselineMeanVelRMSE);
fprintf("\n");

fprintf("%12s %16s %16s %16s %16s\n", ...
    "betaOracle", "MeanPosRMSE", "PosRatio", "MeanVelRMSE", "VelRatio");

for b = 1:nBeta

    posRatio = safeRatioOracle(meanPosRMSE(b), baselineMeanPosRMSE);
    velRatio = safeRatioOracle(meanVelRMSE(b), baselineMeanVelRMSE);

    fprintf("%12.3f %16.6f %16.6f %16.6f %16.6f\n", ...
        betaList(b), meanPosRMSE(b), posRatio, meanVelRMSE(b), velRatio);

end

fprintf("\nBest betaOracle by mean position RMSE: %.3f\n", bestBeta);
fprintf("Best mean position RMSE: %.6f\n", bestPosRMSE);
fprintf("Best ratio against Step 02: %.6f\n", ...
    safeRatioOracle(bestPosRMSE, baselineMeanPosRMSE));
fprintf("============================================================\n\n");

% -------------------------------------------------------------------------
% Sanity check: betaOracle = 0 should match Step 02
% -------------------------------------------------------------------------
idxZero = find(abs(betaList) < 1e-12, 1, "first");

if ~isempty(idxZero)
    ratioZero = safeRatioOracle(meanPosRMSE(idxZero), baselineMeanPosRMSE);

    if abs(ratioZero - 1) > 1e-8
        warning("betaOracle = 0 does not match Step 02 baseline. Check random stream or oracle implementation.");
    else
        fprintf("Sanity check passed: betaOracle = 0 matches Step 02 baseline.\n\n");
    end
end

% -------------------------------------------------------------------------
% Plots
% -------------------------------------------------------------------------
figure;
hold on;
grid on;

plot(betaList, meanPosRMSE, "-o", "LineWidth", 1.5, ...
    "DisplayName", "Oracle residual EKF");

yline(baselineMeanPosRMSE, "--", "Step 02 baseline", ...
    "LineWidth", 1.2, "DisplayName", "Step 02 baseline");

xlabel("\beta_{oracle}");
ylabel("Mean Position RMSE");
title("Oracle Residual Prediction Test: Position RMSE");
legend("Location", "best");

figure;
hold on;
grid on;

plot(betaList, meanVelRMSE, "-o", "LineWidth", 1.5, ...
    "DisplayName", "Oracle residual EKF");

yline(baselineMeanVelRMSE, "--", "Step 02 baseline", ...
    "LineWidth", 1.2, "DisplayName", "Step 02 baseline");

xlabel("\beta_{oracle}");
ylabel("Mean Velocity RMSE");
title("Oracle Residual Prediction Test: Velocity RMSE");
legend("Location", "best");

figure;
hold on;
grid on;

for i = 1:cfg02.Nw
    plot(betaList, posRMSE(i,:), "-o", "LineWidth", 1.2, ...
        "DisplayName", "Watcher " + string(i));
end

xlabel("\beta_{oracle}");
ylabel("Position RMSE");
title("Per-Watcher Position RMSE: Oracle Residual Prediction Test");
legend("Location", "best");

% -------------------------------------------------------------------------
% Local helper functions
% -------------------------------------------------------------------------
function metrics = computeOraclePhysicalRMSE(results, cfg)
%{
Function:
    computeOraclePhysicalRMSE

Purpose:
    Compute physical target tracking RMSE without printing diagnostics.
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

function r = safeRatioOracle(a, b)
% Compute a/b while avoiding division by zero.

    if abs(b) < eps
        r = NaN;
    else
        r = a / b;
    end

end