%{
Script:
    run_sweep_residualAmp_oracle_step03.m

Purpose:
    Sweep the unknown truth residual amplitude and compare:

    1) Step 02 physical EKF:
        Truth:
            dot v = a_nom + d_true

        Estimator:
            dot v_hat = a_nom

    2) Step 03 oracle residual EKF:
        Truth:
            dot v = a_nom + d_true

        Estimator:
            dot v_hat = a_nom + d_true(t, eta_hat)

    The oracle residual EKF is not realizable. It uses the hidden true
    residual function inside the estimator prediction. This is only an
    upper-bound diagnostic.

Main question:
    Does perfect residual compensation actually improve tracking when the
    residual amplitude is increased?

Inputs:
    None directly.

Outputs:
    resAmpSweep - structure containing sweep results.

Assumptions:
    - DNN_EKF_Predict_Local.m supports

          cfg.dnn.predictionResidualSource = "oracle"

    - initLocalDNNEKF.m has been fixed so that theta0_std = 0 does not call
      randn.
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

residualAmpList = [0, 1e-4, 2e-4, 5e-4, 1e-3];
nAmp = numel(residualAmpList);

% -------------------------------------------------------------------------
% Allocate logs
% -------------------------------------------------------------------------
meanPosPhysical = NaN(nAmp,1);
meanVelPhysical = NaN(nAmp,1);

meanPosOracle = NaN(nAmp,1);
meanVelOracle = NaN(nAmp,1);

posRatioOraclePhysical = NaN(nAmp,1);
velRatioOraclePhysical = NaN(nAmp,1);

posImprovePercent = NaN(nAmp,1);
velImprovePercent = NaN(nAmp,1);

trueResidualRMS = NaN(nAmp,1);

% Per-watcher logs
cfgTmp = config_step02_residual_physical_EKF();
Nw = cfgTmp.Nw;

posPhysical = NaN(Nw,nAmp);
velPhysical = NaN(Nw,nAmp);

posOracle = NaN(Nw,nAmp);
velOracle = NaN(Nw,nAmp);

resultsPhysicalCell = cell(nAmp,1);
resultsOracleCell = cell(nAmp,1);
cfgPhysicalCell = cell(nAmp,1);
cfgOracleCell = cell(nAmp,1);

% -------------------------------------------------------------------------
% Main residualAmp sweep
% -------------------------------------------------------------------------
for a = 1:nAmp

    residualAmp = residualAmpList(a);

    fprintf("\n============================================================\n");
    fprintf(" Running residualAmp sweep case: residualAmp = %.3e\n", residualAmp);
    fprintf("============================================================\n");

    % ---------------------------------------------------------------------
    % Physical EKF baseline under residual truth
    % ---------------------------------------------------------------------
    rng(rngSeed);

    cfgPhys = config_step02_residual_physical_EKF();

    cfgPhys.truth.useResidual = true;
    cfgPhys.truth.residualModel = "branchwise";
    cfgPhys.truth.residualAmp = residualAmp;

    resultsPhys = simulatePhysicalEKF(cfgPhys);
    metricsPhys = computeResidualAmpSweepRMSE(resultsPhys, cfgPhys);

    % ---------------------------------------------------------------------
    % Oracle residual EKF under same residual truth
    % ---------------------------------------------------------------------
    rng(rngSeed);

    cfgOracle = config_step03_local_DNN_EKF();

    cfgOracle.truth.useResidual = true;
    cfgOracle.truth.residualModel = "branchwise";
    cfgOracle.truth.residualAmp = residualAmp;

    % Fair initialization.
    cfgOracle.dnn.theta0_std = 0.0;

    % Oracle residual prediction.
    cfgOracle.dnn.useResidualInPrediction = true;
    cfgOracle.dnn.predictionResidualSource = "oracle";
    cfgOracle.dnn.residualInjectionGain = 1.0;

    % Oracle residual does not use theta, so Qtheta adaptation is irrelevant.
    cfgOracle.dnn.adaptQThetaEnabled = false;

    resultsOracle = simulateLocalDNNEKF(cfgOracle);
    metricsOracle = computeResidualAmpSweepRMSE(resultsOracle, cfgOracle);

    % ---------------------------------------------------------------------
    % Store logs
    % ---------------------------------------------------------------------
    meanPosPhysical(a) = metricsPhys.meanPosRMSE;
    meanVelPhysical(a) = metricsPhys.meanVelRMSE;

    meanPosOracle(a) = metricsOracle.meanPosRMSE;
    meanVelOracle(a) = metricsOracle.meanVelRMSE;

    posPhysical(:,a) = metricsPhys.posRMSE;
    velPhysical(:,a) = metricsPhys.velRMSE;

    posOracle(:,a) = metricsOracle.posRMSE;
    velOracle(:,a) = metricsOracle.velRMSE;

    posRatioOraclePhysical(a) = safeRatioResidualAmp(meanPosOracle(a), meanPosPhysical(a));
    velRatioOraclePhysical(a) = safeRatioResidualAmp(meanVelOracle(a), meanVelPhysical(a));

    posImprovePercent(a) = 100 * (1 - posRatioOraclePhysical(a));
    velImprovePercent(a) = 100 * (1 - velRatioOraclePhysical(a));

    trueResidualRMS(a) = computeTrueResidualRMSForSweep(resultsPhys, cfgPhys);

    resultsPhysicalCell{a} = resultsPhys;
    resultsOracleCell{a} = resultsOracle;
    cfgPhysicalCell{a} = cfgPhys;
    cfgOracleCell{a} = cfgOracle;

end

% -------------------------------------------------------------------------
% Build output structure
% -------------------------------------------------------------------------
resAmpSweep = struct();

resAmpSweep.rngSeed = rngSeed;
resAmpSweep.residualAmpList = residualAmpList(:);

resAmpSweep.meanPosPhysical = meanPosPhysical;
resAmpSweep.meanVelPhysical = meanVelPhysical;

resAmpSweep.meanPosOracle = meanPosOracle;
resAmpSweep.meanVelOracle = meanVelOracle;

resAmpSweep.posRatioOraclePhysical = posRatioOraclePhysical;
resAmpSweep.velRatioOraclePhysical = velRatioOraclePhysical;

resAmpSweep.posImprovePercent = posImprovePercent;
resAmpSweep.velImprovePercent = velImprovePercent;

resAmpSweep.trueResidualRMS = trueResidualRMS;

resAmpSweep.posPhysical = posPhysical;
resAmpSweep.velPhysical = velPhysical;

resAmpSweep.posOracle = posOracle;
resAmpSweep.velOracle = velOracle;

resAmpSweep.resultsPhysicalCell = resultsPhysicalCell;
resAmpSweep.resultsOracleCell = resultsOracleCell;

resAmpSweep.cfgPhysicalCell = cfgPhysicalCell;
resAmpSweep.cfgOracleCell = cfgOracleCell;

% -------------------------------------------------------------------------
% Print summary
% -------------------------------------------------------------------------
fprintf("\n");
fprintf("============================================================\n");
fprintf(" ResidualAmp Sweep with Oracle Residual Prediction\n");
fprintf("============================================================\n");
fprintf("Random seed: %d\n", rngSeed);
fprintf("\n");

fprintf("%14s %16s %16s %16s %16s %16s\n", ...
    "residualAmp", "ResidualRMS", "PhysPosRMSE", ...
    "OraclePosRMSE", "OracleRatio", "Improve[%]");

for a = 1:nAmp
    fprintf("%14.3e %16.6e %16.6f %16.6f %16.6f %16.2f\n", ...
        residualAmpList(a), trueResidualRMS(a), ...
        meanPosPhysical(a), meanPosOracle(a), ...
        posRatioOraclePhysical(a), posImprovePercent(a));
end

fprintf("\n");

fprintf("%14s %16s %16s %16s %16s\n", ...
    "residualAmp", "PhysVelRMSE", "OracleVelRMSE", ...
    "OracleRatio", "Improve[%]");

for a = 1:nAmp
    fprintf("%14.3e %16.6f %16.6f %16.6f %16.2f\n", ...
        residualAmpList(a), ...
        meanVelPhysical(a), meanVelOracle(a), ...
        velRatioOraclePhysical(a), velImprovePercent(a));
end

fprintf("============================================================\n\n");

% -------------------------------------------------------------------------
% Sanity check: residualAmp = 0 should make physical and oracle identical
% -------------------------------------------------------------------------
idxZeroAmp = find(abs(residualAmpList) < 1e-15, 1, "first");

if ~isempty(idxZeroAmp)

    ratioZero = posRatioOraclePhysical(idxZeroAmp);

    if abs(ratioZero - 1) > 1e-8
        warning("residualAmp = 0 oracle case does not match physical EKF. Check random stream or oracle implementation.");
    else
        fprintf("Sanity check passed: residualAmp = 0 oracle matches physical EKF.\n\n");
    end

end

% -------------------------------------------------------------------------
% Plots
% -------------------------------------------------------------------------
figure;
hold on;
grid on;

plot(residualAmpList, meanPosPhysical, "-o", "LineWidth", 1.5, ...
    "DisplayName", "Physical EKF");

plot(residualAmpList, meanPosOracle, "-o", "LineWidth", 1.5, ...
    "DisplayName", "Oracle residual EKF");

xlabel("Residual amplitude");
ylabel("Mean Position RMSE");
title("Position RMSE vs Residual Amplitude");
legend("Location", "best");

figure;
hold on;
grid on;

plot(residualAmpList, meanVelPhysical, "-o", "LineWidth", 1.5, ...
    "DisplayName", "Physical EKF");

plot(residualAmpList, meanVelOracle, "-o", "LineWidth", 1.5, ...
    "DisplayName", "Oracle residual EKF");

xlabel("Residual amplitude");
ylabel("Mean Velocity RMSE");
title("Velocity RMSE vs Residual Amplitude");
legend("Location", "best");

figure;
hold on;
grid on;

plot(residualAmpList, posImprovePercent, "-o", "LineWidth", 1.5);

xlabel("Residual amplitude");
ylabel("Oracle improvement [%]");
title("Oracle Residual Improvement vs Residual Amplitude");

figure;
hold on;
grid on;

plot(residualAmpList, trueResidualRMS, "-o", "LineWidth", 1.5);

xlabel("Residual amplitude");
ylabel("True residual RMS acceleration");
title("True Residual RMS vs Residual Amplitude");

figure;
hold on;
grid on;

for i = 1:Nw
    plot(residualAmpList, posPhysical(i,:), "--o", "LineWidth", 1.1, ...
        "DisplayName", "Physical W" + string(i));

    plot(residualAmpList, posOracle(i,:), "-o", "LineWidth", 1.1, ...
        "DisplayName", "Oracle W" + string(i));
end

xlabel("Residual amplitude");
ylabel("Position RMSE");
title("Per-Watcher Position RMSE vs Residual Amplitude");
legend("Location", "best");

% -------------------------------------------------------------------------
% Local helper functions
% -------------------------------------------------------------------------

function metrics = computeResidualAmpSweepRMSE(results, cfg)
%{
Function:
    computeResidualAmpSweepRMSE

Purpose:
    Compute physical target tracking RMSE without printing diagnostics.

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

function residualRMS = computeTrueResidualRMSForSweep(results, cfg)
%{
Function:
    computeTrueResidualRMSForSweep

Purpose:
    Compute RMS norm of the true residual acceleration.

Output:
    residualRMS = sqrt(mean(||d_true(k)||^2)).
%}

    dim = cfg.dim;

    if ~isfield(results, "trueResidual")
        residualRMS = NaN;
        return;
    end

    dTrue = results.trueResidual;

    if isempty(dTrue)
        residualRMS = NaN;
        return;
    end

    if size(dTrue,1) ~= dim && size(dTrue,2) == dim
        dTrue = dTrue.';
    end

    if size(dTrue,1) ~= dim
        residualRMS = NaN;
        return;
    end

    residualRMS = sqrt(mean(sum(dTrue.^2, 1)));

end

function r = safeRatioResidualAmp(a, b)
% Compute a/b while avoiding division by zero.

    if abs(b) < eps
        r = NaN;
    else
        r = a / b;
    end

end