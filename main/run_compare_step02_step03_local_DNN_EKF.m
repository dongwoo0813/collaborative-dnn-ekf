%{
Script:
    run_compare_step02_step03_local_DNN_EKF.m

Purpose:
    Compare Step 02 physical EKF and Step 03 local DNN-EKF.

    Step 02:
        Truth includes unknown residual acceleration.
        Estimator is physical EKF only:

            eta_i = [r_t; v_t].

        The EKF prediction model does not know the residual.

    Step 03:
        Truth also includes unknown residual acceleration.
        Estimator is local augmented DNN-EKF:

            X_i = [eta_i; theta_i].

        Each watcher uses its own local fixed-feature DNN branch:

            d_hat_i(eta;theta_i) = W_i phi_i(eta).

    This script checks whether the local DNN-EKF improves over the
    residual-mismatch physical EKF baseline.

Inputs:
    None directly.

Outputs:
    results02, metrics02
    results03, metrics03
    nisStats03
    dnnStats03

Notes:
    - The same top-level random seed is used before each simulation.
    - Because Step 03 initializes additional DNN parameters, the random
      number stream may not be perfectly identical to Step 02 after
      initialization. This is still useful as a first fair comparison.
    - A stricter comparison can later use pre-generated measurement logs.
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
% Random seed
% -------------------------------------------------------------------------
rngSeed = 1;

% -------------------------------------------------------------------------
% Step 02: residual truth + physical EKF
% -------------------------------------------------------------------------
rng(rngSeed);

cfg02 = config_step02_residual_physical_EKF();

results02 = simulatePhysicalEKF(cfg02);
metrics02 = computePhysicalEKFMetrics(results02, cfg02);

% -------------------------------------------------------------------------
% Step 03: residual truth + local DNN-EKF
% -------------------------------------------------------------------------
rng(rngSeed);

cfg03 = config_step03_local_DNN_EKF();

results03 = simulateLocalDNNEKF(cfg03);
metrics03 = computePhysicalEKFMetrics(results03, cfg03);

nisStats03 = computeNISDiagnostics(results03, cfg03);
dnnStats03 = plotDNNLearningDiagnostics(results03, cfg03);

if isfield(results03, "gammaTheta")
    plotCovMatchingDiagnostics(results03, cfg03);
end

% -------------------------------------------------------------------------
% Print comparison
% -------------------------------------------------------------------------
fprintf("\n");
fprintf("============================================================\n");
fprintf(" Step 02 Physical EKF vs Step 03 Local DNN-EKF Comparison\n");
fprintf("============================================================\n");
fprintf("Random seed: %d\n", rngSeed);
fprintf("Number of watchers: %d\n", cfg03.Nw);
fprintf("Dimension: %d\n", cfg03.dim);
fprintf("Residual model: %s\n", cfg03.truth.residualModel);
fprintf("Residual amplitude: %.3e\n", cfg03.truth.residualAmp);

fprintf("DNN residual switch: %d\n", cfg03.dnn.useResidualInPrediction);

if isfield(cfg03.dnn, "residualInjectionGain")
    fprintf("DNN residual injection gain betaDNN: %.3f\n", ...
        cfg03.dnn.residualInjectionGain);
else
    fprintf("DNN residual injection gain betaDNN: %.3f\n", 1.0);
end

fprintf("DNN theta dynamics: %s\n", cfg03.dnn.thetaDynamics);

fprintf("\n");

fprintf("Mean position RMSE:\n");
fprintf("    Step 02 physical EKF    = %.6f\n", metrics02.meanPosRMSE);
fprintf("    Step 03 local DNN-EKF   = %.6f\n", metrics03.meanPosRMSE);
fprintf("    Ratio Step03 / Step02   = %.6f\n", ...
    safeRatioLocal(metrics03.meanPosRMSE, metrics02.meanPosRMSE));
fprintf("    Improvement             = %.2f %%\n", ...
    100 * (1 - safeRatioLocal(metrics03.meanPosRMSE, metrics02.meanPosRMSE)));
fprintf("\n");

fprintf("Mean velocity RMSE:\n");
fprintf("    Step 02 physical EKF    = %.6f\n", metrics02.meanVelRMSE);
fprintf("    Step 03 local DNN-EKF   = %.6f\n", metrics03.meanVelRMSE);
fprintf("    Ratio Step03 / Step02   = %.6f\n", ...
    safeRatioLocal(metrics03.meanVelRMSE, metrics02.meanVelRMSE));
fprintf("    Improvement             = %.2f %%\n", ...
    100 * (1 - safeRatioLocal(metrics03.meanVelRMSE, metrics02.meanVelRMSE)));
fprintf("\n");

fprintf("%10s %16s %16s %16s %16s\n", ...
    "Watcher", "PosRMSE_02", "PosRMSE_03", "Ratio", "Improve[%]");

for i = 1:cfg03.Nw
    ratio_i = safeRatioLocal(metrics03.posRMSE(i), metrics02.posRMSE(i));
    improve_i = 100 * (1 - ratio_i);

    fprintf("%10d %16.6f %16.6f %16.6f %16.2f\n", ...
        i, metrics02.posRMSE(i), metrics03.posRMSE(i), ratio_i, improve_i);
end

fprintf("\n");

fprintf("%10s %16s %16s %16s %16s\n", ...
    "Watcher", "VelRMSE_02", "VelRMSE_03", "Ratio", "Improve[%]");

for i = 1:cfg03.Nw
    ratio_i = safeRatioLocal(metrics03.velRMSE(i), metrics02.velRMSE(i));
    improve_i = 100 * (1 - ratio_i);

    fprintf("%10d %16.6f %16.6f %16.6f %16.2f\n", ...
        i, metrics02.velRMSE(i), metrics03.velRMSE(i), ratio_i, improve_i);
end

fprintf("============================================================\n\n");

% -------------------------------------------------------------------------
% Bar plots: RMSE comparison
% -------------------------------------------------------------------------
watcherLabels = categorical("Watcher " + string(1:cfg03.Nw));
watcherLabels = reordercats(watcherLabels, "Watcher " + string(1:cfg03.Nw));

figure;
bar(watcherLabels, [metrics02.posRMSE, metrics03.posRMSE]);
grid on;
xlabel("Watcher");
ylabel("Position RMSE");
title("Position RMSE: Step 02 Physical EKF vs Step 03 Local DNN-EKF");
legend("Step 02 Physical EKF", "Step 03 Local DNN-EKF", "Location", "best");

figure;
bar(watcherLabels, [metrics02.velRMSE, metrics03.velRMSE]);
grid on;
xlabel("Watcher");
ylabel("Velocity RMSE");
title("Velocity RMSE: Step 02 Physical EKF vs Step 03 Local DNN-EKF");
legend("Step 02 Physical EKF", "Step 03 Local DNN-EKF", "Location", "best");

% -------------------------------------------------------------------------
% Tracking error plots for Step 03
% -------------------------------------------------------------------------
plotTrajectories(results03, cfg03);
plotTrackingErrors(results03, cfg03);
plotTrueResidual(results03, cfg03);
plotMeasurementAvailability(results03, cfg03);
plotNIS(results03, cfg03);

if exist("plotWatcherMotion", "file") == 2
    plotWatcherMotion(results03, cfg03);
end

if exist("plotRelativeGeometry", "file") == 2
    plotRelativeGeometry(results03, cfg03);
end

if exist("plotWatcherControl", "file") == 2
    plotWatcherControl(results03, cfg03);
end

% -------------------------------------------------------------------------
% Local helper
% -------------------------------------------------------------------------
function r = safeRatioLocal(a, b)
% safeRatioLocal
%
% Purpose:
%     Compute a/b while avoiding division by zero.

    if abs(b) < eps
        r = NaN;
    else
        r = a / b;
    end

end