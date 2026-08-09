%{
Script:
    run_step03_local_DNN_EKF.m

Purpose:
    Entry-point script for Step 03 local DNN-EKF simulation.

    Step 03 runs the local augmented DNN-EKF:

        X_i = [eta_i; theta_i],

    where

        eta_i = [r_t; v_t]

    and theta_i is the local fixed-feature output-layer DNN branch parameter.

    The true target dynamics include an unknown residual acceleration.
    Each watcher predicts using its own local DNN residual model and updates
    with bearing-only measurements.

    There is still no ground-station sharing and no peer-to-peer sharing.

Inputs:
    None directly.

    The script loads simulation parameters from

        config_step03_local_DNN_EKF.m

Outputs:
    results - Simulation output structure returned by simulateLocalDNNEKF.
    metrics - Physical tracking metrics returned by computePhysicalEKFMetrics.

Notes:
    - This script assumes simulation/simulateLocalDNNEKF.m exists.
    - The result field results.xhat is expected to remain eta-only so that
      the existing physical EKF plotting and metric functions still work.
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
% Check required Step 03 simulation function
% -------------------------------------------------------------------------
if exist("simulateLocalDNNEKF", "file") ~= 2
    error("simulateLocalDNNEKF.m was not found. Please save it under simulation/simulateLocalDNNEKF.m first.");
end

if exist("DNN_EKF_Update_Local", "file") ~= 2
    error("DNN_EKF_Update_Local.m was not found. Please save it under ekf/DNNEKFUpdateLocal.m first.");
end

% -------------------------------------------------------------------------
% Load Step 03 configuration
% -------------------------------------------------------------------------
cfg = config_step03_local_DNN_EKF();

% -------------------------------------------------------------------------
% Run local DNN-EKF simulation
% -------------------------------------------------------------------------
results = simulateLocalDNNEKF(cfg);

% -------------------------------------------------------------------------
% Compute physical tracking metrics, NIS statistics, and DNN statistics.
% -------------------------------------------------------------------------
metrics  = computePhysicalEKFMetrics(results, cfg);
nisStats = computeNISDiagnostics(results, cfg);
dnnStats = plotDNNLearningDiagnostics(results, cfg);

fprintf("\n=== Step 03 Local DNN-EKF Summary ===\n");
fprintf("Estimator type: %s\n", cfg.estimator.type);
fprintf("Communication mode: %s\n", cfg.comm.mode);
fprintf("Number of watchers: %d\n", cfg.Nw);
fprintf("State dimension eta: %d\n", 2*cfg.dim);
fprintf("Theta dimension per branch: %d\n", cfg.dnn.nThetaPerBranch);
fprintf("Mean position RMSE: %.6f\n", metrics.meanPosRMSE);
fprintf("Mean velocity RMSE: %.6f\n", metrics.meanVelRMSE);
fprintf("=====================================\n\n");

% -------------------------------------------------------------------------
% Plot physical tracking results
% -------------------------------------------------------------------------
plotTrajectories(results, cfg);
plotTrackingErrors(results, cfg);
plotTrueResidual(results, cfg);
plotMeasurementAvailability(results, cfg);
plotNIS(results, cfg);

if exist("plotWatcherMotion", "file") == 2
    plotWatcherMotion(results, cfg);
end

if exist("plotRelativeGeometry", "file") == 2
    plotRelativeGeometry(results, cfg);
end

if exist("plotWatcherControl", "file") == 2
    plotWatcherControl(results, cfg);
end