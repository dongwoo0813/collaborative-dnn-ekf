%{
Script:
    run_step02_residual_physical_EKF.m

Purpose:
    Entry-point script for Step 02 of the collaborative DNN-EKF simulation.

    Step 02 runs the physical EKF baseline under a residual-mismatch truth
    model.

    The true target dynamics include an unknown residual acceleration,

        dot r_t = v_t,

        dot v_t = a_nom(r_t,v_t,t) + d_unk(eta,t),

    but the physical EKF prediction model still assumes only the nominal
    model,

        dot r_t = v_t,

        dot v_t = a_nom(r_t,v_t,t).

    Therefore, this script is used to check how much the baseline physical
    EKF degrades when the target has unmodeled residual dynamics.

Inputs:
    None directly.

    The script loads simulation parameters from

        config_step02_residual_physical_EKF.m

Outputs:
    results - Simulation output structure returned by simulatePhysicalEKF.
    metrics - Tracking metrics returned by computePhysicalEKFMetrics.

Main procedure:
    1. Clear MATLAB workspace.
    2. Add all project subfolders to the MATLAB path.
    3. Load Step 02 configuration.
    4. Run the physical EKF simulation.
    5. Compute tracking metrics.
    6. Plot trajectories, tracking errors, measurement availability, watcher
       motion, relative geometry, and control placeholders.

Notes:
    - Watcher control is still disabled in this step.
    - Watcher motion remains prescribed.
    - This script is not yet a DNN-EKF simulation.
    - This step prepares the model-mismatch case that the DNN-EKF will later
      try to improve.
%}

clear; clc; close all;

% Add project root and all subfolders to path.
%
% This is more robust than addpath(genpath(pwd)) if the script is executed
% from inside the main/ folder.
thisFile = mfilename("fullpath");
mainDir = fileparts(thisFile);
projectRoot = fileparts(mainDir);

addpath(genpath(projectRoot));
rehash;

% Load Step 02 residual-mismatch configuration.
cfg = config_step02_residual_physical_EKF();

% Run physical EKF under residual truth dynamics.
results = simulatePhysicalEKF(cfg);

% Compute baseline physical EKF metrics.
metrics = computePhysicalEKFMetrics(results, cfg);

% Plot results.
plotTrajectories(results, cfg);
plotTrackingErrors(results, cfg);
plotTrueResidual(results, cfg);
plotMeasurementAvailability(results, cfg);

if exist("plotWatcherMotion", "file") == 2
    plotWatcherMotion(results, cfg);
end

if exist("plotRelativeGeometry", "file") == 2
    plotRelativeGeometry(results, cfg);
end

if exist("plotWatcherControl", "file") == 2
    plotWatcherControl(results, cfg);
end