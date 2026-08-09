%{
Script:
    run_step01_physical_EKF.m

Purpose:
    Entry-point script for Step 01 of the collaborative DNN-EKF simulation.

    Step 01 runs a dimension-generic physical EKF baseline. The target state is

        eta = [r_t; v_t],

    and each watcher estimates eta using intermittent or continuous bearing-only
    measurements.

    No DNN residual approximation, no branch sharing, and no controller are
    included in this step.

Inputs:
    None directly.
    The script loads simulation parameters from

        config_step01_physical_EKF.m

Outputs:
    results - Simulation output structure returned by simulatePhysicalEkf.

Main procedure:
    1. Clear MATLAB workspace.
    2. Add all subfolders to the MATLAB path.
    3. Load configuration.
    4. Run physical EKF simulation.
    5. Plot trajectories and tracking errors if plotting functions exist.

Notes:
    - This script should be run from the repository root folder.
    - The simulation dimension is controlled by cfg.dim, not by this filename.
%}

clear; clc; close all;

addpath(genpath(pwd));

cfg = config_step01_physical_EKF();

results = simulatePhysicalEKF(cfg);

metrics = computePhysicalEKFMetrics(results, cfg);

plotTrajectories(results, cfg);
plotTrackingErrors(results, cfg);
plotMeasurementAvailability(results, cfg);
plotWatcherMotion(results, cfg);
plotRelativeGeometry(results, cfg);
plotWatcherControl(results, cfg);