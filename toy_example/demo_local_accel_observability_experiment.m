%% DEMO_LOCAL_ACCEL_OBSERVABILITY_EXPERIMENT
% Step 1: reproduce the successful selective-maneuver baseline.
% Step 2: compare representative maneuver masks with paired noise/seed.

clear;
close all;
clc;

dim = 2;
makePlots = true;
simulationTime = 120;
dt = 0.05;
seed = 73;

options = struct();

options.truthAccelerationMode = 'constant';
options.constantAcceleration = [0.15;0.20];

% All four independent local EKFs receive angle measurements.
options.activeWatcherMode = 'all';
options.activeWatcherIndex = 1;

% Watchers 1 and 3 maneuver; watchers 2 and 4 only measure.
options.maneuverWatcherMask = [true false true false];

options.initialRangeScale = [0.55,0.80,1.25,1.55];

options.burnAcceleration = 2.0;
options.burnStartTime = 10.0;
options.burnDuration = 5.0;

options.sigmaBearingDeg = 0.2;

% Constant-acceleration validation: do not allow acceleration random walk.
options.sigmaJerk = 0;

% Weighted fusion uses only maneuvering watchers after burn start.
options.nonManeuverFusionWeight = 0;
options.postBurnSettlingTime = 20.0;

resultSelective = run_local_accel_observability_experiment( ...
    dim,makePlots,simulationTime,dt,seed,options);

%% Representative maneuver-mask sweep
% Set the final argument to 'all_combinations' to test all 16 masks.
sweep = run_maneuver_mask_sweep( ...
    dim,simulationTime,dt,seed,'core');
