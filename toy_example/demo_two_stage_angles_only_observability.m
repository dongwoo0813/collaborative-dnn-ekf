%% DEMO_TWO_STAGE_ANGLES_ONLY_OBSERVABILITY
% Stage 1: reproduce the paper-state [r;v] maneuver mechanism.
% Stage 2: benchmark [r;v;a] using one common multi-watcher EKF.

clear;
close all;
clc;

dim = 2;
makePlots = true;
simulationTime = 60;
dt = 0.05;

%% Stage 1: [r;v], local angle-only filters, finite calibrated pulse pair
rvOptions = struct();
rvOptions.burnAcceleration = 2.0;
rvOptions.burnStartTime = 10.0;
rvOptions.burnDuration = 5.0;
rvOptions.initialRangeScale = [0.55,1.45,0.70,1.30];
rvOptions.watcherRadius = 1000;
rvOptions.sigmaBearingDeg = 0.2;

rvResult = run_paper_rv_observability_validation( ...
    dim,makePlots,simulationTime,dt,33,rvOptions);

%% Stage 2: one common [r;v;a] EKF using all four bearings
commonOptions = struct();
commonOptions.burnAcceleration = 2.0;
commonOptions.burnStartTime = 10.0;
commonOptions.burnDuration = 5.0;
commonOptions.watcherRadius = 1000;
commonOptions.sigmaBearingDeg = 0.2;

% White-jerk process model for the unknown acceleration.
commonOptions.sigmaJerk = 1.0;

% Start with the original smooth acceleration frequency, not the x10 case.
commonOptions.truthAccelerationScale = 1.0;
commonOptions.truthFrequencyScale = 1.0;

commonResult = run_common_multisensor_accel_EKF( ...
    dim,makePlots,simulationTime,dt,53,commonOptions);
