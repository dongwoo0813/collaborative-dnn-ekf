%% DEMO_ANGLE_ONLY_OBSERVABILITY_MANEUVER
% Angle-only local EKFs with calibrated LOS-transverse watcher maneuvers.

clear;
close all;
clc;

dim = 2;
makePlots = true;
simulationTime = 200;
dt = 0.05;
seed = 9;

initialEstimate = struct();
initialEstimate.acceleration = [1.5; 2.0];

options = struct();
options.measurementMode = 'angle_only';
options.maneuverMode = 'observability_aware';

% Known watcher maneuver settings.
options.maneuverAcceleration = 2.0;
options.maneuverFrequencyHz = 0.5;
options.maneuverStartTime = 2.0;
options.maneuverStopTime = simulationTime;

% Measurement and process-noise settings.
options.sigmaBearingDeg = 0.02;
options.sigmaAzimuthDeg = 0.02;
options.sigmaElevationDeg = 0.02;
options.sigmaRange = 5.0;
options.sigmaAccelIncrement = 1;

result = run_4watcher_accel_fusion_EKF( ...
    dim,makePlots,simulationTime,dt,seed, ...
    initialEstimate,options);
