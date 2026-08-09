%% DEMO_FOUR_WATCHER_QUALITY_WEIGHTED_GEOMETRY
% Four independent angle-only [r;v;a] EKFs with:
%
%   1. local acceleration mean,
%   2. equal geometry fusion,
%   3. maneuver-mask geometry fusion,
%   4. adaptive covariance + NIS quality-weighted geometry fusion.
%
% All four watchers measure. Watchers 1 and 3 perform the finite pulse pair.

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

options.activeWatcherMode = 'all';
options.maneuverWatcherMask = [true false true false];
options.initialRangeScale = [0.55,0.80,1.25,1.55];

options.burnAcceleration = 2.0;
options.burnStartTime = 10.0;
options.burnDuration = 5.0;

options.sigmaBearingDeg = 0.2;

% Constant-acceleration observability validation.
options.sigmaJerk = 0;

% Existing maneuver-mask baseline:
% W1 and W3 receive weight 1; W2 and W4 receive weight 0.
options.nonManeuverFusionWeight = 0;

% New adaptive quality weight:
% alpha_i proportional to the inverse transverse acceleration variance,
% with an additional normalized-NIS penalty.
options.qualityWeightMinimum = 0.05;
options.qualityWeightEpsilon = 1e-10;
options.nisEwmaFactor = 0.95;

% Reject/hold a quality-weighted fusion when its geometry is too weak.
options.maxQualityGeometryCondition = 20;

options.postBurnSettlingTime = 20;

resultQuality = run_local_accel_observability_experiment( ...
    dim,makePlots,simulationTime,dt,seed,options);

disp(resultQuality.summary);

fprintf('\nPulse-pair post-burn comparison\n');
fprintf('  local mean RMSE              : %.8f\n', ...
    resultQuality.pulsePair.postActiveMeanAccelerationRMSE);
fprintf('  equal geometry RMSE          : %.8f\n', ...
    resultQuality.pulsePair.postEqualGeometryAccelerationRMSE);
fprintf('  maneuver-mask geometry RMSE  : %.8f\n', ...
    resultQuality.pulsePair. ...
        postManeuverWeightedGeometryAccelerationRMSE);
fprintf('  quality-weighted geometry RMSE: %.8f\n', ...
    resultQuality.pulsePair. ...
        postQualityWeightedGeometryAccelerationRMSE);
fprintf('  quality-fusion valid rate    : %.4f\n', ...
    resultQuality.pulsePair.postQualityFusionValidRate);
