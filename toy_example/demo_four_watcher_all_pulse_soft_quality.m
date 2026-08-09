%% DEMO_FOUR_WATCHER_QUALITY_WEIGHTED_GEOMETRY
% Four independent angle-only [r;v;a] EKFs with:
%
%   1. local acceleration mean,
%   2. equal geometry fusion,
%   3. maneuver-mask geometry fusion,
%   4. adaptive covariance + NIS quality-weighted geometry fusion.
%
% All four watchers measure and all four perform the finite pulse pair.

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
options.maneuverWatcherMask = [true true true true];
options.initialRangeScale = [0.55,0.80,1.25,1.55];

options.burnAcceleration = 2.0;
options.burnStartTime = 10.0;
options.burnDuration = 5.0;

options.sigmaBearingDeg = 0.2;

% Constant-acceleration observability validation.
options.sigmaJerk = 0;

% All four watchers maneuver, so the maneuver-mask fusion is the same as
% equal-geometry fusion. It remains in the output as a consistency check.
options.nonManeuverFusionWeight = 1;

% Bounded soft quality weight:
%
%   raw_i = 1/sqrt(transverse variance_i)
%           /sqrt(max(1,normalized NIS EWMA_i))
%
% and alpha_i is mapped to [0.25,1]. This prevents one watcher from
% dominating while still letting more confident watchers contribute more.
options.qualityWeightMinimum = 0.25;
options.qualityWeightEpsilon = 1e-10;
options.qualityVarianceExponent = 0.5;
options.qualityNISExponent = 0.5;
options.nisEwmaFactor = 0.95;

% Reject/hold a quality-weighted fusion when its geometry is too weak.
options.maxQualityGeometryCondition = 20;

options.postBurnSettlingTime = 20;

resultAllPulse = run_local_accel_observability_experiment( ...
    dim,makePlots,simulationTime,dt,seed,options);

disp(resultAllPulse.summary);

fprintf('\nAll-four-pulse post-burn comparison\n');
fprintf('  local mean RMSE              : %.8f\n', ...
    resultAllPulse.pulsePair.postActiveMeanAccelerationRMSE);
fprintf('  equal geometry RMSE          : %.8f\n', ...
    resultAllPulse.pulsePair.postEqualGeometryAccelerationRMSE);
fprintf('  all-pulse equal-mask RMSE    : %.8f\n', ...
    resultAllPulse.pulsePair. ...
        postManeuverWeightedGeometryAccelerationRMSE);
fprintf('  bounded quality geometry RMSE: %.8f\n', ...
    resultAllPulse.pulsePair. ...
        postQualityWeightedGeometryAccelerationRMSE);
fprintf('  quality-fusion valid rate    : %.4f\n', ...
    resultAllPulse.pulsePair.postQualityFusionValidRate);
