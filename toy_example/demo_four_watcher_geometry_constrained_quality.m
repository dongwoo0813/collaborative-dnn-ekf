%% DEMO_FOUR_WATCHER_TIME_VARYING_ACCELERATION
% Four independent angle-only [r;v;a] EKFs.
%
% Test case:
%   - all four watchers measure,
%   - all four watchers execute the finite pulse pair,
%   - each watcher contributes its LOS-transverse acceleration component,
%   - bounded covariance + NIS quality weights are used,
%   - the weights are relaxed toward equal geometry only when required by
%     the rank/condition constraint,
%   - no arithmetic-mean fallback or last-value hold is used,
%   - target acceleration varies smoothly instead of remaining constant.
%
% The EKF still uses the [r;v;a] constant-acceleration transition model.
% Nonzero jerk process noise lets the acceleration state follow the
% time-varying truth.

clear;
close all;
clc;

dim = 2;
makePlots = true;
simulationTime = 180;
dt = 0.05;
seed = 73;

options = struct();

%% Smooth time-varying target acceleration
options.truthAccelerationMode = 'slow_sinusoid';

% a_i(t) = bias_i + amplitude_i*sin(omega_i*t + phase_i)
options.constantAcceleration = [0.15;0.20];
options.accelerationAmplitude = [0.06;0.05]*8;
options.accelerationFrequency = [0.035;0.050]*10;
options.accelerationPhase = [0;pi/2];

%% Four watchers: measurement and maneuver
options.activeWatcherMode = 'all';
options.maneuverWatcherMask = [true true true true];
options.initialRangeScale = [0.55,0.80,1.25,1.55];

options.burnAcceleration = 2.0;
options.burnStartTime = 10.0;
options.burnDuration = 5.0;

options.sigmaBearingDeg = 0.05;

% Nonzero jerk process noise is required for time-varying acceleration.
% The truth jerk amplitudes are approximately amplitude.*frequency,
% which are around 2e-3 to 3e-3 in this example.
options.sigmaJerk = 0.5;

%% Geometry-aware fusion
% All four maneuver, so the maneuver-mask output equals equal geometry.
options.nonManeuverFusionWeight = 1;

% Bounded soft quality weighting.
options.qualityWeightMinimum = 0.25;
options.qualityWeightEpsilon = 1e-10;
options.qualityVarianceExponent = 0.5;
options.qualityNISExponent = 0.5;
options.nisEwmaFactor = 0.95;

options.maxQualityGeometryCondition = 20;
options.qualityGeometryBlendGridSize = 101;
options.postBurnSettlingTime = 20;

resultGeometryConstrained = run_local_accel_observability_experiment( ...
    dim,makePlots,simulationTime,dt,seed,options);

disp(resultGeometryConstrained.summary);

fprintf('\nGeometry-constrained quality fusion comparison\n');
fprintf('  local mean RMSE               : %.8f\n', ...
    resultGeometryConstrained.pulsePair.postActiveMeanAccelerationRMSE);
fprintf('  equal geometry RMSE           : %.8f\n', ...
    resultGeometryConstrained.pulsePair.postEqualGeometryAccelerationRMSE);
fprintf('  bounded quality geometry RMSE : %.8f\n', ...
    resultGeometryConstrained.pulsePair. ...
        postQualityWeightedGeometryAccelerationRMSE);
fprintf('  quality-fusion valid rate     : %.4f\n', ...
    resultGeometryConstrained.pulsePair.postQualityFusionValidRate);
fprintf('  total maneuver Delta-V        : %.4f\n', ...
    resultGeometryConstrained.pulsePair.totalDeltaV);

fprintf('  mean geometry blend factor   : %.4f\n', ...
    mean(resultGeometryConstrained.pulsePair. ...
        qualityGeometryBlendFactor( ...
        resultGeometryConstrained.pulsePair.postBurnIndex), ...
        'omitnan'));
