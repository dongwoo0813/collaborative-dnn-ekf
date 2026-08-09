%% DEMO_FOUR_WATCHER_ACCELERATION_BLUE_FUSION
% Four independent local angle-only [r;v;a] EKFs.
%
% Fusion methods:
%   1. equal geometry baseline,
%   2. transverse BLUE assuming independent local acceleration errors,
%   3. transverse BLUE using propagated local-error cross-covariances.
%
% No heuristic quality weight, clipping, low-pass filter, beta blending,
% arithmetic-mean fallback, or previous-value hold is used.

clear;
close all;
clc;

dim = 2;
makePlots = true;
simulationTime = 180;
dt = 0.05;
seed = 73;

options = struct();

%% Time-varying target acceleration
options.truthAccelerationMode = 'slow_sinusoid';

% a_i(t) = bias_i + amplitude_i*sin(omega_i*t+phase_i)
options.constantAcceleration = [0.15;0.20];
options.accelerationAmplitude = [0.06;0.05];
options.accelerationFrequency = [0.035;0.050];
options.accelerationPhase = [0;pi/2];

%% Four watchers all measure and all maneuver
options.activeWatcherMode = 'all';
options.maneuverWatcherMask = [true true true true];
options.initialRangeScale = [0.55,0.80,1.25,1.55];

options.burnAcceleration = 2.0;
options.burnStartTime = 10.0;
options.burnDuration = 5.0;

options.sigmaBearingDeg = 0.2;

% The local [r;v;a] EKFs require nonzero jerk process noise to track the
% smooth time-varying acceleration.
options.sigmaJerk = 0.005;

%% Explicit cross-covariance assumptions
% 'independent':
%   P_ij(0)=0 for i~=j.
%
% 'common_prior':
%   P_ij(0)=P0 for i~=j.
%
% The current scaled initial estimates are different for each watcher, so
% the demo starts with the explicit independent-prior assumption.
options.initialCrossCovarianceMode = 'independent';

% All local filters estimate the same target process. Therefore the demo
% treats the EKF process noise as fully common across local filters.
options.commonProcessNoiseFraction = 1.0;

options.postBurnSettlingTime = 20;

resultBLUE = run_four_watcher_acceleration_blue_experiment( ...
    dim,makePlots,simulationTime,dt,seed,options);

disp(resultBLUE.summary);

fprintf('\nPulse-pair post-burn acceleration comparison\n');
fprintf('  local mean, diagnostic only : %.8f\n', ...
    resultBLUE.pulsePair.postLocalMeanAccelerationRMSE);
fprintf('  equal geometry              : %.8f\n', ...
    resultBLUE.pulsePair.postEqualGeometryAccelerationRMSE);
fprintf('  diagonal transverse BLUE    : %.8f\n', ...
    resultBLUE.pulsePair.postDiagonalBLUEAccelerationRMSE);
fprintf('  correlated transverse BLUE  : %.8f\n', ...
    resultBLUE.pulsePair.postCorrelatedBLUEAccelerationRMSE);
fprintf('  correlated BLUE full-rank rate: %.4f\n', ...
    resultBLUE.pulsePair.postCorrelatedBLUEFullRankRate);
fprintf('  correlated BLUE median condition: %.4f\n', ...
    resultBLUE.pulsePair.postCorrelatedBLUEMedianCondition);
fprintf('  total maneuver Delta-V      : %.4f\n', ...
    resultBLUE.pulsePair.totalDeltaV);
