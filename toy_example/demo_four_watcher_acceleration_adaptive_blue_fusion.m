%% DEMO_FOUR_WATCHER_ACCELERATION_ADAPTIVE_BLUE_FUSION
% Four independent local angle-only [r;v;a] EKFs.
%
% Fusion includes the previous transverse BLUE implementation plus a
% reliability-gated directional reconstruction:
%   1. equal geometry baseline,
%   2. diagonal transverse BLUE,
%   3. correlated transverse BLUE with propagated cross-covariances.
%   4. gated directional estimate, sum_i alpha_i(I-u_i*u_i') aHat_i.
%
% u_i is the predicted LOS unit vector (l_i in the DNN-EKF formulation).
% alpha_i is a covariance-reliability softmax weight, fixed from EKF
% quantities rather than learned.  The local EKF estimates are therefore a
% branch-output proxy before replacing them with watcher-local DNN outputs.
%
% Maneuver logic:
%   1. one initial LOS-transverse pulse pair,
%   2. calibrate a healthy rolling acceleration-information level,
%   3. repeat a pulse pair when temporal information or fusion geometry
%      remains weak for the specified dwell time.

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
options.constantAcceleration = [0.15;0.20];
options.accelerationAmplitude = [0.01;0.01];
options.accelerationFrequency = [0.035;0.050];
options.accelerationPhase = [0;pi/2];

%% Watcher geometry and measurements
options.watcherRadius = 1000;
options.initialRangeScale = [0.55,0.80,1.25,1.55];
options.sigmaBearingDeg = 0.05;
options.sigmaJerk = 0.01;

%% Directional reliability gate (pre-DNN branch-routing test)
% Smaller temperatures make routing more exclusive.
options.directionalGateTemperature = 0.25;

%% Transverse BLUE cross-covariance assumptions
options.initialCrossCovarianceMode = 'independent';
options.commonProcessNoiseFraction = 1.0;

%% Initial and event-triggered observability maneuvers
options.maneuverWatcherMask = [true true true true];
options.burnAcceleration = 2.0;
options.burnDuration = 5.0;
options.initialBurnStartTime = 10.0;

% Rolling acceleration-information window.
options.observabilityWindowTime = 15.0;

% Healthy-reference calibration starts after the initial pulse pair and
% settling. The median lambda_min(J_a) over this interval is retained.
options.postBurnSettlingTime = 10.0;
options.referenceCalibrationDuration = 15.0;

% Re-maneuver when lambda_min(J_a) falls below 55% of its calibrated
% healthy value, or when the normalized fusion-geometry score is below
% 0.35. The condition must persist for 2 s.
options.temporalTriggerFraction = 0.55;
options.geometryTriggerScore = 0.35;
options.triggerDwellTime = 2.0;

% Prevent rapid repeated burns and cap maneuver expenditure.
options.maneuverCooldown = 20.0;
options.maximumAdaptiveBurns = 3;

resultAdaptiveBLUE = ...
    run_four_watcher_acceleration_adaptive_blue_experiment( ...
    dim,makePlots,simulationTime,dt,seed,options);

disp(resultAdaptiveBLUE.summary);
disp(resultAdaptiveBLUE.eventTable);

fprintf('\nAdaptive maneuver summary\n');
fprintf('  reference lambda_min(J_a) : %.6e\n', ...
    resultAdaptiveBLUE.referenceTemporalInformation);
fprintf('  total maneuver events     : %d\n', ...
    height(resultAdaptiveBLUE.eventTable));
fprintf('  total maneuver Delta-V    : %.6f\n', ...
    resultAdaptiveBLUE.totalDeltaV);
