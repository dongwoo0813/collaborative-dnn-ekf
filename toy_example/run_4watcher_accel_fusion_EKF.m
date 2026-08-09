function result = run_4watcher_accel_fusion_EKF( ...
    dim,makePlots,simulationTime,dt,seed,initialEstimate,options)
%RUN_4WATCHER_ACCEL_FUSION_EKF Local EKFs with selectable measurements.
%
% PURPOSE
% -------
% Simulate four independent local EKFs that estimate the same inertial
% target state
%
%   x_k = [r_k; v_k; a_k]
%
% while the watcher locations are known and may execute calibrated
% observability-aware maneuvers.
%
% MEASUREMENT MODES
% -----------------
% options.measurementMode = 'angle_only'
%
%   dim = 2: bearing
%   dim = 3: azimuth and elevation
%
% options.measurementMode = 'angle_range'
%
%   dim = 2: bearing and range
%   dim = 3: azimuth, elevation, and range
%
% MANEUVER MODES
% --------------
% options.maneuverMode = 'none'
% options.maneuverMode = 'observability_aware'
%
% The observability-aware mode commands a known watcher acceleration that
% is transverse to the local prior-predicted line of sight (LOS):
%
%   a_w,i = A cos(omega*tau) d_perp,i,
%   d_perp,i^T uHat_i = 0.
%
% The maneuver changes the LOS measurement profile while avoiding a
% deliberately LOS-aligned maneuver. This is a practical implementation of
% the maneuver principle used in angles-only observability analysis. It is
% not, by itself, a proof that the augmented acceleration state is globally
% observable.
%
% ACCELERATION FUSION
% -------------------
% Each local EKF produces a filtered current-acceleration point estimate
%
%   aHat_i(k|k) = E[a_k | z_i(1:k)].
%
% The fusion projector uses the prior predicted LOS used by the local EKF:
%
%   Pperp_i = I-uHat_i^- (uHat_i^-)^T.
%
% The equal-confidence geometry-aware fused estimate is
%
%   aHat_f = [eps*I + sum_i Pperp_i]^(-1)
%            sum_i Pperp_i aHat_i.
%
% INPUTS
% ------
% dim             Spatial dimension: 2 or 3.
% makePlots       Logical flag for plotting.
% simulationTime  Total simulation duration [s].
% dt              Sampling interval [s].
% seed            Random-number seed.
% initialEstimate Optional scalar structure with fields:
%                   position, velocity, acceleration.
%                 Each field may be dim-by-1 or dim-by-4.
% options         Optional scalar structure. Supported fields:
%
%   measurementMode       'angle_only' or 'angle_range'
%   maneuverMode          'none' or 'observability_aware'
%   maneuverAcceleration  watcher maneuver amplitude
%   maneuverFrequencyHz   maneuver frequency [Hz]
%   maneuverStartTime     maneuver start time [s]
%   maneuverStopTime      maneuver stop time [s]
%   sigmaBearingDeg       2-D bearing noise [deg]
%   sigmaAzimuthDeg       3-D azimuth noise [deg]
%   sigmaElevationDeg     3-D elevation noise [deg]
%   sigmaRange            range noise [position units]
%   sigmaAccelIncrement   acceleration random-walk increment std
%
% OUTPUT
% ------
% result          Truth, local estimates, watcher trajectories, maneuver
%                 commands, measurement logs, fusion results, diagnostics,
%                 plots, and RMSE summaries.

if nargin < 2 || isempty(makePlots),       makePlots = true; end
if nargin < 3 || isempty(simulationTime),  simulationTime = 40; end
if nargin < 4 || isempty(dt),              dt = 0.05; end
if nargin < 5 || isempty(seed),            seed = 7 + dim; end
if nargin < 6 || isempty(initialEstimate), initialEstimate = struct(); end
if nargin < 7 || isempty(options),         options = struct(); end

validateattributes(dim,{'numeric'},{'scalar','integer'});
if ~ismember(dim,[2,3])
    error('dim must be 2 or 3.');
end
validateattributes(simulationTime,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(dt,{'numeric'},{'scalar','positive','finite'});
validateattributes(seed,{'numeric'}, ...
    {'scalar','integer','nonnegative'});
if ~isstruct(initialEstimate) || ~isscalar(initialEstimate)
    error('initialEstimate must be a scalar structure or empty.');
end
if ~isstruct(options) || ~isscalar(options)
    error('options must be a scalar structure or empty.');
end

rng(seed);

%% Configuration
Nw = 4;
t  = (0:dt:simulationTime).';
N  = numel(t);

options = resolveOptions(options,simulationTime);

measurementMode = options.measurementMode;
maneuverMode    = options.maneuverMode;

sigmaBearing = deg2rad(options.sigmaBearingDeg);
sigmaAz      = deg2rad(options.sigmaAzimuthDeg);
sigmaEl      = deg2rad(options.sigmaElevationDeg);
sigmaRange   = options.sigmaRange;

epsGeometry = 1e-9;

if dim == 2
    % Initial planar watcher geometry.
    pWatcher0 = 100*[ ...
       -50,  50,   0,   0;
          0,   0, -50,  70];
else
    % Initial noncoplanar tetrahedral geometry.
    d = 30/sqrt(3);
    pWatcher0 = d*[ ...
         1,  1, -1, -1;
         1, -1,  1, -1;
         1, -1, -1,  1];
end

%% Smooth unknown target acceleration
aTrue = zeros(dim,N);

for k = 1:N
    tk = t(k);

    if dim == 2
        aTrue(:,k) = 100*[ ...
            0.025*sin(10*0.35*tk) + 0.015*cos(10*0.13*tk);
            0.020*cos(10*0.31*tk) - 0.010*sin(10*0.19*tk)];
    else
        aTrue(:,k) = 10*[ ...
            0.025*sin(0.35*tk) + 0.012*cos(0.11*tk);
            0.020*cos(0.31*tk) - 0.010*sin(0.19*tk);
            0.018*sin(0.27*tk + 0.6) + 0.009*cos(0.15*tk)];
    end
end

%% Integrate target truth trajectory
rTrue = zeros(dim,N);
vTrue = zeros(dim,N);

for k = 1:N-1
    vTrue(:,k+1) = vTrue(:,k) + aTrue(:,k)*dt;
    rTrue(:,k+1) = rTrue(:,k) + vTrue(:,k)*dt ...
        + 0.5*aTrue(:,k)*dt^2;
end

%% Constant-acceleration local EKF model
n = 3*dim;

F = [ ...
    eye(dim),        dt*eye(dim), 0.5*dt^2*eye(dim);
    zeros(dim),      eye(dim),    dt*eye(dim);
    zeros(dim),      zeros(dim),  eye(dim)];

% Discrete acceleration-random-walk increment model.
G = [ ...
    0.5*dt^2*eye(dim);
    dt*eye(dim);
    eye(dim)];

sigmaAccelIncrement = options.sigmaAccelIncrement;
Q = sigmaAccelIncrement^2*(G*G.');

P0 = diag([ ...
    50^2*ones(1,dim), ...
    0.1^2*ones(1,dim), ...
    0.05^2*ones(1,dim)]);

%% User-configurable local EKF initial means
defaultInitialPosition = repmat(rTrue(:,1),1,Nw) ...
    + 20*randn(dim,Nw);
defaultInitialVelocity = 0.1*randn(dim,Nw);
defaultInitialAcceleration = repmat(aTrue(:,1),1,Nw) ...
    + 0.05*randn(dim,Nw);

initialPosition = resolveInitialComponent( ...
    initialEstimate,'position',defaultInitialPosition,dim,Nw);
initialVelocity = resolveInitialComponent( ...
    initialEstimate,'velocity',defaultInitialVelocity,dim,Nw);
initialAcceleration = resolveInitialComponent( ...
    initialEstimate,'acceleration',defaultInitialAcceleration,dim,Nw);

initialState = [ ...
    initialPosition;
    initialVelocity;
    initialAcceleration];

%% Local-filter, watcher, and fusion storage
xhat       = zeros(n,N,Nw);
Plog       = zeros(n,n,Nw,N);
aLocal     = zeros(dim,N,Nw);

watcherPositionHistory = zeros(dim,N,Nw);
watcherVelocityHistory = zeros(dim,N,Nw);
watcherManeuverAcceleration = zeros(dim,N,Nw);

uMeasLog = zeros(dim,N,Nw);
uPredLog = zeros(dim,N,Nw);

bearingMeasurementLog   = nan(N,Nw);
azimuthMeasurementLog   = nan(N,Nw);
elevationMeasurementLog = nan(N,Nw);
rangeMeasurementLog     = nan(N,Nw);
rangePredictionLog      = nan(N,Nw);

nisLog = nan(N,Nw);
accelerationGainNorm = nan(N,Nw);

WLog        = zeros(dim,dim,Nw,N);
geometrySum = zeros(dim,dim,N);
geometryCondition = nan(1,N);

for j = 1:Nw
    xhat(:,1,j) = initialState(:,j);
    Plog(:,:,j,1) = P0;
    aLocal(:,1,j) = initialAcceleration(:,j);

    watcherPositionHistory(:,1,j) = pWatcher0(:,j);
    watcherVelocityHistory(:,1,j) = zeros(dim,1);
end

%% Initial measurements and LOS geometry
for j = 1:Nw
    pW0 = watcherPositionHistory(:,1,j);
    rhoTrue0 = rTrue(:,1)-pW0;
    rangeTrue0 = norm(rhoTrue0);

    if dim == 2
        bearingTrue0 = atan2(rhoTrue0(2),rhoTrue0(1));
        zBearing0 = wrapAngle( ...
            bearingTrue0 + sigmaBearing*randn);

        bearingMeasurementLog(1,j) = zBearing0;
        uMeasLog(:,1,j) = [cos(zBearing0);sin(zBearing0)];
    else
        [azTrue0,elTrue0] = cartesianToAzEl(rhoTrue0);

        zAz0 = wrapAngle(azTrue0 + sigmaAz*randn);
        zEl0 = clampElevation(elTrue0 + sigmaEl*randn);

        azimuthMeasurementLog(1,j) = zAz0;
        elevationMeasurementLog(1,j) = zEl0;
        uMeasLog(:,1,j) = azElToUnitLOS(zAz0,zEl0);
    end

    if strcmp(measurementMode,'angle_range')
        rangeMeasurementLog(1,j) = ...
            rangeTrue0 + sigmaRange*randn;
    end

    rh0 = xhat(1:dim,1,j)-pW0;
    rangePredictionLog(1,j) = norm(rh0);
    uPredLog(:,1,j) = normalizeVector(rh0);
end

[aSum,aGeo,aMean,W0,Omega0] = fusePredictedGeometry( ...
    aLocal(:,1,:),uPredLog(:,1,:),epsGeometry);

WLog(:,:,:,1)        = W0;
geometrySum(:,:,1)   = Omega0;
geometryCondition(1) = safeConditionNumber(Omega0);

%% EKF recursion with known watcher maneuver
for k = 1:N-1
    tk = t(k);

    for j = 1:Nw
        pW = watcherPositionHistory(:,k,j);
        vW = watcherVelocityHistory(:,k,j);

        % Known observability-aware watcher command. The controller uses
        % the local current target estimate, not the true target state.
        aW = computeWatcherManeuver( ...
            maneuverMode,tk,pW,xhat(1:dim,k,j), ...
            dim,j,Nw,options);

        watcherManeuverAcceleration(:,k,j) = aW;

        % Propagate known watcher kinematics.
        pWNext = pW + vW*dt + 0.5*aW*dt^2;
        vWNext = vW + aW*dt;

        watcherPositionHistory(:,k+1,j) = pWNext;
        watcherVelocityHistory(:,k+1,j) = vWNext;

        % Predict the inertial target state. The watcher maneuver does not
        % enter F because the EKF state is the inertial target state.
        xp = F*xhat(:,k,j);
        Pp = F*Plog(:,:,j,k)*F.' + Q;
        Pp = 0.5*(Pp+Pp.');

        % Relative geometry at time k+1.
        rhoTrue = rTrue(:,k+1)-pWNext;
        rangeTrue = norm(rhoTrue);

        rh = xp(1:dim)-pWNext;
        rr = max(norm(rh),1e-9);
        uh = rh/rr;

        uPredLog(:,k+1,j) = uh;
        rangePredictionLog(k+1,j) = rr;

        if dim == 2
            %% 2-D bearing measurement
            bearingTrue = atan2(rhoTrue(2),rhoTrue(1));
            zBearing = wrapAngle( ...
                bearingTrue + sigmaBearing*randn);
            bearingHat = atan2(uh(2),uh(1));

            HposBearing = [-uh(2),uh(1)]/rr;
            uMeasLog(:,k+1,j) = ...
                [cos(zBearing);sin(zBearing)];
            bearingMeasurementLog(k+1,j) = zBearing;

            if strcmp(measurementMode,'angle_only')
                innov = wrapAngle(zBearing-bearingHat);
                H = [HposBearing,zeros(1,2*dim)];
                R = sigmaBearing^2;
            else
                zRange = rangeTrue + sigmaRange*randn;
                rangeMeasurementLog(k+1,j) = zRange;

                innov = [ ...
                    wrapAngle(zBearing-bearingHat);
                    zRange-rr];

                HposRange = uh.';
                Hpos = [HposBearing;HposRange];

                H = [Hpos,zeros(2,2*dim)];
                R = diag([sigmaBearing^2,sigmaRange^2]);
            end

        else
            %% 3-D azimuth/elevation measurement
            [azTrue,elTrue] = cartesianToAzEl(rhoTrue);

            zAz = wrapAngle(azTrue + sigmaAz*randn);
            zEl = clampElevation(elTrue + sigmaEl*randn);

            dxh = rh(1);
            dyh = rh(2);
            dzh = rh(3);

            rhoXYh = max(hypot(dxh,dyh),1e-9);
            rho2h  = max(dot(rh,rh),1e-12);

            azHat = atan2(dyh,dxh);
            elHat = atan2(dzh,rhoXYh);

            HposAngle = [ ...
                -dyh/(rhoXYh^2), ...
                 dxh/(rhoXYh^2), ...
                 0; ...
                -(dxh*dzh)/(rho2h*rhoXYh), ...
                -(dyh*dzh)/(rho2h*rhoXYh), ...
                  rhoXYh/rho2h];

            uMeasLog(:,k+1,j) = azElToUnitLOS(zAz,zEl);
            azimuthMeasurementLog(k+1,j) = zAz;
            elevationMeasurementLog(k+1,j) = zEl;

            if strcmp(measurementMode,'angle_only')
                innov = [ ...
                    wrapAngle(zAz-azHat);
                    zEl-elHat];

                H = [HposAngle,zeros(2,2*dim)];
                R = diag([sigmaAz^2,sigmaEl^2]);
            else
                zRange = rangeTrue + sigmaRange*randn;
                rangeMeasurementLog(k+1,j) = zRange;

                innov = [ ...
                    wrapAngle(zAz-azHat);
                    zEl-elHat;
                    zRange-rr];

                HposRange = uh.';
                Hpos = [HposAngle;HposRange];

                H = [Hpos,zeros(3,2*dim)];
                R = diag([ ...
                    sigmaAz^2,sigmaEl^2,sigmaRange^2]);
            end
        end

        % EKF correction.
        S = H*Pp*H.' + R;
        S = 0.5*(S+S.');

        K = (Pp*H.')/S;
        xn = xp + K*innov;

        IminusKH = eye(n)-K*H;
        Pn = IminusKH*Pp*IminusKH.' + K*R*K.';
        Pn = 0.5*(Pn+Pn.');

        xhat(:,k+1,j)   = xn;
        Plog(:,:,j,k+1) = Pn;
        aLocal(:,k+1,j) = xn(2*dim+(1:dim));

        nisLog(k+1,j) = innov.'*(S\innov);
        accelerationGainNorm(k+1,j) = norm( ...
            K(2*dim+(1:dim),:),'fro');
    end

    % Fuse the local current-acceleration point estimates using the current
    % prior-predicted LOS geometry.
    [aSum(:,k+1),aGeo(:,k+1),aMean(:,k+1),Wk,Omegak] = ...
        fusePredictedGeometry( ...
            aLocal(:,k+1,:),uPredLog(:,k+1,:),epsGeometry);

    WLog(:,:,:,k+1)        = Wk;
    geometrySum(:,:,k+1)   = Omegak;
    geometryCondition(k+1) = safeConditionNumber(Omegak);
end

% Log the final-time command for plotting.
for j = 1:Nw
    watcherManeuverAcceleration(:,N,j) = ...
        computeWatcherManeuver( ...
        maneuverMode,t(N),watcherPositionHistory(:,N,j), ...
        xhat(1:dim,N,j),dim,j,Nw,options);
end

%% Local acceleration error diagnostics
transverseErrorNorm = zeros(N,Nw);
radialErrorAbs      = zeros(N,Nw);

for k = 1:N
    for j = 1:Nw
        u = normalizeVector(uPredLog(:,k,j));
        Pperp = eye(dim)-u*u.';
        eLocal = aLocal(:,k,j)-aTrue(:,k);

        transverseErrorNorm(k,j) = norm(Pperp*eLocal);
        radialErrorAbs(k,j) = abs(u.'*eLocal);
    end
end

%% Maneuver observability-excitation diagnostic
% This diagnostic compares the maneuver-induced relative displacement with
% the nominal LOS profile produced by a stationary watcher. Truth is used
% here only for post-processing, not for the controller or EKF.
observabilityExcitation = zeros(N,Nw);
maneuverLOSAlignment = nan(N,Nw);

for k = 1:N
    for j = 1:Nw
        rhoNominal = rTrue(:,k)-pWatcher0(:,j);
        uNominal = normalizeVector(rhoNominal);
        PperpNominal = eye(dim)-uNominal*uNominal.';

        deltaRelativePosition = -( ...
            watcherPositionHistory(:,k,j)-pWatcher0(:,j));

        observabilityExcitation(k,j) = norm( ...
            PperpNominal*deltaRelativePosition);

        deltaNorm = norm(deltaRelativePosition);
        if deltaNorm > 1e-12
            maneuverLOSAlignment(k,j) = abs( ...
                uNominal.'*deltaRelativePosition)/deltaNorm;
        end
    end
end

%% Package result
result = struct( ...
    'time',t, ...
    'targetPosition',rTrue, ...
    'targetVelocity',vTrue, ...
    'trueAcceleration',aTrue, ...
    'watcherInitialPosition',pWatcher0, ...
    'watcherPosition',watcherPositionHistory, ...
    'watcherPositionHistory',watcherPositionHistory, ...
    'watcherVelocityHistory',watcherVelocityHistory, ...
    'watcherManeuverAcceleration', ...
        watcherManeuverAcceleration, ...
    'measurementMode',measurementMode, ...
    'maneuverMode',maneuverMode, ...
    'options',options, ...
    'localState',xhat, ...
    'initialStateEstimate',initialState, ...
    'initialPositionEstimate',initialPosition, ...
    'initialVelocityEstimate',initialVelocity, ...
    'initialAccelerationEstimate',initialAcceleration, ...
    'localCovariance',Plog, ...
    'localAcceleration',aLocal, ...
    'measuredLOS',uMeasLog, ...
    'predictedLOS',uPredLog, ...
    'fusionLOS',uPredLog, ...
    'fusionLOSSource','EKF prior predicted LOS', ...
    'bearingMeasurement',bearingMeasurementLog, ...
    'azimuthMeasurement',azimuthMeasurementLog, ...
    'elevationMeasurement',elevationMeasurementLog, ...
    'rangeMeasurement',rangeMeasurementLog, ...
    'rangePrediction',rangePredictionLog, ...
    'rangeSigma',sigmaRange, ...
    'NIS',nisLog, ...
    'accelerationGainNorm',accelerationGainNorm, ...
    'rawSum',aSum, ...
    'geometryFusion',aGeo, ...
    'rawMean',aMean, ...
    'geometryWeight',WLog, ...
    'geometryInformationSum',geometrySum, ...
    'geometryCondition',geometryCondition, ...
    'transverseErrorNorm',transverseErrorNorm, ...
    'radialErrorAbs',radialErrorAbs, ...
    'observabilityExcitation',observabilityExcitation, ...
    'maneuverLOSAlignment',maneuverLOSAlignment, ...
    'seed',seed, ...
    'dimension',dim, ...
    'bearingSigmaDeg',options.sigmaBearingDeg, ...
    'azimuthSigmaDeg',options.sigmaAzimuthDeg, ...
    'elevationSigmaDeg',options.sigmaElevationDeg);

%% RMSE summary
caseName = ["raw-sum";"geometry-aware";"raw-mean"];
estimate = {aSum,aGeo,aMean};

accelerationRMSE = zeros(3,1);
finalAccelerationRMSE = zeros(3,1);
idxFinal = max(1,round(0.9*N)):N;

for i = 1:3
    e = vecnorm(estimate{i}-aTrue,2,1);
    accelerationRMSE(i) = sqrt(mean(e.^2));
    finalAccelerationRMSE(i) = sqrt( ...
        mean(e(idxFinal).^2));
end

result.summary = table( ...
    caseName,accelerationRMSE,finalAccelerationRMSE);

fprintf(['\n4-watcher %d-D local EKF fusion\n' ...
    'measurement mode: %s\nmaneuver mode: %s\n'], ...
    dim,measurementMode,maneuverMode);
disp(result.summary);

if makePlots
    result.figure = plotResult(result);
end
end

function options = resolveOptions(options,simulationTime)
%RESOLVEOPTIONS Fill and validate measurement and maneuver options.

defaults = struct( ...
    'measurementMode','angle_only', ...
    'maneuverMode','observability_aware', ...
    'maneuverAcceleration',2.0, ...
    'maneuverFrequencyHz',0.05, ...
    'maneuverStartTime',2.0, ...
    'maneuverStopTime',simulationTime, ...
    'sigmaBearingDeg',0.2, ...
    'sigmaAzimuthDeg',0.2, ...
    'sigmaElevationDeg',0.2, ...
    'sigmaRange',5.0, ...
    'sigmaAccelIncrement',0.9);

names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(options,name) || isempty(options.(name))
        options.(name) = defaults.(name);
    end
end

options.measurementMode = validatestring( ...
    options.measurementMode,{'angle_only','angle_range'});
options.maneuverMode = validatestring( ...
    options.maneuverMode,{'none','observability_aware'});

validateattributes(options.maneuverAcceleration, ...
    {'numeric'},{'scalar','nonnegative','finite'});
validateattributes(options.maneuverFrequencyHz, ...
    {'numeric'},{'scalar','positive','finite'});
validateattributes(options.maneuverStartTime, ...
    {'numeric'},{'scalar','nonnegative','finite'});
validateattributes(options.maneuverStopTime, ...
    {'numeric'},{'scalar','nonnegative','finite'});

if options.maneuverStopTime < options.maneuverStartTime
    error(['options.maneuverStopTime must be greater than or ' ...
        'equal to options.maneuverStartTime.']);
end

validateattributes(options.sigmaBearingDeg, ...
    {'numeric'},{'scalar','positive','finite'});
validateattributes(options.sigmaAzimuthDeg, ...
    {'numeric'},{'scalar','positive','finite'});
validateattributes(options.sigmaElevationDeg, ...
    {'numeric'},{'scalar','positive','finite'});
validateattributes(options.sigmaRange, ...
    {'numeric'},{'scalar','positive','finite'});
validateattributes(options.sigmaAccelIncrement, ...
    {'numeric'},{'scalar','nonnegative','finite'});
end

function aW = computeWatcherManeuver( ...
    maneuverMode,t,pWatcher,targetPositionEstimate, ...
    dim,watcherIndex,Nw,options)
%COMPUTEWATCHERMANEUVER Known LOS-transverse watcher maneuver.
%
% The command uses estimated geometry only. In observability-aware mode,
% the instantaneous command direction is orthogonal to the predicted LOS.

aW = zeros(dim,1);

if strcmp(maneuverMode,'none')
    return;
end

if t < options.maneuverStartTime || ...
        t > options.maneuverStopTime
    return;
end

relativeEstimate = targetPositionEstimate-pWatcher;
if norm(relativeEstimate) < 1e-9
    return;
end

u = normalizeVector(relativeEstimate);
tau = t-options.maneuverStartTime;
carrier = cos(2*pi*options.maneuverFrequencyHz*tau);

if dim == 2
    transverseDirection = [-u(2);u(1)];

    % Alternate the signed direction across watchers.
    signedDirection = (-1)^(watcherIndex-1) ...
        * transverseDirection;
else
    Pperp = eye(3)-u*u.';

    candidateAxes = [ ...
        eye(3), ...
        normalizeVector([1;1;1])];

    % Start each watcher from a different preferred reference axis.
    preferred = 1+mod(watcherIndex-1,size(candidateAxes,2));
    order = [preferred, ...
        setdiff(1:size(candidateAxes,2),preferred)];

    tangent = zeros(3,1);
    for q = order
        candidate = Pperp*candidateAxes(:,q);
        if norm(candidate) > 1e-8
            tangent = normalizeVector(candidate);
            break;
        end
    end

    if norm(tangent) < 1e-8
        return;
    end

    signedDirection = (-1)^(watcherIndex-1)*tangent;
end

aW = options.maneuverAcceleration ...
    * carrier*normalizeVector(signedDirection);

% Numerical check: command should be transverse to the estimated LOS.
if abs(u.'*aW) > 1e-9*max(1,norm(aW))
    error('Generated maneuver is not transverse to the predicted LOS.');
end

% Nw is retained in the interface for future phase scheduling.
if Nw < 1
    error('Nw must be positive.');
end
end

function [aSum,aGeo,aMean,B,OmegaSum] = ...
    fusePredictedGeometry(aLocal,uPred,epsReg)
%FUSEPREDICTEDGEOMETRY Fuse local accelerations using predicted LOS.
%
%   Omega_i = I-uHat_i^- (uHat_i^-)^T
%
%   a_geo = (eps*I + sum_i Omega_i)^(-1)
%           sum_i Omega_i a_i.

dim = size(aLocal,1);
Nw  = size(aLocal,3);

Omega = zeros(dim,dim,Nw);
OmegaSum = epsReg*eye(dim);

for j = 1:Nw
    u = normalizeVector(uPred(:,1,j));
    Omega(:,:,j) = eye(dim)-u*u.';
    Omega(:,:,j) = 0.5*(Omega(:,:,j)+Omega(:,:,j).');
    OmegaSum = OmegaSum+Omega(:,:,j);
end

B = zeros(dim,dim,Nw);
for j = 1:Nw
    B(:,:,j) = OmegaSum\Omega(:,:,j);
end

aSum = sum(aLocal,3);
aMean = aSum/Nw;

aGeo = zeros(dim,1);
for j = 1:Nw
    aGeo = aGeo+B(:,:,j)*aLocal(:,:,j);
end
end

function value = resolveInitialComponent( ...
    initialEstimate,fieldName,defaultValue,dim,Nw)
%RESOLVEINITIALCOMPONENT Validate and expand one initial-state component.

if ~isfield(initialEstimate,fieldName) || ...
        isempty(initialEstimate.(fieldName))
    value = defaultValue;
    return;
end

candidate = initialEstimate.(fieldName);
validateattributes(candidate,{'numeric'}, ...
    {'real','finite','nonempty'}, ...
    mfilename,['initialEstimate.' fieldName]);

if isvector(candidate) && numel(candidate) == dim
    value = repmat(candidate(:),1,Nw);
elseif isequal(size(candidate),[dim,Nw])
    value = candidate;
else
    error(['initialEstimate.%s must be dim-by-1 or dim-by-%d. ' ...
        'For dim=%d, the received size was %s.'], ...
        fieldName,Nw,dim,mat2str(size(candidate)));
end
end

function [az,el] = cartesianToAzEl(r)
%CARTESIANTOAZEL Convert Cartesian relative position to azimuth/elevation.

dx = r(1);
dy = r(2);
dz = r(3);

rhoXY = max(hypot(dx,dy),1e-12);
az = atan2(dy,dx);
el = atan2(dz,rhoXY);
end

function u = azElToUnitLOS(az,el)
%AZELTOUNITLOS Convert azimuth/elevation to a unit LOS vector.

u = [ ...
    cos(el)*cos(az);
    cos(el)*sin(az);
    sin(el)];

u = normalizeVector(u);
end

function x = wrapAngle(x)
%WRAPANGLE Wrap angles elementwise to [-pi,pi].

x = atan2(sin(x),cos(x));
end

function el = clampElevation(el)
%CLAMPELEVATION Keep elevation away from chart singularities.

margin = 1e-8;
el = min(max(el,-pi/2+margin),pi/2-margin);
end

function u = normalizeVector(v)
%NORMALIZEVECTOR Safely normalize a vector.

nv = norm(v);
if nv < realmin
    error('Cannot normalize a zero vector.');
end
u = v/nv;
end

function c = safeConditionNumber(A)
%SAFECONDITIONNUMBER Return Inf for singular matrices.

s = svd(A);
if isempty(s) || s(end) <= eps(max(s))
    c = Inf;
else
    c = s(1)/s(end);
end
end

function fig = plotResult(r)
%PLOTRESULT Plot fusion, local estimates, and watcher maneuvers.

t   = r.time;
a   = r.trueAcceleration;
dim = r.dimension;
Nw  = size(r.localState,3);

fig = gobjects(3,1);

%% Figure 1: fused acceleration comparison
fig(1) = figure( ...
    'Name',sprintf('%d-D %s EKF acceleration fusion', ...
    dim,r.measurementMode), ...
    'Color','w');

tiledlayout(dim+2,1, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for q = 1:dim
    nexttile;
    plot(t,a(q,:),'k','LineWidth',1.3);
    hold on;
    plot(t,r.rawSum(q,:),'b');
    plot(t,r.geometryFusion(q,:),'r--','LineWidth',1.1);
    plot(t,r.rawMean(q,:),'g-.');
    grid on;
    ylabel(sprintf('a_%d',q));

    if q == 1
        legend( ...
            'truth','raw sum','geometry-aware','raw mean', ...
            'Location','best');
    end
end

nexttile;
plot(t,vecnorm(r.rawSum-a,2,1),'b');
hold on;
plot(t,vecnorm(r.geometryFusion-a,2,1), ...
    'r--','LineWidth',1.1);
plot(t,vecnorm(r.rawMean-a,2,1),'g-.');
grid on;
ylabel('error norm');
legend( ...
    'raw sum','geometry-aware','raw mean', ...
    'Location','best');

nexttile;
plot(t,r.geometryCondition,'k');
grid on;
xlabel('time [s]');
ylabel('cond(\Sigma P_i^\perp)');

%% Figure 2: local position, velocity, and acceleration estimates
fig(2) = figure( ...
    'Name',sprintf('%d-D local watcher EKF state estimates',dim), ...
    'Color','w');

tiledlayout(3,dim, ...
    'TileSpacing','compact', ...
    'Padding','compact');

truthByRow = { ...
    r.targetPosition, ...
    r.targetVelocity, ...
    r.trueAcceleration};

stateNames   = {'position','velocity','acceleration'};
stateSymbols = {'r','v','a'};
stateOffsets = [0,dim,2*dim];

legendText = [{'truth'}, ...
    arrayfun(@(j)sprintf('watcher %d',j), ...
    1:Nw,'UniformOutput',false)];

for row = 1:3
    for q = 1:dim
        nexttile;

        plot(t,truthByRow{row}(q,:),'k','LineWidth',1.5);
        hold on;

        stateIndex = stateOffsets(row)+q;
        for j = 1:Nw
            plot(t,squeeze(r.localState(stateIndex,:,j)), ...
                'LineWidth',0.9);
        end

        grid on;
        xlabel('time [s]');
        ylabel(sprintf('%s_%d',stateSymbols{row},q));
        title(sprintf('%s component %d',stateNames{row},q));

        if row == 1 && q == 1
            legend(legendText,'Location','best');
        end
    end
end

%% Figure 3: watcher trajectories and excitation diagnostics
fig(3) = figure( ...
    'Name',sprintf('%d-D watcher observability maneuver',dim), ...
    'Color','w');

tiledlayout(3,1, ...
    'TileSpacing','compact', ...
    'Padding','compact');

nexttile;
if dim == 2
    plot(r.targetPosition(1,:),r.targetPosition(2,:), ...
        'k','LineWidth',1.5);
    hold on;
    for j = 1:Nw
        pW = squeeze(r.watcherPositionHistory(:,:,j));
        plot(pW(1,:),pW(2,:),'LineWidth',1.0);
        plot(pW(1,1),pW(2,1),'o');
    end
    xlabel('x');
    ylabel('y');
    axis equal;
else
    plot3(r.targetPosition(1,:), ...
        r.targetPosition(2,:), ...
        r.targetPosition(3,:), ...
        'k','LineWidth',1.5);
    hold on;
    for j = 1:Nw
        pW = squeeze(r.watcherPositionHistory(:,:,j));
        plot3(pW(1,:),pW(2,:),pW(3,:), ...
            'LineWidth',1.0);
        plot3(pW(1,1),pW(2,1),pW(3,1),'o');
    end
    xlabel('x');
    ylabel('y');
    zlabel('z');
    axis equal;
    view(3);
end
grid on;
title(sprintf('target and watcher trajectories: %s', ...
    r.maneuverMode));

nexttile;
hold on;
for j = 1:Nw
    aW = squeeze(r.watcherManeuverAcceleration(:,:,j));
    plot(t,vecnorm(aW,2,1),'LineWidth',1.0);
end
grid on;
ylabel('||a_w||');
title('known watcher maneuver magnitude');
legend(arrayfun(@(j)sprintf('watcher %d',j), ...
    1:Nw,'UniformOutput',false),'Location','best');

nexttile;
plot(t,r.observabilityExcitation,'LineWidth',1.0);
grid on;
xlabel('time [s]');
ylabel('transverse displacement');
title('maneuver-induced transverse LOS-profile excitation');
end
