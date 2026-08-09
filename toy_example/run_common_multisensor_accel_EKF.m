function result = run_common_multisensor_accel_EKF( ...
    dim,makePlots,simulationTime,dt,seed,options)
%RUN_COMMON_MULTISENSOR_ACCEL_EKF Common-state angle-only acceleration EKF.
%
% This function is the second-stage benchmark. One common EKF estimates
%
%   x = [r; v; a]
%
% using all four watchers' simultaneous angle measurements in one stacked
% measurement update. It runs two cases with identical angle-noise samples:
%
%   1. stationary watchers,
%   2. finite calibrated pulse-pair watcher maneuvers.
%
% This file intentionally does not perform local acceleration fusion. Its
% purpose is to determine whether the combined multi-watcher geometry can
% estimate the augmented acceleration state before returning to distributed
% local filters and output fusion.
%
% INPUT options fields
% --------------------
% sigmaBearingDeg, sigmaAzimuthDeg, sigmaElevationDeg
% burnAcceleration, burnStartTime, burnDuration
% watcherRadius
% sigmaJerk
% initialPositionError, initialVelocityError, initialAcceleration
% initialPositionSigma, initialVelocitySigma, initialAccelerationSigma
% truthAccelerationScale, truthFrequencyScale
%
% OUTPUT
% ------
% result.noManeuver
% result.pulsePair
% result.summary

if nargin < 2 || isempty(makePlots),      makePlots = true; end
if nargin < 3 || isempty(simulationTime), simulationTime = 60; end
if nargin < 4 || isempty(dt),             dt = 0.05; end
if nargin < 5 || isempty(seed),           seed = 51 + dim; end
if nargin < 6 || isempty(options),        options = struct(); end

validateattributes(dim,{'numeric'},{'scalar','integer'});
if ~ismember(dim,[2,3])
    error('dim must be 2 or 3.');
end
if ~isstruct(options) || ~isscalar(options)
    error('options must be a scalar structure or empty.');
end

options = resolveOptions(options,dim,simulationTime);
rng(seed);

Nw = 4;
t = (0:dt:simulationTime).';
N = numel(t);

pWatcher0 = makeWatcherGeometry(dim,options.watcherRadius);

aTrue = makeTruthAcceleration(dim,t, ...
    options.truthAccelerationScale, ...
    options.truthFrequencyScale);

rTrue = zeros(dim,N);
vTrue = zeros(dim,N);
for k = 1:N-1
    vTrue(:,k+1) = vTrue(:,k)+aTrue(:,k)*dt;
    rTrue(:,k+1) = rTrue(:,k)+vTrue(:,k)*dt ...
        + 0.5*aTrue(:,k)*dt^2;
end

x0 = [ ...
    rTrue(:,1)+options.initialPositionError(:); ...
    vTrue(:,1)+options.initialVelocityError(:); ...
    options.initialAcceleration(:)];

% Freeze maneuver directions from the common initial estimated geometry.
burnDirection = zeros(dim,Nw);
for j = 1:Nw
    u0 = normalizeVector(x0(1:dim)-pWatcher0(:,j));
    burnDirection(:,j) = makeTransverseDirection(u0,j,dim);
end

noise = struct();
if dim == 2
    noise.bearing = deg2rad(options.sigmaBearingDeg)*randn(N,Nw);
else
    noise.azimuth = deg2rad(options.sigmaAzimuthDeg)*randn(N,Nw);
    noise.elevation = deg2rad(options.sigmaElevationDeg)*randn(N,Nw);
end

common = struct( ...
    'time',t, ...
    'targetPosition',rTrue, ...
    'targetVelocity',vTrue, ...
    'trueAcceleration',aTrue, ...
    'watcherInitialPosition',pWatcher0, ...
    'initialStateEstimate',x0, ...
    'burnDirection',burnDirection, ...
    'options',options, ...
    'dimension',dim, ...
    'seed',seed);

noManeuver = simulateCase('none',common,noise,dt);
pulsePair = simulateCase('pulse_pair',common,noise,dt);

result = struct( ...
    'time',t, ...
    'dimension',dim, ...
    'options',options, ...
    'targetPosition',rTrue, ...
    'targetVelocity',vTrue, ...
    'trueAcceleration',aTrue, ...
    'noManeuver',noManeuver, ...
    'pulsePair',pulsePair);

caseName = ["no-maneuver";"finite-pulse-pair"];
positionRMSE = [noManeuver.positionRMSE;pulsePair.positionRMSE];
velocityRMSE = [noManeuver.velocityRMSE;pulsePair.velocityRMSE];
accelerationRMSE = [ ...
    noManeuver.accelerationRMSE;
    pulsePair.accelerationRMSE];

result.summary = table( ...
    caseName,positionRMSE,velocityRMSE,accelerationRMSE);

fprintf('\nCommon-state multi-watcher angle-only [r;v;a] EKF\n');
disp(result.summary);

if makePlots
    result.figure = plotResult(result);
end
end

function caseResult = simulateCase(mode,common,noise,dt)

t = common.time;
N = numel(t);
dim = common.dimension;
Nw = size(common.watcherInitialPosition,2);
n = 3*dim;

F = [ ...
    eye(dim),dt*eye(dim),0.5*dt^2*eye(dim); ...
    zeros(dim),eye(dim),dt*eye(dim); ...
    zeros(dim),zeros(dim),eye(dim)];

sj2 = common.options.sigmaJerk^2;
Qaxis = sj2*[ ...
    dt^5/20,dt^4/8,dt^3/6; ...
    dt^4/8,dt^3/3,dt^2/2; ...
    dt^3/6,dt^2/2,dt];
Q = kron(Qaxis,eye(dim));

P0 = diag([ ...
    common.options.initialPositionSigma^2*ones(1,dim), ...
    common.options.initialVelocitySigma^2*ones(1,dim), ...
    common.options.initialAccelerationSigma^2*ones(1,dim)]);

xhat = zeros(n,N);
Plog = zeros(n,n,N);
xhat(:,1) = common.initialStateEstimate;
Plog(:,:,1) = P0;

watcherPosition = zeros(dim,N,Nw);
watcherVelocity = zeros(dim,N,Nw);
watcherAcceleration = zeros(dim,N,Nw);
for j = 1:Nw
    watcherPosition(:,1,j) = common.watcherInitialPosition(:,j);
end

nis = nan(1,N);
normalizedNIS = nan(1,N);

for k = 1:N-1
    for j = 1:Nw
        aW = pulsePairCommand( ...
            mode,t(k),common.burnDirection(:,j),common.options);
        watcherAcceleration(:,k,j) = aW;

        pW = watcherPosition(:,k,j);
        vW = watcherVelocity(:,k,j);

        watcherPosition(:,k+1,j) = ...
            pW+vW*dt+0.5*aW*dt^2;
        watcherVelocity(:,k+1,j) = vW+aW*dt;
    end

    xp = F*xhat(:,k);
    Pp = F*Plog(:,:,k)*F.'+Q;
    Pp = 0.5*(Pp+Pp.');

    if dim == 2
        H = zeros(Nw,n);
        innov = zeros(Nw,1);
        R = deg2rad(common.options.sigmaBearingDeg)^2*eye(Nw);

        for j = 1:Nw
            pW = watcherPosition(:,k+1,j);
            rhoTrue = common.targetPosition(:,k+1)-pW;
            rhoHat = xp(1:dim)-pW;
            rr = max(norm(rhoHat),1e-9);
            u = rhoHat/rr;

            zTrue = atan2(rhoTrue(2),rhoTrue(1));
            z = wrapAngle(zTrue+noise.bearing(k+1,j));
            zHat = atan2(u(2),u(1));

            innov(j) = wrapAngle(z-zHat);
            Hpos = [-u(2),u(1)]/rr;
            H(j,:) = [Hpos,zeros(1,2*dim)];
        end
    else
        H = zeros(2*Nw,n);
        innov = zeros(2*Nw,1);
        Rsingle = diag([ ...
            deg2rad(common.options.sigmaAzimuthDeg)^2, ...
            deg2rad(common.options.sigmaElevationDeg)^2]);
        R = kron(eye(Nw),Rsingle);

        for j = 1:Nw
            pW = watcherPosition(:,k+1,j);
            rhoTrue = common.targetPosition(:,k+1)-pW;
            rhoHat = xp(1:dim)-pW;

            [azTrue,elTrue] = cartesianToAzEl(rhoTrue);
            zAz = wrapAngle(azTrue+noise.azimuth(k+1,j));
            zEl = clampElevation( ...
                elTrue+noise.elevation(k+1,j));

            dx = rhoHat(1);
            dy = rhoHat(2);
            dz = rhoHat(3);
            rhoXY = max(hypot(dx,dy),1e-9);
            rho2 = max(dot(rhoHat,rhoHat),1e-12);

            azHat = atan2(dy,dx);
            elHat = atan2(dz,rhoXY);

            rows = 2*j-1:2*j;
            innov(rows) = [wrapAngle(zAz-azHat);zEl-elHat];

            Hpos = [ ...
                -dy/rhoXY^2,dx/rhoXY^2,0; ...
                -(dx*dz)/(rho2*rhoXY), ...
                -(dy*dz)/(rho2*rhoXY), ...
                 rhoXY/rho2];

            H(rows,:) = [Hpos,zeros(2,2*dim)];
        end
    end

    S = H*Pp*H.'+R;
    S = 0.5*(S+S.');
    K = (Pp*H.')/S;

    xn = xp+K*innov;
    ImKH = eye(n)-K*H;
    Pn = ImKH*Pp*ImKH.'+K*R*K.';
    Pn = 0.5*(Pn+Pn.');

    xhat(:,k+1) = xn;
    Plog(:,:,k+1) = Pn;

    nis(k+1) = innov.'*(S\innov);
    normalizedNIS(k+1) = nis(k+1)/numel(innov);
end

for j = 1:Nw
    watcherAcceleration(:,N,j) = pulsePairCommand( ...
        mode,t(N),common.burnDirection(:,j),common.options);
end

ePosition = xhat(1:dim,:)-common.targetPosition;
eVelocity = xhat(dim+(1:dim),:)-common.targetVelocity;
eAcceleration = xhat(2*dim+(1:dim),:)-common.trueAcceleration;

caseResult = struct( ...
    'maneuverMode',mode, ...
    'stateEstimate',xhat, ...
    'covariance',Plog, ...
    'watcherPositionHistory',watcherPosition, ...
    'watcherVelocityHistory',watcherVelocity, ...
    'watcherManeuverAcceleration',watcherAcceleration, ...
    'NIS',nis, ...
    'normalizedNIS',normalizedNIS, ...
    'positionErrorNorm',vecnorm(ePosition,2,1), ...
    'velocityErrorNorm',vecnorm(eVelocity,2,1), ...
    'accelerationErrorNorm',vecnorm(eAcceleration,2,1), ...
    'positionRMSE',sqrt(mean(vecnorm(ePosition,2,1).^2)), ...
    'velocityRMSE',sqrt(mean(vecnorm(eVelocity,2,1).^2)), ...
    'accelerationRMSE',sqrt(mean( ...
        vecnorm(eAcceleration,2,1).^2)));
end

function aW = pulsePairCommand(mode,t,direction,options)

aW = zeros(size(direction));
if strcmp(mode,'none')
    return;
end

t0 = options.burnStartTime;
Tb = options.burnDuration;

if t >= t0 && t < t0+Tb
    aW = options.burnAcceleration*direction;
elseif t >= t0+Tb && t < t0+2*Tb
    aW = -options.burnAcceleration*direction;
end
end

function options = resolveOptions(options,dim,simulationTime)

defaults = struct( ...
    'sigmaBearingDeg',0.2, ...
    'sigmaAzimuthDeg',0.2, ...
    'sigmaElevationDeg',0.2, ...
    'burnAcceleration',2.0, ...
    'burnStartTime',10.0, ...
    'burnDuration',5.0, ...
    'watcherRadius',1000, ...
    'sigmaJerk',1.0, ...
    'initialPositionError',defaultPositionError(dim), ...
    'initialVelocityError',defaultVelocityError(dim), ...
    'initialAcceleration',zeros(dim,1), ...
    'initialPositionSigma',500, ...
    'initialVelocitySigma',20, ...
    'initialAccelerationSigma',5, ...
    'truthAccelerationScale',1.0, ...
    'truthFrequencyScale',1.0);

names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(options,name) || isempty(options.(name))
        options.(name) = defaults.(name);
    end
end

if options.burnStartTime+2*options.burnDuration > simulationTime
    error('The two-pulse maneuver must finish before simulationTime.');
end
validateattributes(options.initialPositionError,{'numeric'}, ...
    {'vector','numel',dim,'finite'});
validateattributes(options.initialVelocityError,{'numeric'}, ...
    {'vector','numel',dim,'finite'});
validateattributes(options.initialAcceleration,{'numeric'}, ...
    {'vector','numel',dim,'finite'});
end

function e = defaultPositionError(dim)
if dim == 2
    e = [250;-180];
else
    e = [250;-180;120];
end
end

function e = defaultVelocityError(dim)
if dim == 2
    e = [4;-3];
else
    e = [4;-3;2];
end
end

function a = makeTruthAcceleration(dim,t,scale,freqScale)

N = numel(t);
a = zeros(dim,N);

for k = 1:N
    tk = t(k);
    if dim == 2
        a(:,k) = scale*100*[ ...
            0.025*sin(freqScale*0.35*tk) ...
                + 0.015*cos(freqScale*0.13*tk); ...
            0.020*cos(freqScale*0.31*tk) ...
                - 0.010*sin(freqScale*0.19*tk)];
    else
        a(:,k) = scale*10*[ ...
            0.025*sin(freqScale*0.35*tk) ...
                + 0.012*cos(freqScale*0.11*tk); ...
            0.020*cos(freqScale*0.31*tk) ...
                - 0.010*sin(freqScale*0.19*tk); ...
            0.018*sin(freqScale*0.27*tk+0.6) ...
                + 0.009*cos(freqScale*0.15*tk)];
    end
end
end

function p = makeWatcherGeometry(dim,radius)
if dim == 2
    p = radius*[ ...
        -1,1,0,0; ...
         0,0,-1,1];
else
    d = radius/sqrt(3);
    p = d*[ ...
         1,1,-1,-1; ...
         1,-1,1,-1; ...
         1,-1,-1,1];
end
end

function d = makeTransverseDirection(u,j,dim)
if dim == 2
    d = [-u(2);u(1)];
else
    P = eye(3)-u*u.';
    axesSet = eye(3);
    preferred = 1+mod(j-1,3);
    order = [preferred,setdiff(1:3,preferred)];
    d = zeros(3,1);
    for q = order
        candidate = P*axesSet(:,q);
        if norm(candidate) > 1e-8
            d = normalizeVector(candidate);
            break;
        end
    end
end
d = (-1)^(j-1)*normalizeVector(d);
end

function [az,el] = cartesianToAzEl(r)
rhoXY = max(hypot(r(1),r(2)),1e-12);
az = atan2(r(2),r(1));
el = atan2(r(3),rhoXY);
end

function x = wrapAngle(x)
x = atan2(sin(x),cos(x));
end

function el = clampElevation(el)
margin = 1e-8;
el = min(max(el,-pi/2+margin),pi/2-margin);
end

function u = normalizeVector(v)
nv = norm(v);
if nv < realmin
    error('Cannot normalize a zero vector.');
end
u = v/nv;
end

function fig = plotResult(result)

t = result.time;
dim = result.dimension;
truth = { ...
    result.targetPosition, ...
    result.targetVelocity, ...
    result.trueAcceleration};
labels = {'r','v','a'};

fig = gobjects(2,1);

fig(1) = figure( ...
    'Name','Common multisensor acceleration EKF', ...
    'Color','w');
tiledlayout(3,dim,'TileSpacing','compact','Padding','compact');

for row = 1:3
    idx = (row-1)*dim+(1:dim);
    for q = 1:dim
        nexttile;
        plot(t,truth{row}(q,:),'k','LineWidth',1.4);
        hold on;
        plot(t,result.noManeuver.stateEstimate(idx(q),:), ...
            'LineWidth',1.0);
        plot(t,result.pulsePair.stateEstimate(idx(q),:), ...
            '--','LineWidth',1.0);
        grid on;
        ylabel(sprintf('%s_%d',labels{row},q));
        if row == 1 && q == 1
            legend('truth','no maneuver','pulse pair', ...
                'Location','best');
        end
        if row == 3
            xlabel('time [s]');
        end
    end
end

fig(2) = figure( ...
    'Name','Common EKF error and consistency diagnostics', ...
    'Color','w');
tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t,result.noManeuver.positionErrorNorm);
hold on;
plot(t,result.pulsePair.positionErrorNorm,'--');
grid on;
ylabel('||e_r||');
legend('no maneuver','pulse pair','Location','best');

nexttile;
plot(t,result.noManeuver.velocityErrorNorm);
hold on;
plot(t,result.pulsePair.velocityErrorNorm,'--');
grid on;
ylabel('||e_v||');

nexttile;
plot(t,result.noManeuver.accelerationErrorNorm);
hold on;
plot(t,result.pulsePair.accelerationErrorNorm,'--');
grid on;
ylabel('||e_a||');

nexttile;
plot(t,result.noManeuver.normalizedNIS);
hold on;
plot(t,result.pulsePair.normalizedNIS,'--');
yline(1,'k:');
grid on;
xlabel('time [s]');
ylabel('NIS / m');
end
