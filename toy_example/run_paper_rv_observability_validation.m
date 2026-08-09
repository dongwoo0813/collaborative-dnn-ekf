function result = run_paper_rv_observability_validation( ...
    dim,makePlots,simulationTime,dt,seed,options)
%RUN_PAPER_RV_OBSERVABILITY_VALIDATION Validate the maneuver mechanism.
%
% This function isolates the state used by the angles-only observability
% criterion:
%
%   x = [r; v].
%
% Four independent local EKFs estimate the same inertial target position and
% velocity from angle-only measurements. Two cases are run with exactly the
% same measurement-noise samples:
%
%   1. stationary watchers,
%   2. calibrated finite pulse-pair watcher maneuvers.
%
% Each filter is initialized on the correct LOS but at an incorrect positive
% range scale. Without a maneuver, the scaled initial relative position and
% velocity generate the same ideal LOS profile. The calibrated watcher
% maneuver changes that profile.
%
% The pulse pair is
%
%   +A*d_perp,  t_b <= t < t_b+T_b,
%   -A*d_perp,  t_b+T_b <= t < t_b+2*T_b,
%    0,          otherwise,
%
% where d_perp is frozen before the burn and is transverse to the nominal
% initial LOS. The pair leaves zero final maneuver velocity but a nonzero
% known transverse displacement.
%
% INPUTS
% ------
% dim             2 or 3.
% makePlots       Plot flag.
% simulationTime  Duration [s].
% dt              Sample time [s].
% seed            Random seed.
% options         Optional scalar structure:
%   sigmaBearingDeg
%   sigmaAzimuthDeg
%   sigmaElevationDeg
%   burnAcceleration
%   burnStartTime
%   burnDuration
%   initialRangeScale       scalar or 1-by-4
%   targetInitialPosition   dim-by-1
%   targetVelocity          dim-by-1
%   processAccelSigma
%   watcherRadius
%
% OUTPUT
% ------
% result.noManeuver
% result.pulsePair
% result.summary

if nargin < 2 || isempty(makePlots),      makePlots = true; end
if nargin < 3 || isempty(simulationTime), simulationTime = 60; end
if nargin < 4 || isempty(dt),             dt = 0.05; end
if nargin < 5 || isempty(seed),           seed = 31 + dim; end
if nargin < 6 || isempty(options),        options = struct(); end

validateattributes(dim,{'numeric'},{'scalar','integer'});
if ~ismember(dim,[2,3])
    error('dim must be 2 or 3.');
end
validateattributes(simulationTime,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(dt,{'numeric'},{'scalar','positive','finite'});
validateattributes(seed,{'numeric'}, ...
    {'scalar','integer','nonnegative'});
if ~isstruct(options) || ~isscalar(options)
    error('options must be a scalar structure or empty.');
end

options = resolveOptions(options,dim,simulationTime);
rng(seed);

Nw = 4;
t = (0:dt:simulationTime).';
N = numel(t);

pWatcher0 = makeWatcherGeometry(dim,options.watcherRadius);

r0 = options.targetInitialPosition(:);
v0 = options.targetVelocity(:);

rTrue = r0 + v0*t.';
vTrue = repmat(v0,1,N);

scale = options.initialRangeScale(:).';
if isscalar(scale)
    scale = repmat(scale,1,Nw);
end
if numel(scale) ~= Nw || any(scale <= 0)
    error('options.initialRangeScale must be positive scalar or 1-by-4.');
end

% Initial estimates lie exactly on each true initial LOS with a wrong scale.
initialState = zeros(2*dim,Nw);
for j = 1:Nw
    rho0 = rTrue(:,1)-pWatcher0(:,j);
    initialState(1:dim,j) = pWatcher0(:,j)+scale(j)*rho0;
    initialState(dim+(1:dim),j) = scale(j)*vTrue(:,1);
end

% Freeze one calibrated transverse direction for each watcher.
burnDirection = zeros(dim,Nw);
for j = 1:Nw
    u0 = normalizeVector(initialState(1:dim,j)-pWatcher0(:,j));
    burnDirection(:,j) = makeTransverseDirection( ...
        u0,j,dim);
end

% Reuse identical angle-noise sequences in both cases.
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
    'watcherInitialPosition',pWatcher0, ...
    'initialStateEstimate',initialState, ...
    'initialRangeScale',scale, ...
    'burnDirection',burnDirection, ...
    'options',options, ...
    'dimension',dim, ...
    'seed',seed);

noManeuver = simulateCase( ...
    'none',common,noise,dt);
pulsePair = simulateCase( ...
    'pulse_pair',common,noise,dt);

result = struct();
result.time = t;
result.dimension = dim;
result.options = options;
result.noManeuver = noManeuver;
result.pulsePair = pulsePair;

caseName = ["no-maneuver";"finite-pulse-pair"];
positionRMSE = [ ...
    noManeuver.networkPositionRMSE;
    pulsePair.networkPositionRMSE];
velocityRMSE = [ ...
    noManeuver.networkVelocityRMSE;
    pulsePair.networkVelocityRMSE];
finalPositionRMSE = [ ...
    noManeuver.finalNetworkPositionRMSE;
    pulsePair.finalNetworkPositionRMSE];

result.summary = table( ...
    caseName,positionRMSE,velocityRMSE,finalPositionRMSE);

fprintf('\nPaper-state [r;v] angle-only observability validation\n');
disp(result.summary);

if makePlots
    result.figure = plotResult(result);
end
end

function caseResult = simulateCase( ...
    maneuverMode,common,noise,dt)

t = common.time;
N = numel(t);
dim = common.dimension;
Nw = size(common.watcherInitialPosition,2);
n = 2*dim;

F = [ ...
    eye(dim),dt*eye(dim);
    zeros(dim),eye(dim)];

q = common.options.processAccelSigma^2;
Qaxis = q*[dt^3/3,dt^2/2;dt^2/2,dt];
Q = kron(Qaxis,eye(dim));

P0 = diag([ ...
    common.options.initialPositionSigma^2*ones(1,dim), ...
    common.options.initialVelocitySigma^2*ones(1,dim)]);

xhat = zeros(n,N,Nw);
Plog = zeros(n,n,Nw,N);
watcherPosition = zeros(dim,N,Nw);
watcherVelocity = zeros(dim,N,Nw);
watcherAcceleration = zeros(dim,N,Nw);
nis = nan(N,Nw);

for j = 1:Nw
    xhat(:,1,j) = common.initialStateEstimate(:,j);
    Plog(:,:,j,1) = P0;
    watcherPosition(:,1,j) = common.watcherInitialPosition(:,j);
end

for k = 1:N-1
    for j = 1:Nw
        aW = pulsePairCommand( ...
            maneuverMode,t(k),common.burnDirection(:,j), ...
            common.options);
        watcherAcceleration(:,k,j) = aW;

        pW = watcherPosition(:,k,j);
        vW = watcherVelocity(:,k,j);

        pWNext = pW+vW*dt+0.5*aW*dt^2;
        vWNext = vW+aW*dt;

        watcherPosition(:,k+1,j) = pWNext;
        watcherVelocity(:,k+1,j) = vWNext;

        xp = F*xhat(:,k,j);
        Pp = F*Plog(:,:,j,k)*F.'+Q;
        Pp = 0.5*(Pp+Pp.');

        rhoTrue = common.targetPosition(:,k+1)-pWNext;
        rhoHat = xp(1:dim)-pWNext;
        rr = max(norm(rhoHat),1e-9);
        u = rhoHat/rr;

        if dim == 2
            zTrue = atan2(rhoTrue(2),rhoTrue(1));
            z = wrapAngle(zTrue+noise.bearing(k+1,j));
            zHat = atan2(u(2),u(1));
            innov = wrapAngle(z-zHat);

            Hpos = [-u(2),u(1)]/rr;
            H = [Hpos,zeros(1,dim)];
            R = deg2rad(common.options.sigmaBearingDeg)^2;
        else
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

            innov = [wrapAngle(zAz-azHat);zEl-elHat];

            Hpos = [ ...
                -dy/rhoXY^2,dx/rhoXY^2,0; ...
                -(dx*dz)/(rho2*rhoXY), ...
                -(dy*dz)/(rho2*rhoXY), ...
                 rhoXY/rho2];

            H = [Hpos,zeros(2,dim)];
            R = diag([ ...
                deg2rad(common.options.sigmaAzimuthDeg)^2, ...
                deg2rad(common.options.sigmaElevationDeg)^2]);
        end

        S = H*Pp*H.'+R;
        S = 0.5*(S+S.');
        K = (Pp*H.')/S;

        xn = xp+K*innov;
        ImKH = eye(n)-K*H;
        Pn = ImKH*Pp*ImKH.'+K*R*K.';
        Pn = 0.5*(Pn+Pn.');

        xhat(:,k+1,j) = xn;
        Plog(:,:,j,k+1) = Pn;
        nis(k+1,j) = innov.'*(S\innov);
    end
end

for j = 1:Nw
    watcherAcceleration(:,N,j) = pulsePairCommand( ...
        maneuverMode,t(N),common.burnDirection(:,j), ...
        common.options);
end

positionErrorNorm = zeros(N,Nw);
velocityErrorNorm = zeros(N,Nw);
rangeError = zeros(N,Nw);
angleSignature = zeros(N,Nw);

sigmaAngle = deg2rad(common.options.sigmaBearingDeg);
if dim == 3
    sigmaAngle = sqrt(0.5*( ...
        deg2rad(common.options.sigmaAzimuthDeg)^2+ ...
        deg2rad(common.options.sigmaElevationDeg)^2));
end

for k = 1:N
    for j = 1:Nw
        pHat = xhat(1:dim,k,j);
        vHat = xhat(dim+(1:dim),k,j);
        pW = watcherPosition(:,k,j);

        positionErrorNorm(k,j) = norm( ...
            pHat-common.targetPosition(:,k));
        velocityErrorNorm(k,j) = norm( ...
            vHat-common.targetVelocity(:,k));

        trueRange = norm(common.targetPosition(:,k)-pW);
        estimatedRange = norm(pHat-pW);
        rangeError(k,j) = estimatedRange-trueRange;

        uM = normalizeVector(common.targetPosition(:,k)-pW);
        uN = normalizeVector(common.targetPosition(:,k)- ...
            common.watcherInitialPosition(:,j));
        angleSignature(k,j) = acos(max(-1,min(1,uM.'*uN)));
    end
end

positionRMSEByWatcher = sqrt(mean(positionErrorNorm.^2,1));
velocityRMSEByWatcher = sqrt(mean(velocityErrorNorm.^2,1));

idxFinal = max(1,round(0.9*N)):N;

caseResult = struct( ...
    'maneuverMode',maneuverMode, ...
    'localState',xhat, ...
    'localCovariance',Plog, ...
    'watcherPositionHistory',watcherPosition, ...
    'watcherVelocityHistory',watcherVelocity, ...
    'watcherManeuverAcceleration',watcherAcceleration, ...
    'positionErrorNorm',positionErrorNorm, ...
    'velocityErrorNorm',velocityErrorNorm, ...
    'rangeError',rangeError, ...
    'NIS',nis, ...
    'angleSignatureRad',angleSignature, ...
    'angleSignatureSigma',angleSignature/max(sigmaAngle,eps), ...
    'positionRMSEByWatcher',positionRMSEByWatcher, ...
    'velocityRMSEByWatcher',velocityRMSEByWatcher, ...
    'networkPositionRMSE',sqrt(mean(positionErrorNorm(:).^2)), ...
    'networkVelocityRMSE',sqrt(mean(velocityErrorNorm(:).^2)), ...
    'finalNetworkPositionRMSE',sqrt(mean( ...
        positionErrorNorm(idxFinal,:).^2,'all')));
end

function aW = pulsePairCommand(mode,t,direction,options)
%PULSEPAIRCOMMAND Finite calibrated burn with zero final velocity.

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
    'initialRangeScale',[0.55,1.45,0.70,1.30], ...
    'processAccelSigma',0.01, ...
    'initialPositionSigma',1000, ...
    'initialVelocitySigma',10, ...
    'watcherRadius',1000, ...
    'targetInitialPosition',zeros(dim,1), ...
    'targetVelocity',defaultTargetVelocity(dim));

names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(options,name) || isempty(options.(name))
        options.(name) = defaults.(name);
    end
end

validateattributes(options.burnAcceleration,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.burnStartTime,{'numeric'}, ...
    {'scalar','nonnegative','finite'});
validateattributes(options.burnDuration,{'numeric'}, ...
    {'scalar','positive','finite'});
if options.burnStartTime+2*options.burnDuration > simulationTime
    error('The two-pulse maneuver must finish before simulationTime.');
end
validateattributes(options.watcherRadius,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.targetInitialPosition,{'numeric'}, ...
    {'vector','numel',dim,'finite'});
validateattributes(options.targetVelocity,{'numeric'}, ...
    {'vector','numel',dim,'finite'});
end

function v = defaultTargetVelocity(dim)
if dim == 2
    v = [2.0;-0.8];
else
    v = [2.0;-0.8;0.5];
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
    order = [1+mod(j-1,3),setdiff(1:3,1+mod(j-1,3))];
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
Nw = size(result.noManeuver.localState,3);

fig = gobjects(3,1);

fig(1) = figure( ...
    'Name','Paper-state RV observability comparison', ...
    'Color','w');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t,mean(result.noManeuver.positionErrorNorm,2), ...
    'LineWidth',1.2);
hold on;
plot(t,mean(result.pulsePair.positionErrorNorm,2), ...
    '--','LineWidth',1.2);
grid on;
ylabel('mean ||e_r||');
legend('no maneuver','pulse pair','Location','best');

nexttile;
plot(t,result.noManeuver.rangeError,'LineWidth',0.9);
grid on;
ylabel('range error');
title('no-maneuver local range errors');

nexttile;
plot(t,result.pulsePair.rangeError,'LineWidth',0.9);
grid on;
xlabel('time [s]');
ylabel('range error');
title('finite-pulse-pair local range errors');

fig(2) = figure( ...
    'Name','Maneuver angular signature', ...
    'Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t,result.pulsePair.angleSignatureSigma,'LineWidth',1.0);
grid on;
ylabel('\Delta angle / \sigma');
title('exact maneuver-induced angular signature');
legend(arrayfun(@(j)sprintf('watcher %d',j), ...
    1:Nw,'UniformOutput',false),'Location','best');

nexttile;
hold on;
for j = 1:Nw
    aW = squeeze( ...
        result.pulsePair.watcherManeuverAcceleration(:,:,j));
    plot(t,vecnorm(aW,2,1),'LineWidth',1.0);
end
grid on;
xlabel('time [s]');
ylabel('||a_w||');
title('finite calibrated pulse pair');

fig(3) = figure( ...
    'Name','RV local state estimates with pulse pair', ...
    'Color','w');
tiledlayout(2,dim,'TileSpacing','compact','Padding','compact');

for q = 1:dim
    nexttile;
    % Truth is reconstructed from the first estimate error data source.
    % Store it through the common result is unnecessary for computation,
    % so use the known initial state and velocity here.
    rTruth = result.options.targetInitialPosition(q) + ...
        result.options.targetVelocity(q)*t.';
    plot(t,rTruth,'k','LineWidth',1.4);
    hold on;
    for j = 1:Nw
        plot(t,squeeze(result.pulsePair.localState(q,:,j)));
    end
    grid on;
    ylabel(sprintf('r_%d',q));
    if q == 1
        legend([{'truth'},arrayfun(@(j)sprintf('watcher %d',j), ...
            1:Nw,'UniformOutput',false)],'Location','best');
    end
end

for q = 1:dim
    nexttile;
    plot(t,result.options.targetVelocity(q)*ones(size(t)), ...
        'k','LineWidth',1.4);
    hold on;
    for j = 1:Nw
        plot(t,squeeze( ...
            result.pulsePair.localState(dim+q,:,j)));
    end
    grid on;
    xlabel('time [s]');
    ylabel(sprintf('v_%d',q));
end
end
