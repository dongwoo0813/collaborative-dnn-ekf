function result = run_local_accel_observability_experiment( ...
    dim,makePlots,simulationTime,dt,seed,options)
%RUN_LOCAL_ACCEL_OBSERVABILITY_EXPERIMENT Local angle-only [r;v;a] EKFs.
%
% PURPOSE
% -------
% Compare independent local angle-only EKFs in two cases:
%
%   1. no watcher maneuver,
%   2. a finite calibrated transverse pulse pair.
%
% Every local EKF estimates the inertial target state
%
%   x = [r; v; a].
%
% The default truth acceleration is unknown and constant. The initial local
% states are deliberately placed on the correct initial LOS with an
% incorrect positive range scale:
%
%   rHat_i(0) = p_wi + lambda_i (r(0)-p_wi),
%   vHat_i(0) = lambda_i v(0),
%   aHat_i(0) = lambda_i a(0).
%
% With stationary watchers and constant acceleration, these scaled states
% can generate the same ideal angle profile. The known watcher maneuver is
% not scaled and can break this ambiguity.
%
% ACTIVE WATCHER MODES
% --------------------
% options.activeWatcherMode:
%
%   'single'        One watcher provides measurements.
%   'parallel_pair' Two watchers with nearly parallel LOS provide data.
%   'all'           All four watchers provide data.
%   'dropout'       One watcher, then a pair, then all four.
%
% MANEUVER
% --------
% The pulse-pair direction is frozen before the burn:
%
%   +A d_perp,  t_b <= t < t_b+T_b,
%   -A d_perp,  t_b+T_b <= t < t_b+2T_b,
%    0,          otherwise.
%
% It leaves zero final maneuver-induced velocity and a nonzero known
% transverse displacement A*T_b^2.
%
% FUSION
% ------
% At each time, only active watchers contribute.
%
%   Pperp_i = I-uHat_i^- (uHat_i^-)^T
%
%   aGeo = [sum_i Pperp_i]^dagger ...
%          sum_i Pperp_i aHat_i.
%
% Three geometry-aware outputs are formed:
%
%   1. equal confidence,
%   2. maneuver-mask confidence,
%   3. adaptive quality confidence.
%
% The adaptive quality weight uses the posterior transverse acceleration
% variance and a normalized NIS EWMA:
%
%   alphaRaw_i proportional to
%       1 / [(tr(Pperp_i Paa_i)/(d-1) + eps)^p_var ...
%            max(1,NISbar_i/m_z)^p_nis].
%
% The default exponents are p_var = p_nis = 1/2. This softens the large
% weight ratios produced by direct inverse variance. The normalized weights
% are then mapped to [alpha_min,1], so every valid watcher contributes:
%
%   alpha_i = alpha_min ...
%             +(1-alpha_min) alphaRaw_i/max_j(alphaRaw_j).
%
% The raw quality weights are then blended toward equal geometry weights
% only as much as required to satisfy the full-rank and condition-number
% constraint. Thus the algorithm never falls back to a geometry-blind
% arithmetic mean:
%
%   alphaEff_i(beta) = beta alphaRaw_i +(1-beta),
%
% where the largest beta in [0,1] satisfying the geometry constraint is
% selected. beta = 1 preserves the quality weights; beta = 0 gives equal
% geometry-aware fusion.
%
% With one active watcher, the geometry matrix is rank deficient. In that
% case a geometry output must not be interpreted as a complete acceleration
% vector.
%
% INPUTS
% ------
% dim             2 or 3.
% makePlots       Plot flag.
% simulationTime  Duration [s].
% dt              Sampling interval [s].
% seed            Random seed.
% options         Optional scalar structure. Supported fields:
%
%   truthAccelerationMode   'constant' or 'slow_sinusoid'
%   constantAcceleration    dim-by-1 constant mode / sinusoid bias
%   accelerationAmplitude   dim-by-1 sinusoid amplitude
%   accelerationFrequency  dim-by-1 angular frequency [rad/s]
%   accelerationPhase      dim-by-1 phase [rad]
%   targetInitialPosition   dim-by-1
%   targetInitialVelocity   dim-by-1
%   activeWatcherMode       'single','parallel_pair','all','dropout'
%   activeWatcherIndex      integer from 1 to 4
%   maneuverWatcherMask     1-by-4 logical/numeric maneuver selection
%   initialRangeScale       positive scalar or 1-by-4
%   nonManeuverFusionWeight scalar weight for nonmaneuvering watchers
%   qualityWeightMinimum   lower bound for every valid watcher weight
%   qualityWeightEpsilon   positive variance regularization
%   qualityVarianceExponent exponent applied to transverse variance
%   qualityNISExponent     exponent applied to normalized-NIS penalty
%   nisEwmaFactor          NIS EWMA forgetting factor in [0,1)
%   maxQualityGeometryCondition condition-number validity threshold
%   qualityGeometryBlendGridSize number of beta candidates in [1,0]
%   postBurnSettlingTime    settling interval before post-burn RMSE [s]
%   watcherRadius
%   parallelSeparation
%   burnAcceleration
%   burnStartTime
%   burnDuration
%   sigmaBearingDeg
%   sigmaAzimuthDeg
%   sigmaElevationDeg
%   sigmaJerk
%   initialPositionSigma
%   initialVelocitySigma
%   initialAccelerationSigma
%
% OUTPUT
% ------
% result.noManeuver
% result.pulsePair
% result.summary
%
% The same measurement-noise realization and active-watcher schedule are
% used in both cases. Measurement availability and maneuver participation
% are controlled independently.

if nargin < 2 || isempty(makePlots),      makePlots = true; end
if nargin < 3 || isempty(simulationTime), simulationTime = 60; end
if nargin < 4 || isempty(dt),             dt = 0.05; end
if nargin < 5 || isempty(seed),           seed = 71 + dim; end
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

pWatcher0 = makeWatcherGeometry(dim,options);

aTrue = makeTruthAcceleration(dim,t,options);
rTrue = zeros(dim,N);
vTrue = zeros(dim,N);
rTrue(:,1) = options.targetInitialPosition(:);
vTrue(:,1) = options.targetInitialVelocity(:);

for k = 1:N-1
    vTrue(:,k+1) = vTrue(:,k)+aTrue(:,k)*dt;
    rTrue(:,k+1) = rTrue(:,k)+vTrue(:,k)*dt ...
        + 0.5*aTrue(:,k)*dt^2;
end

rangeScale = options.initialRangeScale(:).';
if isscalar(rangeScale)
    rangeScale = repmat(rangeScale,1,Nw);
end
if numel(rangeScale) ~= Nw || any(rangeScale <= 0)
    error('options.initialRangeScale must be positive scalar or 1-by-4.');
end

initialState = zeros(3*dim,Nw);
burnDirection = zeros(dim,Nw);

for j = 1:Nw
    relativePosition0 = rTrue(:,1)-pWatcher0(:,j);

    initialState(1:dim,j) = ...
        pWatcher0(:,j)+rangeScale(j)*relativePosition0;
    initialState(dim+(1:dim),j) = ...
        rangeScale(j)*vTrue(:,1);
    initialState(2*dim+(1:dim),j) = ...
        rangeScale(j)*aTrue(:,1);

    u0 = normalizeVector( ...
        initialState(1:dim,j)-pWatcher0(:,j));
    burnDirection(:,j) = makeTransverseDirection(u0,j,dim);
end

% Identical measurement noise in no-maneuver and pulse-pair cases.
noise = struct();
if dim == 2
    noise.bearing = deg2rad(options.sigmaBearingDeg)*randn(N,Nw);
else
    noise.azimuth = deg2rad(options.sigmaAzimuthDeg)*randn(N,Nw);
    noise.elevation = deg2rad(options.sigmaElevationDeg)*randn(N,Nw);
end

activeMask = makeActiveMask( ...
    options.activeWatcherMode,options.activeWatcherIndex,t,Nw);

% Measurement availability and maneuver participation are independent.
maneuverWatcherMask = logical(options.maneuverWatcherMask(:).');

common = struct( ...
    'time',t, ...
    'targetPosition',rTrue, ...
    'targetVelocity',vTrue, ...
    'trueAcceleration',aTrue, ...
    'watcherInitialPosition',pWatcher0, ...
    'initialStateEstimate',initialState, ...
    'initialRangeScale',rangeScale, ...
    'burnDirection',burnDirection, ...
    'activeMask',activeMask, ...
    'maneuverWatcherMask',maneuverWatcherMask, ...
    'options',options, ...
    'dimension',dim, ...
    'seed',seed);

noManeuver = simulateCase('none',common,noise,dt);
pulsePair = simulateCase('pulse_pair',common,noise,dt);

% Exact angular change caused by the maneuver, computed with truth only for
% post-processing. It is not used by the EKF or controller.
angleSignatureRad = computeAngleSignature( ...
    common,pulsePair.watcherPositionHistory);

sigmaAngle = deg2rad(options.sigmaBearingDeg);
if dim == 3
    sigmaAngle = sqrt(0.5*( ...
        deg2rad(options.sigmaAzimuthDeg)^2+ ...
        deg2rad(options.sigmaElevationDeg)^2));
end

pulsePair.angleSignatureRad = angleSignatureRad;
pulsePair.angleSignatureSigma = ...
    angleSignatureRad/max(sigmaAngle,eps);

result = struct( ...
    'time',t, ...
    'dimension',dim, ...
    'options',options, ...
    'targetPosition',rTrue, ...
    'targetVelocity',vTrue, ...
    'trueAcceleration',aTrue, ...
    'activeMask',activeMask, ...
    'maneuverWatcherMask',maneuverWatcherMask, ...
    'burnDirection',burnDirection, ...
    'watcherInitialPosition',pWatcher0, ...
    'initialStateEstimate',initialState, ...
    'initialRangeScale',rangeScale, ...
    'noManeuver',noManeuver, ...
    'pulsePair',pulsePair);

caseName = ["no-maneuver";"finite-pulse-pair"];
positionRMSE = [ ...
    noManeuver.activeMeanPositionRMSE;
    pulsePair.activeMeanPositionRMSE];
velocityRMSE = [ ...
    noManeuver.activeMeanVelocityRMSE;
    pulsePair.activeMeanVelocityRMSE];
localAccelerationRMSE = [ ...
    noManeuver.activeMeanAccelerationRMSE;
    pulsePair.activeMeanAccelerationRMSE];
equalGeometryAccelerationRMSE = [ ...
    noManeuver.equalGeometryAccelerationRMSE;
    pulsePair.equalGeometryAccelerationRMSE];
weightedGeometryAccelerationRMSE = [ ...
    noManeuver.maneuverWeightedGeometryAccelerationRMSE;
    pulsePair.maneuverWeightedGeometryAccelerationRMSE];
qualityWeightedGeometryAccelerationRMSE = [ ...
    noManeuver.qualityWeightedGeometryAccelerationRMSE;
    pulsePair.qualityWeightedGeometryAccelerationRMSE];

postLocalAccelerationRMSE = [ ...
    noManeuver.postActiveMeanAccelerationRMSE;
    pulsePair.postActiveMeanAccelerationRMSE];
postEqualGeometryAccelerationRMSE = [ ...
    noManeuver.postEqualGeometryAccelerationRMSE;
    pulsePair.postEqualGeometryAccelerationRMSE];
postWeightedGeometryAccelerationRMSE = [ ...
    noManeuver.postManeuverWeightedGeometryAccelerationRMSE;
    pulsePair.postManeuverWeightedGeometryAccelerationRMSE];
postQualityWeightedGeometryAccelerationRMSE = [ ...
    noManeuver.postQualityWeightedGeometryAccelerationRMSE;
    pulsePair.postQualityWeightedGeometryAccelerationRMSE];

totalDeltaV = [ ...
    noManeuver.totalDeltaV;
    pulsePair.totalDeltaV];

result.summary = table( ...
    caseName,positionRMSE,velocityRMSE, ...
    localAccelerationRMSE,equalGeometryAccelerationRMSE, ...
    weightedGeometryAccelerationRMSE, ...
    qualityWeightedGeometryAccelerationRMSE, ...
    postLocalAccelerationRMSE, ...
    postEqualGeometryAccelerationRMSE, ...
    postWeightedGeometryAccelerationRMSE, ...
    postQualityWeightedGeometryAccelerationRMSE,totalDeltaV);

fprintf('\nIndependent local angle-only [r;v;a] EKFs\n');
fprintf('active watcher mode: %s\n',options.activeWatcherMode);
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

xhat = zeros(n,N,Nw);
Plog = zeros(n,n,Nw,N);

watcherPosition = zeros(dim,N,Nw);
watcherVelocity = zeros(dim,N,Nw);
watcherAcceleration = zeros(dim,N,Nw);

uPred = zeros(dim,N,Nw);
nis = nan(N,Nw);
nisEWMA = ones(N,Nw);
measurementAvailable = common.activeMask;

activeMeanState = nan(n,N);
activeMeanAcceleration = nan(dim,N);

% Equal-confidence and maneuver-aware geometry reconstructions.
equalGeometryAcceleration = nan(dim,N);
maneuverWeightedGeometryAcceleration = nan(dim,N);
qualityWeightedGeometryAcceleration = nan(dim,N);

geometryRank = zeros(1,N);
geometryCondition = inf(1,N);
weightedGeometryRank = zeros(1,N);
weightedGeometryCondition = inf(1,N);
qualityGeometryRank = zeros(1,N);
qualityGeometryCondition = inf(1,N);
qualityFusionValid = false(1,N);

rawQualityFusionAlpha = zeros(N,Nw);
qualityFusionAlpha = zeros(N,Nw);
qualityGeometryBlendFactor = zeros(N,1);
transverseAccelerationVariance = nan(N,Nw);

% Before the maneuver begins, both fusion rules are identical. In the
% pulse-pair case, designated maneuvering watchers receive unit weight after
% burn start and nonmaneuvering watchers receive a reduced scalar weight.
% With the default all-four maneuver mask, the maneuver-mask output is
% intentionally identical to equal-geometry fusion and acts as a check.
fusionAlpha = ones(N,Nw);
if strcmp(mode,'pulse_pair')
    idxWeighted = t >= common.options.burnStartTime;
    fusionAlpha(idxWeighted,:) = ...
        common.options.nonManeuverFusionWeight;
    fusionAlpha(idxWeighted,common.maneuverWatcherMask) = 1.0;
end

for j = 1:Nw
    xhat(:,1,j) = common.initialStateEstimate(:,j);
    Plog(:,:,j,1) = P0;
    watcherPosition(:,1,j) = common.watcherInitialPosition(:,j);

    relativeHat0 = xhat(1:dim,1,j)-watcherPosition(:,1,j);
    uPred(:,1,j) = normalizeVector(relativeHat0);
end

[rawQualityFusionAlpha(1,:), ...
    transverseAccelerationVariance(1,:)] = computeQualityWeights( ...
        squeeze(Plog(:,:,:,1)),squeeze(uPred(:,1,:)), ...
        measurementAvailable(1,:),nisEWMA(1,:), ...
        common.options,dim);

[qualityFusionAlpha(1,:),qualityGeometryBlendFactor(1)] = ...
    enforceQualityGeometryCondition( ...
        rawQualityFusionAlpha(1,:),squeeze(uPred(:,1,:)), ...
        measurementAvailable(1,:),common.options,dim);

[activeMeanState(:,1),activeMeanAcceleration(:,1), ...
    equalGeometryAcceleration(:,1), ...
    maneuverWeightedGeometryAcceleration(:,1), ...
    qualityWeightedGeometryAcceleration(:,1), ...
    geometryRank(1),geometryCondition(1), ...
    weightedGeometryRank(1),weightedGeometryCondition(1), ...
    qualityGeometryRank(1),qualityGeometryCondition(1)] = ...
    networkOutputs( ...
        squeeze(xhat(:,1,:)),squeeze(uPred(:,1,:)), ...
        measurementAvailable(1,:),fusionAlpha(1,:), ...
        qualityFusionAlpha(1,:),dim);

qualityFusionValid(1) = ...
    qualityGeometryRank(1) == dim && ...
    qualityGeometryCondition(1) <= ...
        common.options.maxQualityGeometryCondition;

if ~qualityFusionValid(1)
    % No geometry-blind fallback is used. A full vector is unavailable only
    % when even equal geometry cannot satisfy the rank/condition constraint.
    qualityWeightedGeometryAcceleration(:,1) = nan(dim,1);
end

for k = 1:N-1
    for j = 1:Nw
        nisEWMA(k+1,j) = nisEWMA(k,j);

        isMeasuring = measurementAvailable(k+1,j);
        isManeuvering = common.maneuverWatcherMask(j);

        aW = pulsePairCommand( ...
            mode,t(k),common.burnDirection(:,j), ...
            common.options,isManeuvering);
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

        relativeHat = xp(1:dim)-pWNext;
        rr = max(norm(relativeHat),1e-9);
        u = relativeHat/rr;
        uPred(:,k+1,j) = u;

        if isMeasuring
            relativeTrue = common.targetPosition(:,k+1)-pWNext;

            if dim == 2
                zTrue = atan2(relativeTrue(2),relativeTrue(1));
                z = wrapAngle(zTrue+noise.bearing(k+1,j));
                zHat = atan2(u(2),u(1));
                innov = wrapAngle(z-zHat);

                Hpos = [-u(2),u(1)]/rr;
                H = [Hpos,zeros(1,2*dim)];
                R = deg2rad(common.options.sigmaBearingDeg)^2;
            else
                [azTrue,elTrue] = cartesianToAzEl(relativeTrue);
                zAz = wrapAngle( ...
                    azTrue+noise.azimuth(k+1,j));
                zEl = clampElevation( ...
                    elTrue+noise.elevation(k+1,j));

                dx = relativeHat(1);
                dy = relativeHat(2);
                dz = relativeHat(3);
                rhoXY = max(hypot(dx,dy),1e-9);
                rho2 = max(dot(relativeHat,relativeHat),1e-12);

                azHat = atan2(dy,dx);
                elHat = atan2(dz,rhoXY);
                innov = [wrapAngle(zAz-azHat);zEl-elHat];

                Hpos = [ ...
                    -dy/rhoXY^2,dx/rhoXY^2,0; ...
                    -(dx*dz)/(rho2*rhoXY), ...
                    -(dy*dz)/(rho2*rhoXY), ...
                     rhoXY/rho2];

                H = [Hpos,zeros(2,2*dim)];
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

            nis(k+1,j) = innov.'*(S\innov);

            measurementDimension = dim-1;
            normalizedNIS = nis(k+1,j)/measurementDimension;
            lambdaNIS = common.options.nisEwmaFactor;
            nisEWMA(k+1,j) = lambdaNIS*nisEWMA(k,j) ...
                +(1-lambdaNIS)*normalizedNIS;
        else
            % Prediction-only local filter during inactivity/dropout.
            xn = xp;
            Pn = Pp;
        end

        xhat(:,k+1,j) = xn;
        Plog(:,:,j,k+1) = Pn;
    end

    [rawQualityFusionAlpha(k+1,:), ...
        transverseAccelerationVariance(k+1,:)] = ...
        computeQualityWeights( ...
            squeeze(Plog(:,:,:,k+1)), ...
            squeeze(uPred(:,k+1,:)), ...
            measurementAvailable(k+1,:), ...
            nisEWMA(k+1,:),common.options,dim);

    [qualityFusionAlpha(k+1,:), ...
        qualityGeometryBlendFactor(k+1)] = ...
        enforceQualityGeometryCondition( ...
            rawQualityFusionAlpha(k+1,:), ...
            squeeze(uPred(:,k+1,:)), ...
            measurementAvailable(k+1,:), ...
            common.options,dim);

    [activeMeanState(:,k+1),activeMeanAcceleration(:,k+1), ...
        equalGeometryAcceleration(:,k+1), ...
        maneuverWeightedGeometryAcceleration(:,k+1), ...
        qualityWeightedGeometryAcceleration(:,k+1), ...
        geometryRank(k+1),geometryCondition(k+1), ...
        weightedGeometryRank(k+1), ...
        weightedGeometryCondition(k+1), ...
        qualityGeometryRank(k+1), ...
        qualityGeometryCondition(k+1)] = networkOutputs( ...
            squeeze(xhat(:,k+1,:)), ...
            squeeze(uPred(:,k+1,:)), ...
            measurementAvailable(k+1,:), ...
            fusionAlpha(k+1,:), ...
            qualityFusionAlpha(k+1,:),dim);

    qualityFusionValid(k+1) = ...
        qualityGeometryRank(k+1) == dim && ...
        qualityGeometryCondition(k+1) <= ...
            common.options.maxQualityGeometryCondition;

    if ~qualityFusionValid(k+1)
        % Do not use an arithmetic mean or a last-value hold. If even the
        % equal-geometry endpoint is invalid, the full acceleration vector
        % is not reconstructible from the current directional geometry.
        qualityWeightedGeometryAcceleration(:,k+1) = nan(dim,1);
    end
end

for j = 1:Nw
    watcherAcceleration(:,N,j) = pulsePairCommand( ...
        mode,t(N),common.burnDirection(:,j), ...
        common.options,common.maneuverWatcherMask(j));
end

positionError = activeMeanState(1:dim,:)-common.targetPosition;
velocityError = activeMeanState(dim+(1:dim),:)- ...
    common.targetVelocity;
localAccelerationError = activeMeanAcceleration- ...
    common.trueAcceleration;
equalGeometryAccelerationError = equalGeometryAcceleration- ...
    common.trueAcceleration;
weightedGeometryAccelerationError = ...
    maneuverWeightedGeometryAcceleration-common.trueAcceleration;
qualityWeightedGeometryAccelerationError = ...
    qualityWeightedGeometryAcceleration-common.trueAcceleration;

% Watcher-level acceleration error reveals which local filters actually
% benefit from the maneuver.
localAccelerationErrorNorm = nan(N,Nw);
for j = 1:Nw
    localAcceleration = squeeze( ...
        xhat(2*dim+(1:dim),:,j));
    localAccelerationErrorNorm(:,j) = vecnorm( ...
        localAcceleration-common.trueAcceleration,2,1).';
end

% Evaluate final performance only after the pulse pair and settling period.
evaluationStartTime = common.options.burnStartTime ...
    + 2*common.options.burnDuration ...
    + common.options.postBurnSettlingTime;
idxPost = t >= evaluationStartTime;

% Maneuver effort: integral of acceleration magnitude.
deltaVByWatcher = zeros(1,Nw);
for j = 1:Nw
    aW = squeeze(watcherAcceleration(:,:,j));
    deltaVByWatcher(j) = trapz(t,vecnorm(aW,2,1));
end
totalDeltaV = sum(deltaVByWatcher);

caseResult = struct( ...
    'maneuverMode',mode, ...
    'localState',xhat, ...
    'localCovariance',Plog, ...
    'predictedLOS',uPred, ...
    'measurementAvailable',measurementAvailable, ...
    'maneuverWatcherMask',common.maneuverWatcherMask, ...
    'watcherPositionHistory',watcherPosition, ...
    'watcherVelocityHistory',watcherVelocity, ...
    'watcherManeuverAcceleration',watcherAcceleration, ...
    'fusionAlpha',fusionAlpha, ...
    'rawQualityFusionAlpha',rawQualityFusionAlpha, ...
    'qualityFusionAlpha',qualityFusionAlpha, ...
    'qualityGeometryBlendFactor',qualityGeometryBlendFactor, ...
    'transverseAccelerationVariance', ...
        transverseAccelerationVariance, ...
    'NIS',nis, ...
    'normalizedNISEWMA',nisEWMA, ...
    'activeMeanState',activeMeanState, ...
    'activeMeanAcceleration',activeMeanAcceleration, ...
    'geometryAcceleration',equalGeometryAcceleration, ...
    'equalGeometryAcceleration',equalGeometryAcceleration, ...
    'maneuverWeightedGeometryAcceleration', ...
        maneuverWeightedGeometryAcceleration, ...
    'qualityWeightedGeometryAcceleration', ...
        qualityWeightedGeometryAcceleration, ...
    'geometryRank',geometryRank, ...
    'geometryCondition',geometryCondition, ...
    'weightedGeometryRank',weightedGeometryRank, ...
    'weightedGeometryCondition',weightedGeometryCondition, ...
    'qualityGeometryRank',qualityGeometryRank, ...
    'qualityGeometryCondition',qualityGeometryCondition, ...
    'qualityFusionValid',qualityFusionValid, ...
    'postQualityFusionValidRate',mean(qualityFusionValid(idxPost)), ...
    'postQualityGeometryMedianCondition',medianFinite( ...
        qualityGeometryCondition(idxPost)), ...
    'postQualityGeometryMaxCondition',maxFinite( ...
        qualityGeometryCondition(idxPost)), ...
    'postWeightedGeometryFullRankRate',mean( ...
        weightedGeometryRank(idxPost) == dim), ...
    'postWeightedGeometryMedianCondition',medianFinite( ...
        weightedGeometryCondition(idxPost)), ...
    'postWeightedGeometryMaxCondition',maxFinite( ...
        weightedGeometryCondition(idxPost)), ...
    'localAccelerationErrorNorm',localAccelerationErrorNorm, ...
    'activeMeanPositionErrorNorm', ...
        vecnorm(positionError,2,1), ...
    'activeMeanVelocityErrorNorm', ...
        vecnorm(velocityError,2,1), ...
    'activeMeanAccelerationErrorNorm', ...
        vecnorm(localAccelerationError,2,1), ...
    'geometryAccelerationErrorNorm', ...
        vecnorm(equalGeometryAccelerationError,2,1), ...
    'equalGeometryAccelerationErrorNorm', ...
        vecnorm(equalGeometryAccelerationError,2,1), ...
    'maneuverWeightedGeometryAccelerationErrorNorm', ...
        vecnorm(weightedGeometryAccelerationError,2,1), ...
    'qualityWeightedGeometryAccelerationErrorNorm', ...
        vecnorm(qualityWeightedGeometryAccelerationError,2,1), ...
    'evaluationStartTime',evaluationStartTime, ...
    'postBurnIndex',idxPost, ...
    'deltaVByWatcher',deltaVByWatcher, ...
    'totalDeltaV',totalDeltaV, ...
    'activeMeanPositionRMSE',rmsFinite( ...
        vecnorm(positionError,2,1)), ...
    'activeMeanVelocityRMSE',rmsFinite( ...
        vecnorm(velocityError,2,1)), ...
    'activeMeanAccelerationRMSE',rmsFinite( ...
        vecnorm(localAccelerationError,2,1)), ...
    'geometryAccelerationRMSE',rmsFinite( ...
        vecnorm(equalGeometryAccelerationError,2,1)), ...
    'equalGeometryAccelerationRMSE',rmsFinite( ...
        vecnorm(equalGeometryAccelerationError,2,1)), ...
    'maneuverWeightedGeometryAccelerationRMSE',rmsFinite( ...
        vecnorm(weightedGeometryAccelerationError,2,1)), ...
    'qualityWeightedGeometryAccelerationRMSE',rmsFinite( ...
        vecnorm(qualityWeightedGeometryAccelerationError,2,1)), ...
    'postActiveMeanAccelerationRMSE',rmsFinite( ...
        vecnorm(localAccelerationError(:,idxPost),2,1)), ...
    'postEqualGeometryAccelerationRMSE',rmsFinite( ...
        vecnorm(equalGeometryAccelerationError(:,idxPost),2,1)), ...
    'postManeuverWeightedGeometryAccelerationRMSE',rmsFinite( ...
        vecnorm(weightedGeometryAccelerationError(:,idxPost),2,1)), ...
    'postQualityWeightedGeometryAccelerationRMSE',rmsFinite( ...
        vecnorm(qualityWeightedGeometryAccelerationError(:,idxPost), ...
        2,1)));
end

function [xMean,aMean,aGeoEqual,aGeoWeighted,aGeoQuality, ...
    rankOmega,condOmega,rankWeighted,condWeighted, ...
    rankQuality,condQuality] = ...
    networkOutputs( ...
        xLocal,uPred,active,alpha,qualityAlpha,dim)
%NETWORKOUTPUTS Active-node mean and three geometry reconstructions.
%
% Equal confidence:
%
%   aGeoEqual = [sum Pperp_i]^dagger sum Pperp_i a_i.
%
% Maneuver-mask confidence:
%
%   aGeoWeighted = [sum alpha_i Pperp_i]^dagger ...
%                  sum alpha_i Pperp_i a_i.
%
% Adaptive quality confidence:
%
%   aGeoQuality = [sum qualityAlpha_i Pperp_i]^dagger ...
%                 sum qualityAlpha_i Pperp_i a_i.

active = logical(active(:).');
alpha = alpha(:).';
qualityAlpha = qualityAlpha(:).';
activeIndex = find(active);

n = size(xLocal,1);
xMean = nan(n,1);
aMean = nan(dim,1);
aGeoEqual = nan(dim,1);
aGeoWeighted = nan(dim,1);
aGeoQuality = nan(dim,1);

rankOmega = 0;
condOmega = Inf;
rankWeighted = 0;
condWeighted = Inf;
rankQuality = 0;
condQuality = Inf;

if isempty(activeIndex)
    return;
end

xMean = mean(xLocal(:,activeIndex),2);
aLocal = xLocal(2*dim+(1:dim),activeIndex);
aMean = mean(aLocal,2);

OmegaEqual = zeros(dim);
informationEqual = zeros(dim,1);
OmegaWeighted = zeros(dim);
informationWeighted = zeros(dim,1);
OmegaQuality = zeros(dim);
informationQuality = zeros(dim,1);

for q = 1:numel(activeIndex)
    j = activeIndex(q);
    u = normalizeVector(uPred(:,j));
    Pperp = eye(dim)-u*u.';
    Pperp = 0.5*(Pperp+Pperp.');

    OmegaEqual = OmegaEqual+Pperp;
    informationEqual = informationEqual+Pperp*aLocal(:,q);

    watcherWeight = alpha(j);
    OmegaWeighted = OmegaWeighted+watcherWeight*Pperp;
    informationWeighted = informationWeighted ...
        + watcherWeight*Pperp*aLocal(:,q);

    qualityWeight = qualityAlpha(j);
    OmegaQuality = OmegaQuality+qualityWeight*Pperp;
    informationQuality = informationQuality ...
        + qualityWeight*Pperp*aLocal(:,q);
end

[OmegaEqualPlus,rankOmega,condOmega] = ...
    symmetricPseudoInverse(OmegaEqual,1e-10);
aGeoEqual = OmegaEqualPlus*informationEqual;

[OmegaWeightedPlus,rankWeighted,condWeighted] = ...
    symmetricPseudoInverse(OmegaWeighted,1e-10);
aGeoWeighted = OmegaWeightedPlus*informationWeighted;

[OmegaQualityPlus,rankQuality,condQuality] = ...
    symmetricPseudoInverse(OmegaQuality,1e-10);
aGeoQuality = OmegaQualityPlus*informationQuality;
end

function [alpha,transverseVariance] = computeQualityWeights( ...
    PLocal,uPred,active,nisEWMA,options,dim)
%COMPUTEQUALITYWEIGHTS Covariance- and NIS-based watcher confidence.
%
% For watcher i:
%
%   sigmaPerp_i^2 = tr(Pperp_i Paa_i)/(dim-1)
%
%   rawAlpha_i =
%       1 / [(sigmaPerp_i^2 + epsilon)^p_var ...
%            max(1,NISbar_i)^p_nis].
%
% The default p_var = p_nis = 1/2 prevents one covariance from dominating
% the fusion. After normalization, an affine lower bound is applied:
%
%   alpha_i = alpha_min +(1-alpha_min) normalizedRaw_i.
%
% Thus all available watchers remain in the directional reconstruction.

Nw = size(uPred,2);
alpha = zeros(1,Nw);
transverseVariance = nan(1,Nw);
rawAlpha = zeros(1,Nw);

accelerationIndex = 2*dim+(1:dim);
active = logical(active(:).');
nisEWMA = nisEWMA(:).';

for j = 1:Nw
    if ~active(j)
        continue;
    end

    u = normalizeVector(uPred(:,j));
    Pperp = eye(dim)-u*u.';
    Pperp = 0.5*(Pperp+Pperp.');

    Paa = PLocal(accelerationIndex,accelerationIndex,j);
    Paa = 0.5*(Paa+Paa.');

    sigmaPerp2 = real(trace(Pperp*Paa))/max(dim-1,1);
    sigmaPerp2 = max(sigmaPerp2,0);
    transverseVariance(j) = sigmaPerp2;

    variancePenalty = ( ...
        sigmaPerp2+options.qualityWeightEpsilon) ...
        ^options.qualityVarianceExponent;
    nisPenalty = max(1,nisEWMA(j)) ...
        ^options.qualityNISExponent;

    rawAlpha(j) = 1/(variancePenalty*nisPenalty);
end

activeRaw = rawAlpha > 0;
if ~any(activeRaw)
    return;
end

normalizedRaw = rawAlpha(activeRaw)/max(rawAlpha(activeRaw));

alpha(activeRaw) = options.qualityWeightMinimum ...
    +(1-options.qualityWeightMinimum)*normalizedRaw;
end

function [alphaEffective,betaSelected] = ...
    enforceQualityGeometryCondition( ...
        alphaQuality,uPred,active,options,dim)
%ENFORCEQUALITYGEOMETRYCONDITION Preserve geometry while regularizing weight.
%
% The raw quality weights are blended toward equal geometry weights:
%
%   alpha_i(beta) = beta alphaQuality_i +(1-beta),
%
% for active watchers. The largest beta satisfying
%
%   rank(Omega(beta)) = dim,
%   cond(Omega(beta)) <= kappaMax
%
% is selected. Hence:
%
%   beta = 1 : pure quality weighting,
%   beta = 0 : equal geometry-aware fusion.
%
% No arithmetic-mean fallback is introduced.

active = logical(active(:).');
alphaQuality = alphaQuality(:).';
alphaEffective = zeros(size(alphaQuality));
betaSelected = nan;

if ~any(active)
    return;
end

gridSize = options.qualityGeometryBlendGridSize;
betaGrid = linspace(1,0,gridSize);

for beta = betaGrid
    alphaCandidate = zeros(size(alphaQuality));
    alphaCandidate(active) = beta*alphaQuality(active) +(1-beta);

    Omega = zeros(dim);
    activeIndex = find(active);

    for q = 1:numel(activeIndex)
        j = activeIndex(q);
        u = normalizeVector(uPred(:,j));
        Pperp = eye(dim)-u*u.';
        Pperp = 0.5*(Pperp+Pperp.');
        Omega = Omega+alphaCandidate(j)*Pperp;
    end

    [~,rankOmega,condOmega] = symmetricPseudoInverse(Omega,1e-10);

    if rankOmega == dim && ...
            condOmega <= options.maxQualityGeometryCondition
        alphaEffective = alphaCandidate;
        betaSelected = beta;
        return;
    end
end

% Even equal geometry is invalid. Keep the equal-geometry endpoint so the
% caller can report the failure honestly rather than replacing it by a mean.
alphaEffective(active) = 1;
betaSelected = 0;
end

function mask = makeActiveMask(mode,index,t,Nw)
%MAKEACTIVEMASK Generate the measurement-availability schedule.

N = numel(t);
mask = false(N,Nw);

switch mode
    case 'single'
        mask(:,index) = true;

    case 'parallel_pair'
        mask(:,1:2) = true;

    case 'all'
        mask(:,:) = true;

    case 'dropout'
        t1 = t(end)/3;
        t2 = 2*t(end)/3;

        mask(t < t1,index) = true;
        mask(t >= t1 & t < t2,[1,2]) = true;
        mask(t >= t2,:) = true;

    otherwise
        error('Unsupported active watcher mode: %s',mode);
end
end

function p = makeWatcherGeometry(dim,options)

R = options.watcherRadius;

if dim == 2
    if strcmp(options.activeWatcherMode,'parallel_pair')
        dy = options.parallelSeparation/2;
        p = [ ...
            -R,-R,0,0; ...
            -dy,dy,-R,R];
    else
        p = R*[ ...
            -1,1,0,0; ...
             0,0,-1,1];
    end
else
    d = R/sqrt(3);
    p = d*[ ...
         1,1,-1,-1; ...
         1,-1,1,-1; ...
         1,-1,-1,1];

    if strcmp(options.activeWatcherMode,'parallel_pair')
        % Make watchers 1 and 2 nearly colocated in angle, while preserving
        % a small nonzero baseline.
        p(:,1) = [-R;0;0];
        p(:,2) = [-R;options.parallelSeparation;0];
    end
end
end

function a = makeTruthAcceleration(dim,t,options)

N = numel(t);
a = zeros(dim,N);

switch options.truthAccelerationMode
    case 'constant'
        a = repmat(options.constantAcceleration(:),1,N);

    case 'slow_sinusoid'
        bias = options.constantAcceleration(:);
        amplitude = options.accelerationAmplitude(:);
        frequency = options.accelerationFrequency(:);
        phase = options.accelerationPhase(:);

        for q = 1:dim
            a(q,:) = bias(q) ...
                + amplitude(q)*sin(frequency(q)*t.'+phase(q));
        end

    otherwise
        error('Unsupported truth acceleration mode: %s', ...
            options.truthAccelerationMode);
end
end

function aW = pulsePairCommand( ...
    mode,t,direction,options,isManeuvering)
%PULSEPAIRCOMMAND Finite known maneuver for designated watchers only.

aW = zeros(size(direction));

if strcmp(mode,'none') || ~isManeuvering
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

function signature = computeAngleSignature(common,pulsePosition)

t = common.time;
N = numel(t);
dim = common.dimension;
Nw = size(common.watcherInitialPosition,2);
signature = zeros(N,Nw);

for k = 1:N
    for j = 1:Nw
        pNominal = common.watcherInitialPosition(:,j);
        pManeuver = pulsePosition(:,k,j);

        uNominal = normalizeVector( ...
            common.targetPosition(:,k)-pNominal);
        uManeuver = normalizeVector( ...
            common.targetPosition(:,k)-pManeuver);

        signature(k,j) = acos(max(-1,min(1, ...
            uNominal.'*uManeuver)));

        if ~common.activeMask(k,j)
            signature(k,j) = nan;
        end
    end
end

if dim < 2
    error('Angle signature requires dim >= 2.');
end
end

function options = resolveOptions(options,dim,simulationTime)

defaults = struct( ...
    'truthAccelerationMode','constant', ...
    'constantAcceleration',defaultAcceleration(dim), ...
    'accelerationAmplitude',defaultAccelerationAmplitude(dim), ...
    'accelerationFrequency',defaultAccelerationFrequency(dim), ...
    'accelerationPhase',defaultAccelerationPhase(dim), ...
    'targetInitialPosition',zeros(dim,1), ...
    'targetInitialVelocity',defaultVelocity(dim), ...
    'activeWatcherMode','single', ...
    'activeWatcherIndex',1, ...
    'maneuverWatcherMask',[true,true,true,true], ...
    'initialRangeScale',[0.55,0.80,1.25,1.55], ...
    'nonManeuverFusionWeight',1.0, ...
    'qualityWeightMinimum',0.25, ...
    'qualityWeightEpsilon',1e-10, ...
    'qualityVarianceExponent',0.5, ...
    'qualityNISExponent',0.5, ...
    'nisEwmaFactor',0.95, ...
    'maxQualityGeometryCondition',20, ...
    'qualityGeometryBlendGridSize',101, ...
    'postBurnSettlingTime',10.0, ...
    'watcherRadius',1000, ...
    'parallelSeparation',100, ...
    'burnAcceleration',2.0, ...
    'burnStartTime',10.0, ...
    'burnDuration',5.0, ...
    'sigmaBearingDeg',0.2, ...
    'sigmaAzimuthDeg',0.2, ...
    'sigmaElevationDeg',0.2, ...
    'sigmaJerk',0.02, ...
    'initialPositionSigma',1000, ...
    'initialVelocitySigma',20, ...
    'initialAccelerationSigma',1.0);

names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(options,name) || isempty(options.(name))
        options.(name) = defaults.(name);
    end
end

options.truthAccelerationMode = validatestring( ...
    options.truthAccelerationMode, ...
    {'constant','slow_sinusoid'});
options.activeWatcherMode = validatestring( ...
    options.activeWatcherMode, ...
    {'single','parallel_pair','all','dropout'});

validateattributes(options.activeWatcherIndex,{'numeric'}, ...
    {'scalar','integer','>=',1,'<=',4});

validateattributes(options.maneuverWatcherMask,{'numeric','logical'}, ...
    {'vector','numel',4});
if any(~ismember(double(options.maneuverWatcherMask(:)),[0,1]))
    error('options.maneuverWatcherMask must contain only true/false values.');
end
options.maneuverWatcherMask = ...
    logical(options.maneuverWatcherMask(:).');

validateattributes(options.nonManeuverFusionWeight,{'numeric'}, ...
    {'scalar','nonnegative','<=',1,'finite'});
validateattributes(options.qualityWeightMinimum,{'numeric'}, ...
    {'scalar','nonnegative','<=',1,'finite'});
validateattributes(options.qualityWeightEpsilon,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.qualityVarianceExponent,{'numeric'}, ...
    {'scalar','nonnegative','finite'});
validateattributes(options.qualityNISExponent,{'numeric'}, ...
    {'scalar','nonnegative','finite'});
validateattributes(options.nisEwmaFactor,{'numeric'}, ...
    {'scalar','>=',0,'<',1,'finite'});
validateattributes(options.maxQualityGeometryCondition,{'numeric'}, ...
    {'scalar','>=',1,'finite'});
validateattributes(options.qualityGeometryBlendGridSize,{'numeric'}, ...
    {'scalar','integer','>=',2,'finite'});
validateattributes(options.postBurnSettlingTime,{'numeric'}, ...
    {'scalar','nonnegative','finite'});
validateattributes(options.constantAcceleration,{'numeric'}, ...
    {'vector','numel',dim,'finite'});
validateattributes(options.accelerationAmplitude,{'numeric'}, ...
    {'vector','numel',dim,'nonnegative','finite'});
validateattributes(options.accelerationFrequency,{'numeric'}, ...
    {'vector','numel',dim,'nonnegative','finite'});
validateattributes(options.accelerationPhase,{'numeric'}, ...
    {'vector','numel',dim,'finite'});
validateattributes(options.targetInitialPosition,{'numeric'}, ...
    {'vector','numel',dim,'finite'});
validateattributes(options.targetInitialVelocity,{'numeric'}, ...
    {'vector','numel',dim,'finite'});
validateattributes(options.watcherRadius,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.parallelSeparation,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.burnAcceleration,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.burnStartTime,{'numeric'}, ...
    {'scalar','nonnegative','finite'});
validateattributes(options.burnDuration,{'numeric'}, ...
    {'scalar','positive','finite'});

if options.burnStartTime+2*options.burnDuration > simulationTime
    error('The pulse pair must finish before simulationTime.');
end

validateattributes(options.sigmaJerk,{'numeric'}, ...
    {'scalar','nonnegative','finite'});
end

function a = defaultAcceleration(dim)
if dim == 2
    a = [0.15;0.20];
else
    a = [0.15;0.20;-0.10];
end
end

function amplitude = defaultAccelerationAmplitude(dim)
if dim == 2
    amplitude = [0.06;0.05];
else
    amplitude = [0.06;0.05;0.04];
end
end

function frequency = defaultAccelerationFrequency(dim)
if dim == 2
    frequency = [0.035;0.050];
else
    frequency = [0.035;0.050;0.040];
end
end

function phase = defaultAccelerationPhase(dim)
if dim == 2
    phase = [0;pi/2];
else
    phase = [0;pi/2;0.4];
end
end

function v = defaultVelocity(dim)
if dim == 2
    v = [2.0;-0.8];
else
    v = [2.0;-0.8;0.5];
end
end

function d = makeTransverseDirection(u,j,dim)

if dim == 2
    d = [-u(2);u(1)];
else
    Pperp = eye(3)-u*u.';
    axesSet = eye(3);
    preferred = 1+mod(j-1,3);
    order = [preferred,setdiff(1:3,preferred)];

    d = zeros(3,1);
    for q = order
        candidate = Pperp*axesSet(:,q);
        if norm(candidate) > 1e-8
            d = normalizeVector(candidate);
            break;
        end
    end
end

d = (-1)^(j-1)*normalizeVector(d);
end

function [Aplus,numericalRank,conditionNumber] = ...
    symmetricPseudoInverse(A,relativeTolerance)

A = 0.5*(A+A.');
[V,D] = eig(A,'vector');
[eigenvalues,order] = sort(real(D),'descend');
V = V(:,order);

lambdaMax = max(eigenvalues);
if isempty(lambdaMax) || lambdaMax <= 0
    Aplus = zeros(size(A));
    numericalRank = 0;
    conditionNumber = Inf;
    return;
end

keep = eigenvalues > relativeTolerance*lambdaMax;
numericalRank = nnz(keep);

lambdaPlus = zeros(size(eigenvalues));
lambdaPlus(keep) = 1./eigenvalues(keep);
Aplus = V*diag(lambdaPlus)*V.';
Aplus = 0.5*(Aplus+Aplus.');

if numericalRank < size(A,1)
    conditionNumber = Inf;
else
    conditionNumber = eigenvalues(1)/eigenvalues(end);
end
end

function value = rmsFinite(x)
x = x(isfinite(x));
if isempty(x)
    value = nan;
else
    value = sqrt(mean(x.^2));
end
end

function value = medianFinite(x)
%MEDIANFINITE Median of finite entries, or NaN when none exist.
x = x(isfinite(x));
if isempty(x)
    value = nan;
else
    value = median(x);
end
end

function value = maxFinite(x)
%MAXFINITE Maximum of finite entries, or NaN when none exist.
x = x(isfinite(x));
if isempty(x)
    value = nan;
else
    value = max(x);
end
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

fig = gobjects(5,1);

%% Figure 1: active-network mean state
fig(1) = figure( ...
    'Name','Local EKF mean state: maneuver comparison', ...
    'Color','w');
tiledlayout(3,dim,'TileSpacing','compact','Padding','compact');

truth = { ...
    result.targetPosition, ...
    result.targetVelocity, ...
    result.trueAcceleration};
symbols = {'r','v','a'};

for row = 1:3
    index = (row-1)*dim+(1:dim);

    for q = 1:dim
        nexttile;
        plot(t,truth{row}(q,:),'k','LineWidth',1.4);
        hold on;
        plot(t,result.noManeuver.activeMeanState(index(q),:), ...
            'LineWidth',1.0);
        plot(t,result.pulsePair.activeMeanState(index(q),:), ...
            '--','LineWidth',1.0);
        xline(result.pulsePair.evaluationStartTime,'k:');
        grid on;
        ylabel(sprintf('%s_%d',symbols{row},q));

        if row == 1 && q == 1
            legend('truth','no maneuver','pulse pair', ...
                'post-burn evaluation start','Location','best');
        end
        if row == 3
            xlabel('time [s]');
        end
    end
end

%% Figure 2: acceleration estimates and geometry
fig(2) = figure( ...
    'Name','Local and geometry-aware acceleration estimates', ...
    'Color','w');
tiledlayout(dim+3,1,'TileSpacing','compact','Padding','compact');

for q = 1:dim
    nexttile;
    plot(t,result.trueAcceleration(q,:),'k','LineWidth',1.4);
    hold on;
    plot(t,result.noManeuver.activeMeanAcceleration(q,:), ...
        'LineWidth',1.0);
    plot(t,result.pulsePair.activeMeanAcceleration(q,:), ...
        '--','LineWidth',1.0);
    plot(t,result.noManeuver.equalGeometryAcceleration(q,:), ...
        ':','LineWidth',1.0);
    plot(t,result.pulsePair.equalGeometryAcceleration(q,:), ...
        '-.','LineWidth',1.0);
    plot(t, ...
        result.pulsePair.maneuverWeightedGeometryAcceleration(q,:), ...
        'LineWidth',1.2);
    plot(t, ...
        result.pulsePair.qualityWeightedGeometryAcceleration(q,:), ...
        '--','LineWidth',1.3);
    xline(result.pulsePair.evaluationStartTime,'k:');
    grid on;
    ylabel(sprintf('a_%d',q));

    if q == 1
        legend( ...
            'truth', ...
            'local mean: no maneuver', ...
            'local mean: pulse pair', ...
            'equal geometry: no maneuver', ...
            'equal geometry: pulse pair', ...
            'maneuver-weighted geometry: pulse pair', ...
            'quality-weighted geometry: pulse pair', ...
            'post-burn evaluation start', ...
            'Location','best');
    end
end

nexttile;
plot(t,result.noManeuver.activeMeanAccelerationErrorNorm);
hold on;
plot(t,result.pulsePair.activeMeanAccelerationErrorNorm,'--');
plot(t,result.noManeuver.equalGeometryAccelerationErrorNorm,':');
plot(t,result.pulsePair.equalGeometryAccelerationErrorNorm,'-.');
plot(t, ...
    result.pulsePair.maneuverWeightedGeometryAccelerationErrorNorm, ...
    'LineWidth',1.2);
plot(t, ...
    result.pulsePair.qualityWeightedGeometryAccelerationErrorNorm, ...
    '--','LineWidth',1.3);
xline(result.pulsePair.evaluationStartTime,'k:');
grid on;
ylabel('accel. error');
legend('local/no','local/pulse','equal/no','equal/pulse', ...
    'maneuver weighted/pulse','quality weighted/pulse', ...
    'post start','Location','best');

nexttile;
stairs(t,result.pulsePair.geometryRank,'LineWidth',1.1);
hold on;
stairs(t,result.pulsePair.weightedGeometryRank,'--','LineWidth',1.1);
stairs(t,result.pulsePair.qualityGeometryRank,':','LineWidth',1.2);
grid on;
ylabel('geometry rank');
ylim([-0.1,dim+0.1]);
legend('equal','maneuver weighted','quality weighted', ...
    'Location','best');

nexttile;
semilogy(t,max(result.pulsePair.geometryCondition,1), ...
    'LineWidth',1.0);
hold on;
semilogy(t,max(result.pulsePair.weightedGeometryCondition,1), ...
    '--','LineWidth',1.0);
semilogy(t,max(result.pulsePair.qualityGeometryCondition,1), ...
    ':','LineWidth',1.2);
yline(result.options.maxQualityGeometryCondition,'k-.');
grid on;
xlabel('time [s]');
ylabel('geometry condition');
legend('equal','maneuver weighted','quality weighted', ...
    'quality threshold','Location','best');

%% Figure 3: availability, maneuver command, and angular signature
fig(3) = figure( ...
    'Name','Measurement availability and maneuver excitation', ...
    'Color','w');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
imagesc(t,1:Nw,result.activeMask.');
axis xy;
ylabel('watcher');
title(sprintf('active mode: %s; maneuver mask: [%s]', ...
    result.options.activeWatcherMode, ...
    sprintf('%d ',result.maneuverWatcherMask)));
colorbar;

nexttile;
hold on;
for j = 1:Nw
    aW = squeeze( ...
        result.pulsePair.watcherManeuverAcceleration(:,:,j));
    signedCommand = ...
        result.burnDirection(:,j).'*aW;
    plot(t,signedCommand,'LineWidth',1.0);
end
grid on;
ylabel('signed a_w');
title('finite pulse-pair command along frozen transverse direction');
legend(arrayfun(@(j)sprintf('watcher %d',j), ...
    1:Nw,'UniformOutput',false),'Location','best');

nexttile;
plot(t,result.pulsePair.angleSignatureSigma,'LineWidth',1.0);
yline(1,'k:');
yline(3,'k--');
grid on;
xlabel('time [s]');
ylabel('\Delta angle / \sigma');
title('maneuver-induced angular signature');

%% Figure 4: watcher-level acceleration error and maneuver effort
fig(4) = figure( ...
    'Name','Watcher-level acceleration error and maneuver cost', ...
    'Color','w');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t,result.noManeuver.localAccelerationErrorNorm, ...
    'LineWidth',0.9);
grid on;
ylabel('||e_{a,i}||');
title('watcher-level acceleration error: no maneuver');
legend(arrayfun(@(j)sprintf('watcher %d',j), ...
    1:Nw,'UniformOutput',false),'Location','best');

nexttile;
plot(t,result.pulsePair.localAccelerationErrorNorm, ...
    'LineWidth',0.9);
xline(result.pulsePair.evaluationStartTime,'k:');
grid on;
ylabel('||e_{a,i}||');
title('watcher-level acceleration error: pulse pair');

nexttile;
bar(1:Nw,result.pulsePair.deltaVByWatcher);
grid on;
xlabel('watcher');
ylabel('\int ||a_w|| dt');
title(sprintf('maneuver effort; total = %.3f', ...
    result.pulsePair.totalDeltaV));

%% Figure 5: adaptive quality-weight diagnostics
fig(5) = figure( ...
    'Name','Adaptive quality-weight diagnostics', ...
    'Color','w');
tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t,result.pulsePair.rawQualityFusionAlpha,':','LineWidth',0.9);
hold on;
plot(t,result.pulsePair.qualityFusionAlpha,'LineWidth',1.1);
grid on;
ylabel('\alpha_i');
ylim([0,1.05]);
title('raw and geometry-constrained quality weights');
legend([ ...
    arrayfun(@(j)sprintf('raw W%d',j),1:Nw,'UniformOutput',false), ...
    arrayfun(@(j)sprintf('effective W%d',j),1:Nw,'UniformOutput',false)], ...
    'Location','best');

nexttile;
plot(t,result.pulsePair.qualityGeometryBlendFactor,'LineWidth',1.1);
grid on;
ylim([-0.05,1.05]);
ylabel('\beta');
title('largest quality-weight fraction satisfying geometry constraint');

nexttile;
semilogy(t,max(result.pulsePair.transverseAccelerationVariance,eps), ...
    'LineWidth',1.0);
grid on;
ylabel('\sigma_{\perp,i}^2');
title('posterior transverse acceleration variance');

nexttile;
plot(t,result.pulsePair.normalizedNISEWMA,'LineWidth',1.0);
yline(1,'k:');
grid on;
xlabel('time [s]');
ylabel('normalized NIS EWMA');
title(sprintf('quality-fusion valid rate after settling = %.3f', ...
    result.pulsePair.postQualityFusionValidRate));
end
