function result = run_four_watcher_acceleration_blue_experiment( ...
    dim,makePlots,simulationTime,dt,seed,options)
%RUN_FOUR_WATCHER_ACCELERATION_BLUE_EXPERIMENT
% Four independent local angle-only [r;v;a] EKFs with rigorous
% geometry-aware acceleration fusion.
%
% PURPOSE
% -------
% Each watcher estimates the same inertial-frame target acceleration:
%
%   aHat_i^I = a^I + e_{a,i}.
%
% Rather than averaging the full vectors or introducing heuristic scalar
% weights, each watcher contributes only its LOS-transverse partial
% acceleration:
%
%   z_i = N_i^T aHat_i^I,
%
% where the columns of N_i form an orthonormal basis for the subspace
% perpendicular to the predicted LOS u_i:
%
%   N_i^T u_i = 0,   N_i^T N_i = I,
%   Pperp_i = N_i N_i^T = I-u_i u_i^T.
%
% The partial estimates are combined by generalized least squares / BLUE:
%
%   z = C a^I + v,
%
%   aHat_BLUE^I = (C^T R^dagger C)^dagger C^T R^dagger z.
%
% Two covariance models are reported:
%
%   1. diagonal BLUE:
%      local acceleration errors are assumed mutually independent;
%
%   2. correlated BLUE:
%      cross-covariances P_ij between local EKF errors are propagated and
%      used in
%
%         R_ij = N_i^T P_{aa,ij} N_j.
%
% Equal geometry-aware fusion is retained only as a baseline:
%
%   aHat_equal = (sum Pperp_i)^dagger sum Pperp_i aHat_i.
%
% No minimum weight, square-root weight, NIS weight, beta blending,
% low-pass filter, arithmetic-mean fallback, or previous-value hold is used.
% If the BLUE information matrix is not full rank, the full acceleration
% estimate is reported as NaN for that sample.
%
% CROSS-COVARIANCE MODEL
% ----------------------
% For i ~= j, with mutually independent measurement noises:
%
%   P_ij^- = F P_ij^+ F^T + gamma_Q Q,
%   P_ij^+ = A_i P_ij^- A_j^T,
%
% where A_i = I-K_i H_i when watcher i updates, and A_i = I during
% prediction-only operation. gamma_Q is
% options.commonProcessNoiseFraction.
%
% The diagonal blocks are the local Joseph-form EKF covariances.
%
% INPUTS
% ------
% dim             2 or 3.
% makePlots       logical plot flag.
% simulationTime  simulation duration [s].
% dt              sample time [s].
% seed            random seed.
% options         optional scalar structure.
%
% Important options:
%   truthAccelerationMode        'constant' or 'slow_sinusoid'
%   constantAcceleration         dim-by-1 constant value / sinusoid bias
%   accelerationAmplitude        dim-by-1
%   accelerationFrequency        dim-by-1 angular frequency [rad/s]
%   accelerationPhase            dim-by-1 phase [rad]
%   activeWatcherMode            'single','parallel_pair','all','dropout'
%   maneuverWatcherMask          1-by-4 logical
%   initialRangeScale            positive scalar or 1-by-4
%   initialCrossCovarianceMode   'independent' or 'common_prior'
%   commonProcessNoiseFraction   scalar in [0,1]
%   postBurnSettlingTime         [s]
%
% OUTPUT
% ------
% result.noManeuver
% result.pulsePair
% result.summary
%
% This is an estimation/fusion simulation. The local-state arithmetic mean
% is logged only as a diagnostic baseline and is never used as a fallback.

if nargin < 2 || isempty(makePlots),      makePlots = true; end
if nargin < 3 || isempty(simulationTime), simulationTime = 120; end
if nargin < 4 || isempty(dt),             dt = 0.05; end
if nargin < 5 || isempty(seed),           seed = 73; end
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
    initialState(dim+(1:dim),j) = rangeScale(j)*vTrue(:,1);
    initialState(2*dim+(1:dim),j) = rangeScale(j)*aTrue(:,1);

    u0 = normalizeVector( ...
        initialState(1:dim,j)-pWatcher0(:,j));
    burnDirection(:,j) = makeTransverseDirection(u0,j,dim);
end

% Identical measurement-noise realization in the no-maneuver and pulse
% cases. Different watcher columns are independent.
noise = struct();
if dim == 2
    noise.bearing = deg2rad(options.sigmaBearingDeg)*randn(N,Nw);
else
    noise.azimuth = deg2rad(options.sigmaAzimuthDeg)*randn(N,Nw);
    noise.elevation = deg2rad(options.sigmaElevationDeg)*randn(N,Nw);
end

activeMask = makeActiveMask( ...
    options.activeWatcherMode,options.activeWatcherIndex,t,Nw);
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
    noManeuver.localMeanPositionRMSE;
    pulsePair.localMeanPositionRMSE];
velocityRMSE = [ ...
    noManeuver.localMeanVelocityRMSE;
    pulsePair.localMeanVelocityRMSE];
localMeanAccelerationRMSE = [ ...
    noManeuver.localMeanAccelerationRMSE;
    pulsePair.localMeanAccelerationRMSE];

equalGeometryAccelerationRMSE = [ ...
    noManeuver.equalGeometryAccelerationRMSE;
    pulsePair.equalGeometryAccelerationRMSE];
diagonalBLUEAccelerationRMSE = [ ...
    noManeuver.diagonalBLUEAccelerationRMSE;
    pulsePair.diagonalBLUEAccelerationRMSE];
correlatedBLUEAccelerationRMSE = [ ...
    noManeuver.correlatedBLUEAccelerationRMSE;
    pulsePair.correlatedBLUEAccelerationRMSE];

postLocalMeanAccelerationRMSE = [ ...
    noManeuver.postLocalMeanAccelerationRMSE;
    pulsePair.postLocalMeanAccelerationRMSE];
postEqualGeometryAccelerationRMSE = [ ...
    noManeuver.postEqualGeometryAccelerationRMSE;
    pulsePair.postEqualGeometryAccelerationRMSE];
postDiagonalBLUEAccelerationRMSE = [ ...
    noManeuver.postDiagonalBLUEAccelerationRMSE;
    pulsePair.postDiagonalBLUEAccelerationRMSE];
postCorrelatedBLUEAccelerationRMSE = [ ...
    noManeuver.postCorrelatedBLUEAccelerationRMSE;
    pulsePair.postCorrelatedBLUEAccelerationRMSE];

postCorrelatedBLUEFullRankRate = [ ...
    noManeuver.postCorrelatedBLUEFullRankRate;
    pulsePair.postCorrelatedBLUEFullRankRate];
postCorrelatedBLUEMedianCondition = [ ...
    noManeuver.postCorrelatedBLUEMedianCondition;
    pulsePair.postCorrelatedBLUEMedianCondition];

totalDeltaV = [noManeuver.totalDeltaV;pulsePair.totalDeltaV];

result.summary = table( ...
    caseName,positionRMSE,velocityRMSE, ...
    localMeanAccelerationRMSE, ...
    equalGeometryAccelerationRMSE, ...
    diagonalBLUEAccelerationRMSE, ...
    correlatedBLUEAccelerationRMSE, ...
    postLocalMeanAccelerationRMSE, ...
    postEqualGeometryAccelerationRMSE, ...
    postDiagonalBLUEAccelerationRMSE, ...
    postCorrelatedBLUEAccelerationRMSE, ...
    postCorrelatedBLUEFullRankRate, ...
    postCorrelatedBLUEMedianCondition, ...
    totalDeltaV);

fprintf('\nFour local angle-only [r;v;a] EKFs\n');
fprintf('Acceleration fusion: transverse partial-estimate BLUE\n');
fprintf('Initial cross-covariance: %s\n', ...
    options.initialCrossCovarianceMode);
fprintf('Common process-noise fraction: %.3f\n', ...
    options.commonProcessNoiseFraction);
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
idxA = 2*dim+(1:dim);

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
Qcommon = common.options.commonProcessNoiseFraction*Q;

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
measurementAvailable = common.activeMask;
nis = nan(N,Nw);

localMeanState = nan(n,N);
localMeanAcceleration = nan(dim,N);

equalGeometryAcceleration = nan(dim,N);
diagonalBLUEAcceleration = nan(dim,N);
correlatedBLUEAcceleration = nan(dim,N);

equalGeometryRank = zeros(1,N);
equalGeometryCondition = inf(1,N);
diagonalBLUERank = zeros(1,N);
diagonalBLUECondition = inf(1,N);
correlatedBLUERank = zeros(1,N);
correlatedBLUECondition = inf(1,N);
directionalCovarianceRank = zeros(1,N);
directionalCovarianceCondition = inf(1,N);
directionalCovarianceMinEigenvalue = nan(1,N);
correlatedBLUECovariance = nan(dim,dim,N);

% Full local-error cross-covariance is needed recursively. Only acceleration
% cross-covariance blocks are logged.
Pcross = initializeCrossCovariance( ...
    P0,Nw,common.options.initialCrossCovarianceMode);
PaaCrossLog = zeros(dim,dim,Nw,Nw,N);

for i = 1:Nw
    for j = 1:Nw
        PaaCrossLog(:,:,i,j,1) = Pcross(idxA,idxA,i,j);
    end
end

for j = 1:Nw
    xhat(:,1,j) = common.initialStateEstimate(:,j);
    Plog(:,:,j,1) = P0;
    watcherPosition(:,1,j) = common.watcherInitialPosition(:,j);

    relativeHat0 = xhat(1:dim,1,j)-watcherPosition(:,1,j);
    uPred(:,1,j) = normalizeVector(relativeHat0);
end

fusion = computeAccelerationFusions( ...
    squeeze(xhat(:,1,:)),squeeze(uPred(:,1,:)), ...
    squeeze(PaaCrossLog(:,:,:,:,1)), ...
    measurementAvailable(1,:),dim);

localMeanState(:,1) = fusion.localMeanState;
localMeanAcceleration(:,1) = fusion.localMeanAcceleration;
equalGeometryAcceleration(:,1) = fusion.equalGeometryAcceleration;
diagonalBLUEAcceleration(:,1) = fusion.diagonalBLUEAcceleration;
correlatedBLUEAcceleration(:,1) = fusion.correlatedBLUEAcceleration;
equalGeometryRank(1) = fusion.equalGeometryRank;
equalGeometryCondition(1) = fusion.equalGeometryCondition;
diagonalBLUERank(1) = fusion.diagonalBLUERank;
diagonalBLUECondition(1) = fusion.diagonalBLUECondition;
correlatedBLUERank(1) = fusion.correlatedBLUERank;
correlatedBLUECondition(1) = fusion.correlatedBLUECondition;
directionalCovarianceRank(1) = fusion.directionalCovarianceRank;
directionalCovarianceCondition(1) = ...
    fusion.directionalCovarianceCondition;
directionalCovarianceMinEigenvalue(1) = ...
    fusion.directionalCovarianceMinEigenvalue;
correlatedBLUECovariance(:,:,1) = fusion.correlatedBLUECovariance;

for k = 1:N-1
    updateErrorMap = repmat(eye(n),1,1,Nw);

    for j = 1:Nw
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
                Rmeas = deg2rad(common.options.sigmaBearingDeg)^2;
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
                Rmeas = diag([ ...
                    deg2rad(common.options.sigmaAzimuthDeg)^2, ...
                    deg2rad(common.options.sigmaElevationDeg)^2]);
            end

            S = H*Pp*H.'+Rmeas;
            S = 0.5*(S+S.');
            K = (Pp*H.')/S;

            xn = xp+K*innov;
            Aupdate = eye(n)-K*H;
            Pn = Aupdate*Pp*Aupdate.'+K*Rmeas*K.';
            Pn = 0.5*(Pn+Pn.');

            nis(k+1,j) = innov.'*(S\innov);
            updateErrorMap(:,:,j) = Aupdate;
        else
            xn = xp;
            Pn = Pp;
            updateErrorMap(:,:,j) = eye(n);
        end

        xhat(:,k+1,j) = xn;
        Plog(:,:,j,k+1) = Pn;
    end

    % Propagate cross-covariances after all local gains/Jacobians are known.
    PcrossNext = zeros(size(Pcross));

    for i = 1:Nw
        PcrossNext(:,:,i,i) = Plog(:,:,i,k+1);
    end

    for i = 1:Nw
        for j = i+1:Nw
            PijPred = F*Pcross(:,:,i,j)*F.'+Qcommon;
            PijPost = updateErrorMap(:,:,i)*PijPred* ...
                updateErrorMap(:,:,j).';

            PcrossNext(:,:,i,j) = PijPost;
            PcrossNext(:,:,j,i) = PijPost.';
        end
    end

    Pcross = PcrossNext;

    for i = 1:Nw
        for j = 1:Nw
            PaaCrossLog(:,:,i,j,k+1) = ...
                Pcross(idxA,idxA,i,j);
        end
    end

    fusion = computeAccelerationFusions( ...
        squeeze(xhat(:,k+1,:)), ...
        squeeze(uPred(:,k+1,:)), ...
        squeeze(PaaCrossLog(:,:,:,:,k+1)), ...
        measurementAvailable(k+1,:),dim);

    localMeanState(:,k+1) = fusion.localMeanState;
    localMeanAcceleration(:,k+1) = fusion.localMeanAcceleration;
    equalGeometryAcceleration(:,k+1) = ...
        fusion.equalGeometryAcceleration;
    diagonalBLUEAcceleration(:,k+1) = ...
        fusion.diagonalBLUEAcceleration;
    correlatedBLUEAcceleration(:,k+1) = ...
        fusion.correlatedBLUEAcceleration;

    equalGeometryRank(k+1) = fusion.equalGeometryRank;
    equalGeometryCondition(k+1) = fusion.equalGeometryCondition;
    diagonalBLUERank(k+1) = fusion.diagonalBLUERank;
    diagonalBLUECondition(k+1) = fusion.diagonalBLUECondition;
    correlatedBLUERank(k+1) = fusion.correlatedBLUERank;
    correlatedBLUECondition(k+1) = fusion.correlatedBLUECondition;
    directionalCovarianceRank(k+1) = ...
        fusion.directionalCovarianceRank;
    directionalCovarianceCondition(k+1) = ...
        fusion.directionalCovarianceCondition;
    directionalCovarianceMinEigenvalue(k+1) = ...
        fusion.directionalCovarianceMinEigenvalue;
    correlatedBLUECovariance(:,:,k+1) = ...
        fusion.correlatedBLUECovariance;
end

for j = 1:Nw
    watcherAcceleration(:,N,j) = pulsePairCommand( ...
        mode,t(N),common.burnDirection(:,j), ...
        common.options,common.maneuverWatcherMask(j));
end

positionError = localMeanState(1:dim,:)-common.targetPosition;
velocityError = localMeanState(dim+(1:dim),:)-common.targetVelocity;
localMeanAccelerationError = ...
    localMeanAcceleration-common.trueAcceleration;
equalGeometryAccelerationError = ...
    equalGeometryAcceleration-common.trueAcceleration;
diagonalBLUEAccelerationError = ...
    diagonalBLUEAcceleration-common.trueAcceleration;
correlatedBLUEAccelerationError = ...
    correlatedBLUEAcceleration-common.trueAcceleration;

localAccelerationErrorNorm = nan(N,Nw);
for j = 1:Nw
    localAcceleration = squeeze(xhat(idxA,:,j));
    localAccelerationErrorNorm(:,j) = vecnorm( ...
        localAcceleration-common.trueAcceleration,2,1).';
end

evaluationStartTime = common.options.burnStartTime ...
    + 2*common.options.burnDuration ...
    + common.options.postBurnSettlingTime;
idxPost = t >= evaluationStartTime;

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
    'localAccelerationCrossCovariance',PaaCrossLog, ...
    'predictedLOS',uPred, ...
    'measurementAvailable',measurementAvailable, ...
    'maneuverWatcherMask',common.maneuverWatcherMask, ...
    'watcherPositionHistory',watcherPosition, ...
    'watcherVelocityHistory',watcherVelocity, ...
    'watcherManeuverAcceleration',watcherAcceleration, ...
    'NIS',nis, ...
    'localMeanState',localMeanState, ...
    'localMeanAcceleration',localMeanAcceleration, ...
    'equalGeometryAcceleration',equalGeometryAcceleration, ...
    'diagonalBLUEAcceleration',diagonalBLUEAcceleration, ...
    'correlatedBLUEAcceleration',correlatedBLUEAcceleration, ...
    'correlatedBLUECovariance',correlatedBLUECovariance, ...
    'equalGeometryRank',equalGeometryRank, ...
    'equalGeometryCondition',equalGeometryCondition, ...
    'diagonalBLUERank',diagonalBLUERank, ...
    'diagonalBLUECondition',diagonalBLUECondition, ...
    'correlatedBLUERank',correlatedBLUERank, ...
    'correlatedBLUECondition',correlatedBLUECondition, ...
    'directionalCovarianceRank',directionalCovarianceRank, ...
    'directionalCovarianceCondition', ...
        directionalCovarianceCondition, ...
    'directionalCovarianceMinEigenvalue', ...
        directionalCovarianceMinEigenvalue, ...
    'localAccelerationErrorNorm',localAccelerationErrorNorm, ...
    'localMeanPositionErrorNorm',vecnorm(positionError,2,1), ...
    'localMeanVelocityErrorNorm',vecnorm(velocityError,2,1), ...
    'localMeanAccelerationErrorNorm', ...
        vecnorm(localMeanAccelerationError,2,1), ...
    'equalGeometryAccelerationErrorNorm', ...
        vecnorm(equalGeometryAccelerationError,2,1), ...
    'diagonalBLUEAccelerationErrorNorm', ...
        vecnorm(diagonalBLUEAccelerationError,2,1), ...
    'correlatedBLUEAccelerationErrorNorm', ...
        vecnorm(correlatedBLUEAccelerationError,2,1), ...
    'evaluationStartTime',evaluationStartTime, ...
    'postBurnIndex',idxPost, ...
    'deltaVByWatcher',deltaVByWatcher, ...
    'totalDeltaV',totalDeltaV, ...
    'localMeanPositionRMSE',rmsFinite( ...
        vecnorm(positionError,2,1)), ...
    'localMeanVelocityRMSE',rmsFinite( ...
        vecnorm(velocityError,2,1)), ...
    'localMeanAccelerationRMSE',rmsFinite( ...
        vecnorm(localMeanAccelerationError,2,1)), ...
    'equalGeometryAccelerationRMSE',rmsFinite( ...
        vecnorm(equalGeometryAccelerationError,2,1)), ...
    'diagonalBLUEAccelerationRMSE',rmsFinite( ...
        vecnorm(diagonalBLUEAccelerationError,2,1)), ...
    'correlatedBLUEAccelerationRMSE',rmsFinite( ...
        vecnorm(correlatedBLUEAccelerationError,2,1)), ...
    'postLocalMeanAccelerationRMSE',rmsFinite( ...
        vecnorm(localMeanAccelerationError(:,idxPost),2,1)), ...
    'postEqualGeometryAccelerationRMSE',rmsFinite( ...
        vecnorm(equalGeometryAccelerationError(:,idxPost),2,1)), ...
    'postDiagonalBLUEAccelerationRMSE',rmsFinite( ...
        vecnorm(diagonalBLUEAccelerationError(:,idxPost),2,1)), ...
    'postCorrelatedBLUEAccelerationRMSE',rmsFinite( ...
        vecnorm(correlatedBLUEAccelerationError(:,idxPost),2,1)), ...
    'postEqualGeometryFullRankRate',mean( ...
        equalGeometryRank(idxPost) == dim), ...
    'postDiagonalBLUEFullRankRate',mean( ...
        diagonalBLUERank(idxPost) == dim), ...
    'postCorrelatedBLUEFullRankRate',mean( ...
        correlatedBLUERank(idxPost) == dim), ...
    'postCorrelatedBLUEMedianCondition',medianFinite( ...
        correlatedBLUECondition(idxPost)), ...
    'postCorrelatedBLUEMaxCondition',maxFinite( ...
        correlatedBLUECondition(idxPost)));
end

function fusion = computeAccelerationFusions( ...
    xLocal,uPred,PaaCross,active,dim)
%COMPUTEACCELERATIONFUSIONS Equal geometry and transverse BLUE rules.

active = logical(active(:).');
activeIndex = find(active);
n = size(xLocal,1);
idxA = 2*dim+(1:dim);

fusion = struct( ...
    'localMeanState',nan(n,1), ...
    'localMeanAcceleration',nan(dim,1), ...
    'equalGeometryAcceleration',nan(dim,1), ...
    'diagonalBLUEAcceleration',nan(dim,1), ...
    'correlatedBLUEAcceleration',nan(dim,1), ...
    'correlatedBLUECovariance',nan(dim), ...
    'equalGeometryRank',0, ...
    'equalGeometryCondition',Inf, ...
    'diagonalBLUERank',0, ...
    'diagonalBLUECondition',Inf, ...
    'correlatedBLUERank',0, ...
    'correlatedBLUECondition',Inf, ...
    'directionalCovarianceRank',0, ...
    'directionalCovarianceCondition',Inf, ...
    'directionalCovarianceMinEigenvalue',nan);

if isempty(activeIndex)
    return;
end

fusion.localMeanState = mean(xLocal(:,activeIndex),2);
aLocal = xLocal(idxA,activeIndex);
fusion.localMeanAcceleration = mean(aLocal,2);

nActive = numel(activeIndex);
mPerWatcher = dim-1;
m = nActive*mPerWatcher;

Ncell = cell(1,nActive);
C = zeros(m,dim);
zStack = zeros(m,1);

OmegaEqual = zeros(dim);
rhsEqual = zeros(dim,1);

InfoDiagonal = zeros(dim);
rhsDiagonal = zeros(dim,1);

for q = 1:nActive
    i = activeIndex(q);
    rows = (q-1)*mPerWatcher+(1:mPerWatcher);

    Ni = makeTransverseBasis(uPred(:,i));
    Ncell{q} = Ni;

    C(rows,:) = Ni.';
    zStack(rows) = Ni.'*aLocal(:,q);

    Pperp = Ni*Ni.';
    OmegaEqual = OmegaEqual+Pperp;
    rhsEqual = rhsEqual+Pperp*aLocal(:,q);

    Rii = Ni.'*PaaCross(:,:,i,i)*Ni;
    Rii = 0.5*(Rii+Rii.');
    [RiiPlus,~,~,~,isPSD] = symmetricPSDInverse(Rii,1e-11);

    if isPSD
        InfoDiagonal = InfoDiagonal+Ni*RiiPlus*Ni.';
        rhsDiagonal = rhsDiagonal+Ni*RiiPlus*Ni.'*aLocal(:,q);
    else
        InfoDiagonal(:) = nan;
        rhsDiagonal(:) = nan;
    end
end

[OmegaEqualPlus,rankEqual,condEqual,~,equalPSD] = ...
    symmetricPSDInverse(OmegaEqual,1e-11);

fusion.equalGeometryRank = rankEqual;
fusion.equalGeometryCondition = condEqual;
if equalPSD && rankEqual == dim
    fusion.equalGeometryAcceleration = OmegaEqualPlus*rhsEqual;
end

if all(isfinite(InfoDiagonal(:)))
    [InfoDiagonalPlus,rankDiagonal,condDiagonal,~,diagonalPSD] = ...
        symmetricPSDInverse(InfoDiagonal,1e-11);

    fusion.diagonalBLUERank = rankDiagonal;
    fusion.diagonalBLUECondition = condDiagonal;

    if diagonalPSD && rankDiagonal == dim
        fusion.diagonalBLUEAcceleration = ...
            InfoDiagonalPlus*rhsDiagonal;
    end
end

Rjoint = zeros(m);

for q = 1:nActive
    i = activeIndex(q);
    rowsQ = (q-1)*mPerWatcher+(1:mPerWatcher);
    Ni = Ncell{q};

    for s = 1:nActive
        j = activeIndex(s);
        rowsS = (s-1)*mPerWatcher+(1:mPerWatcher);
        Nj = Ncell{s};

        Rjoint(rowsQ,rowsS) = ...
            Ni.'*PaaCross(:,:,i,j)*Nj;
    end
end

Rjoint = 0.5*(Rjoint+Rjoint.');
[RjointPlus,rankR,condR,minEigR,isPSD_R] = ...
    symmetricPSDInverse(Rjoint,1e-11);

fusion.directionalCovarianceRank = rankR;
fusion.directionalCovarianceCondition = condR;
fusion.directionalCovarianceMinEigenvalue = minEigR;

if isPSD_R
    InfoCorrelated = C.'*RjointPlus*C;
    InfoCorrelated = 0.5*(InfoCorrelated+InfoCorrelated.');
    rhsCorrelated = C.'*RjointPlus*zStack;

    [InfoCorrelatedPlus,rankCorrelated,condCorrelated,~,isPSD_Info] = ...
        symmetricPSDInverse(InfoCorrelated,1e-11);

    fusion.correlatedBLUERank = rankCorrelated;
    fusion.correlatedBLUECondition = condCorrelated;

    if isPSD_Info && rankCorrelated == dim
        fusion.correlatedBLUEAcceleration = ...
            InfoCorrelatedPlus*rhsCorrelated;
        fusion.correlatedBLUECovariance = InfoCorrelatedPlus;
    end
end
end

function Pcross = initializeCrossCovariance(P0,Nw,mode)
%INITIALIZECROSSCOVARIANCE Explicit prior-correlation assumption.

n = size(P0,1);
Pcross = zeros(n,n,Nw,Nw);

for i = 1:Nw
    Pcross(:,:,i,i) = P0;
end

switch mode
    case 'independent'
        % Off-diagonal blocks remain zero.

    case 'common_prior'
        for i = 1:Nw
            for j = i+1:Nw
                Pcross(:,:,i,j) = P0;
                Pcross(:,:,j,i) = P0;
            end
        end

    otherwise
        error('Unsupported initialCrossCovarianceMode: %s',mode);
end
end

function N = makeTransverseBasis(u)
%MAKETRANSVERSEBASIS Orthonormal basis perpendicular to unit LOS u.

u = normalizeVector(u);
dim = numel(u);

if dim == 2
    N = [-u(2);u(1)];
else
    axesSet = eye(3);
    [~,index] = min(abs(axesSet.'*u));
    referenceAxis = axesSet(:,index);

    n1 = normalizeVector(cross(u,referenceAxis));
    n2 = normalizeVector(cross(u,n1));
    N = [n1,n2];
end
end

function mask = makeActiveMask(mode,index,t,Nw)
%MAKEACTIVEMASK Generate measurement availability.

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
%PULSEPAIRCOMMAND Finite known maneuver.

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
end

function options = resolveOptions(options,dim,simulationTime)

defaults = struct( ...
    'truthAccelerationMode','slow_sinusoid', ...
    'constantAcceleration',defaultAcceleration(dim), ...
    'accelerationAmplitude',defaultAccelerationAmplitude(dim), ...
    'accelerationFrequency',defaultAccelerationFrequency(dim), ...
    'accelerationPhase',defaultAccelerationPhase(dim), ...
    'targetInitialPosition',zeros(dim,1), ...
    'targetInitialVelocity',defaultVelocity(dim), ...
    'activeWatcherMode','all', ...
    'activeWatcherIndex',1, ...
    'maneuverWatcherMask',[true,true,true,true], ...
    'initialRangeScale',[0.55,0.80,1.25,1.55], ...
    'initialCrossCovarianceMode','independent', ...
    'commonProcessNoiseFraction',1.0, ...
    'postBurnSettlingTime',20.0, ...
    'watcherRadius',1000, ...
    'parallelSeparation',100, ...
    'burnAcceleration',2.0, ...
    'burnStartTime',10.0, ...
    'burnDuration',5.0, ...
    'sigmaBearingDeg',0.2, ...
    'sigmaAzimuthDeg',0.2, ...
    'sigmaElevationDeg',0.2, ...
    'sigmaJerk',0.005, ...
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
options.initialCrossCovarianceMode = validatestring( ...
    options.initialCrossCovarianceMode, ...
    {'independent','common_prior'});

validateattributes(options.activeWatcherIndex,{'numeric'}, ...
    {'scalar','integer','>=',1,'<=',4});

validateattributes(options.maneuverWatcherMask,{'numeric','logical'}, ...
    {'vector','numel',4});
if any(~ismember(double(options.maneuverWatcherMask(:)),[0,1]))
    error('options.maneuverWatcherMask must contain only true/false values.');
end
options.maneuverWatcherMask = ...
    logical(options.maneuverWatcherMask(:).');

validateattributes(options.initialRangeScale,{'numeric'}, ...
    {'vector','positive','finite'});
validateattributes(options.commonProcessNoiseFraction,{'numeric'}, ...
    {'scalar','>=',0,'<=',1,'finite'});
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
validateattributes(options.sigmaBearingDeg,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.sigmaAzimuthDeg,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.sigmaElevationDeg,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.sigmaJerk,{'numeric'}, ...
    {'scalar','nonnegative','finite'});
validateattributes(options.initialPositionSigma,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.initialVelocitySigma,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(options.initialAccelerationSigma,{'numeric'}, ...
    {'scalar','positive','finite'});

if options.burnStartTime+2*options.burnDuration > simulationTime
    error('The pulse pair must finish before simulationTime.');
end
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

N = makeTransverseBasis(u);
d = N(:,1);
d = (-1)^(j-1)*normalizeVector(d);

if dim == 3 && size(N,2) > 1 && mod(j,2) == 0
    d = (-1)^(j-1)*N(:,2);
end
end

function [Aplus,numericalRank,conditionNumber,minEigenvalue,isPSD] = ...
    symmetricPSDInverse(A,relativeTolerance)
%SYMMETRICPSDINVERSE Moore-Penrose inverse for a symmetric PSD matrix.
%
% Eigenvalues below relativeTolerance*lambdaMax are treated as numerical
% zeros. A materially negative eigenvalue marks the matrix non-PSD; no
% estimator is returned from that covariance/information matrix.

A = 0.5*(A+A.');
[V,D] = eig(A,'vector');
[eigenvalues,order] = sort(real(D),'descend');
V = V(:,order);

if isempty(eigenvalues)
    Aplus = zeros(size(A));
    numericalRank = 0;
    conditionNumber = Inf;
    minEigenvalue = nan;
    isPSD = false;
    return;
end

lambdaMax = max(eigenvalues);
minEigenvalue = min(eigenvalues);
scale = max(abs(lambdaMax),1);
isPSD = minEigenvalue >= -relativeTolerance*scale;

if lambdaMax <= relativeTolerance*scale
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
x = x(isfinite(x));
if isempty(x)
    value = nan;
else
    value = median(x);
end
end

function value = maxFinite(x)
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

%% Figure 1: local-network mean state, diagnostic only
fig(1) = figure( ...
    'Name','Local EKF mean state: diagnostic', ...
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
        plot(t,result.noManeuver.localMeanState(index(q),:), ...
            'LineWidth',1.0);
        plot(t,result.pulsePair.localMeanState(index(q),:), ...
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

%% Figure 2: acceleration fusion
fig(2) = figure( ...
    'Name','Transverse partial-estimate BLUE acceleration fusion', ...
    'Color','w');
tiledlayout(dim+1,1,'TileSpacing','compact','Padding','compact');

for q = 1:dim
    nexttile;
    plot(t,result.trueAcceleration(q,:),'k','LineWidth',1.5);
    hold on;
    plot(t,result.pulsePair.localMeanAcceleration(q,:), ...
        ':','LineWidth',1.0);
    plot(t,result.pulsePair.equalGeometryAcceleration(q,:), ...
        '-.','LineWidth',1.0);
    plot(t,result.pulsePair.diagonalBLUEAcceleration(q,:), ...
        '--','LineWidth',1.2);
    plot(t,result.pulsePair.correlatedBLUEAcceleration(q,:), ...
        'LineWidth',1.3);
    xline(result.pulsePair.evaluationStartTime,'k:');
    grid on;
    ylabel(sprintf('a_%d',q));

    if q == 1
        legend('truth','local mean: diagnostic','equal geometry', ...
            'diagonal BLUE','correlated BLUE','post start', ...
            'Location','best');
    end
end

nexttile;
semilogy(t,max(result.pulsePair.localMeanAccelerationErrorNorm,eps), ...
    ':','LineWidth',1.0);
hold on;
semilogy(t,max( ...
    result.pulsePair.equalGeometryAccelerationErrorNorm,eps), ...
    '-.','LineWidth',1.0);
semilogy(t,max( ...
    result.pulsePair.diagonalBLUEAccelerationErrorNorm,eps), ...
    '--','LineWidth',1.2);
semilogy(t,max( ...
    result.pulsePair.correlatedBLUEAccelerationErrorNorm,eps), ...
    'LineWidth',1.3);
xline(result.pulsePair.evaluationStartTime,'k:');
grid on;
xlabel('time [s]');
ylabel('acceleration error');
legend('local mean','equal geometry','diagonal BLUE', ...
    'correlated BLUE','post start','Location','best');

%% Figure 3: information rank and condition
fig(3) = figure( ...
    'Name','BLUE information diagnostics', ...
    'Color','w');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
stairs(t,result.pulsePair.equalGeometryRank,'LineWidth',1.0);
hold on;
stairs(t,result.pulsePair.diagonalBLUERank,'--','LineWidth',1.1);
stairs(t,result.pulsePair.correlatedBLUERank,':','LineWidth',1.2);
grid on;
ylim([-0.1,dim+0.1]);
ylabel('rank');
legend('equal geometry','diagonal BLUE','correlated BLUE', ...
    'Location','best');

nexttile;
semilogy(t,max(result.pulsePair.equalGeometryCondition,1), ...
    'LineWidth',1.0);
hold on;
semilogy(t,max(result.pulsePair.diagonalBLUECondition,1), ...
    '--','LineWidth',1.1);
semilogy(t,max(result.pulsePair.correlatedBLUECondition,1), ...
    ':','LineWidth',1.2);
grid on;
ylabel('information condition');
legend('equal geometry','diagonal BLUE','correlated BLUE', ...
    'Location','best');

nexttile;
semilogy(t,max(result.pulsePair.directionalCovarianceCondition,1), ...
    'LineWidth',1.1);
grid on;
xlabel('time [s]');
ylabel('cond(R)');
title('stacked transverse-estimate covariance');

%% Figure 4: watcher excitation and local errors
fig(4) = figure( ...
    'Name','Watcher maneuver and local acceleration error', ...
    'Color','w');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
hold on;
for j = 1:Nw
    aW = squeeze( ...
        result.pulsePair.watcherManeuverAcceleration(:,:,j));
    signedCommand = result.burnDirection(:,j).'*aW;
    plot(t,signedCommand,'LineWidth',1.0);
end
grid on;
ylabel('signed a_w');
title('finite pulse-pair commands');
legend(arrayfun(@(j)sprintf('watcher %d',j), ...
    1:Nw,'UniformOutput',false),'Location','best');

nexttile;
plot(t,result.pulsePair.localAccelerationErrorNorm, ...
    'LineWidth',0.9);
xline(result.pulsePair.evaluationStartTime,'k:');
grid on;
ylabel('||e_{a,i}||');
title('watcher-level local acceleration errors');

nexttile;
bar(1:Nw,result.pulsePair.deltaVByWatcher);
grid on;
xlabel('watcher');
ylabel('\int ||a_w||dt');
title(sprintf('total maneuver effort = %.3f', ...
    result.pulsePair.totalDeltaV));

%% Figure 5: covariance validity
fig(5) = figure( ...
    'Name','Cross-covariance model diagnostics', ...
    'Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t,result.pulsePair.directionalCovarianceMinEigenvalue, ...
    'LineWidth',1.0);
yline(0,'k:');
grid on;
ylabel('min eig(R)');
title('stacked transverse covariance PSD check');

nexttile;
plot(t,tracePage(result.pulsePair.correlatedBLUECovariance), ...
    'LineWidth',1.1);
grid on;
xlabel('time [s]');
ylabel('tr(P_{a,f})');
title('correlated-BLUE reported acceleration covariance');
end

function y = tracePage(A)
%TRACEPAGE Trace of every page of a square 3-D array.

N = size(A,3);
y = nan(1,N);
for k = 1:N
    Ak = A(:,:,k);
    if all(isfinite(Ak(:)))
        y(k) = trace(Ak);
    end
end
end
