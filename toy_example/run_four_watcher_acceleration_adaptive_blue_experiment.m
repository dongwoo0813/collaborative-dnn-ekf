function result = run_four_watcher_acceleration_adaptive_blue_experiment( ...
    dim,makePlots,simulationTime,dt,seed,options)
%RUN_FOUR_WATCHER_ACCELERATION_ADAPTIVE_BLUE_EXPERIMENT
% Four independent angle-only [r;v;a] EKFs with:
%   1) equal LOS-transverse geometry fusion,
%   2) diagonal transverse BLUE,
%   3) correlated transverse BLUE using propagated cross-covariances,
%   4) an initial observability maneuver, and
%   5) repeated maneuvers triggered by weak rolling observability, and
%   6) a gated directional reconstruction for the future multi-branch DNN.
%
% The maneuver trigger combines two diagnostics.
%
% Rolling acceleration information:
%
%   J_a(k) = sum_{ell in W_k} Gamma_a(tau_ell)' H_r(ell)' R^(-1)
%            H_r(ell) Gamma_a(tau_ell),
%
%   Gamma_a(tau) = 0.5*tau^2*I.
%
% Current network fusion geometry:
%
%   G(k) = sum_i (I-u_i*u_i').
%
% After the initial pulse pair, the code calibrates a healthy reference
% value of lambda_min(J_a). A new pulse pair is triggered when either
%
%   lambda_min(J_a) < triggerFraction*referenceLambda,
%
% or the normalized geometry score becomes too small for a prescribed
% dwell time. This is a practical online diagnostic, not a formal proof of
% nonlinear observability.
%
% The fusion method is unchanged by the maneuver logic. For watcher i,
%
%   z_i = N_i' aHat_i,
%   N_i' u_i = 0,
%
% and the correlated BLUE uses
%
%   R_ij = N_i' P_aa,ij N_j,
%   aHat = (C' R^dagger C)^dagger C' R^dagger z.
%
% The additional branch-routing proxy is
%
%   aHat_gate = [sum_i alpha_i Pperp_i]^dagger
%               sum_i alpha_i Pperp_i aHat_i,
%   Pperp_i = I-u_i*u_i'.
%
% u_i is the predicted LOS unit vector (l_i in the DNN-EKF notation).
% alpha_i is a softmax of negative log transverse acceleration uncertainty.
% It is fixed by EKF quantities rather than trained, so the gate does not
% introduce another trainable branch-assignment ambiguity.
%
% INPUTS
% ------
% dim             Spatial dimension, 2 or 3.
% makePlots       Logical plot flag.
% simulationTime  Simulation duration [s].
% dt              Sample time [s].
% seed            Random seed.
% options         Optional scalar structure. See resolveOptions().
%
% OUTPUT
% ------
% result          Truth, local EKFs, fusion outputs, observability metrics,
%                 maneuver events, cross-covariance diagnostics, and plots.

if nargin < 2 || isempty(makePlots),      makePlots = true; end
if nargin < 3 || isempty(simulationTime), simulationTime = 180; end
if nargin < 4 || isempty(dt),             dt = 0.05; end
if nargin < 5 || isempty(seed),           seed = 73; end
if nargin < 6 || isempty(options),        options = struct(); end

validateattributes(dim,{'numeric'},{'scalar','integer'});
if ~ismember(dim,[2,3])
    error('dim must be 2 or 3.');
end
validateattributes(makePlots,{'logical','numeric'},{'scalar'});
validateattributes(simulationTime,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(dt,{'numeric'},{'scalar','positive','finite'});
validateattributes(seed,{'numeric'}, ...
    {'scalar','integer','nonnegative'});
if ~isstruct(options) || ~isscalar(options)
    error('options must be a scalar structure.');
end

rng(seed,'twister');
options = resolveOptions(options,dim,simulationTime,dt);

Nw = 4;
t = (0:dt:simulationTime).';
N = numel(t);
n = 3*dim;
idxR = 1:dim;
idxV = dim+(1:dim);
idxA = 2*dim+(1:dim);

%% Truth trajectory
trueAcceleration = makeTruthAcceleration(t,dim,options);
truePosition = zeros(dim,N);
trueVelocity = zeros(dim,N);
truePosition(:,1) = options.targetInitialPosition;
trueVelocity(:,1) = options.targetInitialVelocity;

for k = 1:N-1
    aAverage = 0.5*(trueAcceleration(:,k)+trueAcceleration(:,k+1));
    truePosition(:,k+1) = truePosition(:,k) ...
        + trueVelocity(:,k)*dt + 0.5*aAverage*dt^2;
    trueVelocity(:,k+1) = trueVelocity(:,k)+aAverage*dt;
end

%% Initial watcher geometry
watcherPosition0 = makeInitialWatcherGeometry(dim,options,Nw);

%% Local EKF model
F = [ ...
    eye(dim),       dt*eye(dim), 0.5*dt^2*eye(dim); ...
    zeros(dim),     eye(dim),    dt*eye(dim); ...
    zeros(dim),     zeros(dim),  eye(dim)];

% Piecewise-constant jerk input integrated over one sample.
Gjerk = [ ...
    (dt^3/6)*eye(dim); ...
    (dt^2/2)*eye(dim); ...
    dt*eye(dim)];
Q = options.sigmaJerk^2*(Gjerk*Gjerk.');
Q = 0.5*(Q+Q.');

P0 = diag([ ...
    options.initialPositionStd^2*ones(1,dim), ...
    options.initialVelocityStd^2*ones(1,dim), ...
    options.initialAccelerationStd^2*ones(1,dim)]);

%% Storage
xLocal = zeros(n,N,Nw);
PLocal = zeros(n,n,Nw,N);
aLocal = zeros(dim,N,Nw);

watcherPosition = zeros(dim,N,Nw);
watcherVelocity = zeros(dim,N,Nw);
watcherCommand = zeros(dim,N,Nw);

uPred = zeros(dim,N,Nw);
rangePred = nan(N,Nw);
activeMask = true(N,Nw);

innovationLog = cell(N,Nw);
nisLog = nan(N,Nw);

% Current full local-error cross covariance. Only the final value is stored
% in the result because the complete five-dimensional history is large.
Pcross = zeros(n,n,Nw,Nw);

%% Initial local estimates and covariances
for j = 1:Nw
    watcherPosition(:,1,j) = watcherPosition0(:,j);

    rhoTrue0 = truePosition(:,1)-watcherPosition0(:,j);
    transverse0 = transverseBasis(normalizeVector(rhoTrue0),j);
    transverseOffset = options.initialTransversePositionStd ...
        * randn*transverse0(:,1);

    positionGuess = watcherPosition0(:,j) ...
        + options.initialRangeScale(j)*rhoTrue0 ...
        + transverseOffset;

    xLocal(:,1,j) = [ ...
        positionGuess; ...
        trueVelocity(:,1) ...
            + options.initialVelocityMeanErrorStd*randn(dim,1); ...
        trueAcceleration(:,1) ...
            + options.initialAccelerationMeanErrorStd*randn(dim,1)];

    PLocal(:,:,j,1) = P0;
    aLocal(:,1,j) = xLocal(idxA,1,j);
    uPred(:,1,j) = normalizeVector( ...
        xLocal(idxR,1,j)-watcherPosition(:,1,j));
    rangePred(1,j) = norm( ...
        xLocal(idxR,1,j)-watcherPosition(:,1,j));

    Pcross(:,:,j,j) = P0;
end

for i = 1:Nw
    for j = 1:Nw
        if i == j
            continue;
        end
        if strcmp(options.initialCrossCovarianceMode,'common_prior')
            Pcross(:,:,i,j) = P0;
        else
            Pcross(:,:,i,j) = zeros(n);
        end
    end
end

%% Fusion and observability storage
localMeanAcceleration = nan(dim,N);
equalGeometryAcceleration = nan(dim,N);
diagonalBLUEAcceleration = nan(dim,N);
correlatedBLUEAcceleration = nan(dim,N);
gatedDirectionalAcceleration = nan(dim,N);
directionalGateWeight = nan(N,Nw);
gatedDirectionalRank = zeros(1,N);
gatedDirectionalCondition = inf(1,N);

equalGeometryRank = zeros(1,N);
equalGeometryCondition = inf(1,N);
equalGeometryScore = zeros(1,N);

diagonalBLUERank = zeros(1,N);
diagonalBLUECondition = inf(1,N);
correlatedBLUERank = zeros(1,N);
correlatedBLUECondition = inf(1,N);
correlatedProjectedCovarianceMinEigenvalue = nan(1,N);

rollingAccelerationInformation = zeros(dim,dim,N);
temporalInformationMinEigenvalue = zeros(1,N);
temporalInformationCondition = inf(1,N);
temporalDirectionScore = zeros(1,N);
temporalTriggerThreshold = nan(1,N);
weakTemporalFlag = false(1,N);
weakGeometryFlag = false(1,N);
triggerConditionFlag = false(1,N);

maneuverActive = false(1,N);
maneuverPhase = zeros(1,N);
maneuverEventIndex = zeros(1,N);

%% Initial fusion and information metrics
fusion0 = fuseTransverseBLUE( ...
    squeeze(aLocal(:,1,:)),squeeze(uPred(:,1,:)), ...
    PLocal(:,:,:,1),Pcross,idxA,options);

localMeanAcceleration(:,1) = fusion0.localMean;
equalGeometryAcceleration(:,1) = fusion0.equalGeometry;
diagonalBLUEAcceleration(:,1) = fusion0.diagonalBLUE;
correlatedBLUEAcceleration(:,1) = fusion0.correlatedBLUE;
gatedDirectionalAcceleration(:,1) = fusion0.gatedDirectional;
directionalGateWeight(1,:) = fusion0.directionalGateWeight;
gatedDirectionalRank(1) = fusion0.gatedDirectionalRank;
gatedDirectionalCondition(1) = fusion0.gatedDirectionalCondition;
equalGeometryRank(1) = fusion0.equalGeometryRank;
equalGeometryCondition(1) = fusion0.equalGeometryCondition;
equalGeometryScore(1) = fusion0.equalGeometryScore;
diagonalBLUERank(1) = fusion0.diagonalBLUERank;
diagonalBLUECondition(1) = fusion0.diagonalBLUECondition;
correlatedBLUERank(1) = fusion0.correlatedBLUERank;
correlatedBLUECondition(1) = fusion0.correlatedBLUECondition;
correlatedProjectedCovarianceMinEigenvalue(1) = ...
    fusion0.correlatedProjectedCovarianceMinEigenvalue;

info0 = computeRollingAccelerationInformation( ...
    1,t,uPred,rangePred,activeMask,dim,options);
rollingAccelerationInformation(:,:,1) = info0.matrix;
temporalInformationMinEigenvalue(1) = info0.minEigenvalue;
temporalInformationCondition(1) = info0.condition;
temporalDirectionScore(1) = info0.directionScore;

%% Event-trigger manager
manager = initializeManeuverManager(dim,options);

%% Main recursion
for k = 1:N-1
    tk = t(k);

    [manager,triggerData] = updateManeuverManager( ...
        manager,tk,rollingAccelerationInformation(:,:,k), ...
        temporalInformationMinEigenvalue(k), ...
        equalGeometryScore(k),uPred(:,k,:),options,dt);

    temporalTriggerThreshold(k) = triggerData.temporalThreshold;
    weakTemporalFlag(k) = triggerData.weakTemporal;
    weakGeometryFlag(k) = triggerData.weakGeometry;
    triggerConditionFlag(k) = triggerData.triggerCondition;
    maneuverActive(k) = manager.active;
    maneuverPhase(k) = currentPulsePhase(manager,tk,options);
    maneuverEventIndex(k) = manager.currentEventIndex;

    % Generate known watcher commands using only predicted LOS geometry.
    for j = 1:Nw
        watcherCommand(:,k,j) = computeAdaptiveManeuverCommand( ...
            manager,tk,uPred(:,k,j),j,options);

        pW = watcherPosition(:,k,j);
        vW = watcherVelocity(:,k,j);
        aW = watcherCommand(:,k,j);

        watcherPosition(:,k+1,j) = pW+vW*dt+0.5*aW*dt^2;
        watcherVelocity(:,k+1,j) = vW+aW*dt;
    end

    %% Local predictions and predicted cross-covariances
    xPred = zeros(n,Nw);
    PPred = zeros(n,n,Nw);
    PcrossPred = zeros(n,n,Nw,Nw);

    for j = 1:Nw
        xPred(:,j) = F*xLocal(:,k,j);
        PPred(:,:,j) = F*PLocal(:,:,j,k)*F.'+Q;
        PPred(:,:,j) = 0.5*(PPred(:,:,j)+PPred(:,:,j).');
    end

    for i = 1:Nw
        for j = 1:Nw
            if i == j
                PcrossPred(:,:,i,j) = PPred(:,:,i);
            else
                PcrossPred(:,:,i,j) = F*Pcross(:,:,i,j)*F.' ...
                    + options.commonProcessNoiseFraction*Q;
            end
        end
    end

    %% Independent local angle-only updates
    updateMap = repmat(eye(n),1,1,Nw);

    for j = 1:Nw
        pWNext = watcherPosition(:,k+1,j);
        rhoTrue = truePosition(:,k+1)-pWNext;
        rhoHat = xPred(idxR,j)-pWNext;
        rhoHatNorm = max(norm(rhoHat),options.minimumRange);
        uHat = rhoHat/rhoHatNorm;

        uPred(:,k+1,j) = uHat;
        rangePred(k+1,j) = rhoHatNorm;

        [innovation,H,R] = makeAngleMeasurementUpdate( ...
            rhoTrue,rhoHat,dim,options);

        S = H*PPred(:,:,j)*H.'+R;
        S = 0.5*(S+S.');
        K = (PPred(:,:,j)*H.')/S;

        xPost = xPred(:,j)+K*innovation;
        A = eye(n)-K*H;
        PPost = A*PPred(:,:,j)*A.'+K*R*K.';
        PPost = 0.5*(PPost+PPost.');

        xLocal(:,k+1,j) = xPost;
        PLocal(:,:,j,k+1) = PPost;
        aLocal(:,k+1,j) = xPost(idxA);
        updateMap(:,:,j) = A;
        innovationLog{k+1,j} = innovation;
        nisLog(k+1,j) = innovation.'*(S\innovation);
    end

    %% Cross-covariance update
    PcrossNew = zeros(n,n,Nw,Nw);
    for i = 1:Nw
        for j = 1:Nw
            if i == j
                PcrossNew(:,:,i,j) = PLocal(:,:,i,k+1);
            else
                PcrossNew(:,:,i,j) = updateMap(:,:,i) ...
                    * PcrossPred(:,:,i,j) * updateMap(:,:,j).';
            end
        end
    end

    % Enforce transpose consistency of off-diagonal blocks.
    for i = 1:Nw
        for j = i+1:Nw
            Cij = 0.5*(PcrossNew(:,:,i,j)+PcrossNew(:,:,j,i).');
            PcrossNew(:,:,i,j) = Cij;
            PcrossNew(:,:,j,i) = Cij.';
        end
    end
    Pcross = PcrossNew;

    %% Same transverse fusion as before
    fusionK = fuseTransverseBLUE( ...
        squeeze(aLocal(:,k+1,:)),squeeze(uPred(:,k+1,:)), ...
        PLocal(:,:,:,k+1),Pcross,idxA,options);

    localMeanAcceleration(:,k+1) = fusionK.localMean;
    equalGeometryAcceleration(:,k+1) = fusionK.equalGeometry;
    diagonalBLUEAcceleration(:,k+1) = fusionK.diagonalBLUE;
correlatedBLUEAcceleration(:,k+1) = fusionK.correlatedBLUE;
gatedDirectionalAcceleration(:,k+1) = fusionK.gatedDirectional;
directionalGateWeight(k+1,:) = fusionK.directionalGateWeight;
gatedDirectionalRank(k+1) = fusionK.gatedDirectionalRank;
gatedDirectionalCondition(k+1) = fusionK.gatedDirectionalCondition;

    equalGeometryRank(k+1) = fusionK.equalGeometryRank;
    equalGeometryCondition(k+1) = fusionK.equalGeometryCondition;
    equalGeometryScore(k+1) = fusionK.equalGeometryScore;
    diagonalBLUERank(k+1) = fusionK.diagonalBLUERank;
    diagonalBLUECondition(k+1) = fusionK.diagonalBLUECondition;
    correlatedBLUERank(k+1) = fusionK.correlatedBLUERank;
    correlatedBLUECondition(k+1) = fusionK.correlatedBLUECondition;
    correlatedProjectedCovarianceMinEigenvalue(k+1) = ...
        fusionK.correlatedProjectedCovarianceMinEigenvalue;

    %% Rolling finite-horizon acceleration information
    infoK = computeRollingAccelerationInformation( ...
        k+1,t,uPred,rangePred,activeMask,dim,options);
    rollingAccelerationInformation(:,:,k+1) = infoK.matrix;
    temporalInformationMinEigenvalue(k+1) = infoK.minEigenvalue;
    temporalInformationCondition(k+1) = infoK.condition;
    temporalDirectionScore(k+1) = infoK.directionScore;
end

% Final manager status and final command log.
[manager,triggerData] = updateManeuverManager( ...
    manager,t(N),rollingAccelerationInformation(:,:,N), ...
    temporalInformationMinEigenvalue(N),equalGeometryScore(N), ...
    uPred(:,N,:),options,dt);
temporalTriggerThreshold(N) = triggerData.temporalThreshold;
weakTemporalFlag(N) = triggerData.weakTemporal;
weakGeometryFlag(N) = triggerData.weakGeometry;
triggerConditionFlag(N) = triggerData.triggerCondition;
maneuverActive(N) = manager.active;
maneuverPhase(N) = currentPulsePhase(manager,t(N),options);
maneuverEventIndex(N) = manager.currentEventIndex;
for j = 1:Nw
    watcherCommand(:,N,j) = computeAdaptiveManeuverCommand( ...
        manager,t(N),uPred(:,N,j),j,options);
end
manager = closeOpenEventAtFinalTime(manager,t(N),options);

%% Error and maneuver diagnostics
positionMean = squeeze(mean(xLocal(idxR,:,:),3));
velocityMean = squeeze(mean(xLocal(idxV,:,:),3));

positionErrorNorm = vecnorm(positionMean-truePosition,2,1);
velocityErrorNorm = vecnorm(velocityMean-trueVelocity,2,1);
localMeanAccelerationErrorNorm = ...
    vecnorm(localMeanAcceleration-trueAcceleration,2,1);
equalGeometryAccelerationErrorNorm = ...
    vecnorm(equalGeometryAcceleration-trueAcceleration,2,1);
diagonalBLUEAccelerationErrorNorm = ...
    vecnorm(diagonalBLUEAcceleration-trueAcceleration,2,1);
correlatedBLUEAccelerationErrorNorm = ...
    vecnorm(correlatedBLUEAcceleration-trueAcceleration,2,1);
gatedDirectionalAccelerationErrorNorm = ...
    vecnorm(gatedDirectionalAcceleration-trueAcceleration,2,1);

deltaVByWatcher = zeros(Nw,1);
for j = 1:Nw
    commandMagnitude = vecnorm(squeeze(watcherCommand(:,:,j)),2,1);
    deltaVByWatcher(j) = trapz(t,commandMagnitude);
end
totalDeltaV = sum(deltaVByWatcher);

calibrationEndTime = options.initialBurnStartTime ...
    + 2*options.burnDuration ...
    + options.postBurnSettlingTime ...
    + options.referenceCalibrationDuration;
idxPostCalibration = t >= calibrationEndTime;
if ~any(idxPostCalibration)
    idxPostCalibration = true(size(t));
end

caseName = [ ...
    "local mean (diagnostic)"; ...
    "equal geometry"; ...
    "diagonal transverse BLUE"; ...
    "correlated transverse BLUE"; ...
    "gated directional reconstruction"];

accelerationRMSE = [ ...
    rmsFinite(localMeanAccelerationErrorNorm); ...
    rmsFinite(equalGeometryAccelerationErrorNorm); ...
    rmsFinite(diagonalBLUEAccelerationErrorNorm); ...
    rmsFinite(correlatedBLUEAccelerationErrorNorm); ...
    rmsFinite(gatedDirectionalAccelerationErrorNorm)];

postCalibrationAccelerationRMSE = [ ...
    rmsFinite(localMeanAccelerationErrorNorm(idxPostCalibration)); ...
    rmsFinite(equalGeometryAccelerationErrorNorm(idxPostCalibration)); ...
    rmsFinite(diagonalBLUEAccelerationErrorNorm(idxPostCalibration)); ...
    rmsFinite(correlatedBLUEAccelerationErrorNorm(idxPostCalibration)); ...
    rmsFinite(gatedDirectionalAccelerationErrorNorm(idxPostCalibration))];

fullRankRate = [ ...
    1.0; ...
    mean(equalGeometryRank == dim); ...
    mean(diagonalBLUERank == dim); ...
    mean(correlatedBLUERank == dim); ...
    mean(gatedDirectionalRank == dim)];

summary = table(caseName,accelerationRMSE, ...
    postCalibrationAccelerationRMSE,fullRankRate);

eventTable = makeEventTable(manager.events,dim);

%% Package result
result = struct();
result.time = t;
result.dimension = dim;
result.options = options;
result.truePosition = truePosition;
result.trueVelocity = trueVelocity;
result.trueAcceleration = trueAcceleration;
result.watcherInitialPosition = watcherPosition0;
result.watcherPosition = watcherPosition;
result.watcherVelocity = watcherVelocity;
result.watcherCommand = watcherCommand;
result.localState = xLocal;
result.localCovariance = PLocal;
result.finalCrossCovariance = Pcross;
result.localAcceleration = aLocal;
result.predictedLOS = uPred;
result.predictedRange = rangePred;
result.innovation = innovationLog;
result.NIS = nisLog;
result.localMeanAcceleration = localMeanAcceleration;
result.equalGeometryAcceleration = equalGeometryAcceleration;
result.diagonalBLUEAcceleration = diagonalBLUEAcceleration;
result.correlatedBLUEAcceleration = correlatedBLUEAcceleration;
result.gatedDirectionalAcceleration = gatedDirectionalAcceleration;
result.directionalGateWeight = directionalGateWeight;
result.gatedDirectionalRank = gatedDirectionalRank;
result.gatedDirectionalCondition = gatedDirectionalCondition;
result.equalGeometryRank = equalGeometryRank;
result.equalGeometryCondition = equalGeometryCondition;
result.equalGeometryScore = equalGeometryScore;
result.diagonalBLUERank = diagonalBLUERank;
result.diagonalBLUECondition = diagonalBLUECondition;
result.correlatedBLUERank = correlatedBLUERank;
result.correlatedBLUECondition = correlatedBLUECondition;
result.correlatedProjectedCovarianceMinEigenvalue = ...
    correlatedProjectedCovarianceMinEigenvalue;
result.rollingAccelerationInformation = rollingAccelerationInformation;
result.temporalInformationMinEigenvalue = ...
    temporalInformationMinEigenvalue;
result.temporalInformationCondition = temporalInformationCondition;
result.temporalDirectionScore = temporalDirectionScore;
result.temporalTriggerThreshold = temporalTriggerThreshold;
result.weakTemporalFlag = weakTemporalFlag;
result.weakGeometryFlag = weakGeometryFlag;
result.triggerConditionFlag = triggerConditionFlag;
result.maneuverActive = maneuverActive;
result.maneuverPhase = maneuverPhase;
result.maneuverEventIndex = maneuverEventIndex;
result.events = manager.events;
result.eventTable = eventTable;
result.referenceTemporalInformation = manager.referenceLambda;
result.deltaVByWatcher = deltaVByWatcher;
result.totalDeltaV = totalDeltaV;
result.positionErrorNorm = positionErrorNorm;
result.velocityErrorNorm = velocityErrorNorm;
result.localMeanAccelerationErrorNorm = ...
    localMeanAccelerationErrorNorm;
result.equalGeometryAccelerationErrorNorm = ...
    equalGeometryAccelerationErrorNorm;
result.diagonalBLUEAccelerationErrorNorm = ...
    diagonalBLUEAccelerationErrorNorm;
result.correlatedBLUEAccelerationErrorNorm = ...
    correlatedBLUEAccelerationErrorNorm;
result.gatedDirectionalAccelerationErrorNorm = ...
    gatedDirectionalAccelerationErrorNorm;
result.summary = summary;

fprintf('\nAdaptive observability-triggered four-watcher experiment\n');
fprintf('  dimension                         : %d\n',dim);
fprintf('  bearing sigma [deg]               : %.4f\n', ...
    options.sigmaBearingDeg);
fprintf('  temporal reference lambda_min     : %.6e\n', ...
    manager.referenceLambda);
fprintf('  number of maneuver events         : %d\n', ...
    numel(manager.events));
fprintf('  adaptive re-maneuver events       : %d\n', ...
    manager.adaptiveEventCount);
fprintf('  total maneuver Delta-V            : %.6f\n',totalDeltaV);
disp(summary);
if ~isempty(eventTable)
    disp(eventTable);
end

if makePlots
    result.figure = plotAdaptiveResult(result);
end
end

%% ========================================================================
function options = resolveOptions(options,dim,simulationTime,dt)
%RESOLVEOPTIONS Fill and validate simulation, trigger, and fusion options.

if dim == 2
    defaultAcceleration = [0.15;0.20];
    defaultAmplitude = [0.06;0.05];
    defaultFrequency = [0.035;0.050];
    defaultPhase = [0;pi/2];
    defaultVelocity = [1.0;-0.5];
else
    defaultAcceleration = [0.15;0.20;-0.10];
    defaultAmplitude = [0.06;0.05;0.04];
    defaultFrequency = [0.035;0.050;0.042];
    defaultPhase = [0;pi/2;0.4];
    defaultVelocity = [1.0;-0.5;0.3];
end

defaults = struct( ...
    'truthAccelerationMode','slow_sinusoid', ...
    'constantAcceleration',defaultAcceleration, ...
    'accelerationAmplitude',defaultAmplitude, ...
    'accelerationFrequency',defaultFrequency, ...
    'accelerationPhase',defaultPhase, ...
    'targetInitialPosition',zeros(dim,1), ...
    'targetInitialVelocity',defaultVelocity, ...
    'watcherRadius',1000, ...
    'initialRangeScale',[0.55,0.80,1.25,1.55], ...
    'sigmaBearingDeg',0.2, ...
    'sigmaAzimuthDeg',0.2, ...
    'sigmaElevationDeg',0.2, ...
    'sigmaJerk',0.005, ...
    'initialPositionStd',150, ...
    'initialVelocityStd',2.0, ...
    'initialAccelerationStd',0.20, ...
    'initialTransversePositionStd',15, ...
    'initialVelocityMeanErrorStd',0.4, ...
    'initialAccelerationMeanErrorStd',0.08, ...
    'initialCrossCovarianceMode','independent', ...
    'commonProcessNoiseFraction',1.0, ...
    'maneuverWatcherMask',[true,true,true,true], ...
    'burnAcceleration',2.0, ...
    'burnDuration',5.0, ...
    'initialBurnStartTime',10.0, ...
    'postBurnSettlingTime',10.0, ...
    'observabilityWindowTime',15.0, ...
    'referenceCalibrationDuration',15.0, ...
    'temporalTriggerFraction',0.55, ...
    'temporalAbsoluteFloor',0.0, ...
    'geometryTriggerScore',0.35, ...
    'triggerDwellTime',2.0, ...
    'maneuverCooldown',20.0, ...
    'maximumAdaptiveBurns',3, ...
    'minimumRange',1.0, ...
    'matrixRankTolerance',1e-9, ...
    'psdTolerance',1e-9);

names = fieldnames(defaults);
for q = 1:numel(names)
    name = names{q};
    if ~isfield(options,name) || isempty(options.(name))
        options.(name) = defaults.(name);
    end
end

options.truthAccelerationMode = validatestring( ...
    options.truthAccelerationMode,{'constant','slow_sinusoid'});
options.initialCrossCovarianceMode = validatestring( ...
    options.initialCrossCovarianceMode,{'independent','common_prior'});

vectorFields = { ...
    'constantAcceleration','accelerationAmplitude', ...
    'accelerationFrequency','accelerationPhase', ...
    'targetInitialPosition','targetInitialVelocity'};
for q = 1:numel(vectorFields)
    name = vectorFields{q};
    value = options.(name);
    validateattributes(value,{'numeric'},{'vector','numel',dim,'finite'});
    options.(name) = value(:);
end

validateattributes(options.initialRangeScale,{'numeric'}, ...
    {'vector','numel',4,'positive','finite'});
options.initialRangeScale = reshape(options.initialRangeScale,1,[]);

validateattributes(options.maneuverWatcherMask,{'logical','numeric'}, ...
    {'vector','numel',4});
options.maneuverWatcherMask = logical( ...
    reshape(options.maneuverWatcherMask,1,[]));

positiveScalarFields = { ...
    'watcherRadius','sigmaBearingDeg','sigmaAzimuthDeg', ...
    'sigmaElevationDeg','initialPositionStd','initialVelocityStd', ...
    'initialAccelerationStd','initialTransversePositionStd', ...
    'initialVelocityMeanErrorStd','initialAccelerationMeanErrorStd', ...
    'burnDuration','observabilityWindowTime', ...
    'referenceCalibrationDuration','triggerDwellTime', ...
    'maneuverCooldown','minimumRange'};
for q = 1:numel(positiveScalarFields)
    name = positiveScalarFields{q};
    validateattributes(options.(name),{'numeric'}, ...
        {'scalar','positive','finite'});
end

nonnegativeScalarFields = { ...
    'sigmaJerk','burnAcceleration','initialBurnStartTime', ...
    'postBurnSettlingTime','temporalAbsoluteFloor', ...
    'matrixRankTolerance','psdTolerance'};
for q = 1:numel(nonnegativeScalarFields)
    name = nonnegativeScalarFields{q};
    validateattributes(options.(name),{'numeric'}, ...
        {'scalar','nonnegative','finite'});
end

validateattributes(options.commonProcessNoiseFraction,{'numeric'}, ...
    {'scalar','>=',0,'<=',1,'finite'});
validateattributes(options.temporalTriggerFraction,{'numeric'}, ...
    {'scalar','positive','<=',1,'finite'});
validateattributes(options.geometryTriggerScore,{'numeric'}, ...
    {'scalar','>=',0,'<=',1,'finite'});
validateattributes(options.maximumAdaptiveBurns,{'numeric'}, ...
    {'scalar','integer','nonnegative'});

if options.initialBurnStartTime+2*options.burnDuration > simulationTime
    warning(['The initial pulse pair extends beyond simulationTime. ' ...
        'The event will be truncated.']);
end
if options.observabilityWindowTime < 2*dt
    error('observabilityWindowTime must contain at least two samples.');
end
end

%% ========================================================================
function a = makeTruthAcceleration(t,dim,options)
%MAKETRUTHACCELERATION Construct constant or smooth sinusoidal truth.

N = numel(t);
a = zeros(dim,N);

switch options.truthAccelerationMode
    case 'constant'
        a = repmat(options.constantAcceleration,1,N);

    case 'slow_sinusoid'
        for k = 1:N
            a(:,k) = options.constantAcceleration ...
                + options.accelerationAmplitude.*sin( ...
                options.accelerationFrequency*t(k) ...
                + options.accelerationPhase);
        end
end
end

%% ========================================================================
function p = makeInitialWatcherGeometry(dim,options,Nw)
%MAKEINITIALWATCHERGEOMETRY Place four watchers around the target region.

if Nw ~= 4
    error('This demo is configured for exactly four watchers.');
end

if dim == 2
    direction = [ ...
         1, -1,  0,  0; ...
         0,  0,  1, -1];
else
    direction = [ ...
         1,  1, -1, -1; ...
         1, -1,  1, -1; ...
         1, -1, -1,  1]/sqrt(3);
end

p = zeros(dim,Nw);
for j = 1:Nw
    p(:,j) = options.targetInitialPosition ...
        + options.watcherRadius*options.initialRangeScale(j) ...
        * direction(:,j);
end
end

%% ========================================================================
function [innovation,H,R] = makeAngleMeasurementUpdate( ...
    rhoTrue,rhoHat,dim,options)
%MAKEANGLEMEASUREMENTUPDATE Generate one noisy angle-only measurement.

n = 3*dim;
rr = max(norm(rhoHat),options.minimumRange);
u = rhoHat/rr;

if dim == 2
    sigma = deg2rad(options.sigmaBearingDeg);
    thetaTrue = atan2(rhoTrue(2),rhoTrue(1));
    thetaHat = atan2(rhoHat(2),rhoHat(1));
    z = wrapAngle(thetaTrue+sigma*randn);
    innovation = wrapAngle(z-thetaHat);
    Hposition = [-u(2),u(1)]/rr;
    H = [Hposition,zeros(1,n-dim)];
    R = sigma^2;
else
    sigmaAz = deg2rad(options.sigmaAzimuthDeg);
    sigmaEl = deg2rad(options.sigmaElevationDeg);

    [azTrue,elTrue] = cartesianToAzEl(rhoTrue);
    [azHat,elHat] = cartesianToAzEl(rhoHat);

    zAz = wrapAngle(azTrue+sigmaAz*randn);
    zEl = clampElevation(elTrue+sigmaEl*randn);
    innovation = [wrapAngle(zAz-azHat);zEl-elHat];

    Hposition = anglePositionJacobian(rhoHat,3,options.minimumRange);
    H = [Hposition,zeros(2,n-dim)];
    R = diag([sigmaAz^2,sigmaEl^2]);
end
end

%% ========================================================================
function info = computeRollingAccelerationInformation( ...
    k,t,uPred,rangePred,activeMask,dim,options)
%COMPUTEROLLINGACCELERATIONINFORMATION Rolling constant-acceleration score.
%
% The score asks how strongly the angle sequence over the recent window
% responds to a constant acceleration perturbation at the window start.

windowSteps = max(2,round(options.observabilityWindowTime/(t(2)-t(1))));
startIndex = max(1,k-windowSteps+1);
J = zeros(dim);

if dim == 2
    Rinv = 1/deg2rad(options.sigmaBearingDeg)^2;
else
    Rinv = diag(1./[ ...
        deg2rad(options.sigmaAzimuthDeg)^2, ...
        deg2rad(options.sigmaElevationDeg)^2]);
end

for ell = startIndex:k
    tau = t(ell)-t(startIndex);
    if tau <= 0
        continue;
    end
    GammaA = 0.5*tau^2*eye(dim);

    for j = 1:size(uPred,3)
        if ~activeMask(ell,j)
            continue;
        end
        u = normalizeVector(uPred(:,ell,j));
        rho = max(rangePred(ell,j),options.minimumRange);

        if dim == 2
            Hposition = [-u(2),u(1)]/rho;
        else
            Hposition = anglePositionJacobian( ...
                rho*u,3,options.minimumRange);
        end

        J = J+GammaA.'*Hposition.'*Rinv*Hposition*GammaA;
    end
end

J = 0.5*(J+J.');
[eigenvectors,eigenvalues] = eig(J,'vector');
[eigenvalues,order] = sort(real(eigenvalues),'ascend');
eigenvectors = real(eigenvectors(:,order));
minEigenvalue = max(0,eigenvalues(1));
traceJ = max(trace(J),eps);
directionScore = min(1,max(0,dim*minEigenvalue/traceJ));

info = struct();
info.matrix = J;
info.minEigenvalue = minEigenvalue;
info.condition = safeConditionNumberPSD(J,options.matrixRankTolerance);
info.directionScore = directionScore;
info.weakDirection = eigenvectors(:,1);
end

%% ========================================================================
function fusion = fuseTransverseBLUE( ...
    aLocal,uPred,PLocal,Pcross,idxA,options)
%FUSETRANSVERSEBLUE Equal geometry, diagonal BLUE, and correlated BLUE.

[dim,Nw] = size(aLocal);
partialDim = dim-1;
stackedDim = Nw*partialDim;

C = zeros(stackedDim,dim);
z = zeros(stackedDim,1);
Rdiagonal = zeros(stackedDim);
Rcorrelated = zeros(stackedDim);
Omega = zeros(dim);
equalInformationVector = zeros(dim,1);
Ncell = cell(Nw,1);

for i = 1:Nw
    rowsI = (i-1)*partialDim+(1:partialDim);
    ui = normalizeVector(uPred(:,i));
    Ni = transverseBasis(ui,i);
    Ncell{i} = Ni;

    C(rowsI,:) = Ni.';
    z(rowsI) = Ni.'*aLocal(:,i);
    Omega = Omega+Ni*Ni.';
    equalInformationVector = equalInformationVector ...
        + Ni*Ni.'*aLocal(:,i);

    PaaI = PLocal(idxA,idxA,i);
    Rdiagonal(rowsI,rowsI) = Ni.'*PaaI*Ni;
end

for i = 1:Nw
    rowsI = (i-1)*partialDim+(1:partialDim);
    for j = 1:Nw
        rowsJ = (j-1)*partialDim+(1:partialDim);
        PaaIJ = Pcross(idxA,idxA,i,j);
        Rcorrelated(rowsI,rowsJ) = ...
            Ncell{i}.'*PaaIJ*Ncell{j};
    end
end

Omega = 0.5*(Omega+Omega.');
Rdiagonal = 0.5*(Rdiagonal+Rdiagonal.');
Rcorrelated = 0.5*(Rcorrelated+Rcorrelated.');

fusion = struct();
fusion.localMean = mean(aLocal,2);
fusion.equalGeometryRank = numericalRankPSD( ...
    Omega,options.matrixRankTolerance);
fusion.equalGeometryCondition = safeConditionNumberPSD( ...
    Omega,options.matrixRankTolerance);
omegaEigen = sort(real(eig(Omega)),'ascend');
fusion.equalGeometryScore = min(1,max(0, ...
    dim*max(0,omegaEigen(1))/max(trace(Omega),eps)));

if fusion.equalGeometryRank == dim
    fusion.equalGeometry = pinvPSD( ...
        Omega,options.matrixRankTolerance)*equalInformationVector;
else
    fusion.equalGeometry = nan(dim,1);
end

[diagonalEstimate,diagonalRank,diagonalCondition,~] = ...
    solveBLUE(C,z,Rdiagonal,dim,options);
fusion.diagonalBLUE = diagonalEstimate;
fusion.diagonalBLUERank = diagonalRank;
fusion.diagonalBLUECondition = diagonalCondition;

[correlatedEstimate,correlatedRank,correlatedCondition,minEigR] = ...
    solveBLUE(C,z,Rcorrelated,dim,options);
fusion.correlatedBLUE = correlatedEstimate;
fusion.correlatedBLUERank = correlatedRank;
fusion.correlatedBLUECondition = correlatedCondition;
fusion.correlatedProjectedCovarianceMinEigenvalue = minEigR;
end

%% ========================================================================
function [estimate,informationRank,informationCondition,minEigR] = ...
    solveBLUE(C,z,R,dim,options)
%SOLVEBLUE Solve generalized least squares without heuristic fallback.

R = 0.5*(R+R.');
eigR = real(eig(R));
minEigR = min(eigR);
scaleR = max(1,max(abs(eigR)));

estimate = nan(dim,1);
informationRank = 0;
informationCondition = Inf;

if minEigR < -options.psdTolerance*scaleR
    return;
end

Rdagger = pinvPSD(R,options.matrixRankTolerance);
information = C.'*Rdagger*C;
information = 0.5*(information+information.');
informationRank = numericalRankPSD( ...
    information,options.matrixRankTolerance);
informationCondition = safeConditionNumberPSD( ...
    information,options.matrixRankTolerance);

if informationRank == dim
    estimate = pinvPSD(information,options.matrixRankTolerance) ...
        *(C.'*Rdagger*z);
end
end

%% ========================================================================
function manager = initializeManeuverManager(dim,options)
%INITIALIZEMANEUVERMANAGER Initialize event-trigger state.

manager = struct();
manager.active = false;
manager.initialEventStarted = false;
manager.initialEventCompleted = false;
manager.currentEventStart = nan;
manager.currentEventIndex = 0;
manager.currentWeakDirection = zeros(dim,1);
manager.currentEventType = '';
manager.lastEventEnd = -inf;
manager.adaptiveEventCount = 0;
manager.dwellCounter = 0;
manager.referenceSamples = zeros(0,1);
manager.referenceLambda = nan;
manager.referenceFrozen = false;
manager.events = struct( ...
    'index',{},'type',{},'startTime',{},'endTime',{}, ...
    'weakDirection',{},'triggerTemporalLambda',{}, ...
    'triggerTemporalThreshold',{},'triggerGeometryScore',{});
manager.calibrationStart = options.initialBurnStartTime ...
    + 2*options.burnDuration+options.postBurnSettlingTime;
manager.calibrationEnd = manager.calibrationStart ...
    + options.referenceCalibrationDuration;
end

%% ========================================================================
function [manager,data] = updateManeuverManager( ...
    manager,tk,J,lambdaMin,geometryScore,uPred,options,dt)
%UPDATEMANEUVERMANAGER Start, finish, calibrate, and retrigger pulse pairs.

% Finish an active event before considering a new event.
if manager.active && tk >= manager.currentEventStart+2*options.burnDuration
    manager.active = false;
    manager.lastEventEnd = manager.currentEventStart+2*options.burnDuration;
    manager.events(manager.currentEventIndex).endTime = manager.lastEventEnd;
    if strcmp(manager.currentEventType,'initial')
        manager.initialEventCompleted = true;
    end
    manager.currentEventIndex = 0;
    manager.currentEventType = '';
end

% Initial pulse pair.
if ~manager.initialEventStarted && tk >= options.initialBurnStartTime
    weakDirection = weakestDirection(J,uPred,options);
    manager = startEvent(manager,'initial',tk,weakDirection, ...
        lambdaMin,nan,geometryScore);
    manager.initialEventStarted = true;
end

% Healthy-reference calibration after the initial pulse pair and settling.
if manager.initialEventCompleted && ~manager.referenceFrozen
    if tk >= manager.calibrationStart && tk <= manager.calibrationEnd ...
            && isfinite(lambdaMin) && lambdaMin > 0
        manager.referenceSamples(end+1,1) = lambdaMin; %#ok<AGROW>
    end

    if tk > manager.calibrationEnd
        if isempty(manager.referenceSamples)
            manager.referenceLambda = max(lambdaMin,eps);
        else
            manager.referenceLambda = median(manager.referenceSamples);
        end
        manager.referenceFrozen = true;
    end
end

if manager.referenceFrozen
    temporalThreshold = max(options.temporalAbsoluteFloor, ...
        options.temporalTriggerFraction*manager.referenceLambda);
else
    temporalThreshold = nan;
end

weakTemporal = manager.referenceFrozen ...
    && isfinite(lambdaMin) && lambdaMin < temporalThreshold;
weakGeometry = manager.referenceFrozen ...
    && geometryScore < options.geometryTriggerScore;
triggerCondition = weakTemporal || weakGeometry;

eligible = manager.referenceFrozen && ~manager.active ...
    && manager.adaptiveEventCount < options.maximumAdaptiveBurns ...
    && tk >= manager.lastEventEnd+options.maneuverCooldown;

if eligible && triggerCondition
    manager.dwellCounter = manager.dwellCounter+1;
else
    manager.dwellCounter = 0;
end

requiredDwellSteps = max(1,ceil(options.triggerDwellTime/dt));
if eligible && manager.dwellCounter >= requiredDwellSteps
    weakDirection = weakestDirection(J,uPred,options);
    manager = startEvent(manager,'adaptive',tk,weakDirection, ...
        lambdaMin,temporalThreshold,geometryScore);
    manager.adaptiveEventCount = manager.adaptiveEventCount+1;
    manager.dwellCounter = 0;
end

data = struct( ...
    'temporalThreshold',temporalThreshold, ...
    'weakTemporal',weakTemporal, ...
    'weakGeometry',weakGeometry, ...
    'triggerCondition',triggerCondition);
end

%% ========================================================================
function manager = startEvent(manager,type,tk,weakDirection, ...
    lambdaMin,temporalThreshold,geometryScore)
%STARTEVENT Register and activate one pulse-pair event.

manager.active = true;
manager.currentEventStart = tk;
manager.currentWeakDirection = normalizeVector(weakDirection);
manager.currentEventType = type;
manager.currentEventIndex = numel(manager.events)+1;

manager.events(manager.currentEventIndex).index = ...
    manager.currentEventIndex;
manager.events(manager.currentEventIndex).type = type;
manager.events(manager.currentEventIndex).startTime = tk;
manager.events(manager.currentEventIndex).endTime = nan;
manager.events(manager.currentEventIndex).weakDirection = ...
    manager.currentWeakDirection;
manager.events(manager.currentEventIndex).triggerTemporalLambda = lambdaMin;
manager.events(manager.currentEventIndex).triggerTemporalThreshold = ...
    temporalThreshold;
manager.events(manager.currentEventIndex).triggerGeometryScore = ...
    geometryScore;
end

%% ========================================================================
function direction = weakestDirection(J,uPred,options)
%WEAKESTDIRECTION Select the least-informed direction of J or geometry.

J = 0.5*(J+J.');
if trace(J) > options.matrixRankTolerance
    [V,D] = eig(J,'vector');
    [~,index] = min(real(D));
    direction = real(V(:,index));
else
    dim = size(J,1);
    Omega = zeros(dim);
    for j = 1:size(uPred,3)
        u = normalizeVector(uPred(:,1,j));
        Omega = Omega+eye(dim)-u*u.';
    end
    Omega = 0.5*(Omega+Omega.');
    [V,D] = eig(Omega,'vector');
    [~,index] = min(real(D));
    direction = real(V(:,index));
end

if norm(direction) < 1e-12
    direction = zeros(size(J,1),1);
    direction(1) = 1;
end
end

%% ========================================================================
function aW = computeAdaptiveManeuverCommand( ...
    manager,tk,uPred,watcherIndex,options)
%COMPUTEADAPTIVEMANEUVERCOMMAND LOS-transverse pulse-pair command.

u = normalizeVector(uPred);
aW = zeros(size(u));

if ~manager.active || ~options.maneuverWatcherMask(watcherIndex)
    return;
end

phase = currentPulsePhase(manager,tk,options);
if phase == 0
    return;
end

Pperp = eye(numel(u))-u*u.';
direction = Pperp*manager.currentWeakDirection;

if norm(direction) < 1e-10
    basis = transverseBasis(u,watcherIndex);
    direction = basis(:,1);
else
    direction = normalizeVector(direction);
end

% Opposite signs spread the watchers rather than translating the whole
% network in one common direction. The sign does not change information
% magnitude, but it improves geometric diversity.
signPattern = [1,-1,-1,1];
direction = signPattern(watcherIndex)*direction;
aW = options.burnAcceleration*phase*direction;

if abs(u.'*aW) > 1e-9*max(1,norm(aW))
    error('The generated watcher command is not LOS-transverse.');
end
end

%% ========================================================================
function phase = currentPulsePhase(manager,tk,options)
%CURRENTPULSEPHASE +1 first burn, -1 second burn, 0 outside event.

phase = 0;
if ~manager.active
    return;
end
elapsed = tk-manager.currentEventStart;
if elapsed >= 0 && elapsed < options.burnDuration
    phase = 1;
elseif elapsed >= options.burnDuration ...
        && elapsed < 2*options.burnDuration
    phase = -1;
end
end

%% ========================================================================
function manager = closeOpenEventAtFinalTime(manager,finalTime,options)
%CLOSEOPENEVENTATFINALTIME Close a truncated event for reporting.

if manager.active && manager.currentEventIndex > 0
    manager.events(manager.currentEventIndex).endTime = min( ...
        finalTime,manager.currentEventStart+2*options.burnDuration);
end
end

%% ========================================================================
function eventTable = makeEventTable(events,dim)
%MAKEEVENTTABLE Convert event structure to a readable MATLAB table.

if isempty(events)
    eventTable = table();
    return;
end

M = numel(events);
eventNumber = zeros(M,1);
eventType = strings(M,1);
startTime = zeros(M,1);
endTime = nan(M,1);
triggerTemporalLambda = nan(M,1);
triggerTemporalThreshold = nan(M,1);
triggerGeometryScore = nan(M,1);
direction = nan(M,dim);

for q = 1:M
    eventNumber(q) = events(q).index;
    eventType(q) = string(events(q).type);
    startTime(q) = events(q).startTime;
    endTime(q) = events(q).endTime;
    triggerTemporalLambda(q) = events(q).triggerTemporalLambda;
    triggerTemporalThreshold(q) = ...
        events(q).triggerTemporalThreshold;
    triggerGeometryScore(q) = events(q).triggerGeometryScore;
    direction(q,:) = events(q).weakDirection(:).';
end

if dim == 2
    weakDirection1 = direction(:,1);
    weakDirection2 = direction(:,2);
    eventTable = table(eventNumber,eventType,startTime,endTime, ...
        triggerTemporalLambda,triggerTemporalThreshold, ...
        triggerGeometryScore,weakDirection1,weakDirection2);
else
    weakDirection1 = direction(:,1);
    weakDirection2 = direction(:,2);
    weakDirection3 = direction(:,3);
    eventTable = table(eventNumber,eventType,startTime,endTime, ...
        triggerTemporalLambda,triggerTemporalThreshold, ...
        triggerGeometryScore,weakDirection1,weakDirection2, ...
        weakDirection3);
end
end

%% ========================================================================
function N = transverseBasis(u,index)
%TRANSVERSEBASIS Orthonormal basis perpendicular to unit vector u.

u = normalizeVector(u);
dim = numel(u);

if dim == 2
    N = [-u(2);u(1)];
    return;
end

Pperp = eye(3)-u*u.';
candidateAxes = [eye(3),normalizeVector([1;1;1])];
preferred = 1+mod(index-1,size(candidateAxes,2));
order = [preferred,setdiff(1:size(candidateAxes,2),preferred)];

v1 = zeros(3,1);
for q = order
    candidate = Pperp*candidateAxes(:,q);
    if norm(candidate) > 1e-10
        v1 = normalizeVector(candidate);
        break;
    end
end
if norm(v1) < 1e-10
    error('Unable to construct a transverse basis.');
end
v2 = normalizeVector(cross(u,v1));
N = [v1,v2];
end

%% ========================================================================
function H = anglePositionJacobian(r,dim,minimumRange)
%ANGLEPOSITIONJACOBIAN Bearing or azimuth/elevation position Jacobian.

if dim == 2
    rr = max(norm(r),minimumRange);
    u = r/rr;
    H = [-u(2),u(1)]/rr;
    return;
end

x = r(1);
y = r(2);
z = r(3);
rhoXY = max(hypot(x,y),minimumRange);
rho2 = max(dot(r,r),minimumRange^2);

H = [ ...
    -y/(rhoXY^2), x/(rhoXY^2), 0; ...
    -(x*z)/(rho2*rhoXY), -(y*z)/(rho2*rhoXY), rhoXY/rho2];
end

%% ========================================================================
function [az,el] = cartesianToAzEl(r)
%CARTESIANTOAZEL Convert Cartesian relative position to azimuth/elevation.

az = atan2(r(2),r(1));
el = atan2(r(3),max(hypot(r(1),r(2)),1e-12));
end

%% ========================================================================
function x = wrapAngle(x)
%WRAPANGLE Wrap angle to [-pi,pi].
x = atan2(sin(x),cos(x));
end

%% ========================================================================
function el = clampElevation(el)
%CLAMPELEVATION Keep elevation away from chart singularities.
margin = 1e-8;
el = min(max(el,-pi/2+margin),pi/2-margin);
end

%% ========================================================================
function u = normalizeVector(v)
%NORMALIZEVECTOR Safely normalize a nonzero vector.
nv = norm(v);
if nv < realmin
    error('Cannot normalize a zero vector.');
end
u = v/nv;
end

%% ========================================================================
function rankA = numericalRankPSD(A,tolerance)
%NUMERICALRANKPSD Numerical rank of a symmetric PSD matrix.
A = 0.5*(A+A.');
e = real(eig(A));
scale = max(1,max(abs(e)));
rankA = sum(e > tolerance*scale);
end

%% ========================================================================
function Ainv = pinvPSD(A,tolerance)
%PINVPSD Symmetric pseudoinverse using nonnegative eigenvalues only.
A = 0.5*(A+A.');
[V,D] = eig(A,'vector');
D = real(D);
scale = max(1,max(abs(D)));
keep = D > tolerance*scale;
Ainv = zeros(size(A));
if any(keep)
    Ainv = V(:,keep)*diag(1./D(keep))*V(:,keep).';
end
Ainv = 0.5*(real(Ainv)+real(Ainv).');
end

%% ========================================================================
function c = safeConditionNumberPSD(A,tolerance)
%SAFECONDITIONNUMBERPSD Condition number on the positive eigenspace.
A = 0.5*(A+A.');
e = sort(real(eig(A)),'descend');
scale = max(1,max(abs(e)));
positive = e(e > tolerance*scale);
if numel(positive) < size(A,1)
    c = Inf;
else
    c = positive(1)/positive(end);
end
end

%% ========================================================================
function value = rmsFinite(x)
%RMSFINITE Root-mean-square over finite entries only.
x = x(isfinite(x));
if isempty(x)
    value = nan;
else
    value = sqrt(mean(x.^2));
end
end

%% ========================================================================
function fig = plotAdaptiveResult(r)
%PLOTADAPTIVERESULT Plot fusion accuracy, trigger metrics, and maneuvers.

t = r.time;
dim = r.dimension;
Nw = size(r.watcherPosition,3);
fig = gobjects(3,1);

%% Figure 1: acceleration fusion
fig(1) = figure('Name','Adaptive maneuver BLUE acceleration fusion', ...
    'Color','w');
tiledlayout(dim+2,1,'TileSpacing','compact','Padding','compact');

for q = 1:dim
    nexttile;
    plot(t,r.trueAcceleration(q,:),'k','LineWidth',1.4);
    hold on;
    plot(t,r.equalGeometryAcceleration(q,:),'--','LineWidth',1.0);
    plot(t,r.diagonalBLUEAcceleration(q,:),':','LineWidth',1.2);
    plot(t,r.correlatedBLUEAcceleration(q,:),'LineWidth',1.1);
    grid on;
    ylabel(sprintf('a_%d',q));
    if q == 1
        legend('truth','equal geometry','diagonal BLUE', ...
            'correlated BLUE','Location','best');
    end
end

nexttile;
plot(t,r.equalGeometryAccelerationErrorNorm,'--','LineWidth',1.0);
hold on;
plot(t,r.diagonalBLUEAccelerationErrorNorm,':','LineWidth',1.2);
plot(t,r.correlatedBLUEAccelerationErrorNorm,'LineWidth',1.1);
grid on;
ylabel('accel. error norm');
legend('equal geometry','diagonal BLUE','correlated BLUE', ...
    'Location','best');

nexttile;
plot(t,r.correlatedBLUERank,'LineWidth',1.0);
hold on;
yline(dim,'--');
grid on;
xlabel('time [s]');
ylabel('BLUE rank');

%% Figure 2: online observability trigger
fig(2) = figure('Name','Online observability trigger diagnostics', ...
    'Color','w');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
semilogy(t,max(r.temporalInformationMinEigenvalue,realmin), ...
    'LineWidth',1.1);
hold on;
semilogy(t,max(r.temporalTriggerThreshold,realmin),'--','LineWidth',1.1);
grid on;
ylabel('\lambda_{min}(J_a)');
legend('rolling information','trigger threshold','Location','best');

nexttile;
plot(t,r.temporalDirectionScore,'LineWidth',1.0);
hold on;
plot(t,r.equalGeometryScore,'LineWidth',1.0);
yline(r.options.geometryTriggerScore,'--');
grid on;
ylabel('normalized score');
legend('temporal direction','fusion geometry', ...
    'geometry threshold','Location','best');

nexttile;
stairs(t,double(r.triggerConditionFlag),'--','LineWidth',1.0);
hold on;
stairs(t,double(r.maneuverActive),'LineWidth',1.2);
grid on;
xlabel('time [s]');
ylabel('logical');
legend('weak-observability condition','maneuver active', ...
    'Location','best');

%% Figure 3: watcher motion and commands
fig(3) = figure('Name','Watcher trajectories and adaptive burns', ...
    'Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile;
if dim == 2
    plot(r.truePosition(1,:),r.truePosition(2,:),'k','LineWidth',1.4);
    hold on;
    for j = 1:Nw
        p = squeeze(r.watcherPosition(:,:,j));
        plot(p(1,:),p(2,:),'LineWidth',1.0);
        plot(p(1,1),p(2,1),'o');
    end
    xlabel('x [m]');
    ylabel('y [m]');
    axis equal;
else
    plot3(r.truePosition(1,:),r.truePosition(2,:), ...
        r.truePosition(3,:),'k','LineWidth',1.4);
    hold on;
    for j = 1:Nw
        p = squeeze(r.watcherPosition(:,:,j));
        plot3(p(1,:),p(2,:),p(3,:),'LineWidth',1.0);
        plot3(p(1,1),p(2,1),p(3,1),'o');
    end
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('z [m]');
    axis equal;
    view(3);
end
grid on;
title('target and watcher trajectories');

nexttile;
hold on;
for j = 1:Nw
    command = squeeze(r.watcherCommand(:,:,j));
    plot(t,vecnorm(command,2,1),'LineWidth',1.0);
end
grid on;
xlabel('time [s]');
ylabel('||a_{w,i}|| [m/s^2]');
legend(arrayfun(@(j)sprintf('watcher %d',j),1:Nw, ...
    'UniformOutput',false),'Location','best');
end
