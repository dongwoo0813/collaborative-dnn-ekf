function out = run_toy_distributed_additive_dnn_ekf(seed,T,makePlots,scenario,dt,maneuverEnabled,maneuverObjective,replayNominalManeuver,nWatchers,wlsWeighting,architecture,communicationMode,adaptParameters,hiddenLayerCount,tuning)
%RUN_TOY_DISTRIBUTED_ADDITIVE_DNN_EKF Controlled additive-DNN EKF toy.
% One physical target has unknown acceleration d_true(eta).  It is
% represented by scalar directional DNN branches s_i(zeta,t_i;theta_i),
% where zeta=[r;v] is each watcher's local estimate of the same target.  Watcher
% i updates only theta_i from its own bearing innovation; it broadcasts
% theta_i and P_theta_i so every watcher can evaluate the full sum.
%
% Every branch estimates its LOS-transverse directional acceleration. The
% common 2-D acceleration is reconstructed by weighted least squares.
% The additive_vector option uses a frozen multi-hidden-layer backbone and
% an EKF-adapted output head.  additive_full_dnn instead augments the EKF
% with every weight and bias of the small branch DNN.  At communication,
% the entire branch parameter posterior (not an acceleration sample) is
% uploaded and used by every watcher in its propagation model.  The
% optional tuning struct permits a controlled ablation of the full-DNN
% parameter process covariance and communication trigger, e.g.
% tuning = struct('parameterProcessScale',.1,'mahalanobisThreshold',8, ...
%                 'minCommunicationInterval',10).

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(makePlots), makePlots = true; end
    if nargin < 4 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 5 || isempty(dt), dt = .1; end
    if nargin < 6 || isempty(maneuverEnabled), maneuverEnabled = true; end
    close all; rng(seed); addpath(genpath(pwd));
    if nargin < 9 || isempty(nWatchers), nWatchers = 4; end
    validateattributes(nWatchers,{'numeric'},{'scalar','integer','>=',1,'<=',8});
    if nargin < 11 || isempty(architecture), architecture = "directional_wls"; end
    architecture = string(architecture);
    validArchitecture = ["directional_wls" "additive_vector" "additive_full_dnn" "partitioned_full_dnn"];
    assert(any(architecture == validArchitecture), ...
        'Unknown architecture. Use directional_wls, additive_vector, additive_full_dnn, or partitioned_full_dnn.');
    if nargin < 14 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    validateattributes(hiddenLayerCount,{'numeric'}, ...
        {'scalar','integer','>=',3,'<=',4});
    cfg = additiveToyConfig(seed,T,scenario,dt,nWatchers,architecture,hiddenLayerCount);
    if nargin < 12 || isempty(communicationMode), communicationMode = "instantaneous"; end
    communicationMode = string(communicationMode);
    validCommunication = ["instantaneous" "event_triggered" "never"];
    assert(any(communicationMode == validCommunication), ...
        'Unknown communication mode. Use instantaneous, event_triggered, or never.');
    cfg.communication.mode = communicationMode;
    if nargin < 13 || isempty(adaptParameters), adaptParameters = true; end
    cfg.training.adaptParameters = logical(adaptParameters);
    if nargin >= 10 && ~isempty(wlsWeighting)
        validWeighting = ["parameter_covariance" "robust_consistency"];
        wlsWeighting = string(wlsWeighting);
        assert(any(wlsWeighting == validWeighting), ...
            'Unknown WLS weighting. Use parameter_covariance or robust_consistency.');
        cfg.residual.weighting = wlsWeighting;
    end
    cfg.maneuver.enabled = logical(maneuverEnabled);
    if nargin >= 7 && ~isempty(maneuverObjective)
        validObjectives = ["local_los_profile" "local_information" "local_position_information" "local_hybrid_position" "local_position" "local_radial" "directional_rank"];
        maneuverObjective = string(maneuverObjective);
        assert(any(maneuverObjective == validObjectives), ...
            'Unknown maneuver objective. Use local_los_profile, local_information, local_position_information, local_hybrid_position, local_position, local_radial, or directional_rank.');
        cfg.maneuver.objective = maneuverObjective;
    end
    if nargin >= 8 && ~isempty(replayNominalManeuver)
        cfg.comparison.replayNominalManeuver = logical(replayNominalManeuver);
    end
    if nargin < 15 || isempty(tuning), tuning = struct; end
    cfg = applyToyTuning(cfg,tuning);
    fprintf('Distributed DNN EKF toy: seed=%d, T=%.0f s, dt=%.2f s, scenario=%s, architecture=%s, hiddenLayers=%d, communication=%s, parameterAdaptation=%d, maneuver=%d\n', ...
        seed,T,cfg.dt,cfg.scenario.name,cfg.residual.architecture,cfg.dnn.hiddenLayerCount,cfg.communication.mode,cfg.training.adaptParameters,cfg.maneuver.enabled);
    % Fair open-loop estimator comparison: derive one maneuver schedule from
    % the nominal controller, then replay it in every DNN case. Set
    % cfg.comparison.replayNominalManeuver=false for closed-loop comparison.
    rng(seed); out.nominal = simulateAdditiveCase(cfg,"nominal");
    if cfg.comparison.replayNominalManeuver
        referenceManeuver = out.nominal.watcherA;
        out.maneuverScheduleSource = "nominal EKF "+cfg.maneuver.objective+" controller";
    else
        referenceManeuver = [];
        out.maneuverScheduleSource = "each case local controller";
    end
    rng(seed); out.localOnly = simulateAdditiveCase( ...
        cfg,"local_only",referenceManeuver);
    % Mean is a required, scale-normalized reference for every vector
    % additive architecture.  It is not only a state-partition baseline.
    if isVectorAdditiveArchitecture(cfg)
        rng(seed); out.meanEnsemble = simulateAdditiveCase( ...
            cfg,"mean_ensemble",referenceManeuver);
    end
    rng(seed); out.sharedAdditive = simulateAdditiveCase( ...
        cfg,"shared_additive",referenceManeuver);
    out.cfg = cfg;
    out.summary = additiveSummary(out);
    disp(out.summary);
    if makePlots, out.figures = plotAdditiveToy(out); end
end

function cfg = applyToyTuning(cfg,tuning)
%APPLYTOYTUNING Small, explicit overrides used by repeatable ablations.
% Reducing Q_theta makes the parameter random walk more conservative;
% threshold and interval reduce abrupt remote-cache replacements.
    if ~isstruct(tuning)
        error('The optional tuning argument must be a struct.');
    end
    if isfield(tuning,'parameterProcessScale') && ~isempty(tuning.parameterProcessScale)
        validateattributes(tuning.parameterProcessScale,{'numeric'}, ...
            {'scalar','positive','finite'});
        cfg.ekf.Qtheta = tuning.parameterProcessScale*cfg.ekf.Qtheta;
    end
    if isfield(tuning,'mahalanobisThreshold') && ~isempty(tuning.mahalanobisThreshold)
        validateattributes(tuning.mahalanobisThreshold,{'numeric'}, ...
            {'scalar','nonnegative','finite'});
        cfg.communication.mahalanobisThreshold = tuning.mahalanobisThreshold;
    end
    if isfield(tuning,'minCommunicationInterval') && ~isempty(tuning.minCommunicationInterval)
        validateattributes(tuning.minCommunicationInterval,{'numeric'}, ...
            {'scalar','nonnegative','finite'});
        cfg.communication.minInterval = tuning.minCommunicationInterval;
    end
    cfg.tuning = tuning;
end

function cfg = additiveToyConfig(seed,T,scenario,dt,nWatchers,architecture,hiddenLayerCount)
    cfg.seed = seed; cfg.T = T; cfg.dt = dt; cfg.time = 0:cfg.dt:T;
    cfg.N = numel(cfg.time); cfg.Nw = nWatchers; cfg.nPhi = 3;
    % Temporary head dimension used while constructing the warm start.
    % It is replaced by fullDnnParameterCount for additive_full_dnn below.
    cfg.nTheta = 2*cfg.nPhi;
    % Outward spiral: rho(t) approaches 100 m while the azimuth rotates.
    % The acceleration below is an arbitrary nonlinear state function, not
    % generated from the DNN used by the estimator.
    cfg.spiral.radiusGoal = 100;
    cfg.spiral.radialRate = .30;
    cfg.spiral.angularRate = .012;
    cfg.spiral.velocityGain = .035;
    cfg.target.r0 = [5;0];
    cfg.target.v0 = spiralDesiredVelocity(cfg.target.r0,cfg);
    % With direct summation, scale=2 produces an O(1e-4) m/s^2 unknown
    % acceleration.  The former value 25 was calibrated for a gate that
    % attenuated every branch and would make the ungated sum unrealistically
    % strong. Change this one value for stress tests.
    cfg.residual.scale = 2;
    cfg.residual.architecture = string(architecture);
    cfg.residual.wlsRidge = 1e-6;
    cfg.residual.weighting = "parameter_covariance";
    % Robust fusion uses branch-to-fused-model disagreement, not truth.
    % It is an IRLS/Huber correction on top of parameter-covariance WLS.
    cfg.residual.robustIterations = 3;
    cfg.residual.robustHuberThreshold = 2.5;
    % This floor prevents an overconfident branch from becoming a numerically
    % dominant, effectively hard constraint in the WLS reconstruction.
    cfg.residual.scalarVarianceFloor = (cfg.residual.scale*0.01e-4)^2;
    cfg.residual.maxWeightRatio = 100;
    cfg.scenario.name = string(scenario);
    switch cfg.scenario.name
        case "well_conditioned"
            % Nested subsets of a mutually non-collinear four-watcher
            % formation.  Two watchers are intentionally 90 degrees apart,
            % rather than opposite, so the 2-D WLS reconstruction is not
            % artificially rank-one at Nw=2.
            % The first four placements are unchanged.  The diagonal
            % placements extend them to nested, well-spread 5--8 watcher
            % formations for scaling experiments.
            formation = 1000*[1 0 -1 0 sqrt(.5) -sqrt(.5) -sqrt(.5) sqrt(.5); ...
                              0 1 0 -1 sqrt(.5)  sqrt(.5) -sqrt(.5) -sqrt(.5)];
            cfg.watchers.r0 = formation(:,1:cfg.Nw);
        case "near_parallel"
            % Deliberately difficult geometry: all cameras initially view the
            % target from nearly the same side, so their transverse directions
            % are almost parallel and WLS has weak 2-D reconstruction rank.
            cfg.watchers.r0 = [-1000*ones(1,cfg.Nw); ...
                linspace(-150,150,cfg.Nw)];
        otherwise
            error('Unknown scenario "%s". Use "well_conditioned" or "near_parallel".', ...
                cfg.scenario.name);
    end
    % Fixed camera boresights point to the initial target.  The gate is
    % deliberately fixed (not a learned router), so branch ownership cannot
    % change merely by permuting DNN parameters.
    cfg.watchers.boresight = cfg.target.r0-cfg.watchers.r0;
    cfg.watchers.boresight = cfg.watchers.boresight./vecnorm(cfg.watchers.boresight);
    % State-dependent, fixed soft partition of the target-position domain.
    % This is intentionally independent of instantaneous LOS geometry and
    % of branch parameters: it gives branch j a persistent state region to
    % represent, rather than allowing every branch to learn the same d(eta).
    partitionAngle = atan2(cfg.watchers.r0(2,:),cfg.watchers.r0(1,:));
    cfg.partition.centers = 75*[cos(partitionAngle);sin(partitionAngle)];
    cfg.partition.width = 85;
    cfg.partition.uniformFloor = .08;
    cfg.meas.sigma = deg2rad(.01);
    cfg.ekf.Peta0 = diag([30^2 30^2 .08^2 .08^2]);
    cfg.ekf.qPhysical = 2e-12;
    cfg.remoteCovInflation = 2.0;
    cfg.gate.mode = "directional"; % "scalar" or "directional"
    cfg.gate.halfAngle = deg2rad(12);
    cfg.gate.rangeDesired = 1000;
    cfg.gate.rangeSigma = 450;
    cfg.gate.minimumScore = 1e-6;
    % "local_information" is the default controller. It selects a known
    % maneuver that maximizes the smallest singular direction of the local
    % finite-horizon bearing observability/Fisher-information matrix.
    % "local_position" and "local_radial" are covariance heuristics.
    cfg.maneuver.enabled = true;
    cfg.maneuver.objective = "local_information";
    cfg.maneuver.startTime = 20;
    cfg.maneuver.acceleration = 0.20;
    cfg.maneuver.burnDuration = 8;
    cfg.maneuver.predictionHorizon = 20;
    cfg.maneuver.triggerRatio = 1.10;
    cfg.maneuver.cooldown = 30;
    cfg.maneuver.rankThreshold = .10;
    cfg.maneuver.infoRankThreshold = 1e-3;
    cfg.maneuver.infoMinRelativeGain = .05;
    % Position-only information controller: require a candidate to improve
    % the weakest marginal position direction by this relative amount.
    cfg.maneuver.positionInfoMinRelativeGain = .05;
    % Hybrid position controller: dimensionless score combines predicted
    % fractional reduction in tr(P_r) and fractional gain in the weakest
    % history-based marginal position-information direction.
    cfg.maneuver.hybridInformationWeight = .25;
    cfg.maneuver.hybridMinBenefit = .03;
    % Retain a finite, recency-weighted bearing history.  The information
    % matrix is propagated into the current state coordinates before each
    % new bearing contribution is added.
    cfg.maneuver.positionInfoHistoryTime = 60;
    cfg.maneuver.losProfileMinScore = (3*cfg.meas.sigma)^2;
    cfg.maneuver.losProfileMaxEvents = 4;
    cfg.comparison.replayNominalManeuver = true;
    % A watcher initially uploads its pretrained posterior.  Subsequent
    % uploads can be event triggered by a statistically meaningful change
    % in its local parameter posterior.  The ground station cache is the
    % only remote-parameter source used during prediction.
    cfg.communication.mode = "instantaneous";
    cfg.communication.mahalanobisThreshold = 4.0;
    cfg.communication.minInterval = 5.0;
    % Fixed, genuinely deep small-width tanh backbone. Only its 2x3 output
    % head is in the EKF state, so theta=vec(Wout) still has six online
    % parameters/branch regardless of backbone depth. The default has three
    % hidden layers; use the final optional function argument 4 for four.
    % Structured rather than fully random frozen features retain the two
    % position coordinates and a mixed velocity feature at the first layer.
    cfg.dnn.hiddenLayerCount = hiddenLayerCount;
    cfg.dnn.hiddenWidth = 3;
    baseW1 = [1 0 .20 0; 0 1 0 .20; 0 0 .70 .70];
    cfg.dnn.W = cell(hiddenLayerCount,1);
    cfg.dnn.b = cell(hiddenLayerCount,1);
    for ell = 1:hiddenLayerCount
        inputWidth = 4;
        if ell > 1, inputWidth = cfg.dnn.hiddenWidth; end
        cfg.dnn.W{ell} = zeros(cfg.dnn.hiddenWidth,inputWidth,cfg.Nw);
        cfg.dnn.b{ell} = zeros(cfg.dnn.hiddenWidth,cfg.Nw);
    end
    for j=1:cfg.Nw
        if isVectorAdditiveArchitecture(cfg)
            % Distinct, fixed feature blocks make an additional branch an
            % actual expansion of the global additive function class.
            rng(cfg.seed+100*j);
            cfg.dnn.W{1}(:,:,j) = baseW1+.18*randn(3,4);
            cfg.dnn.b{1}(:,j) = .03*randn(3,1);
            for ell=2:hiddenLayerCount
                cfg.dnn.W{ell}(:,:,j) = eye(3)+.12*randn(3,3);
                cfg.dnn.b{ell}(:,j) = .03*randn(3,1);
            end
        else
            cfg.dnn.W{1}(:,:,j) = baseW1;
            for ell=2:hiddenLayerCount
                cfg.dnn.W{ell}(:,:,j) = eye(3);
            end
        end
    end
    cfg.dnn.inputScale = [100;100;.8;.8];
    % Offline least-squares head fit gives all branches a realistic warm
    % start, while the true spiral acceleration remains an independent,
    % arbitrary nonlinear function.
    % First fit an output-head warm start.  The full-DNN option packs this
    % head together with *all* hidden weights and biases into theta_i, so
    % every layer is subsequently estimated and communicated.
    headTheta = fitOfflineSpiralHead(cfg);
    if isFullDnnArchitecture(cfg)
        cfg.nTheta = fullDnnParameterCount(cfg);
        cfg.thetaOffline = zeros(cfg.nTheta,cfg.Nw);
        for j=1:cfg.Nw
            cfg.thetaOffline(:,j) = packFullDnnBranch(cfg,j,headTheta(:,j));
        end
        cfg.ekf.Ptheta0 = fullDnnInitialCovariance(cfg);
        cfg.ekf.Qtheta = fullDnnProcessCovariance(cfg);
    else
        cfg.nTheta = 2*cfg.nPhi;
        cfg.thetaOffline = headTheta;
        cfg.ekf.Ptheta0 = (3e-3)^2*eye(cfg.nTheta);
        cfg.ekf.Qtheta = 2e-8*eye(cfg.nTheta);
    end
    cfg.training.pretrainedFraction = .95;
    cfg.training.pretrainedNoiseStd = 1.5e-4;
    cfg.training.adaptParameters = true;
end

function res = simulateAdditiveCase(cfg,mode,referenceManeuver)
    if nargin < 3, referenceManeuver = []; end
    N = cfg.N; Nw = cfg.Nw; dt = cfg.dt; nx = 4+cfg.nTheta;
    etaTrue = zeros(4,N); etaTrue(:,1) = [cfg.target.r0;cfg.target.v0];
    watcherR = zeros(2,N,Nw); watcherR(:,1,:) = cfg.watchers.r0;
    watcherV = zeros(2,N,Nw); watcherA = zeros(2,N,Nw);
    xhat = zeros(nx,N,Nw); P = zeros(nx,nx,Nw);
    thetaCache = zeros(cfg.nTheta,Nw); thetaInitial = zeros(cfg.nTheta,Nw);
    PthetaCache = repmat(cfg.ekf.Ptheta0,1,1,Nw);
    dHat = zeros(2,N,Nw); dTrue = zeros(2,N); NIS = nan(N,Nw);
    % Post-hoc diagnostics use the true state only as a common evaluation
    % input.  They never enter the EKF, parameter update, or communication
    % trigger, and therefore do not leak supervision into the experiment.
    branchOutputDiagnostic = nan(2,Nw,N);
    branchContributionNorm = nan(Nw,N);
    parameterCovarianceTrace = nan(Nw,N);
    parameterChangeNorm = nan(Nw,N);
    alpha = zeros(Nw,N,Nw); los = zeros(2,N,Nw); gateScore = zeros(Nw,N,Nw);
    radialVariance = nan(N,Nw); maneuverEvent = false(N,Nw);
    communicationEvent = false(N,Nw); communicationMetric = nan(N,Nw);
    lastCommunicationTime = -inf(Nw,1);
    geometryLambdaMin = nan(N,Nw); geometryCondition = nan(N,Nw);
    directionalScalarError = nan(Nw,N,Nw);
    directionalWeights = nan(Nw,N,Nw);
    positionInfoLambdaMin = nan(N,Nw);
    maneuver = repmat(initLocalManeuverManager(),Nw,1);
    for i=1:Nw
        theta0 = cfg.training.pretrainedFraction*cfg.thetaOffline(:,i) + ...
            cfg.training.pretrainedNoiseStd*randn(cfg.nTheta,1);
        thetaCache(:,i) = theta0;
        thetaInitial(:,i) = theta0;
        parameterCovarianceTrace(i,1) = trace(cfg.ekf.Ptheta0);
        parameterChangeNorm(i,1) = 0;
        lastCommunicationTime(i) = 0; % initial pretrained upload
        xhat(:,1,i) = [cfg.target.r0+[25;-18]; cfg.target.v0+[.035;-.025]; theta0];
        P(:,:,i) = blkdiag(cfg.ekf.Peta0,cfg.ekf.Ptheta0);
        maneuver(i).referenceVariance = localPositionUncertainty(xhat(:,1,i),P(:,:,i));
        maneuver(i) = updatePositionInformationHistory( ...
            maneuver(i),xhat(1:4,1,i),watcherR(:,1,i),cfg,true);
        positionInfoLambdaMin(1,i) = marginalPositionInformationMetric( ...
            maneuver(i).positionInformation);
        radialVariance(1,i) = maneuver(i).referenceVariance;
        % These diagnostics are defined only for the directional-WLS
        % parameterization s_i = psi_i' theta_i.  A full vector DNN has
        % no scalar directional feature psi_i, so retain NaNs rather than
        % incorrectly applying a 6-D diagnostic to its full P_theta.
        if cfg.residual.architecture == "directional_wls"
            [geometryLambdaMin(1,i),geometryCondition(1,i)] = ...
                directionalGeometryMetrics(xhat(1:4,1,i),watcherR(:,1,:),PthetaCache,cfg);
            directionalWeights(:,1,i) = directionalWLSWeights( ...
                xhat(1:4,1,i),watcherR(:,1,:),PthetaCache,cfg);
        end
    end
    for k=1:N-1
        t = cfg.time(k);
        dTrue(:,k) = trueResidual(etaTrue(:,k),cfg);
        etaTrue(:,k+1) = propagateTruthAdditive(etaTrue(:,k),cfg);
        for i=1:Nw
            if isempty(referenceManeuver)
                PthetaForManeuver = PthetaCache;
                PthetaForManeuver(:,:,i) = P(5:end,5:end,i);
                [aW,maneuver(i),started] = selectObservabilityManeuver( ...
                    xhat(:,k,i),P(:,:,i),watcherR(:,k,:),watcherV(:,k,:),i, ...
                    PthetaForManeuver,cfg.time(k),maneuver(i),cfg);
            else
                aW = referenceManeuver(:,k,i);
                started = false;
            end
            watcherA(:,k,i) = aW; maneuverEvent(k,i) = started;
            watcherR(:,k+1,i) = watcherR(:,k,i)+watcherV(:,k,i)*dt+.5*aW*dt^2;
            watcherV(:,k+1,i) = watcherV(:,k,i)+aW*dt;
            thetaForPrediction = thetaCache;
            thetaForPrediction(:,i) = xhat(5:end,k,i);
            [xp,Pp] = predictAdditive(xhat(:,k,i),P(:,:,i),thetaForPrediction,PthetaCache, ...
                watcherR(:,k,:),i,mode,cfg);
            z = bearingAdditive(etaTrue(1:2,k+1),watcherR(:,k+1,i)) + cfg.meas.sigma*randn;
            [xu,Pu,innovation] = updateAdditive(xp,Pp,z,watcherR(:,k+1,i),cfg);
            % Frozen-head ablations retain precisely the initial output head.
            % Resetting the cross blocks prevents a measurement update from
            % indirectly changing the physical state through theta covariance.
            if ~cfg.training.adaptParameters
                xu(5:end) = thetaInitial(:,i);
                Pu(5:end,:) = 0;
                Pu(:,5:end) = 0;
                Pu(5:end,5:end) = cfg.ekf.Ptheta0;
            end
            xhat(:,k+1,i) = xu; P(:,:,i) = Pu;
            parameterCovarianceTrace(i,k+1) = trace(Pu(5:end,5:end));
            parameterChangeNorm(i,k+1) = norm(xu(5:end)-thetaInitial(:,i));
            NIS(k+1,i) = innovation.nu^2/innovation.S;
            radialVariance(k+1,i) = radialPositionVariance(xu,Pu,watcherR(:,k+1,i));
            maneuver(i) = updatePositionInformationHistory( ...
                maneuver(i),xu(1:4),watcherR(:,k+1,i),cfg,false);
            positionInfoLambdaMin(k+1,i) = marginalPositionInformationMetric( ...
                maneuver(i).positionInformation);
            % The owner alone adapts branch i from its angle-only update.
            % A remote watcher never receives the state; it sees this
            % branch only through the ground-station parameter cache.
            if mode ~= "nominal"
                [upload,communicationMetric(k+1,i)] = shouldUploadParameterPosterior( ...
                    xu(5:end),Pu(5:end,5:end),thetaCache(:,i),PthetaCache(:,:,i), ...
                    cfg.time(k+1),lastCommunicationTime(i),cfg);
                if upload
                    thetaCache(:,i) = xu(5:end);
                    PthetaCache(:,:,i) = Pu(5:end,5:end);
                    lastCommunicationTime(i) = cfg.time(k+1);
                    communicationEvent(k+1,i) = true;
                end
            end
        end
        % Evaluate all current ground-station cached branches at one common
        % input for correlation/contribution diagnostics.  The cache is the
        % same object used by remote predictions; etaTrue is diagnostic only.
        for j=1:Nw
            bj = vectorAdditiveBranchOutput(etaTrue(:,k+1),thetaCache(:,j),j,cfg);
            branchOutputDiagnostic(:,j,k+1) = bj;
            branchContributionNorm(j,k+1) = norm( ...
                vectorBranchContributionWeight(etaTrue(:,k+1),j,1,mode,cfg)*bj);
        end
        for i=1:Nw
            % The owner's newest posterior is available locally even before
            % its next ground-station upload.  Remote branches use the cache.
            thetaForDiagnostic = thetaCache;
            thetaForDiagnostic(:,i) = xhat(5:end,k+1,i);
            PthetaForDiagnostic = PthetaCache;
            PthetaForDiagnostic(:,:,i) = P(5:end,5:end,i);
            [dHat(:,k+1,i),gate] = additiveResidual(xhat(1:4,k+1,i),thetaForDiagnostic, ...
                PthetaForDiagnostic,watcherR(:,k+1,:),i,mode,cfg);
            if cfg.residual.architecture == "directional_wls"
                [geometryLambdaMin(k+1,i),geometryCondition(k+1,i)] = ...
                    directionalGeometryMetrics(xhat(1:4,k+1,i),watcherR(:,k+1,:),PthetaForDiagnostic,cfg);
                [~,directionalWeights(:,k+1,i)] = directionalWLSReconstruct( ...
                    xhat(1:4,k+1,i),thetaForDiagnostic,PthetaForDiagnostic,watcherR(:,k+1,:),cfg);
                for j=1:Nw
                    tj = transverseDirection(xhat(1:4,k+1,i),watcherR(:,k+1,j));
                    sj = thetaForDiagnostic(:,j)'*directionalFeatures(xhat(1:4,k+1,i),tj,cfg);
                    directionalScalarError(j,k+1,i) = sj-tj'*trueResidual(etaTrue(:,k+1),cfg);
                end
            end
            alpha(:,k+1,i) = gate.alpha;
            los(:,k+1,i) = gate.los(:,i);
            gateScore(:,k+1,i) = gate.score;
        end
    end
    dTrue(:,N) = trueResidual(etaTrue(:,N),cfg);
    for j=1:Nw
        bj = vectorAdditiveBranchOutput(etaTrue(:,1),thetaCache(:,j),j,cfg);
        branchOutputDiagnostic(:,j,1) = bj;
        branchContributionNorm(j,1) = norm( ...
            vectorBranchContributionWeight(etaTrue(:,1),j,1,mode,cfg)*bj);
    end
    for i=1:Nw
        [dHat(:,1,i),gate] = additiveResidual(xhat(1:4,1,i),thetaCache, ...
            PthetaCache,watcherR(:,1,:),i,mode,cfg);
        if cfg.residual.architecture == "directional_wls"
            [~,directionalWeights(:,1,i)] = directionalWLSReconstruct( ...
                xhat(1:4,1,i),thetaCache,PthetaCache,watcherR(:,1,:),cfg);
        end
        alpha(:,1,i) = gate.alpha; los(:,1,i) = gate.los(:,i); gateScore(:,1,i) = gate.score;
        if cfg.residual.architecture == "directional_wls"
            for j=1:Nw
                tj = transverseDirection(xhat(1:4,1,i),watcherR(:,1,j));
                sj = thetaCache(:,j)'*directionalFeatures(xhat(1:4,1,i),tj,cfg);
                directionalScalarError(j,1,i) = sj-tj'*trueResidual(etaTrue(:,1),cfg);
            end
        end
    end
    res = struct('mode',mode,'time',cfg.time,'etaTrue',etaTrue,'watcherR',watcherR, ...
        'xhat',xhat,'P',P,'thetaInitial',thetaInitial, ...
        'thetaCache',thetaCache,'PthetaCache',PthetaCache, ...
        'dHat',dHat,'dTrue',dTrue,'NIS',NIS,'alpha',alpha, ...
        'branchOutputDiagnostic',branchOutputDiagnostic, ...
        'branchContributionNorm',branchContributionNorm, ...
        'parameterCovarianceTrace',parameterCovarianceTrace, ...
        'parameterChangeNorm',parameterChangeNorm, ...
        'los',los,'gateScore',gateScore,'gateMode',cfg.gate.mode, ...
        'watcherV',watcherV,'watcherA',watcherA, ...
        'radialVariance',radialVariance,'maneuverEvent',maneuverEvent, ...
        'communicationMode',cfg.communication.mode, ...
        'communicationEvent',communicationEvent, ...
        'communicationMetric',communicationMetric, ...
        'lastCommunicationTime',lastCommunicationTime, ...
        'maneuverObjective',cfg.maneuver.objective, ...
        'maneuverScheduleReplayed',~isempty(referenceManeuver), ...
        'positionInfoLambdaMin',positionInfoLambdaMin, ...
        'geometryLambdaMin',geometryLambdaMin,'geometryCondition',geometryCondition, ...
        'directionalScalarError',directionalScalarError,'directionalWeights',directionalWeights);
end

function manager = initLocalManeuverManager()
    manager = struct('referenceVariance',nan,'burnTimeRemaining',0, ...
        'cooldownUntil',-inf,'direction',zeros(2,1),'eventCount',0, ...
        'positionInformation',zeros(4));
end

function [aW,manager,started] = selectObservabilityManeuver( ...
        x,P,rWAll,vWAll,owner,PthetaAll,tk,manager,cfg)
%SELECTOBSERVABILITYMANEUVER Candidate-rollout watcher controller.
% local_los_profile implements the LOS-profile criterion: maneuvers must
% induce position change transverse to the natural LOS. local_information maximizes the weakest direction of a 4-D state information matrix.
% local_position_information instead marginalizes velocity and maximizes the
% weakest *position* direction. local_hybrid_position combines predicted
% position covariance reduction with that history-information gain.
% local_position uses the owner's covariance and
% targets tr(P_r); local_radial retains the LOS-only uncertainty objective.
% directional_rank uses communicated geometry/P_theta for WLS conditioning.
    aW = zeros(2,1); started = false;
    if ~cfg.maneuver.enabled || tk < cfg.maneuver.startTime, return; end
    rWAll = reshape(rWAll,2,[]); vWAll = reshape(vWAll,2,[]);
    rW = rWAll(:,owner); vW = vWAll(:,owner);
    if manager.burnTimeRemaining > 0
        aW = cfg.maneuver.acceleration*manager.direction;
        manager.burnTimeRemaining = max(0,manager.burnTimeRemaining-cfg.dt);
        if manager.burnTimeRemaining == 0
            manager.cooldownUntil = tk+cfg.maneuver.cooldown;
            if cfg.maneuver.objective == "local_position"
                manager.referenceVariance = localPositionUncertainty(x,P);
            else
                manager.referenceVariance = radialPositionVariance(x,P,rW);
            end
        end
        return;
    end
    if cfg.maneuver.objective == "local_los_profile"
        qNow = radialPositionVariance(x,P,rW);
        if ~isfinite(manager.referenceVariance), manager.referenceVariance = qNow; end
        if manager.eventCount >= cfg.maneuver.losProfileMaxEvents || ...
                tk < manager.cooldownUntil || qNow < cfg.maneuver.triggerRatio*manager.referenceVariance
            manager.referenceVariance = min(manager.referenceVariance,qNow);
            return;
        end
    elseif cfg.maneuver.objective == "local_information"
        infoCoast = rolloutLocalInformation(x(1:4),rW,vW,zeros(2,1),cfg);
        if tk < manager.cooldownUntil || infoCoast >= cfg.maneuver.infoRankThreshold
            return;
        end
    elseif cfg.maneuver.objective == "local_position_information" || ...
            cfg.maneuver.objective == "local_hybrid_position"
        infoCoast = rolloutLocalPositionInformation( ...
            x(1:4),rW,vW,zeros(2,1),manager.positionInformation,cfg);
        if tk < manager.cooldownUntil
            return;
        end
    elseif cfg.maneuver.objective == "local_position" || cfg.maneuver.objective == "local_radial"
        if cfg.maneuver.objective == "local_position"
            qNow = localPositionUncertainty(x,P);
        else
            qNow = radialPositionVariance(x,P,rW);
        end
        if ~isfinite(manager.referenceVariance), manager.referenceVariance = qNow; end
        if tk < manager.cooldownUntil || qNow < cfg.maneuver.triggerRatio*manager.referenceVariance
            manager.referenceVariance = min(manager.referenceVariance,qNow);
            return;
        end
    else
        [rankNow,~] = directionalGeometryMetrics(x(1:4),rWAll,PthetaAll,cfg);
        if tk < manager.cooldownUntil || rankNow >= cfg.maneuver.rankThreshold
            return;
        end
    end
    if cfg.maneuver.objective == "local_position_information" || ...
            cfg.maneuver.objective == "local_hybrid_position"
        ell = x(1:2)-rW; ell = ell/max(norm(ell),eps);
        transverse = [-ell(2);ell(1)];
        directions = [ell -ell transverse -transverse];
    else
        directions = [1 -1 0 0; 0 0 1 -1];
    end
    score = inf(1,size(directions,2));
    for c=1:size(directions,2)
        aCandidate = cfg.maneuver.acceleration*directions(:,c);
        if cfg.maneuver.objective == "local_los_profile"
            score(c) = -rolloutLOSProfileChange(x(1:4),rW,vW,aCandidate,cfg);
        elseif cfg.maneuver.objective == "local_information"
            score(c) = -rolloutLocalInformation(x(1:4),rW,vW,aCandidate,cfg);
        elseif cfg.maneuver.objective == "local_position_information"
            score(c) = -rolloutLocalPositionInformation( ...
                x(1:4),rW,vW,aCandidate,manager.positionInformation,cfg);
        elseif cfg.maneuver.objective == "local_hybrid_position"
            qCoast = rolloutLocalPositionUncertainty(x,P,rW,vW,zeros(2,1),cfg);
            qCandidate = rolloutLocalPositionUncertainty(x,P,rW,vW,aCandidate,cfg);
            infoCandidate = rolloutLocalPositionInformation( ...
                x(1:4),rW,vW,aCandidate,manager.positionInformation,cfg);
            covarianceBenefit = (qCoast-qCandidate)/max(qCoast,eps);
            informationBenefit = (infoCandidate-infoCoast)/max(infoCoast,eps);
            score(c) = -(covarianceBenefit + ...
                cfg.maneuver.hybridInformationWeight*informationBenefit);
        elseif cfg.maneuver.objective == "local_position"
            score(c) = rolloutLocalPositionUncertainty(x,P,rW,vW,aCandidate,cfg);
        elseif cfg.maneuver.objective == "local_radial"
            score(c) = rolloutLocalRadialVariance(x,P,rW,vW,aCandidate,cfg);
        else
            score(c) = -rolloutDirectionalRank(x(1:4),rWAll,vWAll,owner, ...
                aCandidate,PthetaAll,cfg);
        end
    end
    [bestScore,best] = min(score);
    if cfg.maneuver.objective == "local_los_profile" && -bestScore < cfg.maneuver.losProfileMinScore
        return;
    elseif cfg.maneuver.objective == "local_information" && ...
            -bestScore < (1+cfg.maneuver.infoMinRelativeGain)*infoCoast
        return;
    elseif cfg.maneuver.objective == "local_position_information" && ...
            -bestScore < (1+cfg.maneuver.positionInfoMinRelativeGain)*infoCoast
        return;
    elseif cfg.maneuver.objective == "local_hybrid_position" && ...
            -bestScore < cfg.maneuver.hybridMinBenefit
        return;
    end
    manager.direction = directions(:,best);
    manager.burnTimeRemaining = cfg.maneuver.burnDuration;
    manager.eventCount = manager.eventCount+1;
    aW = cfg.maneuver.acceleration*manager.direction;
    started = true;
end

function lambdaMin = rolloutLocalPositionInformation(eta,rW,vW,aW,information,cfg)
%ROLLOUTLOCALPOSITIONINFORMATION History + candidate bearing information.
% INFORMATION is the local, recency-weighted bearing Fisher information
% already accumulated through the current time.  It is expressed in the
% current [position; velocity] coordinates.  Each candidate advances that
% history through the state transition and appends its predicted bearings.
% Velocity is marginalized before scoring the weakest position direction.
    dt = cfg.dt; steps = max(1,round(cfg.maneuver.predictionHorizon/dt));
    burnSteps = min(steps,max(1,round(cfg.maneuver.burnDuration/dt)));
    F = [eye(2),dt*eye(2);zeros(2),eye(2)];
    Finv = [eye(2),-dt*eye(2);zeros(2),eye(2)];
    forget = exp(-dt/max(cfg.maneuver.positionInfoHistoryTime,dt));
    for n=1:steps
        aStep = zeros(2,1);
        if n <= burnSteps, aStep = aW; end
        rW = rW+vW*dt+.5*aStep*dt^2; vW = vW+aStep*dt;
        eta = F*eta;
        information = forget*(Finv'*information*Finv);
        information = information+bearingInformationContribution(eta,rW,cfg);
    end
    lambdaMin = marginalPositionInformationMetric(information);
end

function manager = updatePositionInformationHistory(manager,eta,rW,cfg,isInitial)
%UPDATEPOSITIONINFORMATIONHISTORY Keep local bearing history at current time.
% Past information is transported from x_{k-1} to x_k using F^{-1}; this
% makes every stored bearing sensitivity refer to the same current state.
    if isInitial
        manager.positionInformation = bearingInformationContribution(eta,rW,cfg);
        return;
    end
    dt = cfg.dt; Finv = [eye(2),-dt*eye(2);zeros(2),eye(2)];
    forget = exp(-dt/max(cfg.maneuver.positionInfoHistoryTime,dt));
    manager.positionInformation = forget*(Finv'*manager.positionInformation*Finv) + ...
        bearingInformationContribution(eta,rW,cfg);
    manager.positionInformation = .5*(manager.positionInformation+manager.positionInformation');
end

function information = bearingInformationContribution(eta,rW,cfg)
    rel = eta(1:2)-rW; r2 = max(rel'*rel,1);
    H = [-rel(2)/r2,rel(1)/r2,0,0];
    information = (H'*H)/(cfg.meas.sigma^2);
end

function lambdaMin = marginalPositionInformationMetric(information)
%MARGINALPOSITIONINFORMATIONMETRIC Position information after eliminating v.
    information = .5*(information+information');
    Irr = information(1:2,1:2);
    Irv = information(1:2,3:4);
    Ivv = information(3:4,3:4);
    ridge = 1e-10*max(1,trace(Ivv)/2);
    Ipos = Irr-Irv*((Ivv+ridge*eye(2))\Irv');
    lambdaMin = max(0,min(eig(.5*(Ipos+Ipos'))));
end

function score = rolloutLOSProfileChange(eta,rW,vW,aW,cfg)
%ROLLOUTLOSPROFILECHANGE Paper Eq. (22) proxy for a finite burn.
% It scores the candidate-induced relative displacement perpendicular to the
% no-maneuver LOS. A purely LOS-parallel displacement receives zero score.
    dt = cfg.dt; steps = max(1,round(cfg.maneuver.predictionHorizon/dt));
    burnSteps = min(steps,max(1,round(cfg.maneuver.burnDuration/dt)));
    F = [eye(2),dt*eye(2);zeros(2),eye(2)];
    rCoast = rW; vCoast = vW; rCandidate = rW; vCandidate = vW;
    score = 0;
    for n=1:steps
        aStep = zeros(2,1);
        if n <= burnSteps, aStep = aW; end
        rCoast = rCoast+vCoast*dt;
        rCandidate = rCandidate+vCandidate*dt+.5*aStep*dt^2;
        vCandidate = vCandidate+aStep*dt;
        eta = F*eta;
        rhoCoast = eta(1:2)-rCoast;
        rho2 = max(rhoCoast'*rhoCoast,1);
        ell = rhoCoast/sqrt(rho2);
        deltaRho = rCoast-rCandidate;
        transverseDelta = (eye(2)-ell*ell')*deltaRho;
        score = score+(transverseDelta'*transverseDelta)/rho2;
    end
end

function lambdaMin = rolloutLocalInformation(eta,rW,vW,aW,cfg)
%ROLLOUTLOCALINFORMATION Finite-horizon local bearing observability score.
% For a candidate known watcher maneuver, form I = sum G_n'G_n, where
% G_n = H_n Phi_n S is the normalized sensitivity of the n-th bearing to
% the initial target [position; velocity].  The maneuver is useful only if
% it increases lambda_min(I), i.e., excites the weakest state direction.
    dt = cfg.dt; steps = max(1,round(cfg.maneuver.predictionHorizon/dt));
    F = [eye(2),dt*eye(2);zeros(2),eye(2)];
    Phi = eye(4);
    Sx = diag([cfg.gate.rangeDesired cfg.gate.rangeDesired .8 .8]);
    information = zeros(4);
    for n=1:steps
        rW = rW+vW*dt+.5*aW*dt^2; vW = vW+aW*dt;
        eta = F*eta; Phi = F*Phi;
        rel = eta(1:2)-rW; r2 = max(rel'*rel,1);
        H = [-rel(2)/r2,rel(1)/r2,0,0];
        G = H*Phi*Sx;
        information = information+G'*G;
    end
    lambdaMin = min(eig(.5*(information+information')));
end

function rankEnd = rolloutDirectionalRank(eta,rWAll,vWAll,owner,aW,PthetaAll,cfg)
%ROLLOUTDIRECTIONALRANK Predict lambda_min(Omega) for one local candidate.
% Other watchers coast; only owner receives the candidate acceleration.
    dt = cfg.dt; steps = max(1,round(cfg.maneuver.predictionHorizon/dt));
    rWAll = reshape(rWAll,2,[]); vWAll = reshape(vWAll,2,[]);
    for n=1:steps
        rWAll = rWAll+vWAll*dt;
        rWAll(:,owner) = rWAll(:,owner)+.5*aW*dt^2;
        vWAll(:,owner) = vWAll(:,owner)+aW*dt;
        eta(1:2) = eta(1:2)+eta(3:4)*dt;
    end
    [rankEnd,~] = directionalGeometryMetrics(eta,rWAll,PthetaAll,cfg);
end

function qEnd = rolloutLocalPositionUncertainty(x,P,rW,vW,aW,cfg)
%ROLLOUTLOCALPOSITIONUNCERTAINTY Predict tr(P_r) under local bearing EKF.
% The rollout uses only watcher i's own state, covariance, and candidate
% trajectory; no other watcher's measurement, LOS, or covariance enters.
    Peta = P(1:4,1:4); eta = x(1:4); dt = cfg.dt;
    steps = max(1,round(cfg.maneuver.predictionHorizon/dt));
    F = [eye(2),dt*eye(2);zeros(2),eye(2)];
    Q = cfg.ekf.qPhysical*dt*eye(4);
    for n=1:steps
        rW = rW+vW*dt+.5*aW*dt^2; vW = vW+aW*dt;
        eta = F*eta;
        Pp = F*Peta*F'+Q;
        rel = eta(1:2)-rW; r2 = max(rel'*rel,1);
        H = [-rel(2)/r2,rel(1)/r2,0,0];
        S = H*Pp*H'+cfg.meas.sigma^2;
        K = Pp*H'/S;
        Peta = Pp-K*S*K'; Peta = .5*(Peta+Peta');
    end
    qEnd = trace(Peta(1:2,1:2));
end

function qEnd = rolloutLocalRadialVariance(x,P,rW,vW,aW,cfg)
%ROLLOUTLOCALRADIALVARIANCE Virtual local bearing-update covariance rollout.
    Peta = P(1:4,1:4); eta = x(1:4); dt = cfg.dt;
    steps = max(1,round(cfg.maneuver.predictionHorizon/dt));
    F = [eye(2),dt*eye(2);zeros(2),eye(2)];
    Q = cfg.ekf.qPhysical*dt*eye(4);
    for n=1:steps
        rW = rW+vW*dt+.5*aW*dt^2; vW = vW+aW*dt;
        eta = F*eta;
        Pp = F*Peta*F'+Q;
        rel = eta(1:2)-rW; r2 = max(rel'*rel,1);
        H = [-rel(2)/r2,rel(1)/r2,0,0];
        S = H*Pp*H'+cfg.meas.sigma^2;
        K = Pp*H'/S;
        Peta = Pp-K*S*K'; Peta = .5*(Peta+Peta');
    end
    u = eta(1:2)-rW; u = u/max(norm(u),eps);
    qEnd = u'*Peta(1:2,1:2)*u;
end

function q = localPositionUncertainty(~,P)
%LOCALPOSITIONUNCERTAINTY Scalar position-estimation objective tr(P_r).
    q = trace(P(1:2,1:2));
end

function q = radialPositionVariance(x,P,rW)
    u = x(1:2)-rW; u = u/max(norm(u),eps);
    q = u'*P(1:2,1:2)*u;
end

function etaNext = propagateTruthAdditive(eta,cfg)
    d = trueResidual(eta,cfg); dt = cfg.dt;
    etaNext = eta;
    etaNext(1:2) = eta(1:2)+eta(3:4)*dt+.5*d*dt^2;
    etaNext(3:4) = eta(3:4)+d*dt;
end

function d = trueResidual(eta,cfg)
%TRUERESIDUAL Independent nonlinear spiral-driving acceleration field.
    d = cfg.spiral.velocityGain*(spiralDesiredVelocity(eta(1:2),cfg)-eta(3:4));
end

function vDesired = spiralDesiredVelocity(r,cfg)
    rho = max(norm(r),.25); uR = r/rho; uT = [-uR(2);uR(1)];
    % A signed radial command makes r=100 m an attractor: once the target
    % overshoots, the desired radial velocity points inward and damps it.
    radialSpeed = cfg.spiral.radialRate*(1-rho/cfg.spiral.radiusGoal);
    tangentialSpeed = cfg.spiral.angularRate*rho;
    vDesired = radialSpeed*uR+tangentialSpeed*uT;
end

function theta = fitOfflineSpiralHead(cfg)
%FITOFFLINESPIRALHEAD Least-squares warm start over the 0--100 m spiral region.
    rhoGrid = linspace(2,cfg.spiral.radiusGoal,17);
    angleGrid = linspace(0,2*pi,25); angleGrid(end) = [];
    H = []; D = [];
    for rho = rhoGrid
        for angle = angleGrid
            r = rho*[cos(angle);sin(angle)]; v = spiralDesiredVelocity(r,cfg);
            % Include small velocity offsets so the fitted head also sees the
            % velocity-restoring component of the unknown acceleration.
            for dv = [0 .08 -.08; 0 .05 -.05]
                eta = [r;v+dv];
                D(:,end+1) = trueResidual(eta,cfg); %#ok<AGROW>
                if isVectorAdditiveArchitecture(cfg)
                    h = [];
                    partitionWeights = statePartitionWeights(eta,cfg);
                    for j=1:cfg.Nw
                        phi = branchFeatures(eta,cfg,j);
                        if isPartitionedArchitecture(cfg)
                            phi = partitionWeights(j)*phi;
                        end
                        h = [h; phi]; %#ok<AGROW>
                    end
                    H(:,end+1) = h; %#ok<AGROW>
                else
                    H(:,end+1) = branchFeatures(eta,cfg,1); %#ok<AGROW>
                end
            end
        end
    end
    Wout = D*H'/(H*H'+1e-7*eye(size(H,1)));
    theta = zeros(cfg.nTheta,cfg.Nw);
    if isVectorAdditiveArchitecture(cfg)
        for j=1:cfg.Nw
            cols = (j-1)*cfg.nPhi+(1:cfg.nPhi);
            Woutj = Wout(:,cols);
            theta(:,j) = Woutj(:);
        end
    else
        theta(:,:) = repmat(Wout(:),1,cfg.Nw);
    end
end

function [upload,metric] = shouldUploadParameterPosterior(thetaLocal,PLocal,thetaCached,PCached,time,lastTime,cfg)
%SHOULDUPLOADPARAMETERPOSTERIOR Event trigger for a ground-station upload.
% The trigger is based on the posterior mean change measured in the sum of
% the local and cached parameter covariances.  Thus a change smaller than
% the uncertainty represented by the two posteriors is not transmitted.
    if cfg.communication.mode == "instantaneous"
        upload = true; metric = inf; return;
    end
    if cfg.communication.mode == "never"
        upload = false; metric = nan; return;
    end
    delta = thetaLocal-thetaCached;
    S = .5*(PLocal+PLocal'+PCached+PCached');
    S = S+1e-12*eye(size(S));
    metric = real(delta'*(S\delta));
    enoughTime = time-lastTime >= cfg.communication.minInterval;
    upload = enoughTime && metric >= cfg.communication.mahalanobisThreshold;
end

function [xp,Pp] = predictAdditive(x,P,thetaCache,PthetaCache,watcherR,owner,mode,cfg)
    PthetaForPrediction = PthetaCache;
    PthetaForPrediction(:,:,owner) = P(5:end,5:end);
    transition = @(xx) additiveTransition(xx,thetaCache,PthetaForPrediction,watcherR,owner,mode,cfg);
    xp = transition(x);
    if isVectorAdditiveArchitecture(cfg)
        % Analytic parameter block: this is essential when every hidden
        % layer is in theta.  A full finite-difference state Jacobian would
        % scale as O(nTheta) transition evaluations at every EKF step.
        thetaAll = thetaCache; thetaAll(:,owner) = x(5:end);
        accel = @(eta)additiveResidual(eta,thetaAll,PthetaForPrediction,watcherR,owner,mode,cfg);
        A = numericalJacobian(accel,x(1:4)); % only four state directions
        if mode == "nominal"
            Bowner = zeros(2,cfg.nTheta);
        else
            Bowner = vectorBranchContributionWeight(x(1:4),owner,owner,mode,cfg)* ...
                vectorAdditiveJacobian(x(1:4),x(5:end),owner,cfg);
        end
        L = [.5*cfg.dt^2*eye(2); cfg.dt*eye(2)];
        F = eye(4+cfg.nTheta);
        F(1:4,1:4) = [eye(2) cfg.dt*eye(2); zeros(2) eye(2)]+L*A;
        F(1:4,5:end) = L*Bowner;
    else
        F = numericalJacobian(transition,x);
    end
    Q = blkdiag(cfg.ekf.qPhysical*cfg.dt*eye(4),cfg.ekf.Qtheta*cfg.dt);
    if any(mode == ["shared_additive" "mean_ensemble"])
        QaRemote = zeros(2);
        for j=1:cfg.Nw
            if j==owner, continue; end
            if isVectorAdditiveArchitecture(cfg)
                B = vectorBranchContributionWeight(x(1:4),j,owner,mode,cfg)* ...
                    vectorAdditiveJacobian(x(1:4),thetaCache(:,j),j,cfg);
            else
                B = directionalWLSJacobian(x(1:4),watcherR,PthetaForPrediction,j,cfg);
            end
            QaRemote = QaRemote + B*PthetaCache(:,:,j)*B';
        end
        L = [.5*cfg.dt^2*eye(2); cfg.dt*eye(2)];
        Q(1:4,1:4) = Q(1:4,1:4) + cfg.remoteCovInflation*(L*QaRemote*L');
    end
    Pp = F*P*F'+Q; Pp = .5*(Pp+Pp');
end

function B = vectorAdditiveJacobian(eta,thetaBranch,branch,cfg)
%VECTORADDITIVEJACOBIAN Parameter-to-vector-acceleration Jacobian.
% For a head-only branch this is analytic.  For the full-DNN branch it is
% an exact backpropagation Jacobian, so covariance propagation accounts for
% every hidden-layer weight/bias and the output layer.
    if isFullDnnArchitecture(cfg)
        B = fullDnnParameterJacobian(eta,thetaBranch,cfg);
    else
        phi = branchFeatures(eta,cfg,branch);
        B = kron(phi',eye(2));
    end
end

function xNext = additiveTransition(x,thetaCache,PthetaAll,watcherR,owner,mode,cfg)
    eta = x(1:4); thetaAll = thetaCache; thetaAll(:,owner) = x(5:end);
    d = additiveResidual(eta,thetaAll,PthetaAll,watcherR,owner,mode,cfg); dt = cfg.dt;
    xNext = x;
    xNext(1:2) = eta(1:2)+eta(3:4)*dt+.5*d*dt^2;
    xNext(3:4) = eta(3:4)+d*dt;
end

function [d,gate] = additiveResidual(eta,thetaAll,PthetaAll,watcherR,owner,mode,cfg)
    [~,gate] = additiveGate(eta,watcherR,cfg);
    d = zeros(2,1);
    if mode == "nominal", return; end
    watcherR = reshape(watcherR,2,[]);
    if isVectorAdditiveArchitecture(cfg) && any(mode == ["shared_additive" "mean_ensemble"])
        for j=1:size(thetaAll,2)
            wj = vectorBranchContributionWeight(eta,j,owner,mode,cfg);
            d = d+wj*vectorAdditiveBranchOutput(eta,thetaAll(:,j),j,cfg);
        end
        if isPartitionedArchitecture(cfg)
            partitionWeights = statePartitionWeights(eta,cfg);
            gate.alpha = partitionWeights;
            gate.score = partitionWeights;
            gate.G = repmat(eye(2),1,1,cfg.Nw);
            for j=1:cfg.Nw, gate.G(:,:,j) = partitionWeights(j)*eye(2); end
        end
        return;
    end
    if mode == "local_only"
        if isVectorAdditiveArchitecture(cfg)
            % A fair local baseline for the vector-additive architecture:
            % watcher i uses only its own vector-valued sub-DNN.
            d = vectorAdditiveBranchOutput(eta,thetaAll(:,owner),owner,cfg);
            return;
        end
        t = transverseDirection(eta,watcherR(:,owner));
        s = thetaAll(:,owner)'*directionalFeatures(eta,t,cfg);
        d = t*s;
        return;
    end
    d = directionalWLSReconstruct(eta,thetaAll,PthetaAll,watcherR,cfg);
end

function [d,weights,standardizedResidual] = directionalWLSReconstruct(eta,thetaAll,PthetaAll,watcherR,cfg)
%DIRECTIONALWLSRECONSTRUCT Reconstruct d with optional robust IRLS weights.
% The robust factor is based only on disagreement between communicated
% directional branch outputs and their common 2-D reconstruction.  Thus it
% detects an overconfident but inconsistent branch without using d_true.
    watcherR = reshape(watcherR,2,[]); Nw = size(watcherR,2);
    [baseWeights,scalarVariance] = directionalWLSWeights(eta,watcherR,PthetaAll,cfg);
    Tdir = zeros(2,Nw); scalar = zeros(Nw,1);
    for j=1:Nw
        Tdir(:,j) = transverseDirection(eta,watcherR(:,j));
        scalar(j) = thetaAll(:,j)'*directionalFeatures(eta,Tdir(:,j),cfg);
    end
    weights = baseWeights;
    d = solveDirectionalWLS(Tdir,scalar,weights,cfg);
    standardizedResidual = zeros(Nw,1);
    if cfg.residual.weighting ~= "robust_consistency" || Nw < 3
        return;
    end
    for iter=1:cfg.residual.robustIterations
        residual = scalar-Tdir'*d;
        standardizedResidual = abs(residual)./sqrt(scalarVariance);
        robustFactor = min(1,cfg.residual.robustHuberThreshold ./ ...
            max(standardizedResidual,eps));
        weights = baseWeights.*robustFactor;
        weights = min(weights,cfg.residual.maxWeightRatio*min(weights));
        weights = weights/max(mean(weights),eps);
        d = solveDirectionalWLS(Tdir,scalar,weights,cfg);
    end
end

function d = solveDirectionalWLS(Tdir,scalar,weights,cfg)
    Omega = cfg.residual.wlsRidge*eye(2);
    rhs = zeros(2,1);
    for j=1:numel(weights)
        t = Tdir(:,j);
        Omega = Omega+weights(j)*t*t';
        rhs = rhs+weights(j)*t*scalar(j);
    end
    d = Omega\rhs;
end

function B = directionalWLSJacobian(eta,watcherR,PthetaAll,branch,cfg)
    watcherR = reshape(watcherR,2,[]);
    [weights,~] = directionalWLSWeights(eta,watcherR,PthetaAll,cfg);
    Omega = cfg.residual.wlsRidge*eye(2);
    for j=1:size(watcherR,2)
        t = transverseDirection(eta,watcherR(:,j));
        Omega = Omega+weights(j)*t*t';
    end
    t = transverseDirection(eta,watcherR(:,branch));
    B = Omega\(weights(branch)*t*directionalFeatures(eta,t,cfg)');
end

function [lambdaMin,condition] = directionalGeometryMetrics(eta,watcherR,PthetaAll,cfg)
    watcherR = reshape(watcherR,2,[]);
    [weights,~] = directionalWLSWeights(eta,watcherR,PthetaAll,cfg);
    Omega = cfg.residual.wlsRidge*eye(2);
    for j=1:size(watcherR,2)
        t = transverseDirection(eta,watcherR(:,j)); Omega = Omega+weights(j)*t*t';
    end
    e = eig(.5*(Omega+Omega')); lambdaMin = min(e);
    condition = max(e)/max(lambdaMin,eps);
end

function [weights,scalarVariance] = directionalWLSWeights(eta,watcherR,PthetaAll,cfg)
%DIRECTIONALWLSWEIGHTS Reliability weights for scalar branch predictions.
% With s_i=theta_i' psi_i, the local parameter covariance implies
% var(s_i) = psi_i' P_theta_i psi_i.  The floor and ratio cap keep the
% fusion well-conditioned when an EKF becomes spuriously overconfident.
    watcherR = reshape(watcherR,2,[]); Nw = size(watcherR,2);
    scalarVariance = zeros(Nw,1);
    for j=1:Nw
        t = transverseDirection(eta,watcherR(:,j)); psi = directionalFeatures(eta,t,cfg);
        Pj = .5*(PthetaAll(:,:,j)+PthetaAll(:,:,j)');
        scalarVariance(j) = max(real(psi'*Pj*psi),cfg.residual.scalarVarianceFloor);
    end
    if cfg.residual.weighting == "parameter_covariance"
        weights = 1./scalarVariance;
        weights = min(weights,cfg.residual.maxWeightRatio*min(weights));
        weights = weights/mean(weights);
    else
        weights = ones(Nw,1);
    end
end

function t = transverseDirection(eta,watcherPosition)
%TRANSVERSEDIRECTION Prior-estimated LOS transverse unit direction in 2-D.
    ell = eta(1:2)-watcherPosition;
    ell = ell/max(norm(ell),eps);
    t = [-ell(2);ell(1)];
end

function psi = directionalFeatures(eta,t,cfg)
%DIRECTIONALFEATURES Features for s=t'W*phi with theta=vec(W).
    psi = kron(branchFeatures(eta,cfg),t);
end

function [G,gate] = additiveGate(eta,watcherR,cfg)
%ADDITIVEGATE Fixed geometry-based routing for watcher-local DNN branches.
% alpha_i selects a reliable camera; directional mode also removes the LOS
% component, which an angle-only camera cannot directly observe.
    watcherR = reshape(watcherR,2,[]); Nw = size(watcherR,2);
    los = zeros(2,Nw); score = zeros(Nw,1);
    for i=1:Nw
        rel = eta(1:2)-watcherR(:,i); rho = max(norm(rel),eps);
        los(:,i) = rel/rho;
        boresight = cfg.watchers.boresight(:,i);
        viewAngle = acos(min(1,max(-1,boresight'*los(:,i))));
        fovScore = max(0,1-viewAngle/cfg.gate.halfAngle);
        rangeScore = exp(-0.5*((rho-cfg.gate.rangeDesired)/cfg.gate.rangeSigma)^2);
        score(i) = max(cfg.gate.minimumScore,fovScore*rangeScore);
    end
    alpha = score/sum(score);
    G = zeros(2,2,Nw);
    for i=1:Nw
        if cfg.gate.mode == "directional"
            projector = eye(2)-los(:,i)*los(:,i)';
        else
            projector = eye(2);
        end
        G(:,:,i) = alpha(i)*projector;
    end
    gate = struct('alpha',alpha,'los',los,'score',score,'G',G);
end

function tf = isVectorAdditiveArchitecture(cfg)
    tf = any(cfg.residual.architecture == ["additive_vector" "additive_full_dnn" "partitioned_full_dnn"]);
end

function tf = isFullDnnArchitecture(cfg)
    tf = any(cfg.residual.architecture == ["additive_full_dnn" "partitioned_full_dnn"]);
end

function tf = isPartitionedArchitecture(cfg)
    tf = cfg.residual.architecture == "partitioned_full_dnn";
end

function w = vectorBranchContributionWeight(eta,branch,owner,mode,cfg)
%VECTORBRANCHCONTRIBUTIONWEIGHT Branch coefficient used in d_hat.
% local_only is a single local branch, mean_ensemble averages the same
% branches, and the partitioned architecture assigns state-dependent soft
% ownership.  Therefore the only distinction between mean and partitioned
% shared cases is the structural responsibility map g_j(r).
    switch mode
        case "nominal"
            w = 0;
        case "local_only"
            w = double(branch == owner);
        case "mean_ensemble"
            w = 1/cfg.Nw;
        case "shared_additive"
            if isPartitionedArchitecture(cfg)
                partitionWeights = statePartitionWeights(eta,cfg);
                w = partitionWeights(branch);
            else
                w = 1;
            end
        otherwise
            error('Unknown additive mode "%s".',mode);
    end
end

function weights = statePartitionWeights(eta,cfg)
%STATEPARTITIONWEIGHTS Fixed RBF soft partition g(r), sum_j g_j(r)=1.
% The small uniform component prevents a branch from becoming identically
% inactive, while the RBF component gives each branch a different region.
    if cfg.Nw == 1
        weights = 1;
        return;
    end
    delta = (eta(1:2)-cfg.partition.centers)./cfg.partition.width;
    logWeight = -.5*sum(delta.^2,1)';
    logWeight = logWeight-max(logWeight);
    softWeight = exp(logWeight);
    softWeight = softWeight/max(sum(softWeight),eps);
    floorWeight = cfg.partition.uniformFloor;
    weights = (1-floorWeight)*softWeight+floorWeight/cfg.Nw;
    weights = weights/max(sum(weights),eps);
end

function n = fullDnnParameterCount(cfg)
    n = 0; inputWidth = 4;
    for ell=1:cfg.dnn.hiddenLayerCount
        n = n+cfg.dnn.hiddenWidth*inputWidth+cfg.dnn.hiddenWidth;
        inputWidth = cfg.dnn.hiddenWidth;
    end
    n = n+2*cfg.dnn.hiddenWidth+2;
end

function theta = packFullDnnBranch(cfg,branch,headTheta)
%PACKFULLDNNBRANCH [vec(W1);b1;...;vec(Wo);bo] for one branch.
    theta = zeros(fullDnnParameterCount(cfg),1); idx = 1;
    for ell=1:cfg.dnn.hiddenLayerCount
        W = cfg.dnn.W{ell}(:,:,branch); b = cfg.dnn.b{ell}(:,branch);
        nW = numel(W); theta(idx:idx+nW-1) = W(:); idx = idx+nW;
        theta(idx:idx+numel(b)-1) = b; idx = idx+numel(b);
    end
    theta(idx:idx+2*cfg.nPhi-1) = headTheta; idx = idx+2*cfg.nPhi;
    theta(idx:idx+1) = 0; % output bias
end

function p = unpackFullDnnBranch(theta,cfg)
    p.W = cell(cfg.dnn.hiddenLayerCount,1); p.b = cell(cfg.dnn.hiddenLayerCount,1);
    idx = 1; inputWidth = 4;
    for ell=1:cfg.dnn.hiddenLayerCount
        nW = cfg.dnn.hiddenWidth*inputWidth;
        p.W{ell} = reshape(theta(idx:idx+nW-1),cfg.dnn.hiddenWidth,inputWidth);
        idx = idx+nW;
        p.b{ell} = theta(idx:idx+cfg.dnn.hiddenWidth-1); idx = idx+cfg.dnn.hiddenWidth;
        inputWidth = cfg.dnn.hiddenWidth;
    end
    p.Wout = reshape(theta(idx:idx+2*cfg.nPhi-1),2,cfg.nPhi); idx = idx+2*cfg.nPhi;
    p.bout = theta(idx:idx+1);
end

function d = fullDnnBranchOutput(eta,theta,cfg)
    p = unpackFullDnnBranch(theta,cfg);
    h = eta(1:4)./cfg.dnn.inputScale;
    for ell=1:cfg.dnn.hiddenLayerCount
        h = tanh(p.W{ell}*h+p.b{ell});
    end
    d = p.Wout*h+p.bout;
end

function B = fullDnnParameterJacobian(eta,theta,cfg)
%FULLDNNPARAMETERJACOBIAN Exact backpropagation Jacobian d d_hat/d theta.
% theta contains every hidden-layer weight/bias and the output weight/bias.
    p = unpackFullDnnBranch(theta,cfg); L = cfg.dnn.hiddenLayerCount;
    h = cell(L+1,1); h{1} = eta(1:4)./cfg.dnn.inputScale;
    for ell=1:L
        h{ell+1} = tanh(p.W{ell}*h{ell}+p.b{ell});
    end
    block = cell(L+1,1); Jdh = p.Wout;
    for ell=L:-1:1
        Jda = Jdh*diag(1-h{ell+1}.^2);
        % W is packed columnwise: kron(h_previous',Jda) has precisely the
        % corresponding two-by-(width*inputWidth) column ordering.
        block{ell} = [kron(h{ell}',Jda), Jda];
        Jdh = Jda*p.W{ell};
    end
    B = [block{1:L}, kron(h{L+1}',eye(2)), eye(2)];
end

function d = vectorAdditiveBranchOutput(eta,theta,branch,cfg)
    if isFullDnnArchitecture(cfg)
        d = fullDnnBranchOutput(eta,theta,cfg);
    else
        d = reshape(theta,2,cfg.nPhi)*branchFeatures(eta,cfg,branch);
    end
end

function P0 = fullDnnInitialCovariance(cfg)
% Larger uncertainty is assigned to the output layer, but every parameter
% has nonzero covariance and can therefore be learned from angle-only data.
    q = 3e-5*ones(fullDnnParameterCount(cfg),1); idx = 1; inputWidth = 4;
    for ell=1:cfg.dnn.hiddenLayerCount
        n = cfg.dnn.hiddenWidth*inputWidth+cfg.dnn.hiddenWidth;
        q(idx:idx+n-1) = 2e-4; idx = idx+n; inputWidth = cfg.dnn.hiddenWidth;
    end
    q(idx:idx+2*cfg.nPhi-1) = 3e-3; idx = idx+2*cfg.nPhi;
    q(idx:idx+1) = 1e-3;
    P0 = diag(q.^2);
end

function Qtheta = fullDnnProcessCovariance(cfg)
    q = 1e-10*ones(fullDnnParameterCount(cfg),1); idx = 1; inputWidth = 4;
    for ell=1:cfg.dnn.hiddenLayerCount
        n = cfg.dnn.hiddenWidth*inputWidth+cfg.dnn.hiddenWidth;
        q(idx:idx+n-1) = 3e-9; idx = idx+n; inputWidth = cfg.dnn.hiddenWidth;
    end
    q(idx:idx+2*cfg.nPhi-1) = 2e-8; idx = idx+2*cfg.nPhi;
    q(idx:idx+1) = 1e-8;
    Qtheta = diag(q);
end

function phi = branchFeatures(eta,cfg,branch)
%BRANCHFEATURES Frozen three/four-hidden-layer, width-three tanh backbone.
    if nargin < 3, branch = 1; end
    phi = eta(1:4)./cfg.dnn.inputScale;
    for ell=1:cfg.dnn.hiddenLayerCount
        phi = tanh(cfg.dnn.W{ell}(:,:,branch)*phi+cfg.dnn.b{ell}(:,branch));
    end
end

function [xu,Pu,info] = updateAdditive(x,P,z,rw,cfg)
    rho = x(1:2)-rw; r2 = max(rho'*rho,1);
    H = [-rho(2)/r2 rho(1)/r2 zeros(1,numel(x)-2)];
    nu = wrapAngleAdditive(z-bearingAdditive(x(1:2),rw));
    S = H*P*H'+cfg.meas.sigma^2; K = P*H'/S;
    xu = x+K*nu; Pu = P-K*S*K'; Pu = .5*(Pu+Pu');
    info = struct('nu',nu,'S',S);
end

function J = numericalJacobian(fun,x)
    n = numel(x); y = fun(x); J = zeros(numel(y),n);
    for q=1:n
        h = 1e-6*max(1,abs(x(q))); xp = x; xm = x; xp(q)=xp(q)+h; xm(q)=xm(q)-h;
        J(:,q) = (fun(xp)-fun(xm))/(2*h);
    end
end

function y = bearingAdditive(rt,rw), y = atan2(rt(2)-rw(2),rt(1)-rw(1)); end
function a = wrapAngleAdditive(a), a = mod(a+pi,2*pi)-pi; end

function summary = additiveSummary(out)
    if isPartitionedArchitecture(out.cfg)
        sharedName = "Shared state-partitioned full-DNN branches";
        names = ["Nominal EKF";"Local branch only";"Mean full-DNN ensemble";sharedName];
        results = {out.nominal,out.localOnly,out.meanEnsemble,out.sharedAdditive};
    elseif any(out.cfg.residual.architecture == ["additive_full_dnn" "additive_vector"])
        if out.cfg.residual.architecture == "additive_full_dnn"
            sharedName = "Shared full-DNN additive branches";
            meanName = "Mean full-DNN ensemble";
        else
            sharedName = "Shared additive-vector branches";
            meanName = "Mean additive-vector ensemble";
        end
        names = ["Nominal EKF";"Local branch only";meanName;sharedName];
        results = {out.nominal,out.localOnly,out.meanEnsemble,out.sharedAdditive};
    else
        sharedName = "Shared directional WLS branches";
        names = ["Nominal EKF";"Local branch only";sharedName];
        results = {out.nominal,out.localOnly,out.sharedAdditive};
    end
    n = numel(results);
    p = zeros(n,1); v = p; d = p; pf = p;
    for c=1:n
        r = results{c}; e = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
        p(c) = sqrt(mean(vecnorm(e(1:2,:,:),2,1).^2,'all'));
        v(c) = sqrt(mean(vecnorm(e(3:4,:,:),2,1).^2,'all'));
        de = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
        d(c) = sqrt(mean(vecnorm(de,2,1).^2,'all'));
        last = round(.9*numel(r.time)):numel(r.time);
        pf(c) = sqrt(mean(vecnorm(e(1:2,last,:),2,1).^2,'all'));
    end
    summary = table(names,p,v,d,pf,'VariableNames', ...
        {'caseName','positionRMSE','velocityRMSE','totalResidualRMSE','finalPositionRMSE'});
end

function figs = plotAdditiveToy(out)
    if isPartitionedArchitecture(out.cfg)
        cases = {out.nominal,out.localOnly,out.meanEnsemble,out.sharedAdditive};
        labels = ["nominal" "local branch only" "mean full-DNN ensemble" ...
            "shared state-partitioned full-DNN"];
    elseif any(out.cfg.residual.architecture == ["additive_full_dnn" "additive_vector"])
        cases = {out.nominal,out.localOnly,out.meanEnsemble,out.sharedAdditive};
        if out.cfg.residual.architecture == "additive_full_dnn"
            labels = ["nominal" "local branch only" "mean full-DNN ensemble" "shared full-DNN additive"];
        else
            labels = ["nominal" "local branch only" "mean additive-vector ensemble" "shared additive vector"];
        end
    else
        cases = {out.nominal,out.localOnly,out.sharedAdditive};
        sharedLabel = "shared directional WLS";
        labels = ["nominal" "local branch only" sharedLabel];
    end
    colors = lines(numel(cases)); figs = struct;
    figs.errors = figure('Name','Distributed additive DNN-EKF toy'); tiledlayout(3,1,'TileSpacing','compact');
    names = ["position estimation error" "velocity estimation error" "total residual approximation error"];
    for q=1:3
        nexttile; hold on;
        for c=1:numel(cases)
            r = cases{c};
            if q==1, err = r.xhat(1:2,:,:)-repmat(r.etaTrue(1:2,:),1,1,size(r.xhat,3));
            elseif q==2, err = r.xhat(3:4,:,:)-repmat(r.etaTrue(3:4,:),1,1,size(r.xhat,3));
            else, err = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3)); end
            value = sqrt(mean(vecnorm(err,2,1).^2,3));
            plot(r.time,reshape(value,1,[]),'Color',colors(c,:),'LineWidth',1.2);
        end
        title(names(q)); ylabel('RMSE'); grid on; if q==1, legend(labels,'Location','best'); end
    end
    xlabel('time [s]');
    figs.residual = figure('Name','Shared additive residual'); tiledlayout(2,1,'TileSpacing','compact');
    r = out.sharedAdditive;
    for q=1:2
        nexttile; hold on; plot(r.time,r.dTrue(q,:),'k','LineWidth',1.4,'DisplayName','truth');
        for i=1:size(r.dHat,3), plot(r.time,squeeze(r.dHat(q,:,i)),'LineWidth',.9,'DisplayName',"watcher "+i); end
        title("total residual component "+q); ylabel('m/s^2'); grid on; if q==1, legend('Location','best'); end
    end
    xlabel('time [s]');
    % Fair case-level residual comparison.  Each curve is the mean of the
    % four local copies of that case's total residual estimate.
    figs.caseResidual = figure('Name','Residual approximation by case');
    tiledlayout(2,1,'TileSpacing','compact');
    caseLabels = labels;
    for q=1:2
        nexttile; hold on;
        plot(out.nominal.time,out.nominal.dTrue(q,:),'k','LineWidth',1.5, ...
            'DisplayName','truth');
        for c=1:numel(cases)
            r = cases{c};
            dMean = mean(r.dHat,3);
            plot(r.time,dMean(q,:),'Color',colors(c,:),'LineWidth',1.2, ...
                'DisplayName',caseLabels(c));
        end
        grid on; ylabel('m/s^2'); title("total residual component "+q);
        if q==1, legend('Location','best'); end
    end
    xlabel('time [s]');
    if out.cfg.residual.architecture == "directional_wls"
    figs.directionalDiagnostics = figure('Name','Directional WLS diagnostics');
    tiledlayout(4,1,'TileSpacing','compact'); r = out.sharedAdditive;
    nexttile; plot(r.time,mean(r.geometryLambdaMin,2),'LineWidth',1.2);
    grid on; ylabel('\lambda_{min}(\Omega)'); title('directional reconstruction rank margin');
    nexttile; semilogy(r.time,mean(r.geometryCondition,2),'LineWidth',1.2);
    grid on; ylabel('\kappa(\Omega)'); title('directional reconstruction condition number');
    nexttile; scalarRMSE = squeeze(sqrt(mean(r.directionalScalarError.^2,3)));
    plot(r.time,scalarRMSE,'LineWidth',1.1); grid on; xlabel('time [s]');
    ylabel('RMSE of s_i'); title('watcher directional-scalar approximation error');
    legend(arrayfun(@(i)"watcher "+i,1:size(scalarRMSE,1), ...
        'UniformOutput',false),'Location','best');
    nexttile; branchWeight = squeeze(mean(r.directionalWeights,3));
    plot(r.time,branchWeight,'LineWidth',1.1); grid on; xlabel('time [s]');
    ylabel('normalized w_i'); title('parameter-covariance WLS reliability weights');
    legend(arrayfun(@(i)"watcher "+i,1:size(branchWeight,1), ...
        'UniformOutput',false),'Location','best');
    else
        % Full-vector DNN branches do not define the scalar directional-WLS
        % quantities plotted above.
        figs.directionalDiagnostics = [];
    end
    figs.maneuver = figure('Name','Local observability-aware maneuvers');
    tiledlayout(2,1,'TileSpacing','compact'); r = out.sharedAdditive;
    nexttile; semilogy(r.time,max(r.radialVariance,eps),'LineWidth',1.1);
    grid on; ylabel('u_i^T P_{r,i}u_i');
    title('local radial-position uncertainty');
    legend(arrayfun(@(i)"watcher "+i,1:size(r.radialVariance,2), ...
        'UniformOutput',false),'Location','best');
    nexttile; hold on;
    for i=1:size(r.watcherA,3)
        aNorm = vecnorm(squeeze(r.watcherA(:,:,i)),2,1);
        plot(r.time,aNorm,'LineWidth',1.1,'DisplayName',"watcher "+i);
    end
    grid on; xlabel('time [s]'); ylabel('||a_{w,i}||');
    title(string(r.maneuverObjective)+" candidate-rollout maneuver commands");
    legend('Location','best');
    figs.spiralTrajectory = figure('Name','True target spiral trajectory');
    tiledlayout(1,2,'TileSpacing','compact'); r = out.sharedAdditive;
    nexttile; hold on; axis equal; grid on;
    plot(r.etaTrue(1,:),r.etaTrue(2,:),'k','LineWidth',1.4,'DisplayName','true target');
    circle = linspace(0,2*pi,300); R = out.cfg.spiral.radiusGoal;
    plot(R*cos(circle),R*sin(circle),'k--','DisplayName','r = 100 m');
    xlabel('r_x [m]'); ylabel('r_y [m]'); title('outward spiral target trajectory');
    legend('Location','best');
    nexttile; rho = vecnorm(r.etaTrue(1:2,:),2,1);
    plot(r.time,rho,'k','LineWidth',1.4); hold on; yline(R,'k--'); grid on;
    xlabel('time [s]'); ylabel('||r_t|| [m]'); title('spiral radius');
end
