function [bestDirection,info] = selectObservabilitySeekingDirection( ...
    watcherState,targetInfo,t,cfg)
%SELECTOBSERVABILITYSEEKINGDIRECTION Lightweight bearing-only planner.
% Roll out candidate directions using the local physical target estimate.
% The default score is the predicted terminal radial variance.  When
% cfg.control.obs.jointScoreEnabled is true and local DNN information is
% supplied in targetInfo, a normalized local-parameter score is included.

    dim = cfg.dim;
    if dim ~= 2
        error("The first observability-seeking planner supports 2-D only.");
    end
    etaHat = targetInfo.etaHat(:);
    if numel(etaHat) ~= 2*dim
        error("targetInfo.etaHat must contain position and velocity.");
    end
    if ~isfield(targetInfo,'PEta') || ...
            any(size(targetInfo.PEta)~=[2*dim 2*dim])
        error("targetInfo.PEta must be a physical-state covariance.");
    end

    obs = cfg.control.obs;
    plannerMode = "information_proxy";
    if isfield(obs,"plannerMode")
        plannerMode = string(obs.plannerMode);
    end
    if plannerMode == "los_profile"
        [bestDirection,info] = losProfileDirection(watcherState,targetInfo,t,cfg);
        return;
    end
    if plannerMode == "covariance_rollout" && isfield(targetInfo,"filter")
        [bestDirection,info] = covarianceRolloutDirection( ...
            watcherState,targetInfo,t,cfg);
        return;
    end
    nDirections = obs.numCandidateDirections;
    angles = 2*pi*(0:nDirections-1)/nDirections;
    candidates = [cos(angles);sin(angles)];
    planTimes = obs.planningDt:obs.planningDt:obs.planningHorizon;
    burnRemaining = max(0,obs.startTime+obs.burnDuration-t);
    P0 = 0.5*(targetInfo.PEta+targetInfo.PEta');
    priorInformation = pinv(P0);
    scores = Inf(1,nDirections);
    geometryScores = NaN(1,nDirections);
    parameterScores = NaN(1,nDirections);
    informationMinEig = NaN(1,nDirections);
    informationCondition = NaN(1,nDirections);
    rTarget0 = etaHat(1:dim);
    vTarget0 = etaHat(dim+(1:dim));
    rWatcher0 = watcherState.r;
    vWatcher0 = watcherState.v;
    I = eye(dim);

    jointEnabled = isfield(obs,"jointScoreEnabled") && ...
        logical(obs.jointScoreEnabled) && ...
        isfield(targetInfo,"thetaHat") && isfield(targetInfo,"Ptheta") && ...
        isfield(targetInfo,"branchID");
    if jointEnabled
        thetaHat = targetInfo.thetaHat(:);
        Ptheta0 = 0.5*(targetInfo.Ptheta+targetInfo.Ptheta');
        Ptheta0 = projectPSD_local(Ptheta0);
        thetaPriorTrace = max(trace(Ptheta0),eps);
        wGeometry = getObsNumeric(obs,"geometryWeight",0.7);
        wTheta = getObsNumeric(obs,"parameterWeight",0.3);
        wSum = max(wGeometry+wTheta,eps);
        wGeometry = wGeometry/wSum;
        wTheta = wTheta/wSum;
    else
        thetaHat = [];
        Ptheta0 = [];
        thetaPriorTrace = NaN;
        wGeometry = 1.0;
        wTheta = 0.0;
    end

    % The local branch Jacobian is evaluated once at the current local
    % estimate.  Reusing this lightweight proxy across the planning horizon
    % avoids repeatedly evaluating the MLP while retaining the candidate-
    % dependent bearing geometry in the information score.
    JthetaPlan = cell(1,numel(planTimes));
    if jointEnabled
        [~,~,JthetaNow] = evaluateBranchResidualModel( ...
            targetInfo.branchID,etaHat,thetaHat,cfg);
        for ip = 1:numel(planTimes)
            JthetaPlan{ip} = JthetaNow;
        end
    end
    
    for ic = 1:nDirections
        direction = candidates(:,ic);
        G = zeros(2*dim);
        terminalLOS = zeros(dim,1);
        thetaInformation = zeros(numel(thetaHat));
        for ip = 1:numel(planTimes)
            tau = planTimes(ip);
            thrustTime = min(tau,burnRemaining);
            coastTime = max(0,tau-burnRemaining);
            deltaV = obs.acceleration*thrustTime*direction;
            deltaR = 0.5*obs.acceleration*thrustTime^2*direction+ ...
                deltaV*coastTime;
            rWatcher = rWatcher0+vWatcher0*tau+deltaR;
            rTarget = rTarget0+vTarget0*tau;
            relative = rTarget-rWatcher;
            range2 = max(relative'*relative,cfg.gate.minRange^2);
            Hpos = [-relative(2),relative(1)]/range2;
            H = [Hpos,zeros(1,dim)];
            Phi = [I,tau*I;zeros(dim),I];
            G = G+Phi'*(H'*(1/cfg.meas.R)*H)*Phi* ...
                obs.planningDt;

            % Local-branch parameter-information proxy.  The branch
            % acceleration sensitivity is mapped through a constant-
            % acceleration position/velocity transition to the predicted
            % bearing.  This is intentionally lightweight; the full
            % augmented sensitivity rollout can be added later.
            if jointEnabled
                Jtheta = JthetaPlan{ip};
                GammaTheta = [0.5*tau^2*I;tau*I]*Jtheta;
                JmeasTheta = H*GammaTheta;
                thetaInformation = thetaInformation + ...
                    (JmeasTheta'*JmeasTheta/max(cfg.meas.R,eps))* ...
                    obs.planningDt;
            end
            terminalLOS = relative/sqrt(range2);
        end
        posterior = pinv(priorInformation+G);
        Prr = posterior(1:dim,1:dim);
        geometryScore = terminalLOS'*Prr*terminalLOS;
        if jointEnabled
            thetaPosterior = pinv(pinv(Ptheta0)+thetaInformation);
            thetaScore = trace(thetaPosterior)/thetaPriorTrace;
            geometryScoreNormalized = geometryScore/max(trace(Prr),eps);
            scores(ic) = wGeometry*geometryScoreNormalized + ...
                wTheta*thetaScore;
        else
            thetaScore = NaN;
            geometryScoreNormalized = geometryScore;
            scores(ic) = geometryScore;
        end
        geometryScores(ic) = geometryScore;
        parameterScores(ic) = thetaScore;
        Gsym = 0.5*(G+G');
        lambdaG = real(eig(Gsym));
        lambdaG = max(lambdaG,0);
        informationMinEig(ic) = min(lambdaG);
        informationCondition(ic) = max(lambdaG) / ...
            max(informationMinEig(ic),eps);
    end

    [score,candidateIndex] = min(scores);
    bestDirection = candidates(:,candidateIndex);
    selectedGeometryScore = geometryScores(candidateIndex);
    selectedParameterScore = parameterScores(candidateIndex);
    info = struct('score',score,'candidateIndex',candidateIndex, ...
        'candidateScores',scores, ...
        'candidateGeometryScores',geometryScores, ...
        'candidateParameterScores',parameterScores, ...
        'selectedGeometryScore',selectedGeometryScore, ...
        'selectedParameterScore',selectedParameterScore, ...
        'jointScoreEnabled',jointEnabled, ...
        'geometryWeight',wGeometry,'parameterWeight',wTheta, ...
        'candidateInformationMinEig',informationMinEig, ...
        'candidateInformationCondition',informationCondition, ...
        'selectedInformationMinEig',informationMinEig(candidateIndex), ...
        'selectedInformationCondition',informationCondition(candidateIndex), ...
        'planningTime',t);
end

function [bestDirection,info] = losProfileDirection( ...
    watcherState,targetInfo,t,cfg)
%LOS_PROFILE_DIRECTION Local calibrated-thrust observability planner.
% The score is the predicted, noise-normalized change between the natural
% LOS profile and the LOS profile produced by this watcher's known thrust.
% It implements the sufficient transverse-displacement idea in
% Woffinden and Geller (2009), Eq. (22), under this project's
% double-integrator watcher model.

    dim = cfg.dim;
    if dim ~= 2
        error("los_profile planner currently supports 2-D only.");
    end
    obs = cfg.control.obs;
    nDirections = max(1,round(obs.numCandidateDirections));
    angles = 2*pi*(0:nDirections-1)/nDirections;
    candidates = [cos(angles);sin(angles)];
    nSteps = max(1,round(getObsNumeric(obs,"rolloutMaxSteps",24)));
    horizon = max(getObsNumeric(obs,"planningHorizon",30),eps);
    dtPlan = horizon/nSteps;
    burnDuration = max(getObsNumeric(obs,"burnDuration", ...
        getObsNumeric(obs,"impulseDuration",0)),0);
    rho0 = targetInfo.etaHat(1:dim)-watcherState.r;
    rhoDot = targetInfo.etaHat(dim+(1:dim))-watcherState.v;
    sigmaBearing = sqrt(max(bearingVarianceForPlanner(cfg),eps));

    scores = Inf(1,nDirections);
    detectability = zeros(1,nDirections);
    peakAngle = zeros(1,nDirections);
    for ic = 1:nDirections
        direction = candidates(:,ic);
        informationScore = 0;
        peak = 0;
        for ip = 1:nSteps
            tau = ip*dtPlan;
            thrustTime = min(tau,burnDuration);
            coastTime = max(0,tau-burnDuration);
            % Known watcher command induces this displacement relative to
            % a no-thrust watcher.  Relative target-to-watcher displacement
            % has the opposite sign.
            deltaWatcher = 0.5*obs.acceleration*thrustTime^2*direction + ...
                obs.acceleration*thrustTime*coastTime*direction;
            deltaRho = -deltaWatcher;
            rhoNatural = rho0 + rhoDot*tau;
            angleChange = abs(cross2d(rhoNatural,deltaRho))/ ...
                max(rhoNatural.'*rhoNatural,cfg.gate.minRange^2);
            informationScore = informationScore + ...
                (angleChange/sigmaBearing)^2*dtPlan;
            peak = max(peak,angleChange);
        end
        detectability(ic) = informationScore;
        peakAngle(ic) = peak;
        scores(ic) = -informationScore; % planner selects minimum score
    end
    [score,candidateIndex] = min(scores);
    bestDirection = candidates(:,candidateIndex);
    info = struct('score',score,'candidateIndex',candidateIndex, ...
        'candidateScores',scores,'candidateGeometryScores',detectability, ...
        'candidateParameterScores',NaN(1,nDirections), ...
        'selectedGeometryScore',detectability(candidateIndex), ...
        'selectedParameterScore',NaN,'jointScoreEnabled',false, ...
        'geometryWeight',1.0,'parameterWeight',0.0, ...
        'candidateInformationMinEig',NaN(1,nDirections), ...
        'candidateInformationCondition',NaN(1,nDirections), ...
        'selectedInformationMinEig',NaN, ...
        'selectedInformationCondition',NaN, ...
        'planningTime',t,'plannerMode',"los_profile", ...
        'expectedPeakLOSChangeRad',peakAngle(candidateIndex), ...
        'expectedPeakLOSChangeSigma',peakAngle(candidateIndex)/sigmaBearing);
end

function value = cross2d(a,b)
    value = a(1)*b(2)-a(2)*b(1);
end

function variance = bearingVarianceForPlanner(cfg)
    if isscalar(cfg.meas.R)
        variance = cfg.meas.R;
    else
        variance = trace(cfg.meas.R)/size(cfg.meas.R,1);
    end
end

function [bestDirection,info] = covarianceRolloutDirection( ...
    watcherState,targetInfo,t,cfg)
% Evaluate each candidate using the same augmented DNN-EKF covariance
% prediction and bearing update used online.  Synthetic measurements equal
% their predicted values, so this is a covariance-only look-ahead and does
% not inject truth or random measurement noise.

    dim = cfg.dim;
    obs = cfg.control.obs;
    nDirections = obs.numCandidateDirections;
    angles = 2*pi*(0:nDirections-1)/nDirections;
    candidates = [cos(angles);sin(angles)];
    nSteps = max(1,round(getObsNumeric(obs,"rolloutMaxSteps",12)));
    horizon = max(getObsNumeric(obs,"planningHorizon",10),eps);
    dtPlan = horizon/nSteps;
    cfgRoll = cfg;
    % The virtual filter must use the same interval as the candidate state
    % propagation.  Otherwise the watcher moves, for example, 10 seconds
    % while the covariance is predicted by only one 0.5-second EKF step.
    cfgRoll.dt = dtPlan;
    burnDuration = max(getObsNumeric(obs,"burnDuration", ...
        getObsNumeric(obs,"impulseDuration",0)),0);

    scores = Inf(1,nDirections);
    geometryScores = Inf(1,nDirections);
    parameterScores = NaN(1,nDirections);
    informationMinEig = NaN(1,nDirections);
    informationCondition = NaN(1,nDirections);
    Ptheta0 = targetInfo.filter.P(targetInfo.filter.idxTheta, ...
        targetInfo.filter.idxTheta);
    thetaPriorTrace = max(trace(Ptheta0),eps);

    for ic = 1:nDirections
        direction = candidates(:,ic);
        filterPlan = targetInfo.filter;
        % A virtual horizon must not retune the live covariance-matching
        % state from artificial zero innovations.
        if isfield(filterPlan,"cm")
            filterPlan.cm.adaptThetaEnabled = false;
            filterPlan.cm.adaptEpsilonEnabled = false;
        end
        watcherPlan = watcherState;
        elapsed = 0;
        for ip = 1:nSteps
            thrustTime = min(dtPlan,max(burnDuration-elapsed,0));
            a = obs.acceleration*direction;
            watcherPlan.r = watcherPlan.r + watcherPlan.v*dtPlan + ...
                0.5*a*thrustTime^2 + a*thrustTime*(dtPlan-thrustTime);
            watcherPlan.v = watcherPlan.v + a*thrustTime;
            filterPlan = DNN_EKF_Predict_Local(filterPlan,t+elapsed,cfgRoll);
            etaPred = filterPlan.xhat(filterPlan.idxEta);
            zVirtual = measurementPrediction(etaPred,watcherPlan,cfgRoll);
            filterPlan = DNN_EKF_Update_Local(filterPlan,zVirtual, ...
                watcherPlan,cfgRoll);
            elapsed = elapsed + dtPlan;
        end
        Peta = filterPlan.P(filterPlan.idxEta,filterPlan.idxEta);
        etaTerminal = filterPlan.xhat(filterPlan.idxEta);
        relative = etaTerminal(1:dim)-watcherPlan.r;
        eRad = relative/max(norm(relative),cfg.gate.minRange);
        geometryScores(ic) = eRad.'*Peta(1:dim,1:dim)*eRad;
        scores(ic) = geometryScores(ic);
        Ptheta = filterPlan.P(filterPlan.idxTheta,filterPlan.idxTheta);
        parameterScores(ic) = trace(Ptheta)/thetaPriorTrace;
        information = pinv(Peta);
        lambda = max(real(eig(0.5*(information+information.'))),0);
        informationMinEig(ic) = min(lambda);
        informationCondition(ic) = max(lambda)/max(min(lambda),eps);
    end
    [score,candidateIndex] = min(scores);
    bestDirection = candidates(:,candidateIndex);
    info = struct('score',score,'candidateIndex',candidateIndex, ...
        'candidateScores',scores,'candidateGeometryScores',geometryScores, ...
        'candidateParameterScores',parameterScores, ...
        'selectedGeometryScore',geometryScores(candidateIndex), ...
        'selectedParameterScore',parameterScores(candidateIndex), ...
        'jointScoreEnabled',false,'geometryWeight',1.0,'parameterWeight',0.0, ...
        'candidateInformationMinEig',informationMinEig, ...
        'candidateInformationCondition',informationCondition, ...
        'selectedInformationMinEig',informationMinEig(candidateIndex), ...
        'selectedInformationCondition',informationCondition(candidateIndex), ...
        'planningTime',t,'plannerMode',"covariance_rollout", ...
        'rolloutDt',dtPlan,'rolloutSteps',nSteps);
end

function value = getObsNumeric(obs,name,defaultValue)
if isfield(obs,name) && isfinite(obs.(name))
    value = double(obs.(name));
else
    value = defaultValue;
end
end

function P = projectPSD_local(P)
P = 0.5*(P+P');
[V,D] = eig(P);
d = real(diag(D));
d(d<0) = 0;
P = V*diag(d)*V';
P = 0.5*(P+P');
end
