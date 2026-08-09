function out = run_toy_acceleration_sharing_observability(seed,T,makePlots,shareMode)
%RUN_TOY_ACCELERATION_SHARING_OBSERVABILITY Small angles-only toy problem.
% Four watchers each run a local EKF for x=[r_x r_y v_x v_y a_x a_y]'.
% They keep local position/velocity estimates.  The baseline shares an
% acceleration consensus after every bearing update; the optional
% "windowed_likelihood" mode instead shares only a Schur-complemented
% acceleration likelihood after a local bearing window.  In the active
% case, each watcher triggers from its own LOS/radial position covariance,
% then chooses its own maneuver direction by a finite-horizon local EKF
% covariance rollout.  Neither controller uses another watcher's LOS,
% measurement, or covariance.

    clc
    % clear
    close all

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(makePlots), makePlots = true; end
    addpath(genpath(pwd));
    cfg = toyConfig(seed,T);
    if nargin >= 4 && ~isempty(shareMode), cfg.share.mode = string(shareMode); end
    fprintf('Toy acceleration-sharing angles-only EKF: seed=%d, T=%.0f s, dt=%.2f s, fusion=%s\n', ...
        seed,T,cfg.dt,cfg.share.mode);
    cfgNoShare = cfg; cfgNoShare.share.acceleration = false;
    rng(seed);
    out.localCoast = simulateToy(cfgNoShare,false);
    rng(seed);
    out.localActive = simulateToy(cfgNoShare,true);
    rng(seed);
    out.coast = simulateToy(cfg,false);
    rng(seed);
    out.active = simulateToy(cfg,true);
    out.cfg = cfg;
    out.summary = summarizeToy(out);
    out.covarianceMatching = summarizeCovarianceMatching(out.active,cfg);
    disp(out.summary);
    if cfg.adaptive.enabled
        cm = out.covarianceMatching;
        fprintf(['Active shared case covariance matching (t >= %.1f s): ' ...
            'median ratio %.3f, log-ratio RMSE %.3f, mean NIS %.3f, ' ...
            'median gammaQ %.3g\n'], ...
            cfg.adaptive.burnIn,cm.medianRatio,cm.logRatioRMSE, ...
            cm.meanNIS,cm.medianGammaQ);
    end
    if makePlots
        out.figures = plot_toy_acceleration_sharing_observability(out,true);
    end
end

function cfg = toyConfig(seed,T)
    cfg.seed = seed; cfg.T = T; cfg.dt = 0.5; cfg.time = 0:cfg.dt:T;
    cfg.N = numel(cfg.time); cfg.Nw = 4; cfg.dim = 2;
    cfg.target.r0 = [500;0]; cfg.target.v0 = [0;0.2];
    cfg.watchers.r0 = 1000*[1 0 -1 0;0 1 0 -1];
    cfg.watchers.v0 = repmat(cfg.target.v0,1,cfg.Nw);
    cfg.meas.sigma = deg2rad(0.02);
    cfg.ekf.P0 = diag([50^2 50^2 .1^2 .1^2 (3e-4)^2 (3e-4)^2]);
    cfg.ekf.qJerk = 2e-12;
    % Innovation covariance matching for the *process* model.  Bearing R
    % remains the known sensor specification; adapting Q and R together
    % from one scalar innovation would not be identifiable.
    cfg.adaptive.enabled = true;
    % Do not let the deliberately large initial state error be interpreted
    % as process noise.  Adapt only after the initial angle-only transient.
    cfg.adaptive.burnIn = 100;
    cfg.adaptive.ewmaGain = 0.02;
    cfg.adaptive.logGain = 0.01;
    cfg.adaptive.gammaBounds = [0.1, 100];
    % The toy residual output is d_toy(xi)=a.  Keep an explicit output
    % covariance floor so the maneuver cost has the same interface as the
    % later DNN-EKF residual-output covariance.
    cfg.model.outputCovFloor = (1e-8)^2*eye(2);
    cfg.share.acceleration = true;
    % A common inertial acceleration prior is broadcast after each fusion
    % epoch.  Watchers return only new, directional acceleration information
    % relative to that prior; they do not average full posteriors.
    % Keep the verified conservative baseline as default.  The two
    % directional modes below remain available for controlled ablations;
    % geometry alone is not a calibrated acceleration likelihood.
    cfg.share.mode = "mean_consensus";
    cfg.share.nisGate = 6.63;              % scalar 99 % chi-square gate
    cfg.share.relativeEigenFloor = 1e-8;   % retain every resolved direction
    cfg.share.minInformationTrace = 1e-12;
    cfg.share.geometryHorizon = 120;
    cfg.share.geometryDt = 10;
    cfg.share.minGeometryCondition = 0.05;
    cfg.share.fallbackCovInflation = 1.25;
    % A likelihood window is the unit of communication for the rigorous
    % directional mode.  Within this interval local filters remain local;
    % at its end each sends only an acceleration likelihood increment.
    cfg.share.windowDuration = 40;
    cfg.share.minWindowSamples = 20;
    cfg.share.minNuisanceRcond = 1e-9;
    cfg.share.minDirectionalEigenvalue = 1e-8;
    % Each watcher owns this controller state.  There is no common event
    % manager and no exchange of LOS vectors or observability matrices.
    % "rollout_position" is deliberately local: radial covariance decides
    % when to maneuver, while a virtual-measurement rollout decides where.
    % "position_only" retains the former fixed-transverse baseline.
    cfg.control.mode = "rollout_position";
    cfg.control.decisionDt = 5; cfg.control.firstDecisionTime = 20;
    cfg.control.cooldown = 100; cfg.control.halfBurnDuration = 15;
    % A pulse pair leaves a*h^2 of transverse displacement.  At 1 km,
    % 8e-4 m/s^2 would yield only 0.18 m (about 0.05 bearing-sigma), so use
    % a 4.5 m geometry-changing displacement in this deliberately scaled toy.
    cfg.control.acceleration = 2e-2;
    cfg.control.horizon = 180; cfg.control.planDt = 10;
    cfg.control.triggerFraction = 0.99;
    cfg.control.rearmFraction = 0.995;
    cfg.control.positionTriggerRatio = 1.10;
    cfg.control.positionRearmRatio = 1.02;
    cfg.control.positionGrowthTriggerRatio = 1.50;
    cfg.control.minCostReduction = 0.02;
    cfg.control.stateScale = diag([1000 1000 .1 .1 1e-4 1e-4]);
    cfg.control.costWeights = [1.0 0.5 0.5 0.05]; % r, v, shared-a, displacement
end

function res = simulateToy(cfg,active)
    nX = 6; N = cfg.N; Nw = cfg.Nw; dt = cfg.dt;
    etaTrue = zeros(nX,N); etaTrue(:,1) = [cfg.target.r0;cfg.target.v0;truthAcceleration(0)];
    watcherR = zeros(2,N,Nw); watcherV = zeros(2,N,Nw); watcherU = zeros(2,N,Nw);
    xhat = zeros(nX,N,Nw); P = zeros(nX,nX,Nw); action = false(N,Nw);
    eventStart = nan(Nw,1); lastEventEnd = -Inf(Nw,1); direction = zeros(2,Nw);
    referenceMetric = nan(Nw,1); nextDecision = cfg.control.firstDecisionTime*ones(Nw,1);
    localMetric = nan(N,Nw); localReference = nan(N,Nw); triggerLog = false(N,Nw);
    localTriggerThreshold = nan(N,Nw);
    localCost = nan(N,Nw); controllerArmed = true(N,Nw);
    lastTriggerMetric = nan(Nw,1);
    eventCount = zeros(Nw,1);
    NIS = nan(N,Nw); Smodel = nan(N,Nw); Sempirical = nan(N,Nw);
    cmRatio = nan(N,Nw); gammaQ = ones(N,Nw); empiricalS = nan(Nw,1);
    commonA = zeros(2,1); commonPa = cfg.ekf.P0(5:6,5:6);
    fusedA = nan(2,N); fusedPa = nan(2,2,N);
    fusionInformation = nan(2,2,N); fusionContribution = nan(N,Nw);
    fusionIncrement = nan(2,2,N,Nw);
    likelihoodWindow = initializeLikelihoodWindows(Nw);
    windowSamples = max(1,round(cfg.share.windowDuration/dt));
    fusedA(:,1) = commonA; fusedPa(:,:,1) = commonPa;
    for i = 1:Nw
        watcherR(:,1,i) = cfg.watchers.r0(:,i); watcherV(:,1,i) = cfg.watchers.v0(:,i);
        xhat(:,1,i) = [cfg.target.r0 + [35;-25]; cfg.target.v0 + [.04;-.03]; zeros(2,1)];
        P(:,:,i) = cfg.ekf.P0;
    end
    for k = 1:N-1
        t = cfg.time(k);
        etaTrue(:,k+1) = propagateTruth(etaTrue(:,k),t,dt);
        if cfg.share.acceleration
            % Reference prior shared by every local branch for this fusion
            % interval.  The average adaptive Q is only used to propagate
            % this common acceleration prior; r,v remain strictly local.
            gammaCommon = mean(gammaQ(k,:),"omitnan");
            commonPriorA = commonA;
            commonPriorPa = commonPa + gammaCommon*cfg.ekf.qJerk*dt*eye(2);
        end
        for i = 1:Nw
            eventActive = active && isfinite(eventStart(i)) && ...
                t < eventStart(i)+2*cfg.control.halfBurnDuration;
            if active && t >= nextDecision(i)-eps && ~eventActive
                if cfg.control.mode == "position_only"
                    candidateDirection = localTransverseDirection(xhat(:,k,i),watcherR(:,k,i));
                    metricNow = localRadialPositionVariance(xhat(:,k,i),P(:,:,i),watcherR(:,k,i));
                    plan = struct('coastMetric',metricNow,'selectedCost',NaN, ...
                        'coastCost',NaN,'beneficial',true);
                elseif cfg.control.mode == "rollout_position"
                    [candidateDirection,plan] = chooseLocalDirection( ...
                        xhat(:,k,i),P(:,:,i),watcherR(:,k,i),watcherV(:,k,i),cfg);
                    % The decision trigger is a directly interpretable
                    % position metric; the rollout only selects direction.
                    plan.coastMetric = localRadialPositionVariance( ...
                        xhat(:,k,i),P(:,:,i),watcherR(:,k,i));
                else
                    [candidateDirection,plan] = chooseLocalDirection( ...
                        xhat(:,k,i),P(:,:,i),watcherR(:,k,i),watcherV(:,k,i),cfg);
                end
                localMetric(k,i) = plan.coastMetric;
                localCost(k,i) = plan.selectedCost;
                if ~isfinite(referenceMetric(i))
                    referenceMetric(i) = plan.coastMetric;
                end
                % A watcher must first recover its own local observability
                % before it can be re-armed for another maneuver episode.
                if cfg.control.mode == "position_only" || cfg.control.mode == "rollout_position"
                    initialThreshold = cfg.control.positionTriggerRatio*referenceMetric(i);
                    if isfinite(lastTriggerMetric(i))
                        growthThreshold = cfg.control.positionGrowthTriggerRatio*lastTriggerMetric(i);
                        triggerThreshold = max(initialThreshold,growthThreshold);
                    else
                        triggerThreshold = initialThreshold;
                    end
                    % Do not require an unrealistic return to the initial
                    % covariance.  After cooldown, fire only when local
                    % radial uncertainty has grown relative to its last
                    % maneuver level.
                    weak = plan.coastMetric > triggerThreshold;
                    beneficial = plan.beneficial;
                    eligible = t >= lastEventEnd(i)+cfg.control.cooldown;
                else
                    triggerThreshold = cfg.control.triggerFraction*referenceMetric(i);
                    if ~controllerArmed(i) && plan.coastMetric >= ...
                            cfg.control.rearmFraction*referenceMetric(i)
                        controllerArmed(i) = true;
                    end
                    weak = plan.coastMetric < triggerThreshold;
                    beneficial = plan.selectedCost <= ...
                        (1-cfg.control.minCostReduction)*plan.coastCost;
                    eligible = controllerArmed(i) && t >= lastEventEnd(i)+cfg.control.cooldown;
                end
                localTriggerThreshold(k,i) = triggerThreshold/referenceMetric(i);
                if weak && eligible && beneficial
                    direction(:,i) = candidateDirection;
                    eventStart(i) = t; eventCount(i) = eventCount(i)+1;
                    triggerLog(k,i) = true;
                    controllerArmed(i) = false;
                    lastTriggerMetric(i) = plan.coastMetric;
                end
                nextDecision(i) = t + cfg.control.decisionDt;
                eventActive = isfinite(eventStart(i)) && ...
                    t < eventStart(i)+2*cfg.control.halfBurnDuration;
            end
            localReference(k,i) = referenceMetric(i);
            if eventActive
                elapsed = t-eventStart(i);
                phase = 1;
                if elapsed >= cfg.control.halfBurnDuration, phase = -1; end
                watcherU(:,k,i) = phase*cfg.control.acceleration*direction(:,i);
                action(k,i) = true;
                if elapsed+dt >= 2*cfg.control.halfBurnDuration
                    lastEventEnd(i) = eventStart(i)+2*cfg.control.halfBurnDuration;
                end
            end
            watcherR(:,k+1,i) = watcherR(:,k,i) + watcherV(:,k,i)*dt + .5*watcherU(:,k,i)*dt^2;
            watcherV(:,k+1,i) = watcherV(:,k,i) + watcherU(:,k,i)*dt;
            [xp,Pp] = predictToy(xhat(:,k,i),P(:,:,i),cfg,gammaQ(k,i));
            z = bearing(etaTrue(1:2,k+1),watcherR(:,k+1,i)) + cfg.meas.sigma*randn;
            [xhat(:,k+1,i),P(:,:,i),innovation] = updateToy( ...
                xp,Pp,z,watcherR(:,k+1,i),cfg);
            NIS(k+1,i) = innovation.nu^2/innovation.S;
            Smodel(k+1,i) = innovation.S;
            if t+dt < cfg.adaptive.burnIn
                % Reinitialize the matching statistic during burn-in rather
                % than carrying the initial acquisition transient forward.
                empiricalS(i) = innovation.S;
            elseif ~isfinite(empiricalS(i))
                empiricalS(i) = innovation.S;
            else
                beta = cfg.adaptive.ewmaGain;
                empiricalS(i) = (1-beta)*empiricalS(i) + beta*innovation.nu^2;
            end
            Sempirical(k+1,i) = empiricalS(i);
            cmRatio(k+1,i) = empiricalS(i)/max(innovation.S,eps);
            gammaQ(k+1,i) = updateProcessNoiseMultiplier( ...
                gammaQ(k,i),cmRatio(k+1,i),t+dt,cfg);
            if cfg.share.acceleration && cfg.share.mode == "windowed_likelihood"
                likelihoodWindow(i) = accumulateAccelerationLikelihood( ...
                    likelihoodWindow(i),xhat(:,k+1,i),z,watcherR(:,k+1,i),cfg);
            end
        end
        if cfg.share.acceleration
            doWindowFusion = cfg.share.mode == "windowed_likelihood" && ...
                mod(k,windowSamples) == 0;
            if cfg.share.mode ~= "windowed_likelihood" || doWindowFusion
                [xhat(:,k+1,:),P,fusion] = shareAcceleration( ...
                    xhat(:,k+1,:),P,commonPriorA,commonPriorPa,NIS(k+1,:), ...
                    squeeze(watcherR(:,k+1,:)),squeeze(watcherV(:,k+1,:)),cfg, ...
                    likelihoodWindow);
                commonA = fusion.a; commonPa = fusion.P;
                fusionInformation(:,:,k+1) = fusion.information;
                fusionContribution(k+1,:) = fusion.contribution;
                fusionIncrement(:,:,k+1,:) = fusion.increment;
                if cfg.share.mode == "windowed_likelihood"
                    likelihoodWindow = initializeLikelihoodWindows(Nw);
                end
            end
            fusedA(:,k+1) = commonA; fusedPa(:,:,k+1) = commonPa;
        end
    end
    res = struct('time',cfg.time,'etaTrue',etaTrue,'watcherR',watcherR, ...
        'watcherV',watcherV,'watcherU',watcherU,'xhat',xhat,'P',P,'action',action, ...
        'localMetric',localMetric,'localReference',localReference, ...
        'localTriggerThreshold',localTriggerThreshold, ...
        'localCost',localCost,'triggerLog',triggerLog, ...
        'controllerArmed',controllerArmed,'lastTriggerMetric',lastTriggerMetric, ...
        'eventCount',eventCount,'NIS',NIS,'Smodel',Smodel, ...
        'Sempirical',Sempirical,'cmRatio',cmRatio,'gammaQ',gammaQ);
    if cfg.share.acceleration
        res.fusedAcceleration = fusedA;
        res.fusedAccelerationCovariance = fusedPa;
        res.fusionInformation = fusionInformation;
        res.fusionContribution = fusionContribution;
        res.fusionIncrement = fusionIncrement;
    end
end

function xNext = propagateTruth(x,t,dt)
    a0 = truthAcceleration(t); a1 = truthAcceleration(t+dt);
    xNext = x;
    xNext(1:2) = x(1:2) + x(3:4)*dt + .5*a0*dt^2;
    xNext(3:4) = x(3:4) + .5*(a0+a1)*dt;
    xNext(5:6) = a1;
end

function a = truthAcceleration(t)
% Bounded unknown target acceleration; not supplied to any filter.
% It is deliberately slowly varying so [r,v,a] is a meaningful local
% augmented state over a maneuver-planning window.
    a = [1.4e-4 + 2.5e-5*sin(2*pi*t/900); ...
        -1.0e-4 + 2.0e-5*cos(2*pi*t/760)];
end

function [xp,Pp] = predictToy(x,P,cfg,gammaQ)
    if nargin < 4 || isempty(gammaQ), gammaQ = 1; end
    dt = cfg.dt; I = eye(2); Z = zeros(2);
    F = [I dt*I .5*dt^2*I; Z I dt*I; Z Z I];
    q = cfg.ekf.qJerk;
    Q1 = q*[dt^5/20 dt^4/8 dt^3/6; dt^4/8 dt^3/3 dt^2/2; dt^3/6 dt^2/2 dt];
    Q = gammaQ*kron(Q1,I);
    xp = F*x; Pp = F*P*F' + Q; Pp = .5*(Pp+Pp');
end

function [xu,Pu,info] = updateToy(x,P,z,rw,cfg)
    rho = x(1:2)-rw; r2 = max(rho.'*rho,1);
    h = atan2(rho(2),rho(1));
    H = [-rho(2)/r2 rho(1)/r2 0 0 0 0];
    nu = wrapAngle(z-h); S = H*P*H' + cfg.meas.sigma^2;
    K = (P*H')/S; xu = x + K*nu; Pu = P-K*S*K'; Pu = .5*(Pu+Pu');
    info = struct('nu',nu,'S',S,'H',H);
end

function gammaNext = updateProcessNoiseMultiplier(gammaNow,ratio,t,cfg)
% Match E[nu^2] to S with a bounded, log-domain Q multiplier.  A positive
% ratio error makes future process covariance more mobile; a negative one
% makes it less mobile.  The update is deliberately slow because bearing
% innovations are noisy scalar samples.
    gammaNext = gammaNow;
    if ~cfg.adaptive.enabled || t < cfg.adaptive.burnIn || ~isfinite(ratio)
        return
    end
    lo = cfg.adaptive.gammaBounds(1); hi = cfg.adaptive.gammaBounds(2);
    logGamma = log(gammaNow) + cfg.adaptive.logGain*log(max(ratio,1e-8));
    gammaNext = min(max(exp(logGamma),lo),hi);
end

function [x,P,fusion] = shareAcceleration(x,P,aPrior,PaPrior,nis,rw,vw,cfg,window)
% Directional incremental-information fusion in the inertial frame.
% Every local posterior contains the common acceleration prior broadcast at
% the previous epoch.  We therefore fuse only its new canonical increment
% DeltaY_i = Y_i^+ - Y_c^-, not the full local posterior information.
% DeltaY_i is generally anisotropic: its eigenvectors identify the inertial
% acceleration directions that watcher i resolved during this interval.
% Local r,v states and bearing data are never transmitted.
    if nargin < 9, window = []; end
    if cfg.share.mode == "windowed_likelihood"
        [x,P,fusion] = windowedLikelihoodAccelerationFusion(x,P,aPrior,PaPrior,window,cfg);
        return
    elseif cfg.share.mode == "geometry_weighted_stateless"
        [x,P,fusion] = geometryWeightedAccelerationFusion( ...
            x,P,aPrior,PaPrior,nis,rw,vw,cfg);
        return
    elseif cfg.share.mode == "mean_consensus"
        [x,P,fusion] = meanConsensusAccelerationFusion(x,P);
        return
    end
    Nw = size(x,3); I2 = eye(2);
    Yprior = inverseSPD(PaPrior); xiPrior = Yprior*aPrior;
    Yfused = Yprior; xifused = xiPrior;
    contribution = zeros(1,Nw);
    increment = zeros(2,2,Nw);
    for i = 1:Nw
        Pai = .5*(P(5:6,5:6,i)+P(5:6,5:6,i)') + 1e-16*I2;
        Yi = inverseSPD(Pai);
        deltaY = .5*((Yi-Yprior)+(Yi-Yprior)');
        % Numerical/nonlinear EKF effects can create a tiny negative
        % increment.  A likelihood cannot have negative information, so
        % retain only the positive semidefinite directional content.
        [U,D] = eig(deltaY); lambda = max(real(diag(D)),0);
        if max(lambda) > 0
            lambda(lambda < cfg.share.relativeEigenFloor*max(lambda)) = 0;
        end
        deltaY = U*diag(lambda)*U';
        if trace(deltaY) <= cfg.share.minInformationTrace
            continue
        end
        % A locally inconsistent bearing should not contaminate either
        % inertial acceleration component.  The NIS gate is scalar; the
        % matrix deltaY retains the watcher-specific directional weighting.
        quality = 1;
        if isfinite(nis(i)) && nis(i) > cfg.share.nisGate
            quality = cfg.share.nisGate/nis(i);
        end
        deltaXiRaw = Yi*x(5:6,1,i) - xiPrior;
        activeDirections = U*diag(lambda > 0)*U';
        deltaXi = activeDirections*deltaXiRaw;
        Yfused = Yfused + quality*deltaY;
        xifused = xifused + quality*deltaXi;
        contribution(i) = quality*trace(deltaY);
        increment(:,:,i) = quality*deltaY;
    end
    Pshared = inverseSPD(Yfused);
    ashared = Pshared*xifused;
    % Broadcast the common inertial acceleration posterior.  Each watcher
    % updates its own r,v conditionally, preserving its local cross block.
    for i = 1:Nw
        Pi = .5*(P(:,:,i)+P(:,:,i)');
        Pya = Pi(1:4,5:6); Paa = Pi(5:6,5:6) + 1e-16*I2;
        G = Pya / Paa;
        x(1:4,1,i) = x(1:4,1,i) + G*(ashared-x(5:6,1,i));
        x(5:6,1,i) = ashared;
        Pyy = Pi(1:4,1:4) - G*(Paa-Pshared)*G';
        P(:,:,i) = [Pyy, G*Pshared; Pshared*G', Pshared];
        P(:,:,i) = .5*(P(:,:,i)+P(:,:,i)');
    end
    fusion = struct('a',ashared,'P',Pshared,'information',Yfused, ...
        'contribution',contribution,'increment',increment);
end

function windows = initializeLikelihoodWindows(Nw)
% Each watcher owns a separate local likelihood window.  No state, LOS, or
% raw measurement is communicated while the window is being accumulated.
    blank = struct('xReference',[],'information',zeros(6),'score',zeros(6,1), ...
        'samples',0);
    windows = repmat(blank,Nw,1);
end

function window = accumulateAccelerationLikelihood(window,x,z,rw,cfg)
% Linearize each bearing about the local state at the beginning of this
% window.  The canonical pair (I,g) represents only this window's bearing
% likelihood, not the EKF posterior and hence not its common prior.
    if isempty(window.xReference)
        window.xReference = x;
    end
    tau = window.samples*cfg.dt;
    Phi = constantAccelerationTransition(tau);
    xNominal = Phi*window.xReference;
    rho = xNominal(1:2)-rw; r2 = max(rho.'*rho,1);
    H = [-rho(2)/r2 rho(1)/r2 0 0 0 0];
    J = H*Phi;
    nu = wrapAngle(z-bearing(xNominal(1:2),rw));
    window.information = window.information + (J'*J)/(cfg.meas.sigma^2);
    window.score = window.score + (J'*nu)/(cfg.meas.sigma^2);
    window.samples = window.samples + 1;
end

function [x,P,fusion] = windowedLikelihoodAccelerationFusion( ...
        x,P,aPrior,PaPrior,windows,cfg)
% Fuse local *measurement likelihoods* for the common inertial
% acceleration.  For watcher i, position and velocity are nuisance
% variables y=[r;v] and are eliminated before anything is shared:
%   Lambda_a = I_aa - I_ay I_yy^{-1} I_ya,
%   xi_a     = g_a - I_ay I_yy^{-1} g_y.
% This avoids treating marginal EKF P_aa or its difference from a prior as
% an independent acceleration observation.
    Nw = size(x,3); I2 = eye(2); Lambda = zeros(2); xi = zeros(2,1);
    increment = zeros(2,2,Nw); contribution = zeros(1,Nw);
    for i = 1:Nw
        [Li,gi,valid] = accelerationLikelihoodFromWindow(windows(i),cfg);
        if ~valid, continue; end
        Lambda = Lambda + Li;
        xi = xi + gi;
        increment(:,:,i) = Li;
        contribution(i) = trace(Li);
    end
    % This is the recursively propagated common prior, not the initial P0.
    % It is introduced once because the transmitted quantities are pure
    % window likelihoods and therefore do not already contain it.
    Pprior = PaPrior + cfg.ekf.qJerk*cfg.share.windowDuration*I2;
    Yprior = inverseSPD(Pprior);
    Yfused = Yprior + Lambda;
    Pshared = inverseSPD(Yfused);
    ashared = aPrior + Pshared*xi;
    [x,P] = broadcastSharedAcceleration(x,P,ashared,Pshared);
    fusion = struct('a',ashared,'P',Pshared,'information',Yfused, ...
        'contribution',contribution,'increment',increment);
end

function [Lambda,xi,valid] = accelerationLikelihoodFromWindow(window,cfg)
    Lambda = zeros(2); xi = zeros(2,1); valid = false;
    if window.samples < cfg.share.minWindowSamples, return; end
    Iwin = .5*(window.information+window.information');
    Iyy = Iwin(1:4,1:4); Iya = Iwin(1:4,5:6); Iaa = Iwin(5:6,5:6);
    % If r,v cannot be locally eliminated, this watcher has not produced a
    % defensible acceleration likelihood during this window.
    if rcond(Iyy) < cfg.share.minNuisanceRcond, return; end
    Yinv = inverseSPD(Iyy);
    Lambda = .5*((Iaa-Iya'*Yinv*Iya) + (Iaa-Iya'*Yinv*Iya)');
    xi = window.score(5:6) - Iya'*Yinv*window.score(1:4);
    [U,D] = eig(Lambda); d = max(real(diag(D)),0);
    if max(d) <= cfg.share.minDirectionalEigenvalue, return; end
    Lambda = U*diag(d)*U';
    xi = U*diag(d > cfg.share.minDirectionalEigenvalue)*U'*xi;
    valid = true;
end

function Phi = constantAccelerationTransition(tau)
    I = eye(2); Z = zeros(2);
    Phi = [I tau*I .5*tau^2*I; Z I tau*I; Z Z I];
end

function [x,P] = broadcastSharedAcceleration(x,P,ashared,Pshared)
    Nw = size(x,3); I2 = eye(2);
    for i = 1:Nw
        Pi = .5*(P(:,:,i)+P(:,:,i)');
        Pya = Pi(1:4,5:6); Paa = Pi(5:6,5:6) + 1e-16*I2;
        G = Pya/Paa;
        x(1:4,1,i) = x(1:4,1,i) + G*(ashared-x(5:6,1,i));
        x(5:6,1,i) = ashared;
        Pyy = Pi(1:4,1:4) - G*(Paa-Pshared)*G';
        P(:,:,i) = [Pyy, G*Pshared; Pshared*G', Pshared];
        P(:,:,i) = .5*(P(:,:,i)+P(:,:,i)');
    end
end

function [x,P,fusion] = meanConsensusAccelerationFusion(x,P)
% Conservative baseline: average local a posteriors and retain their
% disagreement as uncertainty.  It avoids an independence assumption but
% does not exploit directional complementarity.
    Nw = size(x,3); I2 = eye(2); aLocal = zeros(2,Nw); PaLocal = zeros(2,2,Nw);
    for i = 1:Nw
        aLocal(:,i) = x(5:6,1,i);
        PaLocal(:,:,i) = .5*(P(5:6,5:6,i)+P(5:6,5:6,i)');
    end
    ashared = mean(aLocal,2); Pshared = mean(PaLocal,3);
    for i = 1:Nw
        da = aLocal(:,i)-ashared;
        Pshared = Pshared + (da*da')/Nw;
    end
    Pshared = .5*(Pshared+Pshared') + 1e-16*I2;
    for i = 1:Nw
        Pi = .5*(P(:,:,i)+P(:,:,i)');
        Pya = Pi(1:4,5:6); Paa = Pi(5:6,5:6) + 1e-16*I2;
        G = Pya/Paa;
        x(1:4,1,i) = x(1:4,1,i) + G*(ashared-x(5:6,1,i));
        x(5:6,1,i) = ashared;
        Pyy = Pi(1:4,1:4) - G*(Paa-Pshared)*G';
        P(:,:,i) = [Pyy, G*Pshared; Pshared*G', Pshared];
        P(:,:,i) = .5*(P(:,:,i)+P(:,:,i)');
    end
    fusion = struct('a',ashared,'P',Pshared,'information',inverseSPD(Pshared), ...
        'contribution',ones(1,Nw)/Nw,'increment',nan(2,2,Nw));
end

function [x,P,fusion] = geometryWeightedAccelerationFusion( ...
        x,P,aPrior,PaPrior,nis,rw,vw,cfg)
% Conservative geometry-only combiner.  W_i is a finite-horizon bearing
% sensitivity to inertial acceleration, built from I-u_i*u_i'.  It is not
% a posterior-information difference, so common prior information is never
% summed repeatedly.  The local a_i estimates are blended only when their
% aggregate geometry spans both inertial acceleration directions.
    Nw = size(x,3); I2 = eye(2); Wsum = zeros(2); rhs = zeros(2,1);
    Wi = zeros(2,2,Nw); quality = ones(1,Nw); contribution = zeros(1,Nw);
    aLocal = zeros(2,Nw); PaLocal = zeros(2,2,Nw);
    for i = 1:Nw
        aLocal(:,i) = x(5:6,1,i);
        PaLocal(:,:,i) = .5*(P(5:6,5:6,i)+P(5:6,5:6,i)') + 1e-16*I2;
        if isfinite(nis(i)) && nis(i) > cfg.share.nisGate
            quality(i) = cfg.share.nisGate/nis(i);
        end
        Wi(:,:,i) = localAccelerationGeometryWeight( ...
            x(:,1,i),rw(:,i),vw(:,i),cfg);
        contribution(i) = quality(i)*trace(Wi(:,:,i));
        Wsum = Wsum + quality(i)*Wi(:,:,i);
        rhs = rhs + quality(i)*Wi(:,:,i)*aLocal(:,i);
    end
    Wsum = .5*(Wsum+Wsum'); eigenvalues = eig(Wsum);
    geometryCondition = min(eigenvalues)/max(max(eigenvalues),eps);
    if trace(Wsum) > cfg.share.minInformationTrace && ...
            isfinite(geometryCondition) && ...
            geometryCondition >= cfg.share.minGeometryCondition
        ashared = Wsum\rhs;
        % Matrix-weighted mixture covariance: it does not assume local
        % acceleration posteriors are independent.  Estimator disagreement
        % is retained explicitly and inflated for unknown cross-correlation.
        Pshared = zeros(2);
        for i = 1:Nw
            Ki = Wsum\(quality(i)*Wi(:,:,i));
            da = aLocal(:,i)-ashared;
            Pshared = Pshared + Ki*(PaLocal(:,:,i)+da*da')*Ki';
        end
        Pshared = cfg.share.fallbackCovInflation*.5*(Pshared+Pshared');
        usedFallback = false;
    else
        ashared = aPrior;
        Pshared = cfg.share.fallbackCovInflation*PaPrior;
        usedFallback = true;
    end
    Pshared = .5*(Pshared+Pshared') + 1e-16*I2;
    for i = 1:Nw
        Pi = .5*(P(:,:,i)+P(:,:,i)');
        Pya = Pi(1:4,5:6); Paa = Pi(5:6,5:6) + 1e-16*I2;
        G = Pya/Paa;
        x(1:4,1,i) = x(1:4,1,i) + G*(ashared-x(5:6,1,i));
        x(5:6,1,i) = ashared;
        Pyy = Pi(1:4,1:4) - G*(Paa-Pshared)*G';
        P(:,:,i) = [Pyy, G*Pshared; Pshared*G', Pshared];
        P(:,:,i) = .5*(P(:,:,i)+P(:,:,i)');
    end
    fusion = struct('a',ashared,'P',Pshared,'information',Wsum, ...
        'contribution',contribution, ...
        'increment',Wi,'usedFallback',usedFallback, ...
        'geometryCondition',geometryCondition);
end

function W = localAccelerationGeometryWeight(x,rw,vw,cfg)
% W = sum G' H'R^-1H G.  For the constant-acceleration toy, G is the
% exact sensitivity of future position to the present inertial a.
    I2 = eye(2); W = zeros(2);
    for tau = cfg.share.geometryDt:cfg.share.geometryDt:cfg.share.geometryHorizon
        rt = x(1:2) + x(3:4)*tau + .5*x(5:6)*tau^2;
        rwt = rw + vw*tau;
        rho = rt-rwt; rho2 = max(rho'*rho,1);
        u = rho/sqrt(rho2);
        HtRH = (I2-u*u')/(cfg.meas.sigma^2*rho2);
        G = .5*tau^2*I2;
        W = W + G'*HtRH*G;
    end
    W = .5*(W+W');
end

function Ainv = inverseSPD(A)
    A = .5*(A+A') + 1e-16*eye(size(A));
    [L,p] = chol(A,'lower');
    if p == 0
        Ainv = L'\(L\eye(size(A)));
    else
        Ainv = pinv(A);
    end
    Ainv = .5*(Ainv+Ainv');
end

function [d,info] = chooseLocalDirection(x,P,rw,vw,cfg)
% Direction is selected only from this watcher's posterior and its known
% future motion.  Virtual bearings carry no truth/noise information: they
% only predict the geometry-induced covariance reduction of each action.
    rho0 = x(1:2)-rw; e = rho0/max(norm(rho0),eps);
    eTan = [-e(2);e(1)];
    % A fixed transverse pulse can miss the useful side of the LOS once
    % relative velocity and target acceleration are included.  Search a
    % modest polar candidate set, with the two transverse directions first
    % for easy interpretation in diagnostics.
    angle = (0:15)*(2*pi/16);
    candidates = [eTan,-eTan];
    candidates = [candidates, [cos(angle);sin(angle)]];
    candidates = candidates./vecnorm(candidates);
    coastMetric = localGramianMetric(x,rw,vw,zeros(2,1),cfg);
    coastCost = localCovarianceRolloutCost(x,P,rw,vw,zeros(2,1),cfg);
    nCandidate = size(candidates,2);
    scores = zeros(1,nCandidate); costs = zeros(1,nCandidate);
    best = Inf; bestIndex = 1; d = eTan;
    for c = 1:nCandidate
        scores(c) = localGramianMetric(x,rw,vw,candidates(:,c),cfg);
        costs(c) = localCovarianceRolloutCost(x,P,rw,vw,candidates(:,c),cfg);
        if costs(c) < best
            best = costs(c); bestIndex = c; d = candidates(:,c);
        end
    end
    info = struct('coastMetric',coastMetric,'coastCost',coastCost, ...
        'candidateMetric',scores, ...
        'candidateCost',costs,'selectedCost',best, ...
        'selectedMetric',scores(bestIndex), ...
        'beneficial',best <= (1-cfg.control.minCostReduction)*coastCost);
end

function d = localTransverseDirection(x,rw)
    rho = x(1:2)-rw; e = rho/max(norm(rho),eps);
    d = [-e(2);e(1)];
end

function metric = localRadialPositionVariance(x,P,rw)
% Simple local position-observability proxy: uncertainty along the LOS,
% the direction that a bearing update observes least directly.
    rho = x(1:2)-rw; e = rho/max(norm(rho),eps);
    Prr = .5*(P(1:2,1:2)+P(1:2,1:2)');
    metric = max(e'*Prr*e,0);
end

function cost = localCovarianceRolloutCost(x,P,rw,vw,d,cfg)
% Local covariance-only look-ahead.  Virtual bearings equal their predicted
% means, so no truth/noise is injected.  The common acceleration posterior
% is represented by the local P(5:6,5:6) block at the planning instant.
    cfgPlan = cfg; cfgPlan.dt = cfg.control.planDt;
    xPlan = x; PPlan = P; nSteps = round(cfg.control.horizon/cfg.control.planDt);
    for ip = 1:nSteps
        tau = ip*cfg.control.planDt;
        rwPlan = rw + vw*tau + pulsePairDisplacement(tau,d,cfg);
        [xPlan,PPlan] = predictToy(xPlan,PPlan,cfgPlan);
        zVirtual = bearing(xPlan(1:2),rwPlan);
        [xPlan,PPlan] = updateToy(xPlan,PPlan,zVirtual,rwPlan,cfgPlan);
    end
    w = cfg.control.costWeights;
    rCost = trace(PPlan(1:2,1:2))/(100^2);
    vCost = trace(PPlan(3:4,3:4))/(.1^2);
    % Do not rank actions by a latent state/parameter covariance directly.
    % Rank them by uncertainty in the acceleration correction that will be
    % injected into the dynamics.  Here d_toy=a, so this reduces to P_aa;
    % the explicit function is the drop-in point for a DNN output Jacobian.
    SigmaD = toyResidualOutputCovariance(xPlan,PPlan,cfgPlan);
    aCost = trace(SigmaD)/(1e-4^2);
    displacementCost = norm(pulsePairDisplacement(cfg.control.horizon,d,cfg))^2/(10^2);
    cost = w(1)*rCost + w(2)*vCost + w(3)*aCost + w(4)*displacementCost;
end

function SigmaD = toyResidualOutputCovariance(x,P,cfg)
%TOYRESIDUALOUTPUTCOVARIANCE Predictive covariance of d_toy(xi)=a.
% xi=[r;v;a] in this toy.  Written in Jacobian form on purpose:
%     Sigma_d = G_xi P_xi G_xi' + Sigma_model.
% For the production DNN-EKF, replace G_xi by [G_eta,G_theta], where
% G_eta=dg/deta and G_theta=dg/dtheta at the rollout state.
    %#ok<INUSD>
    Gxi = [zeros(2,4), eye(2)];
    SigmaD = Gxi*P*Gxi' + cfg.model.outputCovFloor;
    SigmaD = .5*(SigmaD+SigmaD');
end

function metric = localGramianMetric(x,rw,vw,d,cfg)
    W = zeros(6); I = eye(2);
    for tau = cfg.control.planDt:cfg.control.planDt:cfg.control.horizon
        deltaW = pulsePairDisplacement(tau,d,cfg);
        rho = x(1:2)-rw + (x(3:4)-vw)*tau + .5*x(5:6)*tau^2-deltaW;
        r2 = max(rho.'*rho,1); Hpos = [-rho(2),rho(1)]/r2;
        Phi = [I tau*I .5*tau^2*I; zeros(2) I tau*I; zeros(2) zeros(2) I];
        H = [Hpos zeros(1,4)];
        W = W + Phi'*H'*H*Phi*cfg.control.planDt/cfg.meas.sigma^2;
    end
    Wn = .5*(cfg.control.stateScale'*W*cfg.control.stateScale + ...
        (cfg.control.stateScale'*W*cfg.control.stateScale)');
    C = chol(eye(6)+Wn,'lower');
    metric = 2*sum(log(diag(C)));
end

function deltaW = pulsePairDisplacement(tau,d,cfg)
    h = cfg.control.halfBurnDuration; a = cfg.control.acceleration*d;
    if tau <= h
        deltaW = .5*a*tau^2;
    elseif tau <= 2*h
        s = tau-h; deltaW = .5*a*h^2 + a*h*s - .5*a*s^2;
    else
        deltaW = a*h^2;
    end
end

function y = bearing(rt,rw), y = atan2(rt(2)-rw(2),rt(1)-rw(1)); end
function y = wrapAngle(y), y = mod(y+pi,2*pi)-pi; end

function summary = summarizeToy(out)
    cases = {out.localCoast,out.localActive,out.coast,out.active};
    activeLabel = "Shared acceleration / active maneuver";
    localActiveLabel = "Local position-triggered maneuver (no sharing)";
    if isfield(out.cfg,'control') && out.cfg.control.mode == "rollout_position"
        localActiveLabel = "Local trigger + covariance-rollout maneuver (no sharing)";
        activeLabel = "Shared acceleration / local covariance-rollout maneuver";
    elseif isfield(out.cfg,'control') && out.cfg.control.mode == "position_only"
        activeLabel = "Shared acceleration / position-triggered maneuver";
    end
    labels = ["Local coast (no sharing)"; ...
        localActiveLabel; ...
        "Shared acceleration / coast";activeLabel];
    summary = table();
    for j = 1:numel(cases)
        r = cases{j}; N = numel(r.time); Nw = size(r.xhat,3);
        truth = repmat(r.etaTrue,1,1,Nw);
        ep = vecnorm(r.xhat(1:2,:,:)-truth(1:2,:,:),2,1);
        ev = vecnorm(r.xhat(3:4,:,:)-truth(3:4,:,:),2,1);
        ea = vecnorm(r.xhat(5:6,:,:)-truth(5:6,:,:),2,1);
        row = table(labels(j),sqrt(mean(ep(:).^2)),sqrt(mean(ev(:).^2)), ...
            sqrt(mean(ea(:).^2)),sqrt(mean(ep(:,round(.9*N):end,:).^2,'all')), ...
            'VariableNames',{'caseName','positionRMSE','velocityRMSE','accelerationRMSE','finalPositionRMSE'});
        if isempty(summary), summary=row; else, summary=[summary;row]; end %#ok<AGROW>
    end
end

function report = summarizeCovarianceMatching(res,cfg)
% Compact diagnostic for the live, shared-acceleration active case.
    idx = res.time >= cfg.adaptive.burnIn;
    ratio = res.cmRatio(idx,:); ratio = ratio(isfinite(ratio) & ratio > 0);
    nis = res.NIS(idx,:); nis = nis(isfinite(nis));
    gamma = res.gammaQ(idx,:); gamma = gamma(isfinite(gamma));
    if isempty(ratio)
        report = struct('medianRatio',NaN,'logRatioRMSE',NaN, ...
            'meanNIS',NaN,'medianGammaQ',NaN);
        return
    end
    report = struct( ...
        'medianRatio',median(ratio), ...
        'logRatioRMSE',sqrt(mean(log(ratio).^2)), ...
        'meanNIS',mean(nis), ...
        'medianGammaQ',median(gamma));
end
