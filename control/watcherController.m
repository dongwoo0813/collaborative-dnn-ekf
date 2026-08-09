function [cmd,controllerState] = watcherController( ...
    i,watcherState,targetInfo,t,cfg)
%{
Function:
    watcherController.m

Purpose:
    Compute the watcher control command.

    In Step 01, this function returns zero translational force and zero
    attitude torque. It is included now so that later thruster control and
    attitude control can be added without restructuring the simulation loop.

Inputs:
    i            - Watcher index.

    watcherState - Current watcher truth-state structure.
                   Required fields:
                       watcherState.r
                       watcherState.v
                       watcherState.q
                       watcherState.omega
                       watcherState.mass

    targetInfo   - Target information structure.
                   Suggested fields:
                       targetInfo.etaHat - watcher-local target estimate
                       targetInfo.etaTrue - target truth, debugging only

    t            - Current simulation time.

    cfg          - Simulation configuration structure.
                   Required fields:
                       cfg.dim
                       cfg.control.translationMode
                       cfg.control.attitudeMode

Outputs:
    cmd          - Control command structure.
                   Fields:
                       cmd.u   - translational force command, cfg.dim x 1
                       cmd.tau - attitude torque command, 3 x 1

Main equations:
    Later translational dynamics will use

        dot r_w = v_w,

        dot v_w = u_w / m_w.

    Later attitude dynamics can use

        dot q = 1/2 Omega(omega) q,

        J dot omega = tau - omega x J omega.

Notes:
    - For now, this function does not use target truth for control.
    - Later, targetInfo.etaHat should be used for relative navigation/control.
%}

    dim = cfg.dim;

    cmd.u = zeros(dim,1);
    cmd.tau = zeros(3,1);
    controllerState = watcherState.controllerState;
    controllerState.replanFlag = false;
    controllerState.activeFlag = false;

    switch cfg.control.translationMode
        case "none"
            % Keep zero force.

        case "observability_seeking"
            obs = cfg.control.obs;
            burnEnd = obs.startTime+obs.burnDuration;
            if t >= obs.startTime && t < burnEnd
                controllerState.activeFlag = true;
                if t+10*eps(max(1,abs(t))) >= ...
                        controllerState.nextReplanTime
                    [direction,planInfo] = ...
                        selectObservabilitySeekingDirection( ...
                        watcherState,targetInfo,t,cfg);
                    controllerState.direction = direction;
                    controllerState.candidateIndex = ...
                        planInfo.candidateIndex;
                    controllerState.score = planInfo.score;
                    controllerState.candidateScores = ...
                        planInfo.candidateScores;
                    if isfield(planInfo,"candidateGeometryScores")
                        controllerState.candidateGeometryScores = ...
                            planInfo.candidateGeometryScores;
                    end
                    if isfield(planInfo,"candidateParameterScores")
                        controllerState.candidateParameterScores = ...
                            planInfo.candidateParameterScores;
                    end
                    if isfield(planInfo,"jointScoreEnabled")
                        controllerState.jointScoreEnabled = ...
                            planInfo.jointScoreEnabled;
                        controllerState.geometryWeight = planInfo.geometryWeight;
                        controllerState.parameterWeight = planInfo.parameterWeight;
                    end
                    if isfield(planInfo,"selectedGeometryScore")
                        controllerState.selectedGeometryScore = ...
                            planInfo.selectedGeometryScore;
                    end
                    if isfield(planInfo,"selectedParameterScore")
                        controllerState.selectedParameterScore = ...
                            planInfo.selectedParameterScore;
                    end
                    if isfield(planInfo,"candidateInformationMinEig")
                        controllerState.candidateInformationMinEig = ...
                            planInfo.candidateInformationMinEig;
                    end
                    if isfield(planInfo,"candidateInformationCondition")
                        controllerState.candidateInformationCondition = ...
                            planInfo.candidateInformationCondition;
                    end
                    if isfield(planInfo,"selectedInformationMinEig")
                        controllerState.selectedInformationMinEig = ...
                            planInfo.selectedInformationMinEig;
                    end
                    if isfield(planInfo,"selectedInformationCondition")
                        controllerState.selectedInformationCondition = ...
                            planInfo.selectedInformationCondition;
                    end
                    if isfield(planInfo,"selectedGeometryScore")
                        controllerState.predictedRadialVariance = ...
                            planInfo.selectedGeometryScore;
                    else
                        controllerState.predictedRadialVariance = planInfo.score;
                    end
                    controllerState.replanFlag = true;
                    controllerState.lastPlanTime = t;
                    controllerState.nextReplanTime = ...
                        t+obs.replanInterval;
                end
                cmd.u = watcherState.mass*obs.acceleration* ...
                    controllerState.direction;
            end

        case "observability_impulse"
            % Coast by default.  Start one finite impulse only when the
            % local filter reports poor radial observability.  The trigger
            % uses etaHat and PEta exclusively; it never uses target truth
            % or the current measurement residual.
            obs = cfg.control.obs;
            controllerState = initializeImpulseState(controllerState,t);
            if t >= obs.startTime
                radialVariance = localRadialVariance( ...
                    watcherState,targetInfo,cfg);
                controllerState.predictedRadialVariance = radialVariance;

                inBurn = t < controllerState.impulseEndTime;
                mayEvaluate = t+10*eps(max(1,abs(t))) >= ...
                    controllerState.nextReplanTime;
                mayTrigger = t+10*eps(max(1,abs(t))) >= ...
                    controllerState.nextEligibleTime;
                threshold = getImpulseNumeric(obs,"triggerRadialVariance",Inf);

                if ~inBurn && mayEvaluate && mayTrigger && ...
                        radialVariance >= threshold
                    % The planner needs a finite burn horizon.  For an
                    % impulse event, make that horizon start now rather
                    % than reusing the global experiment start time.
                    cfgPlan = cfg;
                    cfgPlan.control.obs.startTime = t;
                    cfgPlan.control.obs.burnDuration = ...
                        getImpulseNumeric(obs,"impulseDuration",0);
                    [direction,planInfo] = selectObservabilitySeekingDirection( ...
                        watcherState,targetInfo,t,cfgPlan);
                    controllerState = storePlan(controllerState,direction, ...
                        planInfo,t,obs);
                    controllerState.impulseEndTime = t + ...
                        getImpulseNumeric(obs,"impulseDuration",0);
                    controllerState.nextEligibleTime = ...
                        controllerState.impulseEndTime + ...
                        getImpulseNumeric(obs,"cooldownDuration",0);
                    controllerState.numImpulses = controllerState.numImpulses+1;
                    inBurn = t < controllerState.impulseEndTime;
                end

                controllerState.activeFlag = inBurn;
                if inBurn
                    cmd.u = watcherState.mass*obs.acceleration* ...
                        controllerState.direction;
                end

                if mayEvaluate
                    controllerState.nextReplanTime = t + ...
                        getImpulseNumeric(obs,"replanInterval",1);
                end
            end

        otherwise
            error("Unknown cfg.control.translationMode.");
    end

    switch cfg.control.attitudeMode
        case "none"
            % Keep zero torque.

        otherwise
            error("Unknown cfg.control.attitudeMode.");
    end

    cmd.u = saturateControl(cmd.u, cfg.watchers.maxThrust);
    cmd.tau = saturateControl(cmd.tau, cfg.watchers.maxTorque);

end

function state = initializeImpulseState(state,t)
% Supply backward-compatible defaults for states created before Step 10-C.
if ~isfield(state,"impulseEndTime"), state.impulseEndTime = -Inf; end
if ~isfield(state,"nextEligibleTime"), state.nextEligibleTime = -Inf; end
if ~isfield(state,"numImpulses"), state.numImpulses = 0; end
if ~isfield(state,"nextReplanTime"), state.nextReplanTime = t; end
end

function radialVariance = localRadialVariance(watcherState,targetInfo,cfg)
dim = cfg.dim;
rRelative = targetInfo.etaHat(1:dim)-watcherState.r;
rNorm = max(norm(rRelative),cfg.gate.minRange);
eRadial = rRelative/rNorm;
Ppos = 0.5*(targetInfo.PEta(1:dim,1:dim)+ ...
    targetInfo.PEta(1:dim,1:dim).');
radialVariance = max(0,real(eRadial.'*Ppos*eRadial));
end

function state = storePlan(state,direction,planInfo,t,obs)
state.direction = direction;
state.candidateIndex = planInfo.candidateIndex;
state.score = planInfo.score;
state.candidateScores = planInfo.candidateScores;
state.replanFlag = true;
state.lastPlanTime = t;
if isfield(planInfo,"candidateInformationMinEig")
    state.candidateInformationMinEig = planInfo.candidateInformationMinEig;
    state.candidateInformationCondition = planInfo.candidateInformationCondition;
    state.selectedInformationMinEig = planInfo.selectedInformationMinEig;
    state.selectedInformationCondition = planInfo.selectedInformationCondition;
end
if isfield(planInfo,"candidateGeometryScores")
    state.candidateGeometryScores = planInfo.candidateGeometryScores;
    state.candidateParameterScores = planInfo.candidateParameterScores;
    state.selectedGeometryScore = planInfo.selectedGeometryScore;
    state.selectedParameterScore = planInfo.selectedParameterScore;
end
if isfield(planInfo,"jointScoreEnabled")
    state.jointScoreEnabled = planInfo.jointScoreEnabled;
    state.geometryWeight = planInfo.geometryWeight;
    state.parameterWeight = planInfo.parameterWeight;
end
state.nextReplanTime = t + getImpulseNumeric(obs,"replanInterval",1);
end

function value = getImpulseNumeric(obs,name,defaultValue)
if isfield(obs,name) && isfinite(obs.(name))
    value = double(obs.(name));
else
    value = defaultValue;
end
end
