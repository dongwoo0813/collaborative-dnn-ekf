function out = run_step10C2_GS_impulse_closed_loop_output_fusion( ...
    seed,simulationTime,makePlots,saveArtifacts,timeStep,obsOverride,runOverride)
%RUN_STEP10C2_GS_IMPULSE_CLOSED_LOOP_OUTPUT_FUSION Independent live runs.
%
% Unlike Step 10-C.1 replay, each estimator/controller pair below generates
% its own watcher maneuvers from its own estimated state and covariance:
%   1) local single-branch DNN-EKF,
%   2) GS additive, and
%   3) GS output-information fusion.
% Initial conditions and the bearing-noise realization are fixed across
% cases, but watcher trajectories are deliberately NOT shared.  This is the
% operational closed-loop experiment: estimation error can affect residual
% evaluation, maneuver timing/direction, and subsequent measurement geometry.

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(simulationTime), simulationTime = 1800; end
    if nargin < 3 || isempty(makePlots), makePlots = false; end
    if nargin < 4 || isempty(saveArtifacts), saveArtifacts = true; end
    if nargin < 5 || isempty(timeStep), timeStep = 0.5; end
    if nargin < 6 || isempty(obsOverride), obsOverride = struct(); end
    if nargin < 7 || isempty(runOverride), runOverride = struct(); end
    addpath(genpath(pwd));

    cfgBase = makeClosedLoopConfig(seed,simulationTime,timeStep,obsOverride);
    if isfield(runOverride,"translationMode")
        cfgBase.control.translationMode = string(runOverride.translationMode);
    end
    cfgLocal = cfgBase;
    cfgLocal.step.name = "step10C2_closed_loop_local_single_branch";
    cfgLocal.estimator.type = "Local_DNN_EKF";
    cfgLocal.dnn.predictionResidualSource = "local_DNN";
    cfgLocal.gs.enabled = false;

    cfgAdd = cfgBase;
    cfgAdd.step.name = "step10C2_closed_loop_GS_additive";
    cfgAdd.dnn.predictionResidualSource = "GS_composite";
    cfgAdd.gs.enabled = true;
    cfgAdd.gs.compositeMode = "additive";

    cfgOutput = cfgBase;
    cfgOutput.step.name = "step10C2_closed_loop_GS_output_information";
    cfgOutput.dnn.predictionResidualSource = "GS_composite";
    cfgOutput.gs.enabled = true;
    cfgOutput.gs.compositeMode = "output_information_fusion";

    fprintf("Step 10-C.2: independent closed-loop impulse comparison\n");
    fprintf("seed=%g, T=%.1f s, dt=%.2f s, impulse=%.1f s, cooldown=%.1f s\n", ...
        seed,cfgBase.T,cfgBase.dt,cfgBase.control.obs.impulseDuration, ...
        cfgBase.control.obs.cooldownDuration);
    fprintf("Each case uses its own estimate/covariance and watcher trajectory.\n");

    rng(seed); resLocal = simulateLocalDNNEKF(cfgLocal);
    rng(seed); resAdd = simulate_GS_DNN_EKF(cfgAdd);
    rng(seed); resOutput = simulate_GS_DNN_EKF(cfgOutput);

    results = {resLocal,resAdd,resOutput};
    configs = {cfgLocal,cfgAdd,cfgOutput};
    labels = ["Local single branch";"GS additive";"GS output-information fusion"];
    summary = table();
    for i = 1:numel(results)
        row = summarizeClosedLoopCase(results{i},configs{i},labels(i));
        if isempty(summary), summary = row; else, summary = [summary;row]; end %#ok<AGROW>
    end
    disp(summary);

    out = struct("seed",seed,"cfgBase",cfgBase,"resLocal",resLocal,"resAdd",resAdd, ...
        "resOutputInformation",resOutput,"cfgLocal",cfgLocal, ...
        "cfgAdd",cfgAdd,"cfgOutputInformation",cfgOutput,"summary",summary);
    requireImpulse = cfgBase.control.translationMode == "observability_impulse";
    out.check = check_step10C2_GS_impulse_closed_loop(out,requireImpulse);

    if saveArtifacts
        if ~isfolder("results"), mkdir("results"); end
        inputMode = string(cfgBase.dnn.mlp.inputMode);
        fileName = fullfile("results",sprintf( ...
            "step10C2_GS_impulse_closed_loop_%s_seed%d_T%d_burn%d_cd%d.mat", ...
            inputMode,seed,round(simulationTime), ...
            round(cfgBase.control.obs.impulseDuration), ...
            round(cfgBase.control.obs.cooldownDuration)));
        save(fileName,"summary","cfgBase","-v7.3");
        out.artifactFile = string(fileName);
        fprintf("Saved closed-loop summary: %s\n",fileName);
    else
        out.artifactFile = "";
    end

    if makePlots
        out.figures = plot_step10C2_GS_impulse_closed_loop(out,true);
    end
end

function cfg = makeClosedLoopConfig(seed,simulationTime,timeStep,obsOverride)
    [cfg,~,~] = config_step09J6_seed101_operational();
    cfg.T = simulationTime;
    cfg.dt = timeStep;
    cfg.time = 0:cfg.dt:cfg.T;
    cfg.N = numel(cfg.time);
    cfg.scenario.watcherModel = "matched_velocity_coast";
    cfg.watchers.motionMode = "controlled";
    cfg.watchers.maxThrust = 0.02;
    cfg.control.translationMode = "observability_impulse";
    cfg.control.obs.mode = "observability_impulse";
    cfg.control.obs.startTime = 40.0;
    cfg.control.obs.acceleration = 6.0e-4;
    % Long burns are required to create meaningful transverse separation at
    % roughly kilometre-scale range; 10 s burns produced sub-metre shifts.
    cfg.control.obs.impulseDuration = 60.0;
    cfg.control.obs.cooldownDuration = 180.0;
    cfg.control.obs.triggerRadialVariance = 25.0^2;
    cfg.control.obs.replanInterval = 5.0;
    cfg.control.obs.numCandidateDirections = 8;
    cfg.control.obs.planningHorizon = 240.0;
    cfg.control.obs.planningDt = 0.5;
    cfg.control.obs.jointScoreEnabled = true;
    cfg.control.obs.geometryWeight = 0.7;
    cfg.control.obs.parameterWeight = 0.3;
    % Local active angles-only navigation: choose the known maneuver that
    % most changes this watcher's natural LOS profile (Woffinden--Geller).
    % "covariance_rollout" remains available through obsOverride.
    cfg.control.obs.plannerMode = "los_profile";
    cfg.control.obs.rolloutMaxSteps = 24;
    overrideNames = fieldnames(obsOverride);
    for i = 1:numel(overrideNames)
        cfg.control.obs.(overrideNames{i}) = obsOverride.(overrideNames{i});
    end
    cfg.gs.useNonlocalBranchCovariance = true;
    cfg.gs.uploadMode = "after_measurement_update";
    cfg.gs.broadcastMode = "every_step";

    % Same measurement-noise realization; no trajectory is replayed.
    rng(seed + 1000003);
    cfg.replay = struct("enabled",false, ...
        "bearingNoise",randn(cfg.N,cfg.Nw));
end

function row = summarizeClosedLoopCase(res,cfg,label)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    last = max(1,round(0.9*N)):N;
    truth = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    positionError = vecnorm(res.xhat(1:dim,:,:)-truth(1:dim,:,:),2,1);
    dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
    dOperational = vecnorm(res.dnnResidual-dTrue,2,1);
    dTrueInput = vecnorm(res.dnnResidualAtTrueEta-dTrue,2,1);
    % Count the physically applied commands, rather than a controller flag.
    % This remains correct if a controller telemetry flag is logged only at
    % planning instants or changes its bookkeeping convention.
    impulseSteps = nnz(sqrt(sum(res.watcherU.^2,1)) > 1e-12);
    radialNEES = computeRadialNEES(res,truth,dim);
    row = table(string(label), ...
        sqrt(mean(positionError(:).^2,"omitnan")), ...
        sqrt(mean(positionError(:,last,:).^2,"all","omitnan")), ...
        sqrt(mean(dOperational(:).^2,"omitnan")), ...
        sqrt(mean(dOperational(:,last,:).^2,"all","omitnan")), ...
        sqrt(mean(dTrueInput(:).^2,"omitnan")), ...
        sqrt(mean(dTrueInput(:,last,:).^2,"all","omitnan")), ...
        impulseSteps,mean(res.cumulativeDeltaV(end,:),"omitnan"), ...
        radialNEES, ...
        'VariableNames',{'caseName','positionRMSE','finalPositionRMSE', ...
        'operationalResidualRMSE','finalOperationalResidualRMSE', ...
        'residualAtTrueEtaRMSE','finalResidualAtTrueEtaRMSE', ...
        'impulseSteps','meanFinalDeltaV','radialNEES'});
end

function value = computeRadialNEES(res,truth,dim)
% Radial consistency of the covariance used by the live controller.
    if ~isfield(res,"predictedRadialVariance")
        value = NaN;
        return;
    end
    [~,N,Nw] = size(res.xhat);
    radialError2 = NaN(N,Nw);
    for i = 1:Nw
        for k = 1:N
            relative = res.xhat(1:dim,k,i)-res.watcherR(:,k,i);
            range = norm(relative);
            if range > eps
                eRad = relative/range;
                errorPosition = res.xhat(1:dim,k,i)-truth(1:dim,k,i);
                radialError2(k,i) = (eRad.'*errorPosition)^2;
            end
        end
    end
    denominator = res.predictedRadialVariance;
    valid = isfinite(radialError2) & isfinite(denominator) & denominator > eps;
    if any(valid,"all")
        value = mean(radialError2(valid)./denominator(valid),"omitnan");
    else
        value = NaN;
    end
end
