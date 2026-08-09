function out = run_step10C1_GS_impulse_replay_output_fusion( ...
    seed,simulationTime,makePlots,saveArtifacts,timeStep)
%RUN_STEP10C1_GS_IMPULSE_REPLAY_OUTPUT_FUSION GS-only residual test.
%
% First, run one GS output-information-fusion estimator closed loop.  Each
% watcher coasts until its own local radial covariance exceeds the configured
% threshold, then executes a short observability-seeking impulse and returns
% to coast.  Next, freeze the realized truth, watcher trajectory, force
% history, and bearing-noise draws, and replay these identical inputs for:
%   1) local single-branch DNN-EKF,
%   2) GS additive, and
%   3) GS output-information fusion.
%
% The replay summaries include residual error evaluated at eta_true.  That
% diagnostic is not supplied to any estimator or controller.

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(simulationTime), simulationTime = 600; end
    if nargin < 3 || isempty(makePlots), makePlots = false; end
    if nargin < 4 || isempty(saveArtifacts), saveArtifacts = true; end
    if nargin < 5 || isempty(timeStep), timeStep = 0.1; end
    addpath(genpath(pwd));

    [cfgSource,~,~] = config_step09J6_seed101_operational();
    cfgSource.T = simulationTime;
    cfgSource.dt = timeStep;
    cfgSource.time = 0:cfgSource.dt:cfgSource.T;
    cfgSource.N = numel(cfgSource.time);
    cfgSource.scenario.watcherModel = "matched_velocity_coast";
    cfgSource.watchers.motionMode = "controlled";
    cfgSource.watchers.maxThrust = 0.02;
    cfgSource.control.translationMode = "observability_impulse";
    cfgSource.control.obs.mode = "observability_impulse";
    cfgSource.control.obs.startTime = 40.0;
    cfgSource.control.obs.acceleration = 6.0e-4;
    cfgSource.control.obs.impulseDuration = 10.0;
    cfgSource.control.obs.cooldownDuration = 45.0;
    cfgSource.control.obs.triggerRadialVariance = 25.0^2;
    cfgSource.control.obs.replanInterval = 5.0;
    cfgSource.control.obs.numCandidateDirections = 8;
    cfgSource.control.obs.planningHorizon = 30.0;
    cfgSource.control.obs.planningDt = 0.5;
    cfgSource.control.obs.jointScoreEnabled = true;
    cfgSource.control.obs.geometryWeight = 0.7;
    cfgSource.control.obs.parameterWeight = 0.3;
    cfgSource.gs.compositeMode = "output_information_fusion";
    cfgSource.gs.useNonlocalBranchCovariance = true;
    cfgSource.gs.uploadMode = "after_measurement_update";
    cfgSource.gs.broadcastMode = "every_step";

    % Freeze the bearing draws independently of model-initialization RNG.
    rng(seed + 1000003);
    bearingNoise = randn(cfgSource.N,cfgSource.Nw);
    cfgSource.replay = struct("enabled",false,"bearingNoise",bearingNoise);

    fprintf("Step 10-C.1: GS output-information closed-loop impulse source\n");
    fprintf("seed=%g, T=%.1f s, impulse=%.1f s, cooldown=%.1f s\n", ...
        seed,cfgSource.T,cfgSource.control.obs.impulseDuration, ...
        cfgSource.control.obs.cooldownDuration);
    rng(seed);
    resSource = simulate_GS_DNN_EKF(cfgSource);

    trajectory = struct();
    trajectory.schema = "step10C1_gs_impulse_replay_v1";
    trajectory.sourceCase = "GS-output-information-fusion closed-loop impulse";
    trajectory.seed = seed;
    trajectory.time = resSource.time;
    trajectory.etaTrue = resSource.etaTrue;
    trajectory.watcherR = resSource.watcherR;
    trajectory.watcherV = resSource.watcherV;
    trajectory.watcherU = resSource.watcherU;
    trajectory.bearingNoise = bearingNoise;
    trajectory.cfg = cfgSource;
    trajectory.controller = struct("controllerActive",resSource.controllerActive, ...
        "replanFlag",resSource.replanFlag, ...
        "predictedRadialVariance",resSource.predictedRadialVariance, ...
        "selectedDirection",resSource.selectedDirection, ...
        "cumulativeImpulse",resSource.cumulativeImpulse, ...
        "cumulativeDeltaV",resSource.cumulativeDeltaV);

    cfgReplay = cfgSource;
    cfgReplay.replay.enabled = true;
    cfgReplay.replay.etaTrue = trajectory.etaTrue;
    cfgReplay.replay.watcherR = trajectory.watcherR;
    cfgReplay.replay.watcherV = trajectory.watcherV;
    cfgReplay.replay.watcherU = trajectory.watcherU;
    cfgReplay.replay.bearingNoise = trajectory.bearingNoise;
    cfgReplay.control.translationMode = "none";
    cfgReplay.watchers.motionMode = "controlled";

    cfgLocal = cfgReplay;
    cfgLocal.step.name = "step10C1_replay_local_single_branch";
    cfgLocal.estimator.type = "Local_DNN_EKF";
    cfgLocal.dnn.predictionResidualSource = "local_DNN";
    cfgLocal.gs.enabled = false;

    cfgAdd = cfgReplay;
    cfgAdd.step.name = "step10C1_replay_GS_additive";
    cfgAdd.dnn.predictionResidualSource = "GS_composite";
    cfgAdd.gs.enabled = true;
    cfgAdd.gs.compositeMode = "additive";

    cfgOutput = cfgReplay;
    cfgOutput.step.name = "step10C1_replay_GS_output_information";
    cfgOutput.dnn.predictionResidualSource = "GS_composite";
    cfgOutput.gs.enabled = true;
    cfgOutput.gs.compositeMode = "output_information_fusion";

    fprintf("Replaying identical trajectory/noise: local, GS additive, GS output-information\n");
    rng(seed); resLocal = simulateLocalDNNEKF(cfgLocal);
    rng(seed); resAdd = simulate_GS_DNN_EKF(cfgAdd);
    rng(seed); resOutput = simulate_GS_DNN_EKF(cfgOutput);

    results = {resLocal,resAdd,resOutput};
    configs = {cfgLocal,cfgAdd,cfgOutput};
    labels = ["Local single branch";"GS additive";"GS output-information fusion"];
    summary = table();
    for i = 1:numel(results)
        row = summarizeReplayCase(results{i},configs{i},labels(i));
        if isempty(summary), summary = row; else, summary = [summary;row]; end %#ok<AGROW>
    end
    disp(summary);

    out = struct("source",resSource,"cfgSource",cfgSource, ...
        "trajectory",trajectory,"resLocal",resLocal,"resAdd",resAdd, ...
        "resOutputInformation",resOutput,"cfgLocal",cfgLocal, ...
        "cfgAdd",cfgAdd,"cfgOutputInformation",cfgOutput,"summary",summary);
    out.check = check_step10C1_GS_impulse_replay(out);

    if saveArtifacts
        if ~isfolder("results"), mkdir("results"); end
        inputMode = string(cfgSource.dnn.mlp.inputMode);
        fileName = fullfile("results",sprintf( ...
            "step10C1_GS_impulse_replay_%s_seed%d_T%d.mat", ...
            inputMode,seed,round(simulationTime)));
        save(fileName,"trajectory","summary","-v7.3");
        out.artifactFile = string(fileName);
        fprintf("Saved trajectory and replay summary: %s\n",fileName);
    else
        out.artifactFile = "";
    end

    if makePlots
        out.figures = plot_step10C1_GS_impulse_replay(out,true);
    end
end

function row = summarizeReplayCase(res,cfg,label)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    last = max(1,round(0.9*N)):N;
    truth = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    positionError = vecnorm(res.xhat(1:dim,:,:)-truth(1:dim,:,:),2,1);
    dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
    dOperational = vecnorm(res.dnnResidual-dTrue,2,1);
    dTrueInput = vecnorm(res.dnnResidualAtTrueEta-dTrue,2,1);
    row = table(string(label), ...
        sqrt(mean(positionError(:).^2,"omitnan")), ...
        sqrt(mean(positionError(:,last,:).^2,"all","omitnan")), ...
        sqrt(mean(dOperational(:).^2,"omitnan")), ...
        sqrt(mean(dOperational(:,last,:).^2,"all","omitnan")), ...
        sqrt(mean(dTrueInput(:).^2,"omitnan")), ...
        sqrt(mean(dTrueInput(:,last,:).^2,"all","omitnan")), ...
        'VariableNames',{'caseName','positionRMSE','finalPositionRMSE', ...
        'operationalResidualRMSE','finalOperationalResidualRMSE', ...
        'residualAtTrueEtaRMSE','finalResidualAtTrueEtaRMSE'});
end
