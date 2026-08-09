function replay = run_step10A2_deterministic_replay(trajectory,makePlots)
%RUN_STEP10A2_DETERMINISTIC_REPLAY Compare additive/FIM on one trajectory.
%
% trajectory may be the struct returned by exportStep10A2Trajectory or a
% MAT-file that contains a variable named trajectory.

    if nargin < 2
        makePlots = false;
    end
    addpath(genpath(pwd));
    if ischar(trajectory) || isstring(trajectory)
        loaded = load(trajectory,"trajectory");
        trajectory = loaded.trajectory;
    end
    assert(isfield(trajectory,"schema") && ...
        trajectory.schema == "step10A2_trajectory_v1", ...
        "Input is not a Step 10-A.2 trajectory export.");

    cfgBase = trajectory.cfg;
    cfgBase.replay.enabled = true;
    cfgBase.replay.etaTrue = trajectory.etaTrue;
    cfgBase.replay.watcherR = trajectory.watcherR;
    cfgBase.replay.watcherV = trajectory.watcherV;
    cfgBase.replay.watcherU = trajectory.watcherU;

    cfgAdd = cfgBase;
    cfgAdd.step.name = "step10A2_replay_additive";
    cfgAdd.gs.compositeMode = "additive";
    cfgFIM = cfgBase;
    cfgFIM.step.name = "step10A2_replay_FIM_weighted_additive";
    cfgFIM.gs.compositeMode = "fim_weighted_additive";

    fprintf("Step 10-A.2 deterministic replay, source=%s\n", ...
        trajectory.sourceCase);
    rng(trajectory.seed); resAdd = simulate_GS_DNN_EKF(cfgAdd);
    rng(trajectory.seed); resFIM = simulate_GS_DNN_EKF(cfgFIM);

    assert(isequaln(resAdd.etaTrue,trajectory.etaTrue) && ...
        isequaln(resFIM.etaTrue,trajectory.etaTrue), ...
        "Replay target truth differs from exported trajectory.");
    assert(isequaln(resAdd.watcherR,trajectory.watcherR) && ...
        isequaln(resFIM.watcherR,trajectory.watcherR), ...
        "Replay watcher trajectory differs from exported trajectory.");

    summary = [summarizeStep10A2Replay(resAdd,cfgAdd,"additive"); ...
        summarizeStep10A2Replay(resFIM,cfgFIM,"FIM-weighted-additive")];
    disp(summary);
    replay = struct("trajectory",trajectory,"resAdd",resAdd,"resFIM",resFIM, ...
        "cfgAdd",cfgAdd,"cfgFIM",cfgFIM,"summary",summary);

    if makePlots
        figure("Name","Step 10-A.2 deterministic replay");
        tiledlayout(1,2);
        nexttile; plot(resAdd.time,meanPositionError(resAdd,cfgAdd), ...
            resFIM.time,meanPositionError(resFIM,cfgFIM)); grid on;
        xlabel("time [s]"); ylabel("mean position error [m]");
        legend("additive","FIM-weighted-additive","Location","best");
        nexttile; bar(categorical(summary.caseName),summary.finalPositionRMSE);
        ylabel("final position RMSE [m]"); grid on;
    end
end

function value = meanPositionError(res,cfg)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    err = vecnorm(res.xhat(1:dim,:,:)-etaTrue(1:dim,:,:),2,1);
    value = mean(reshape(err,N,Nw),2,"omitnan");
end
