function frozen = run_step10A5_frozen_estimator_ablation(trajectory,makePlots)
%RUN_STEP10A5_FROZEN_ESTIMATOR_ABLATION Compare estimators on one trajectory.
%
% The target truth, watcher trajectory, and bearing-noise sequence are held
% fixed.  This separates estimator effects from active-motion effects.

    if nargin < 2, makePlots = false; end
    addpath(genpath(pwd));
    if ischar(trajectory) || isstring(trajectory)
        loaded = load(trajectory,"trajectory");
        trajectory = loaded.trajectory;
    end
    assert(isfield(trajectory,"cfg"),"Trajectory export has no cfg field.");
    cfgBase = trajectory.cfg;
    [~,N,Nw] = size(trajectory.watcherR);
    rng(trajectory.seed);
    bearingNoise = randn(N,Nw);

    cfgCommon = cfgBase;
    cfgCommon.replay.enabled = true;
    cfgCommon.replay.etaTrue = trajectory.etaTrue;
    cfgCommon.replay.watcherR = trajectory.watcherR;
    cfgCommon.replay.watcherV = trajectory.watcherV;
    if isfield(trajectory,"watcherU")
        cfgCommon.replay.watcherU = trajectory.watcherU;
    end
    cfgCommon.replay.bearingNoise = bearingNoise;
    cfgCommon.meas.type = "bearing";
    cfgCommon.meas.availabilityMode = "always";
    cfgCommon.fov.enabled = false;

    cfgPhys = cfgCommon;
    cfgPhys.step.name = "step10A5_physical_EKF";
    cfgPhys.gs.enabled = false;
    cfgPhys.control.translationMode = "none";
    cfgPhys.estimator.type = "physical_EKF";

    cfgOracle = cfgCommon;
    cfgOracle.step.name = "step10A5_oracle_residual_EKF";
    cfgOracle.estimator.type = "oracle_residual_EKF";
    cfgOracle.dnn.predictionResidualSource = "oracle";
    cfgOracle.gs.enabled = false;

    cfgLocal = cfgCommon;
    cfgLocal.step.name = "step10A5_local_DNN_EKF";
    cfgLocal.dnn.predictionResidualSource = "local_DNN";
    cfgLocal.gs.enabled = false;

    cfgAdd = cfgCommon;
    cfgAdd.step.name = "step10A5_GS_additive";
    cfgAdd.gs.compositeMode = "additive";

    cfgFIM = cfgCommon;
    cfgFIM.step.name = "step10A5_GS_FIM_weighted_additive";
    cfgFIM.gs.compositeMode = "fim_weighted_additive";

    fprintf("Step 10-A.5 frozen-motion estimator ablation\n");
    fprintf("source trajectory=%s, seed=%g, T=%.1f s, dt=%.4g s\n", ...
        trajectory.sourceCase,trajectory.seed,cfgCommon.T,cfgCommon.dt);

    rng(trajectory.seed); resPhys = simulatePhysicalEKF(cfgPhys);
    rng(trajectory.seed); resOracle = simulateLocalDNNEKF(cfgOracle);
    rng(trajectory.seed); resLocal = simulateLocalDNNEKF(cfgLocal);
    rng(trajectory.seed); resAdd = simulate_GS_DNN_EKF(cfgAdd);
    rng(trajectory.seed); resFIM = simulate_GS_DNN_EKF(cfgFIM);

    results = {resPhys,resOracle,resLocal,resAdd,resFIM};
    cfgs = {cfgPhys,cfgOracle,cfgLocal,cfgAdd,cfgFIM};
    labels = ["physical-EKF";"oracle-residual-EKF";"local-DNN-EKF"; ...
        "GS-additive";"GS-FIM-weighted-additive"];
    summary = table();
    for i = 1:numel(results)
        row = summarizeFrozenResult(results{i},cfgs{i},labels(i));
        if isempty(summary), summary = row; else, summary = [summary;row]; end %#ok<AGROW>
    end
    disp(summary);
    frozen = struct("trajectory",trajectory,"summary",summary, ...
        "resPhysical",resPhys,"resOracle",resOracle,"resLocal",resLocal, ...
        "resAdd",resAdd,"resFIM",resFIM,"cfgPhysical",cfgPhys, ...
        "cfgOracle",cfgOracle,"cfgLocal",cfgLocal,"cfgAdd",cfgAdd,"cfgFIM",cfgFIM);

    if makePlots
        figure("Name","Step 10-A.5 frozen estimator ablation");
        tiledlayout(1,2);
        nexttile; hold on;
        for i = 1:numel(results)
            plot(results{i}.time,meanPositionErrorFrozen(results{i},cfgs{i}));
        end
        grid on; xlabel("time [s]"); ylabel("mean position error [m]");
        legend(labels,"Location","best");
        nexttile; bar(categorical(summary.caseName),summary.finalPositionRMSE);
        grid on; ylabel("final position RMSE [m]"); xtickangle(35);
    end
end

function row = summarizeFrozenResult(res,cfg,label)
    dim = cfg.dim; [~,N,Nw] = size(res.xhat);
    finalIdx = max(1,round(.9*N)):N;
    truth = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    p = reshape(vecnorm(res.xhat(1:dim,:,:)-truth(1:dim,:,:),2,1),N,Nw);
    v = reshape(vecnorm(res.xhat(dim+(1:dim),:,:)-truth(dim+(1:dim),:,:),2,1),N,Nw);
    rTruth = repmat(reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
    rangeErr = reshape(vecnorm(res.xhat(1:dim,:,:)-res.watcherR,2,1)- ...
        vecnorm(rTruth-res.watcherR,2,1),N,Nw);
    if isfield(res,"dnnResidual")
        dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
        dErr = reshape(vecnorm(res.dnnResidual-dTrue,2,1),N,Nw);
        residualRMSE = sqrt(mean(dErr(:).^2,"omitnan"));
        finalResidualRMSE = sqrt(mean(dErr(finalIdx,:).^2,"all","omitnan"));
    else
        residualRMSE = NaN; finalResidualRMSE = NaN;
    end
    meanNIS = NaN;
    if isfield(res,"NIS"), meanNIS = mean(res.NIS(isfinite(res.NIS)),"omitnan"); end
    row = table(string(label),sqrt(mean(p(:).^2,"omitnan")), ...
        sqrt(mean(p(finalIdx,:).^2,"all","omitnan")), ...
        sqrt(mean(rangeErr(:).^2,"omitnan")), ...
        sqrt(mean(rangeErr(finalIdx,:).^2,"all","omitnan")), ...
        sqrt(mean(v(:).^2,"omitnan")),sqrt(mean(v(finalIdx,:).^2,"all","omitnan")), ...
        residualRMSE,finalResidualRMSE,meanNIS, ...
        'VariableNames',{'caseName','positionRMSE','finalPositionRMSE', ...
        'rangeRMSE','finalRangeRMSE','velocityRMSE','finalVelocityRMSE', ...
        'residualVectorRMSE','finalResidualVectorRMSE','meanNIS'});
end

function e = meanPositionErrorFrozen(res,cfg)
    dim = cfg.dim; [~,N,Nw] = size(res.xhat);
    truth = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    e = mean(reshape(vecnorm(res.xhat(1:dim,:,:)-truth(1:dim,:,:),2,1),N,Nw),2,"omitnan");
end
