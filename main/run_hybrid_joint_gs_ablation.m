function study = run_hybrid_joint_gs_ablation( ...
    seed,T,dt,nWatchers,hiddenLayerCount,makePlots)
%RUN_HYBRID_JOINT_GS_ABLATION Test the proposed two-timescale architecture.
%
% Compares the existing all-layer FOGM block-diagonal GS update against a
% canonical-factor hybrid.  The final case adds independent direct position
% measurements (20 m 1-sigma) at every watcher as an observability ablation.
%
% The joint-GS case is intentionally a first one-step Gauss-Newton test.
% It does not claim an exact centralized MAP/IEKS solution because watcher
% state-copy cross-covariances are still omitted.
%
% Example:
%   study = run_hybrid_joint_gs_ablation(101,600,0.1,4,3,true);

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = 0.1; end
    if nargin < 4 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end

    blockDiagonal = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "collaborative_all_fogm_cm_60s");
    jointGS = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "hybrid_canonical_factor_joint_gs_60s");
    positionAided = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "hybrid_canonical_factor_joint_gs_position_aided_60s");

    assert(isequal(blockDiagonal.truth,jointGS.truth), ...
        'Joint-GS case must use the same truth trajectory.');
    assert(isequal(blockDiagonal.trueAcceleration,jointGS.trueAcceleration), ...
        'Joint-GS case must use the same true acceleration.');
    assert(isequal(blockDiagonal.truth,positionAided.truth), ...
        'Position-aided case must use the same truth trajectory.');

    study.localOnly = blockDiagonal.localOnly;
    study.blockDiagonalGS = blockDiagonal.sharedBlock;
    study.hybridJointGS = jointGS.sharedBlock;
    study.positionAidedHybridJointGS = positionAided.sharedBlock;
    study.truth = jointGS.truth;
    study.trueAcceleration = jointGS.trueAcceleration;
    study.cfg = jointGS.cfg;
    study.architecture = jointGS.architecture;
    study.summary = makeHybridSummary(study);
    disp(study.summary);
    if makePlots
        study.figures = plotHybridStudy(study);
    else
        study.figures = struct;
    end
end

function summary = makeHybridSummary(study)
    names = ["Local independent block"; ...
        "Block-diagonal all-layer FOGM GS"; ...
        "Hybrid owner-EKF + canonical full-joint GS"; ...
        "Hybrid canonical full-joint GS + 20 m position aid"];
    cases = {study.localOnly,study.blockDiagonalGS,study.hybridJointGS, ...
        study.positionAidedHybridJointGS};
    values = zeros(numel(cases),7);
    for c = 1:numel(cases)
        result = cases{c};
        values(c,1) = result.positionRMSE;
        values(c,2) = result.velocityRMSE;
        values(c,3) = result.accelerationRMSE;
        values(c,4) = result.finalPositionRMSE;
        dof = 1;
        if isfield(result,'nisDegreesOfFreedom'), dof = result.nisDegreesOfFreedom; end
        values(c,5) = mean(result.nis(2:end,:),'all','omitnan')/dof;
        values(c,6) = result.parameterUploads;
        if isfield(result,'gsLinearizedCostDecrease')
            values(c,7) = sum(result.gsLinearizedCostDecrease);
        elseif isfield(result,'gsCostBefore')
            values(c,7) = sum(result.gsCostBefore-result.gsCostAfter,'omitnan');
        else
            values(c,7) = NaN;
        end
    end
    summary = table(names,values(:,1),values(:,2),values(:,3),values(:,4), ...
        values(:,5),values(:,6),values(:,7),'VariableNames', ...
        {'caseName','positionRMSE','velocityRMSE','accelerationRMSE', ...
        'finalPositionRMSE','meanNIS','parameterUploads', ...
        'sumAcceptedGSWindowCostDecrease'});
end

function figs = plotHybridStudy(study)
    t = study.cfg.time;
    cases = {study.localOnly,study.blockDiagonalGS,study.hybridJointGS, ...
        study.positionAidedHybridJointGS};
    labels = {'local independent block','block-diagonal GS', ...
        'hybrid canonical full-joint GS','hybrid + 20 m position aid'};
    colors = lines(numel(cases));
    figs.errors = figure('Name','Hybrid owner-EKF and joint-GS ablation');
    tiledlayout(4,1,'TileSpacing','compact');
    fields = {'positionError','velocityError','accelerationError'};
    titles = {'position estimation error','velocity estimation error', ...
        'acceleration approximation error'};
    units = {'RMSE [m]','RMSE [m/s]','RMSE [m/s^2]'};
    for row = 1:3
        nexttile; hold on; grid on;
        for c = 1:numel(cases)
            plot(t,cases{c}.(fields{row}),'LineWidth',1.15,'Color',colors(c,:));
        end
        title(titles{row}); ylabel(units{row});
        if row == 1, legend(labels,'Location','best'); end
    end
    nexttile; hold on; grid on;
    for c = 1:numel(cases)
        nisMean = mean(cases{c}.nis,2,'omitnan');
        dof = 1;
        if isfield(cases{c},'nisDegreesOfFreedom'), dof = cases{c}.nisDegreesOfFreedom; end
        nisMean = nisMean/dof;
        nisMean(1) = NaN;
        plot(t,movmean(nisMean,max(1,round(5/study.cfg.dt)),'omitnan'), ...
            'LineWidth',1.15,'Color',colors(c,:));
    end
    yline(1,'k--','NIS reference');
    title('bearing-innovation consistency (5 s moving mean)');
    ylabel('mean NIS'); xlabel('time [s]');

    figs.acceleration = figure('Name','Hybrid joint-GS acceleration reconstruction');
    tiledlayout(2,1,'TileSpacing','compact');
    for axisID = 1:2
        nexttile; hold on; grid on;
        plot(t,study.trueAcceleration(axisID,:),'k','LineWidth',1.5);
        for c = 1:numel(cases)
            dMean = mean(cases{c}.dHat,3);
            plot(t,dMean(axisID,:), ...
                'LineWidth',1.1,'Color',colors(c,:));
        end
        ylabel(sprintf('d_%c [m/s^2]','x'+axisID-1));
        if axisID == 1, legend([{'truth'},labels],'Location','best'); end
    end
    xlabel('time [s]');
end
