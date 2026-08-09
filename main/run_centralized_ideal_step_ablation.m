function study = run_centralized_ideal_step_ablation( ...
    seed,T,dt,nWatchers,hiddenLayerCount,makePlots)
%RUN_CENTRALIZED_IDEAL_STEP_ABLATION Establish an every-step GS upper bound.
%
% This is the first rung of the realism ladder.  At every dt each watcher
% sends its raw bearing to the ground station, which runs one common-state,
% common-DNN augmented EKF.  It is a sequential centralized reference, not
% an offline oracle.  The other cases are unchanged and use the same truth,
% noisy bearings, initialization, DNN architecture, and FOGM prior.
%
% Example:
%   study = run_centralized_ideal_step_ablation(101,600,0.1,4,3,true);

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = 0.1; end
    if nargin < 4 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end

    blockDiagonal = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "collaborative_all_fogm_cm_60s");
    hybrid60 = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "hybrid_canonical_factor_joint_gs_60s");
    idealStep = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "centralized_common_dnn_step");

    assert(isequal(blockDiagonal.truth,idealStep.truth), ...
        'All cases must use exactly the same truth trajectory.');
    assert(isequal(blockDiagonal.trueAcceleration,idealStep.trueAcceleration), ...
        'All cases must use exactly the same unknown acceleration.');

    study.localOnly = blockDiagonal.localOnly;
    study.blockDiagonalGS = blockDiagonal.sharedBlock;
    study.hybrid60s = hybrid60.sharedBlock;
    study.centralizedIdealStep = idealStep.sharedBlock;
    study.truth = idealStep.truth;
    study.trueAcceleration = idealStep.trueAcceleration;
    study.cfg = idealStep.cfg;
    study.architecture = idealStep.architecture;
    study.summary = makeSummary(study);
    disp(study.summary);
    if makePlots
        study.figures = makePlotsForStudy(study);
    else
        study.figures = struct;
    end
end

function summary = makeSummary(study)
    names = ["Local independent block"; ...
        "Block-diagonal all-layer FOGM GS: 60 s"; ...
        "Hybrid canonical full-joint GS: 60 s"; ...
        "Centralized common state/DNN EKF: every step"];
    cases = {study.localOnly,study.blockDiagonalGS,study.hybrid60s, ...
        study.centralizedIdealStep};
    values = zeros(numel(cases),7);
    for c = 1:numel(cases)
        r = cases{c};
        values(c,1:4) = [r.positionRMSE,r.velocityRMSE,r.accelerationRMSE, ...
            r.finalPositionRMSE];
        values(c,5) = mean(r.nis(2:end,:),'all','omitnan');
        values(c,6) = r.parameterUploads;
        if isfield(r,'measurementPacketsToGS')
            values(c,7) = r.measurementPacketsToGS;
        else
            values(c,7) = NaN;
        end
    end
    summary = table(names,values(:,1),values(:,2),values(:,3),values(:,4), ...
        values(:,5),values(:,6),values(:,7),'VariableNames', ...
        {'caseName','positionRMSE','velocityRMSE','accelerationRMSE', ...
        'finalPositionRMSE','meanNIS','parameterUploads', ...
        'rawMeasurementPacketsToGS'});
end

function figs = makePlotsForStudy(study)
    t = study.cfg.time;
    cases = {study.localOnly,study.blockDiagonalGS,study.hybrid60s, ...
        study.centralizedIdealStep};
    labels = {'local independent block','block-diagonal GS: 60 s', ...
        'hybrid full-joint GS: 60 s','centralized common DNN: every step'};
    colors = lines(numel(cases));
    figs.errors = figure('Name','Centralized ideal-step GS reference');
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
        nisMean = mean(cases{c}.nis,2,'omitnan'); nisMean(1) = NaN;
        plot(t,movmean(nisMean,max(1,round(5/study.cfg.dt)),'omitnan'), ...
            'LineWidth',1.15,'Color',colors(c,:));
    end
    yline(1,'k--','NIS reference');
    title('bearing-innovation consistency (5 s moving mean)');
    ylabel('mean NIS'); xlabel('time [s]');

    figs.acceleration = figure('Name','Centralized ideal-step acceleration reconstruction');
    tiledlayout(2,1,'TileSpacing','compact');
    for axisID = 1:2
        nexttile; hold on; grid on;
        plot(t,study.trueAcceleration(axisID,:),'k','LineWidth',1.5);
        for c = 1:numel(cases)
            plot(t,mean(cases{c}.dHat(axisID,:,:),3), ...
                'LineWidth',1.1,'Color',colors(c,:));
        end
        ylabel(sprintf('d_%c [m/s^2]','x'+axisID-1));
        if axisID == 1, legend([{'truth'},labels],'Location','best'); end
    end
    xlabel('time [s]');
end
