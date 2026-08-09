function study = run_distributed_block_information_ablation( ...
    seed,T,dt,nWatchers,hiddenLayerCount,makePlots)
%RUN_DISTRIBUTED_BLOCK_INFORMATION_ABLATION Compare five DNN learning cases.
%
% Preserves the existing three communication-ablation cases and adds a
% fourth case in which all watchers contribute bearing-innovation
% information to every DNN parameter block, and a fifth Phase-1 case that
% updates only the final output-layer blocks.  No full global DNN covariance
% is formed.  The information packets are exchanged every 60 seconds.
%
% Example:
%   study = run_distributed_block_information_ablation(101,600,0.1,4,3,true);

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = 0.1; end
    if nargin < 4 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end

    base = run_block_structured_communication_ablation( ...
        seed,T,dt,nWatchers,hiddenLayerCount,false);
    collaborativeRun = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount,"collaborative_info_60s");
    outputOnlyRun = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "collaborative_output_info_60s");
    fogmRun = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount,"collaborative_all_fogm_cm_60s");

    assert(isequal(base.truth,collaborativeRun.truth), ...
        'Collaborative-information case did not use the same truth data.');
    assert(isequal(base.trueAcceleration,collaborativeRun.trueAcceleration), ...
        'Collaborative-information case did not use the same acceleration data.');
    assert(isequal(base.truth,outputOnlyRun.truth), ...
        'Output-only collaborative case did not use the same truth data.');

    study.localOnly = base.localOnly;
    study.noSharing = base.noSharing;
    study.periodic60s = base.periodic60s;
    study.collaborativeInformation60s = collaborativeRun.sharedBlock;
    study.collaborativeOutputInformation60s = outputOnlyRun.sharedBlock;
    study.collaborativeAllFOGM60s = fogmRun.sharedBlock;
    study.cfg = collaborativeRun.cfg;
    study.truth = collaborativeRun.truth;
    study.trueAcceleration = collaborativeRun.trueAcceleration;
    study.architecture = collaborativeRun.architecture;
    study.summary = makeSummary(study);
    disp(study.summary);

    if makePlots
        study.figures = plotStudy(study);
    else
        study.figures = struct;
    end
end

function summary = makeSummary(study)
    names = ["Local independent block"; ...
        "Shared block: no communication"; ...
        "Shared block: raw periodic 60 s"; ...
        "Distributed all-parameter information EKF: 60 s"; ...
        "Distributed output-layer information EKF: 60 s"; ...
        "Distributed all-layer FOGM + covariance matching EKF: 60 s"];
    cases = {study.localOnly,study.noSharing,study.periodic60s, ...
        study.collaborativeInformation60s, ...
        study.collaborativeOutputInformation60s,study.collaborativeAllFOGM60s};
    nCases = numel(cases);
    values = zeros(nCases,6);
    for c = 1:nCases
        result = cases{c};
        values(c,1) = result.positionRMSE;
        values(c,2) = result.velocityRMSE;
        values(c,3) = result.accelerationRMSE;
        values(c,4) = result.finalPositionRMSE;
        values(c,5) = finiteMean(result.nis(2:end,:));
        values(c,6) = result.parameterUploads;
    end
    summary = table(names,values(:,1),values(:,2),values(:,3),values(:,4), ...
        values(:,5),values(:,6),'VariableNames',{'caseName','positionRMSE', ...
        'velocityRMSE','accelerationRMSE','finalPositionRMSE','meanNIS', ...
        'parameterUploads'});
end

function value = finiteMean(x)
    x = x(isfinite(x));
    if isempty(x), value = NaN; else, value = mean(x); end
end

function figs = plotStudy(study)
    t = study.cfg.time;
    cases = {study.localOnly,study.noSharing,study.periodic60s, ...
        study.collaborativeInformation60s, ...
        study.collaborativeOutputInformation60s,study.collaborativeAllFOGM60s};
    labels = {'local independent block','shared: no communication', ...
        'shared: raw periodic 60 s','distributed all-parameter EKF', ...
        'distributed output-layer EKF','distributed all-layer FOGM EKF'};
    colors = lines(numel(cases));

    figs.errors = figure('Name','Distributed block-information EKF ablation');
    tiledlayout(4,1,'TileSpacing','compact');
    fields = {'positionError','velocityError','accelerationError'};
    titles = {'position estimation error','velocity estimation error', ...
        'acceleration approximation error'};
    units = {'RMSE [m]','RMSE [m/s]','RMSE [m/s^2]'};
    for row = 1:3
        nexttile; hold on; grid on;
        for c = 1:numel(cases)
            plot(t,cases{c}.(fields{row}),'LineWidth',1.1,'Color',colors(c,:));
        end
        title(titles{row}); ylabel(units{row});
        if row == 1, legend(labels,'Location','best'); end
    end
    nexttile; hold on; grid on;
    for c = 1:numel(cases)
        nisMean = mean(cases{c}.nis,2,'omitnan');
        nisMean(1) = NaN;
        plot(t,movmean(nisMean,max(1,round(5/study.cfg.dt)),'omitnan'), ...
            'LineWidth',1.1,'Color',colors(c,:));
    end
    yline(1,'k--','NIS reference');
    title('bearing-innovation consistency (5 s moving mean)');
    ylabel('mean NIS'); xlabel('time [s]');

    figs.acceleration = figure('Name','Distributed information acceleration');
    tiledlayout(2,1,'TileSpacing','compact');
    for axisID = 1:2
        nexttile; hold on; grid on;
        plot(t,study.trueAcceleration(axisID,:),'k','LineWidth',1.5);
        for c = 1:numel(cases)
            dMean = mean(cases{c}.dHat,3);
            plot(t,dMean(axisID,:),'LineWidth',1.0,'Color',colors(c,:));
        end
        ylabel(sprintf('d_%c [m/s^2]','x'+axisID-1));
        if axisID == 1, legend([{'truth'},labels],'Location','best'); end
    end
    xlabel('time [s]');
end
