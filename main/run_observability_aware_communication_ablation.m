function study = run_observability_aware_communication_ablation( ...
    seed,T,dt,nWatchers,hiddenLayerCount,makePlots,informationThreshold)
%RUN_OBSERVABILITY_AWARE_COMMUNICATION_ABLATION Compare GS schedules.
%
% Periodic case: canonical-factor full-joint GS every 60 s.
% Information-aware case: GS update after every block has accumulated a
% minimum prior-normalized information score of 0.05, subject to 15 s
% minimum interval and 120 s maximum silence.  The threshold is optional
% (default 1.0) and should be swept rather than treated as universal.
%
% Example:
%   study = run_observability_aware_communication_ablation(101,600,0.1,4,3,true);

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = 0.1; end
    if nargin < 4 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end
    if nargin < 7 || isempty(informationThreshold), informationThreshold = 1.0; end

    periodic = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "hybrid_canonical_factor_joint_gs_60s");
    informationAware = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "hybrid_canonical_factor_joint_gs_observability_aware",3,[], ...
        "all_to_all",informationThreshold);
    assert(isequal(periodic.truth,informationAware.truth), ...
        'Both schedules must use the same truth trajectory.');

    study.periodic60s = periodic.sharedBlock;
    study.informationAware = informationAware.sharedBlock;
    study.cfg = informationAware.cfg;
    study.truth = informationAware.truth;
    study.trueAcceleration = informationAware.trueAcceleration;
    study.summary = makeScheduleSummary(study);
    disp(study.summary);
    if makePlots
        study.figures = plotScheduleStudy(study);
    else
        study.figures = struct;
    end
end

function summary = makeScheduleSummary(study)
    cases = {study.periodic60s,study.informationAware};
    names = ["Periodic 60 s canonical full-joint GS"; ...
        "Observability-aware canonical full-joint GS"];
    values = zeros(2,7);
    for c = 1:2
        result = cases{c};
        uploadTimes = unique([result.uploadTimes{:}]);
        values(c,1:4) = [result.positionRMSE,result.velocityRMSE, ...
            result.accelerationRMSE,result.finalPositionRMSE];
        values(c,5) = mean(result.nis(2:end,:),'all','omitnan');
        values(c,6) = numel(uploadTimes);
        if numel(uploadTimes) > 1
            values(c,7) = mean(diff(uploadTimes));
        else
            values(c,7) = NaN;
        end
    end
    summary = table(names,values(:,1),values(:,2),values(:,3),values(:,4), ...
        values(:,5),values(:,6),values(:,7),'VariableNames', ...
        {'caseName','positionRMSE','velocityRMSE','accelerationRMSE', ...
        'finalPositionRMSE','meanNIS','gsSyncCount','meanSyncInterval'});
end

function figs = plotScheduleStudy(study)
    t = study.cfg.time;
    cases = {study.periodic60s,study.informationAware};
    labels = {'periodic 60 s','observability-aware'};
    colors = lines(2);
    figs.errors = figure('Name','Observability-aware GS communication');
    tiledlayout(4,1,'TileSpacing','compact');
    fields = {'positionError','velocityError','accelerationError'};
    titles = {'position estimation error','velocity estimation error', ...
        'acceleration approximation error'};
    units = {'RMSE [m]','RMSE [m/s]','RMSE [m/s^2]'};
    for row = 1:3
        nexttile; hold on; grid on;
        for c = 1:2
            plot(t,cases{c}.(fields{row}),'Color',colors(c,:), ...
                'LineWidth',1.2);
        end
        title(titles{row}); ylabel(units{row});
        if row == 1, legend(labels,'Location','best'); end
    end
    nexttile; hold on; grid on;
    for c = 1:2
        score = cases{c}.informationScore;
        plot(t,score,'Color',colors(c,:),'LineWidth',1.1);
        syncTimes = unique([cases{c}.uploadTimes{:}]);
        if ~isempty(syncTimes)
            scoreAtSync = interp1(t,score,syncTimes,'previous','extrap');
            scatter(syncTimes,scoreAtSync,28,colors(c,:),'filled');
        end
    end
    yline(study.cfg.collaborative.informationTriggerThreshold,'k--','information threshold');
    title('minimum prior-normalized block information');
    ylabel('score'); xlabel('time [s]');
    legend([labels,{'threshold'}],'Location','best');
end
