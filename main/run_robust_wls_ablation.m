function study = run_robust_wls_ablation(seeds,T,dt,watcherCounts,scenario,makePlots)
%RUN_ROBUST_WLS_ABLATION Test whether robust branch-consistency weighting
% makes an added watcher beneficial rather than harmful.  Every paired run
% resets the seed and replays the same nominal-derived local-radial burns.

    if nargin < 1 || isempty(seeds), seeds = 101:105; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(watcherCounts), watcherCounts = 3:4; end
    if nargin < 5 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end
    seeds = reshape(seeds,1,[]); watcherCounts = reshape(watcherCounts,1,[]);
    caseNames = ["Nominal EKF";"Local branch only"; ...
        "Shared covariance WLS";"Shared robust WLS"];
    metricNames = ["positionRMSE","velocityRMSE","accelerationRMSE","finalPositionRMSE"];
    raw = nan(numel(watcherCounts),numel(seeds),numel(caseNames),numel(metricNames));

    for w = 1:numel(watcherCounts)
        for s = 1:numel(seeds)
            common = {seeds(s),T,false,scenario,dt,true,"local_radial",true,watcherCounts(w)};
            standard = run_toy_distributed_additive_dnn_ekf(common{:},"parameter_covariance");
            robust = run_toy_distributed_additive_dnn_ekf(common{:},"robust_consistency");
            raw(w,s,1,:) = [standard.summary.positionRMSE(1),standard.summary.velocityRMSE(1), ...
                standard.summary.totalResidualRMSE(1),standard.summary.finalPositionRMSE(1)];
            raw(w,s,2,:) = [standard.summary.positionRMSE(2),standard.summary.velocityRMSE(2), ...
                standard.summary.totalResidualRMSE(2),standard.summary.finalPositionRMSE(2)];
            raw(w,s,3,:) = [standard.summary.positionRMSE(3),standard.summary.velocityRMSE(3), ...
                standard.summary.totalResidualRMSE(3),standard.summary.finalPositionRMSE(3)];
            raw(w,s,4,:) = [robust.summary.positionRMSE(3),robust.summary.velocityRMSE(3), ...
                robust.summary.totalResidualRMSE(3),robust.summary.finalPositionRMSE(3)];
        end
    end

    nRows = numel(watcherCounts)*numel(caseNames);
    summary = table(zeros(nRows,1),strings(nRows,1),zeros(nRows,1), ...
        zeros(nRows,1),zeros(nRows,1),zeros(nRows,1),zeros(nRows,1), ...
        zeros(nRows,1),zeros(nRows,1),zeros(nRows,1),zeros(nRows,1), ...
        'VariableNames',{'nWatchers','caseName','nSeeds', ...
        'positionRMSEMean','positionRMSEStd','velocityRMSEMean','velocityRMSEStd', ...
        'accelerationRMSEMean','accelerationRMSEStd', ...
        'finalPositionRMSEMean','finalPositionRMSEStd'});
    row = 0;
    for w = 1:numel(watcherCounts)
        for c = 1:numel(caseNames)
            row = row+1; values = reshape(raw(w,:,c,:),numel(seeds),numel(metricNames));
            summary.nWatchers(row) = watcherCounts(w); summary.caseName(row) = caseNames(c);
            summary.nSeeds(row) = numel(seeds);
            summary.positionRMSEMean(row) = mean(values(:,1)); summary.positionRMSEStd(row) = std(values(:,1));
            summary.velocityRMSEMean(row) = mean(values(:,2)); summary.velocityRMSEStd(row) = std(values(:,2));
            summary.accelerationRMSEMean(row) = mean(values(:,3)); summary.accelerationRMSEStd(row) = std(values(:,3));
            summary.finalPositionRMSEMean(row) = mean(values(:,4)); summary.finalPositionRMSEStd(row) = std(values(:,4));
        end
    end
    study = struct('seeds',seeds,'watcherCounts',watcherCounts,'scenario',string(scenario), ...
        'raw',raw,'metricNames',metricNames,'summary',summary);
    disp(summary);

    if makePlots
        fig = figure('Name',"Robust WLS ablation: "+string(scenario));
        tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
        titles = ["position RMSE","velocity RMSE","acceleration approximation RMSE","final position RMSE"];
        for q = 1:4
            nexttile; hold on;
            for c = 1:numel(caseNames)
                y = squeeze(mean(raw(:,:,c,q),2)); e = squeeze(std(raw(:,:,c,q),0,2));
                errorbar(watcherCounts,y,e,'-o','LineWidth',1.2,'DisplayName',caseNames(c));
            end
            grid on; xticks(watcherCounts); xlabel('number of watchers');
            ylabel(metricNames(q)); title(titles(q));
            if q == 1, legend('Location','best'); end
        end
        study.figure = fig;
    end
end
