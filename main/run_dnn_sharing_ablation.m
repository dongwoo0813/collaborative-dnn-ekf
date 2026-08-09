function study = run_dnn_sharing_ablation(seeds,T,dt,watcherCounts,scenario,makePlots)
%RUN_DNN_SHARING_ABLATION Controlled scaling test for distributed DNN sharing.
%   Compares nominal EKF, local-only branch DNN, and shared directional-WLS
%   branches while holding the target, bearing-noise realization, watcher
%   geometry, and a nominal-derived local-radial maneuver schedule fixed.
%   Each watcher-count row aggregates independent random seeds.

    if nargin < 1 || isempty(seeds), seeds = 101:105; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(watcherCounts), watcherCounts = 1:4; end
    if nargin < 5 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end
    seeds = reshape(seeds,1,[]); watcherCounts = reshape(watcherCounts,1,[]);
    validateattributes(watcherCounts,{'numeric'},{'integer','>=',1,'<=',4});

    caseNames = ["Nominal EKF";"Local branch only";"Shared directional WLS"];
    metricNames = ["positionRMSE","velocityRMSE","accelerationRMSE","finalPositionRMSE","fleetDeltaV"];
    raw = nan(numel(watcherCounts),numel(seeds),numel(caseNames),numel(metricNames));

    for w = 1:numel(watcherCounts)
        for s = 1:numel(seeds)
            out = run_toy_distributed_additive_dnn_ekf(seeds(s),T,false, ...
                scenario,dt,true,"local_radial",true,watcherCounts(w));
            runSummary = out.summary;
            raw(w,s,:,1:4) = reshape([runSummary.positionRMSE,runSummary.velocityRMSE, ...
                runSummary.totalResidualRMSE,runSummary.finalPositionRMSE],1,1,numel(caseNames),4);
            % All three cases replay the same nominal-derived maneuver.  The
            % fleet delta-v is recorded once per case to make this explicit.
            fleetDeltaV = sum(vecnorm(out.nominal.watcherA,2,1),'all')*dt;
            raw(w,s,:,5) = fleetDeltaV;
        end
    end

    nRows = numel(watcherCounts)*numel(caseNames);
    summary = table(zeros(nRows,1),strings(nRows,1),repmat(string(scenario),nRows,1), ...
        zeros(nRows,1),zeros(nRows,1),zeros(nRows,1),zeros(nRows,1),zeros(nRows,1), ...
        zeros(nRows,1),zeros(nRows,1),zeros(nRows,1),zeros(nRows,1),zeros(nRows,1), ...
        'VariableNames',{'nWatchers','caseName','scenario','nSeeds', ...
        'positionRMSEMean','positionRMSEStd','velocityRMSEMean','velocityRMSEStd', ...
        'accelerationRMSEMean','accelerationRMSEStd', ...
        'finalPositionRMSEMean','finalPositionRMSEStd','fleetDeltaVMean'});
    row = 0;
    for w = 1:numel(watcherCounts)
        for c = 1:numel(caseNames)
            row = row+1;
            % Keep the [seed x metric] layout even for a one-seed smoke run.
            values = reshape(raw(w,:,c,:),numel(seeds),numel(metricNames));
            summary.nWatchers(row) = watcherCounts(w);
            summary.caseName(row) = caseNames(c);
            summary.nSeeds(row) = numel(seeds);
            summary.positionRMSEMean(row) = mean(values(:,1));
            summary.positionRMSEStd(row) = std(values(:,1),0);
            summary.velocityRMSEMean(row) = mean(values(:,2));
            summary.velocityRMSEStd(row) = std(values(:,2),0);
            summary.accelerationRMSEMean(row) = mean(values(:,3));
            summary.accelerationRMSEStd(row) = std(values(:,3),0);
            summary.finalPositionRMSEMean(row) = mean(values(:,4));
            summary.finalPositionRMSEStd(row) = std(values(:,4),0);
            summary.fleetDeltaVMean(row) = mean(values(:,5));
        end
    end

    study = struct('seeds',seeds,'watcherCounts',watcherCounts, ...
        'scenario',string(scenario),'maneuverObjective',"local_radial", ...
        'maneuverScheduleSource',"nominal EKF replayed in every DNN case", ...
        'raw',raw,'metricNames',metricNames,'summary',summary);
    disp(summary);
    if makePlots
        fig = figure('Name',"DNN sharing scaling ablation: "+string(scenario));
        tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
        plotMetrics = [1 2 3 4];
        plotTitles = ["position RMSE","velocity RMSE", ...
            "acceleration approximation RMSE","final position RMSE"];
        for q = 1:numel(plotMetrics)
            nexttile; hold on;
            for c = 1:numel(caseNames)
                y = squeeze(mean(raw(:,:,c,plotMetrics(q)),2));
                e = squeeze(std(raw(:,:,c,plotMetrics(q)),0,2));
                errorbar(watcherCounts,y,e,'-o','LineWidth',1.2, ...
                    'DisplayName',caseNames(c));
            end
            grid on; xticks(watcherCounts); xlabel('number of watchers');
            ylabel(metricNames(plotMetrics(q))); title(plotTitles(q));
            if q == 1, legend('Location','best'); end
        end
        study.figure = fig;
    end
end
