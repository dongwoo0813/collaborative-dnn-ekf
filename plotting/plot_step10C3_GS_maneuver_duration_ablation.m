function figures = plot_step10C3_GS_maneuver_duration_ablation(out,saveFigures)
%PLOT_STEP10C3_GS_MANEUVER_DURATION_ABLATION Compact duration-ablation plot.
    if nargin < 2, saveFigures = false; end
    s = out.summary;
    f = figure('Name','Step 10-C.3 maneuver-duration ablation');
    tiledlayout(1,2,'TileSpacing','compact');
    nexttile;
    bar(categorical(strcat(s.schedule," | ",s.caseName)),s.finalPositionRMSE);
    title('Final position RMSE'); ylabel('RMSE [m]'); grid on;
    xtickangle(35);
    nexttile;
    bar(categorical(strcat(s.schedule," | ",s.caseName)),s.radialNEES);
    yline(1,'k--','consistent target');
    title('Radial NEES: controller covariance consistency'); ylabel('mean NEES'); grid on;
    xtickangle(35);
    methodFields = {"resLocal","resAdd","resOutputInformation"};
    methodLabels = ["Local single branch";"GS additive"; ...
        "GS output-information fusion"];
    f2 = figure('Name','Step 10-C.3 watcher displacement from coast');
    tiledlayout(3,1,'TileSpacing','compact');
    for i = 1:numel(methodFields)
        coast = out.coast.(methodFields{i});
        short = out.short.(methodFields{i});
        long = out.long.(methodFields{i});
        dShort = meanWatcherDisplacement(short.watcherR-coast.watcherR);
        dLong = meanWatcherDisplacement(long.watcherR-coast.watcherR);
        nexttile; hold on;
        plot(coast.time,dShort,'LineWidth',1.2);
        plot(coast.time,dLong,'LineWidth',1.2);
        title(methodLabels(i)); ylabel('mean |r_w-r_{w,coast}| [m]'); grid on;
        if i == 1, legend('10 s impulse','60 s impulse','Location','northwest'); end
        if i == numel(methodFields), xlabel('time [s]'); end
    end
    f3 = figure('Name','Step 10-C.3 residual-ablation metrics');
    tiledlayout(1,2,'TileSpacing','compact');
    labels = categorical(strcat(s.schedule," | ",s.caseName));
    nexttile;
    bar(labels,s.residualAtTrueEtaRMSE);
    title('Residual approximation error at true state');
    ylabel('RMSE [m/s^2]'); grid on; xtickangle(35);
    nexttile;
    bar(labels,s.operationalResidualRMSE);
    title('Operational residual-correction error');
    ylabel('RMSE [m/s^2]'); grid on; xtickangle(35);
    f4 = figure('Name','Step 10-C.3 position-error time history');
    tiledlayout(3,1,'TileSpacing','compact');
    scheduleResults = {out.coast,out.short,out.long};
    scheduleLabels = ["Coast only";"10 s impulse";"60 s impulse"];
    scheduleColors = lines(3);
    for i = 1:numel(methodFields)
        nexttile; hold on;
        for j = 1:numel(scheduleResults)
            res = scheduleResults{j}.(methodFields{i});
            rmse = positionRmseAcrossWatchers(res);
            plot(res.time,rmse,'Color',scheduleColors(j,:), ...
                'LineWidth',1.25);
        end
        title(methodLabels(i)); ylabel('position RMSE [m]'); grid on;
        if i == 1, legend(scheduleLabels,'Location','northwest'); end
        if i == numel(methodFields), xlabel('time [s]'); end
    end
    figures = struct('ablation',f,'watcherDisplacement',f2, ...
        'residualMetrics',f3,'positionTimeHistory',f4);
    if saveFigures
        if ~isfolder('results'), mkdir('results'); end
        exportgraphics(f,fullfile('results','step10C3_GS_maneuver_duration_ablation.png'),'Resolution',180);
        exportgraphics(f2,fullfile('results','step10C3_GS_watcher_displacement_from_coast.png'),'Resolution',180);
        exportgraphics(f3,fullfile('results','step10C3_GS_residual_ablation_metrics.png'),'Resolution',180);
        exportgraphics(f4,fullfile('results','step10C3_GS_position_error_time_history.png'),'Resolution',180);
    end
end

function d = meanWatcherDisplacement(deltaR)
    magnitude = squeeze(sqrt(sum(deltaR.^2,1)));
    d = mean(magnitude,2,"omitnan");
end

function rmse = positionRmseAcrossWatchers(res)
    dim = size(res.watcherR,1);
    N = numel(res.time);
    Nw = size(res.xhat,3);
    truth = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    errorNorm = vecnorm(res.xhat(1:dim,:,:)-truth(1:dim,:,:),2,1);
    rmse = reshape(sqrt(mean(errorNorm.^2,3,"omitnan")),1,[]);
end
