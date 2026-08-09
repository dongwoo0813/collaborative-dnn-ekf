function figures = plot_step10C2_GS_impulse_closed_loop(out,saveFigures)
%PLOT_STEP10C2_GS_IMPULSE_CLOSED_LOOP Plot independent live-run comparison.

    if nargin < 2, saveFigures = false; end
    results = {out.resLocal,out.resAdd,out.resOutputInformation};
    labels = ["Local single branch";"GS additive";"GS output-information fusion"];
    colors = lines(numel(results));
    cfg = out.cfgBase;
    N = numel(results{1}.time);
    posRmse = zeros(numel(results),N);
    forceNorm = zeros(numel(results),N);
    for i = 1:numel(results)
        res = results{i};
        truth = repmat(reshape(res.etaTrue,2*cfg.dim,N,1),1,1,cfg.Nw);
        posRmse(i,:) = reshape(sqrt(mean(vecnorm( ...
            res.xhat(1:cfg.dim,:,:)-truth(1:cfg.dim,:,:),2,1).^2,3,"omitnan")),1,[]);
        % watcherU is dim-by-time-by-watcher.  Average force magnitude
        % across watchers at each time, retaining the time dimension.
        forceByWatcher = squeeze(sqrt(sum(res.watcherU.^2,1)));
        forceNorm(i,:) = reshape(mean(forceByWatcher,2,"omitnan"),1,[]);
    end

    f1 = figure('Name','Step 10-C.2 independent closed-loop performance');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on;
    for i = 1:numel(results), plot(results{i}.time,posRmse(i,:), ...
            'Color',colors(i,:),'LineWidth',1.25); end
    title('Target position error: independent closed-loop runs');
    ylabel('RMSE across watchers [m]'); legend(labels,'Location','best'); grid on;
    nexttile; hold on;
    for i = 1:numel(results), plot(results{i}.time,forceNorm(i,:), ...
            'Color',colors(i,:),'LineWidth',1.1); end
    title('Mean watcher force magnitude'); xlabel('time [s]'); ylabel('Force [N]'); grid on;

    f2 = figure('Name','Step 10-C.2 impulse trigger telemetry');
    hold on;
    for i = 1:numel(results)
        radial = mean(results{i}.predictedRadialVariance,2,"omitnan");
        plot(results{i}.time,radial,'Color',colors(i,:),'LineWidth',1.15);
    end
    yline(cfg.control.obs.triggerRadialVariance,'k--','trigger threshold');
    title('Mean local radial covariance used by each live controller');
    xlabel('time [s]'); ylabel('radial variance [m^2]'); legend(labels,'Location','best'); grid on;
    figures = struct('performance',f1,'triggerTelemetry',f2);

    if saveFigures
        if ~isfolder('results'), mkdir('results'); end
        suffix = sprintf('%s_seed%d_T%d',string(cfg.dnn.mlp.inputMode), ...
            out.seed,round(cfg.T));
        exportgraphics(f1,fullfile('results',"step10C2_GS_impulse_closed_loop_" + suffix + "_performance.png"),'Resolution',180);
        exportgraphics(f2,fullfile('results',"step10C2_GS_impulse_closed_loop_" + suffix + "_trigger.png"),'Resolution',180);
    end
end
