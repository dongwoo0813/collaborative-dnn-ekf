function figures = plot_step10C1_GS_impulse_replay(out,saveFigures)
%PLOT_STEP10C1_GS_IMPULSE_REPLAY Plot the GS impulse replay comparison.
%
% figures = plot_step10C1_GS_impulse_replay(out)
% Uses the output structure returned by
% run_step10C1_GS_impulse_replay_output_fusion.  The first figure contains
% replayed estimator histories; the second identifies when the source
% controller applied impulses and how its local radial covariance evolved.

    if nargin < 2 || isempty(saveFigures), saveFigures = true; end
    required = ["source","resLocal","resAdd","resOutputInformation", ...
        "summary","cfgSource"];
    assert(all(isfield(out,required)), ...
        "Input must be a Step 10-C.1 output structure.");

    labels = ["Local single branch","GS additive", ...
        "GS output-information fusion"];
    results = {out.resLocal,out.resAdd,out.resOutputInformation};
    styles = {'-','--','-.'};
    colors = lines(3);
    histories = cell(3,1);
    for i = 1:3
        histories{i} = replayHistory(results{i},out.cfgSource);
    end

    f1 = figure('Name','Step 10-C.1 GS impulse replay performance');
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    nexttile; hold on;
    for i = 1:3
        plot(histories{i}.time,histories{i}.positionRMSE, ...
            'LineStyle',styles{i},'Color',colors(i,:),'LineWidth',1.35);
    end
    title('Target position error'); ylabel('RMSE across watchers [m]'); grid on;
    legend(labels,'Location','best');

    nexttile; hold on;
    for i = 1:3
        plot(histories{i}.time,histories{i}.residualTrueEtaRMSE, ...
            'LineStyle',styles{i},'Color',colors(i,:),'LineWidth',1.35);
    end
    title('Residual function error at true state');
    ylabel('RMSE [m/s^2]'); grid on; legend(labels,'Location','best');

    nexttile; hold on;
    for i = 1:3
        plot(histories{i}.time,histories{i}.operationalResidualRMSE, ...
            'LineStyle',styles{i},'Color',colors(i,:),'LineWidth',1.35);
    end
    title('Operational residual correction error');
    xlabel('time [s]'); ylabel('RMSE [m/s^2]'); grid on;

    nexttile;
    bar(categorical(out.summary.caseName), ...
        [out.summary.residualAtTrueEtaRMSE,out.summary.finalResidualAtTrueEtaRMSE]);
    ylabel('Residual RMSE [m/s^2]'); grid on;
    title('Whole-run and final-window residual error');
    legend('whole run','final 10%','Location','best'); xtickangle(18);

    f2 = figure('Name','Step 10-C.1 source impulse telemetry');
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
    forceNorm = sqrt(squeeze(sum(out.source.watcherU.^2,1)));
    if isvector(forceNorm), forceNorm = forceNorm(:); end
    nexttile; hold on;
    plot(out.source.time,forceNorm,'LineWidth',1.05);
    title('Source-run impulse force'); ylabel('Force [N]'); grid on;
    legend(compose('watcher %d',1:size(forceNorm,2)), ...
        'Location','bestoutside');

    nexttile; hold on;
    radialVar = out.source.predictedRadialVariance;
    plot(out.source.time,radialVar,'LineWidth',1.05);
    yline(out.cfgSource.control.obs.triggerRadialVariance,'k--', ...
        'trigger threshold','LabelHorizontalAlignment','left');
    title('Local radial covariance used for impulse triggering');
    xlabel('time [s]'); ylabel('radial variance [m^2]'); grid on;
    legend(compose('watcher %d',1:size(radialVar,2)), ...
        'Location','bestoutside');

    % Plot the learned unknown acceleration directly at one common input:
    % eta_true.  Averaging only combines the four watcher estimates for
    % readability; each estimator's raw per-watcher values remain in out.
    f3 = figure('Name','Step 10-C.1 true and approximated residual acceleration');
    dTrue = out.resLocal.trueResidual;
    dHat = zeros(size(dTrue,1),numel(out.resLocal.time),3);
    for i = 1:3
        dHat(:,:,i) = mean(results{i}.dnnResidualAtTrueEta,3,'omitnan');
    end
    componentNames = ["x component","y component","z component"];
    for q = 1:size(dTrue,1)
        subplot(size(dTrue,1),1,q); hold on;
        plot(out.resLocal.time,dTrue(q,:),'k','LineWidth',1.7);
        for i = 1:3
            plot(out.resLocal.time,dHat(q,:,i), ...
                'LineStyle',styles{i},'Color',colors(i,:),'LineWidth',1.15);
        end
        grid on;
        title("Unknown residual acceleration: " + componentNames(q));
        ylabel('acceleration [m/s^2]');
        if q == size(dTrue,1)
            xlabel('time [s]');
        end
        if q == 1
            legend(["true residual"; labels(:)],'Location','best');
        end
    end

    figures = struct('performance',f1,'sourceTelemetry',f2, ...
        'residualAcceleration',f3);
    if saveFigures
        if ~isfolder('results'), mkdir('results'); end
        seed = out.trajectory.seed;
        T = round(out.cfgSource.T);
        inputMode = string(out.cfgSource.dnn.mlp.inputMode);
        exportgraphics(f1,fullfile('results',sprintf( ...
            'step10C1_GS_impulse_replay_%s_performance_seed%d_T%d.png', ...
            inputMode,seed,T)), ...
            'Resolution',200);
        exportgraphics(f2,fullfile('results',sprintf( ...
            'step10C1_GS_impulse_replay_%s_impulses_seed%d_T%d.png', ...
            inputMode,seed,T)), ...
            'Resolution',200);
        exportgraphics(f3,fullfile('results',sprintf( ...
            'step10C1_GS_impulse_replay_%s_acceleration_seed%d_T%d.png', ...
            inputMode,seed,T)), ...
            'Resolution',200);
    end
end

function h = replayHistory(res,cfg)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    truth = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
    positionNorm2 = sum((res.xhat(1:dim,:,:)-truth(1:dim,:,:)).^2,1);
    operationalNorm2 = sum((res.dnnResidual-dTrue).^2,1);
    trueInputNorm2 = sum((res.dnnResidualAtTrueEta-dTrue).^2,1);
    h = struct();
    h.time = res.time(:);
    h.positionRMSE = sqrt(mean(reshape(positionNorm2,N,Nw),2,'omitnan'));
    h.operationalResidualRMSE = sqrt(mean(reshape(operationalNorm2,N,Nw),2,'omitnan'));
    h.residualTrueEtaRMSE = sqrt(mean(reshape(trueInputNorm2,N,Nw),2,'omitnan'));
end
