function study = run_centralized_communication_period_sweep( ...
    seed,T,dt,nWatchers,hiddenLayerCount,periods,makePlots)
%RUN_CENTRALIZED_COMMUNICATION_PERIOD_SWEEP Raw-data GS latency sweep.
%
% Bearings remain sampled at dt.  Only the GS communication/canonical-DNN
% update period changes.  Packets received at a sync are replayed in time
% order, so this is a delay experiment rather than a sensor-rate experiment.
%
% Example:
%   study = run_centralized_communication_period_sweep( ...
%       101,600,0.1,4,3,[0.1 1 5 10 30 60],true);

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = 0.1; end
    if nargin < 4 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(periods), periods = [0.1 1 5 10 30 60]; end
    if nargin < 7 || isempty(makePlots), makePlots = true; end
    periods = unique(periods(:)');
    assert(all(periods >= dt),'Each communication period must be at least dt.');

    cases = cell(1,numel(periods));
    for c = 1:numel(periods)
        if abs(periods(c)-dt) <= 1e-10
            % Use the dedicated every-step implementation as the exact
            % zero-latency baseline, not the periodic replay bookkeeping.
            cases{c} = run_toy_block_structured_global_head_dnn_ekf( ...
                seed,T,false,nWatchers,dt,hiddenLayerCount, ...
                "centralized_common_dnn_step");
        else
            cases{c} = run_toy_block_structured_global_head_dnn_ekf( ...
                seed,T,false,nWatchers,dt,hiddenLayerCount, ...
                "centralized_common_dnn_periodic",[],[],[],[],periods(c));
        end
    end
    truth = cases{1}.truth;
    assert(all(cellfun(@(s) isequal(s.truth,truth),cases)), ...
        'All sweep cases must share the same truth.');

    positionRMSE = zeros(numel(periods),1); velocityRMSE = positionRMSE;
    accelerationRMSE = positionRMSE; finalPositionRMSE = positionRMSE;
    meanNIS = positionRMSE; syncCount = positionRMSE;
    for c = 1:numel(periods)
        r = cases{c}.sharedBlock;
        positionRMSE(c) = r.positionRMSE;
        velocityRMSE(c) = r.velocityRMSE;
        accelerationRMSE(c) = r.accelerationRMSE;
        finalPositionRMSE(c) = r.finalPositionRMSE;
        meanNIS(c) = mean(r.nis(2:end,:),'all','omitnan');
        if isfield(r,'syncTimes')
            syncCount(c) = numel(r.syncTimes);
        else
            syncCount(c) = T/dt;
        end
    end
    study.periods = periods; study.cases = cases; study.truth = truth;
    study.trueAcceleration = cases{1}.trueAcceleration; study.cfg = cases{1}.cfg;
    study.summary = table(periods(:),positionRMSE,velocityRMSE,accelerationRMSE, ...
        finalPositionRMSE,meanNIS,syncCount,'VariableNames', ...
        {'communicationPeriod','positionRMSE','velocityRMSE','accelerationRMSE', ...
        'finalPositionRMSE','meanNIS','numberOfGSSyncs'});
    disp(study.summary);
    if makePlots, study.figures = plotSweep(study); else, study.figures = struct; end
end

function figs = plotSweep(study)
    p = study.periods; s = study.summary;
    figs.metrics = figure('Name','Centralized GS communication-period sweep');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on; grid on;
    semilogx(p,s.positionRMSE,'o-','LineWidth',1.3,'DisplayName','position RMSE');
    semilogx(p,s.velocityRMSE,'o-','LineWidth',1.3,'DisplayName','velocity RMSE');
    semilogx(p,s.accelerationRMSE,'o-','LineWidth',1.3,'DisplayName','acceleration RMSE');
    xlabel('GS communication period [s]'); ylabel('RMSE (native units)');
    title('Raw bearings remain at 10 Hz; only GS synchronization is delayed'); legend('Location','best');
    nexttile; hold on; grid on;
    semilogx(p,s.meanNIS,'o-','LineWidth',1.3); yline(1,'k--','NIS reference');
    xlabel('GS communication period [s]'); ylabel('mean bearing NIS');
    title('Consistency versus communication period');

    figs.position = figure('Name','Centralized GS period sweep: position error');
    hold on; grid on; colors = lines(numel(p)); t = study.cfg.time;
    for c = 1:numel(p)
        plot(t,study.cases{c}.sharedBlock.positionError,'Color',colors(c,:), ...
            'LineWidth',1.1,'DisplayName',sprintf('%.3g s',p(c)));
    end
    xlabel('time [s]'); ylabel('position RMSE [m]');
    title('Real-time error: buffered bearings applied at GS synchronization');
    legend('Location','best');
end
