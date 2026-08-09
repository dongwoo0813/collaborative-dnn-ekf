function figureHandle = plot_step10C3_unknown_dynamics_estimation(out,saveFigure)
%PLOT_STEP10C3_UNKNOWN_DYNAMICS_ESTIMATION Presentation summary for the
% selected long LOS-profile maneuver case.  It separates online tracking
% accuracy from residual-function approximation at the true state.

    if nargin < 2, saveFigure = false; end
    selected = out.long;
    results = {selected.resLocal,selected.resAdd, ...
        selected.resOutputInformation};
    labels = ["Local single branch";"GS additive"; ...
        "GS output-information fusion"];
    colors = lines(numel(results));
    figureHandle = figure('Name','Unknown dynamics estimation: long LOS maneuver');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on;
    for i = 1:numel(results)
        plot(results{i}.time,positionRmse(results{i}), ...
            'Color',colors(i,:),'LineWidth',1.3);
    end
    title('State estimation under unknown target dynamics');
    ylabel('position RMSE across watchers [m]'); grid on;
    legend(labels,'Location','northwest');
    nexttile; hold on;
    for i = 1:numel(results)
        plot(results{i}.time,residualFunctionRmse(results{i}), ...
            'Color',colors(i,:),'LineWidth',1.3);
    end
    title('Unknown-residual function approximation');
    xlabel('time [s]'); ylabel('RMSE at true state [m/s^2]'); grid on;
    legend(labels,'Location','northeast');
    if saveFigure
        if ~isfolder('results'), mkdir('results'); end
        exportgraphics(figureHandle,fullfile('results', ...
            'step10C3_unknown_dynamics_estimation_time_history.png'), ...
            'Resolution',180);
    end
end

function value = positionRmse(res)
    dim = size(res.watcherR,1);
    N = numel(res.time);
    Nw = size(res.xhat,3);
    truth = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    e = vecnorm(res.xhat(1:dim,:,:)-truth(1:dim,:,:),2,1);
    value = reshape(sqrt(mean(e.^2,3,"omitnan")),1,[]);
end

function value = residualFunctionRmse(res)
    dim = size(res.watcherR,1);
    N = numel(res.time);
    Nw = size(res.xhat,3);
    truth = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
    e = vecnorm(res.dnnResidualAtTrueEta-truth,2,1);
    value = reshape(sqrt(mean(e.^2,3,"omitnan")),1,[]);
end
