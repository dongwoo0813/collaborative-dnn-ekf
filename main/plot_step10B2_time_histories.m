function fig = plot_step10B2_time_histories(resultB2,saveFigure)
%PLOT_STEP10B2_TIME_HISTORIES Plot an existing Step 10-B2 result.
%   fig = plot_step10B2_time_histories(resultB2) reuses the trajectories
%   already stored in resultB2; it does not rerun any simulation.

if nargin < 2 || isempty(saveFigure), saveFigure = true; end

caseNames = ["GS-additive","GS-global-geometry", ...
    "P2P-additive","P2P-global-geometry"];
cases = { ...
    resultB2.runs{1}.out.resGSAdd, ...
    resultB2.runs{1}.out.resGSFIM, ...
    resultB2.runs{2}.out.resGSAdd, ...
    resultB2.runs{2}.out.resGSFIM};
colors = lines(4);
lineStyles = {'-','--','-','--'};

fig = figure('Name','Step 10-B2 range-bearing geometry time histories', ...
    'Color','w');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

for ic = 1:numel(cases)
    h = computeTimeHistories(cases{ic});
    nexttile(1);
    plot(h.time,h.positionRMSE,'Color',colors(ic,:), ...
        'LineStyle',lineStyles{ic},'LineWidth',1.35); hold on;
    nexttile(2);
    plot(h.time,h.rangeRMSE,'Color',colors(ic,:), ...
        'LineStyle',lineStyles{ic},'LineWidth',1.35); hold on;
    nexttile(3);
    plot(h.time,h.velocityRMSE,'Color',colors(ic,:), ...
        'LineStyle',lineStyles{ic},'LineWidth',1.35); hold on;
    nexttile(4);
    plot(h.time,h.residualRMSE,'Color',colors(ic,:), ...
        'LineStyle',lineStyles{ic},'LineWidth',1.35); hold on;
end

nexttile(1); title('Position error'); ylabel('RMSE [m]'); grid on;
legend(caseNames,'Location','best','Interpreter','none');
nexttile(2); title('Exact-range error'); ylabel('RMSE [m]'); grid on;
nexttile(3); title('Velocity error'); ylabel('RMSE [m/s]');
xlabel('time [s]'); grid on;
nexttile(4); title('Residual-vector error'); ylabel('RMSE [m/s^2]');
xlabel('time [s]'); grid on;

if saveFigure
    if ~exist('results','dir'), mkdir('results'); end
    fileName = sprintf('step10B2_time_history_seed%d_T%d.png', ...
        resultB2.seed,round(resultB2.simulationTime));
    exportgraphics(fig,fullfile('results',fileName),'Resolution',200);
    fprintf('Saved time-history plot: %s\n',fullfile('results',fileName));
end
end

function h = computeTimeHistories(res)
% RMS over watchers at every stored simulation time.
[nEta,N,Nw] = size(res.xhat);
dim = nEta/2;
truth = repmat(reshape(res.etaTrue,nEta,N,1),1,1,Nw);
ep = res.xhat(1:dim,:,:) - truth(1:dim,:,:);
ev = res.xhat(dim+(1:dim),:,:) - truth(dim+(1:dim),:,:);
positionNorm2 = reshape(sum(ep.^2,1),N,Nw);
velocityNorm2 = reshape(sum(ev.^2,1),N,Nw);

truthR = repmat(reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
rhoTrue = truthR-res.watcherR;
rhoHat = res.xhat(1:dim,:,:)-res.watcherR;
rangeTrue = reshape(sqrt(sum(rhoTrue.^2,1)),N,Nw);
rangeHat = reshape(sqrt(sum(rhoHat.^2,1)),N,Nw);
rangeError2 = (rangeHat-rangeTrue).^2;

trueResidual = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
residualError2 = reshape(sum((res.dnnResidual-trueResidual).^2,1),N,Nw);

h.time = res.time(:);
h.positionRMSE = sqrt(mean(positionNorm2,2,'omitnan'));
h.rangeRMSE = sqrt(mean(rangeError2,2,'omitnan'));
h.velocityRMSE = sqrt(mean(velocityNorm2,2,'omitnan'));
h.residualRMSE = sqrt(mean(residualError2,2,'omitnan'));
end
