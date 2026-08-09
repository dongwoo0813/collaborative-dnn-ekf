function figs = plot_toy_global_fused_angles_only_observability(out,saveFigures)
%PLOT_TOY_GLOBAL_FUSED_ANGLES_ONLY_OBSERVABILITY Diagnostics for global toy.
    if nargin<2, saveFigures=false; end
    f1=figure('Name','Global fused angles-only toy errors'); tiledlayout(3,1,'TileSpacing','compact');
    labels=["position RMSE [m]";"velocity RMSE [m/s]";"acceleration RMSE [m/s^2]"];
    for q=1:3
        nexttile; hold on; plotMetric(out.coast,q); plotMetric(out.active,q); title(labels(q)); grid on;
        if q==1, legend('global coast','global + local maneuvers','Location','northwest'); end
        if q==3, xlabel('time [s]'); end
    end
    f2=figure('Name','Global fused local trigger diagnostics'); tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on;
    for i=1:4
        ratio=out.active.metric(:,i)./out.active.reference(:,i); valid=isfinite(ratio); plot(out.active.time(valid),ratio(valid),'-o','MarkerSize',3);
        th=out.active.threshold(:,i); valid=isfinite(th); plot(out.active.time(valid),th(valid),'--','Color',[.3 .3 .3],'HandleVisibility','off');
    end
    title('Watcher-local normalized radial covariance'); ylabel('variance / initial reference'); legend("watcher "+string((1:4)'),'Location','best'); grid on;
    nexttile; hold on;
    for i=1:4, stairs(out.active.time,double(out.active.trigger(:,i))+2*(i-1),'LineWidth',1.1); end
    title('Independent watcher maneuver triggers'); xlabel('time [s]'); ylabel('triggered watcher index'); grid on;
    figs=struct('errors',f1,'triggers',f2);
    if saveFigures
        if ~isfolder('results'), mkdir('results'); end
        exportgraphics(f1,fullfile('results','toy_global_fused_errors.png'),'Resolution',180);
        exportgraphics(f2,fullfile('results','toy_global_fused_triggers.png'),'Resolution',180);
    end
end
function plotMetric(r,q)
    rows=(q-1)*2+(1:2); e=vecnorm(r.xhat(rows,:)-r.xTrue(rows,:),2,1); plot(r.time,e,'LineWidth',1.2);
end
