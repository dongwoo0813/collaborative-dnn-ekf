function ablation = run_spiral_maneuver_ablation(seed,T,dt,scenario,makePlots)
%RUN_SPIRAL_MANEUVER_ABLATION Isolate maneuver benefit to DNN approximation.
% Runs the identical spiral truth twice: once with all watcher maneuvers
% disabled and once with the distributed directional-rank controller.  The
% comparison uses only the shared directional-WLS estimator in each run.
% scenario may be "near_parallel", "well_conditioned", or "both".

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(scenario), scenario = "near_parallel"; end
    if nargin < 5 || isempty(makePlots), makePlots = true; end

    if string(scenario) == "both"
        ablation = struct;
        ablation.nearParallel = run_spiral_maneuver_ablation( ...
            seed,T,dt,"near_parallel",makePlots);
        ablation.wellConditioned = run_spiral_maneuver_ablation( ...
            seed,T,dt,"well_conditioned",makePlots);
        ablation.summary = [ablation.nearParallel.summary; ablation.wellConditioned.summary];
        fprintf('\nCombined geometry-ablation summary:\n');
        disp(ablation.summary);
        return;
    end

    fprintf('Spiral maneuver ablation: seed=%d, T=%.0f s, dt=%.2f s, scenario=%s\n', ...
        seed,T,dt,string(scenario));
    noManeuver = run_toy_distributed_additive_dnn_ekf( ...
        seed,T,false,scenario,dt,false);
    withManeuver = run_toy_distributed_additive_dnn_ekf( ...
        seed,T,false,scenario,dt,true);

    r0 = noManeuver.sharedAdditive;
    r1 = withManeuver.sharedAdditive;
    ablation.noManeuver = r0;
    ablation.withManeuver = r1;
    ablation.cfg = withManeuver.cfg;
    ablation.summary = ablationSummary(r0,r1);
    ablation.summary.scenario = repmat(string(scenario),height(ablation.summary),1);
    ablation.summary = movevars(ablation.summary,'scenario','Before','caseName');
    disp(ablation.summary);
    if makePlots, ablation.figures = plotManeuverAblation(r0,r1,string(scenario)); end
end

function summary = ablationSummary(r0,r1)
    [p0,v0,d0,pf0] = ablationMetrics(r0);
    [p1,v1,d1,pf1] = ablationMetrics(r1);
    names = ["Shared WLS: no maneuver"; "Shared WLS: directional-rank maneuver"];
    summary = table(names,[p0;p1],[v0;v1],[d0;d1],[pf0;pf1], ...
        'VariableNames',{'caseName','positionRMSE','velocityRMSE', ...
        'totalResidualRMSE','finalPositionRMSE'});
end

function [p,v,d,pf] = ablationMetrics(r)
    e = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
    p = sqrt(mean(vecnorm(e(1:2,:,:),2,1).^2,'all'));
    v = sqrt(mean(vecnorm(e(3:4,:,:),2,1).^2,'all'));
    de = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
    d = sqrt(mean(vecnorm(de,2,1).^2,'all'));
    last = round(.9*numel(r.time)):numel(r.time);
    pf = sqrt(mean(vecnorm(e(1:2,last,:),2,1).^2,'all'));
end

function figs = plotManeuverAblation(r0,r1,scenario)
    labels = ["no maneuver" "directional-rank maneuver"];
    colors = [0.85 0.25 0.10; 0.15 0.55 0.20];
    figs = struct;
    figs.ablation = figure('Name',"Spiral maneuver ablation: "+scenario);
    tiledlayout(4,1,'TileSpacing','compact');

    nexttile; hold on;
    plot(r0.time,caseRMSE(r0,'position'),'Color',colors(1,:),'LineWidth',1.2);
    plot(r1.time,caseRMSE(r1,'position'),'Color',colors(2,:),'LineWidth',1.2);
    grid on; ylabel('position RMSE [m]'); title('state-estimation effect');
    legend(labels,'Location','best');

    nexttile; hold on;
    plot(r0.time,caseRMSE(r0,'residual'),'Color',colors(1,:),'LineWidth',1.2);
    plot(r1.time,caseRMSE(r1,'residual'),'Color',colors(2,:),'LineWidth',1.2);
    grid on; ylabel('RMSE(\hat a_t-a_t)');
    title('acceleration-approximation effect');

    nexttile; hold on;
    plot(r0.time,mean(r0.geometryLambdaMin,2),'Color',colors(1,:),'LineWidth',1.2);
    plot(r1.time,mean(r1.geometryLambdaMin,2),'Color',colors(2,:),'LineWidth',1.2);
    grid on; ylabel('\lambda_{min}(\Omega)'); title('directional reconstruction rank margin');

    nexttile; hold on;
    plot(r0.time,mean(r0.geometryCondition,2),'Color',colors(1,:),'LineWidth',1.2);
    plot(r1.time,mean(r1.geometryCondition,2),'Color',colors(2,:),'LineWidth',1.2);
    grid on; xlabel('time [s]'); ylabel('\kappa(\Omega)');
    title('directional reconstruction condition number');

    figs.maneuvers = figure('Name',"Directional-rank maneuvers: "+scenario); hold on;
    for i=1:size(r1.watcherA,3)
        aNorm = vecnorm(squeeze(r1.watcherA(:,:,i)),2,1);
        plot(r1.time,aNorm,'LineWidth',1.1,'DisplayName',"watcher "+i);
    end
    grid on; xlabel('time [s]'); ylabel('||a_{w,i}||');
    title('maneuvers used in the ablation'); legend('Location','best');
end

function value = caseRMSE(r,kind)
    switch kind
        case 'position'
            e = r.xhat(1:2,:,:)-repmat(r.etaTrue(1:2,:),1,1,size(r.xhat,3));
        case 'residual'
            e = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
        otherwise
            error('Unknown metric kind.');
    end
    value = reshape(sqrt(mean(vecnorm(e,2,1).^2,3)),1,[]);
end
