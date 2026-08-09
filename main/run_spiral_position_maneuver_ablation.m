function ablation = run_spiral_position_maneuver_ablation(seed,T,dt,scenario,makePlots)
%RUN_SPIRAL_POSITION_MANEUVER_ABLATION Compare local position objectives.
% Same target/initial seed; compares no maneuver, two covariance heuristics,
% and the LOS-profile information-rank controller.

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 5 || isempty(makePlots), makePlots = true; end

    if string(scenario) == "both"
        ablation = struct;
        ablation.nearParallel = run_spiral_position_maneuver_ablation(seed,T,dt,"near_parallel",makePlots);
        ablation.wellConditioned = run_spiral_position_maneuver_ablation(seed,T,dt,"well_conditioned",makePlots);
        ablation.summary = [ablation.nearParallel.summary; ablation.wellConditioned.summary];
        disp(ablation.summary); return;
    end

    fprintf('Position-objective maneuver ablation: seed=%d, T=%.0f s, dt=%.2f, scenario=%s\n', ...
        seed,T,dt,string(scenario));
    baseline = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,false);
    % A maneuver must be selected from the estimator that will actually use
    % it. Replaying the nominal-only schedule would use a badly biased state
    % estimate when the unknown residual is large, defeating this ablation.
    radial = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,true,"local_radial",false);
    position = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,true,"local_position",false);
    information = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,true,"local_information",false);

    ablation.noManeuver = baseline.sharedAdditive;
    ablation.localRadial = radial.sharedAdditive;
    ablation.localPosition = position.sharedAdditive;
    ablation.localInformation = information.sharedAdditive;
    ablation.cfg = position.cfg;
    results = {ablation.noManeuver,ablation.localRadial,ablation.localPosition,ablation.localInformation};
    names = ["Shared WLS: no maneuver"; "Shared WLS: local radial"; "Shared WLS: local position trace"; "Shared WLS: local information rank"];
    summary = table(names,zeros(4,1),zeros(4,1),zeros(4,1),zeros(4,1), ...
        'VariableNames',{'caseName','positionRMSE','velocityRMSE','totalResidualRMSE','finalPositionRMSE'});
    for n=1:4
        [summary.positionRMSE(n),summary.velocityRMSE(n),summary.totalResidualRMSE(n),summary.finalPositionRMSE(n)] = metrics(results{n});
    end
    summary.scenario = repmat(string(scenario),height(summary),1);
    ablation.summary = movevars(summary,'scenario','Before','caseName');
    disp(ablation.summary);
    if makePlots, ablation.figures = plotAblation(results,names,string(scenario)); end
end

function [p,v,d,pf] = metrics(r)
    e = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
    p = sqrt(mean(vecnorm(e(1:2,:,:),2,1).^2,'all'));
    v = sqrt(mean(vecnorm(e(3:4,:,:),2,1).^2,'all'));
    de = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
    d = sqrt(mean(vecnorm(de,2,1).^2,'all'));
    idx = round(.9*numel(r.time)):numel(r.time);
    pf = sqrt(mean(vecnorm(e(1:2,idx,:),2,1).^2,'all'));
end

function figs = plotAblation(r,names,scenario)
    colors = [0.35 0.35 0.35; 0.85 0.30 0.10; 0.15 0.55 0.20; 0.20 0.35 0.85];
    figs.comparison = figure('Name',"Position maneuver ablation: "+scenario);
    tiledlayout(3,1,'TileSpacing','compact');
    kinds = {'position','velocity','residual'};
    titles = ["position-estimation effect" "velocity-estimation effect" "acceleration-approximation effect"];
    ylabels = ["position RMSE [m]" "velocity RMSE [m/s]" "RMSE(ahat-a)"];
    for q=1:3
        nexttile; hold on;
        for n=1:numel(r), plot(r{n}.time,seriesRMSE(r{n},kinds{q}),'Color',colors(n,:),'LineWidth',1.2); end
        grid on; title(titles(q)); ylabel(ylabels(q));
        if q == 1, legend(names,'Location','best'); end
        if q == 3, xlabel('time [s]'); end
    end
    figs.maneuvers = figure('Name',"Local maneuver commands: "+scenario);
    tiledlayout(3,1,'TileSpacing','compact');
    for q=1:3
        nexttile; hold on;
        rr = r{q+1};
        for i=1:size(rr.watcherA,3)
            plot(rr.time,vecnorm(squeeze(rr.watcherA(:,:,i)),2,1),'LineWidth',1.0, ...
                'DisplayName',"watcher "+i);
        end
        grid on; ylabel('||a_{w,i}||'); title(names(q+1)); legend('Location','best');
        if q == 3, xlabel('time [s]'); end
    end
end

function value = seriesRMSE(r,kind)
    switch kind
        case 'position', e = r.xhat(1:2,:,:)-repmat(r.etaTrue(1:2,:),1,1,size(r.xhat,3));
        case 'velocity', e = r.xhat(3:4,:,:)-repmat(r.etaTrue(3:4,:),1,1,size(r.xhat,3));
        case 'residual', e = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
    end
    value = reshape(sqrt(mean(vecnorm(e,2,1).^2,3)),1,[]);
end
