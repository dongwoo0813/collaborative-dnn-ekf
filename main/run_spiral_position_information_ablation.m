function ablation = run_spiral_position_information_ablation(seed,T,dt,scenario,makePlots)
%RUN_SPIRAL_POSITION_INFORMATION_ABLATION Position-observability controller test.
% Compares no maneuver, radial covariance, history-information, and a
% covariance-plus-history hybrid position controller.
    if nargin < 1 || isempty(seed), seed=101; end
    if nargin < 2 || isempty(T), T=600; end
    if nargin < 3 || isempty(dt), dt=.1; end
    if nargin < 4 || isempty(scenario), scenario="both"; end
    if nargin < 5 || isempty(makePlots), makePlots=true; end
    if string(scenario)=="both"
        ablation.nearParallel = run_spiral_position_information_ablation(seed,T,dt,"near_parallel",makePlots);
        ablation.wellConditioned = run_spiral_position_information_ablation(seed,T,dt,"well_conditioned",makePlots);
        ablation.summary = [ablation.nearParallel.summary;ablation.wellConditioned.summary];
        disp(ablation.summary); return;
    end

    base = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,false);
    radial = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,true,"local_radial",false);
    positionInfo = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,true,"local_position_information",false);
    hybrid = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,true,"local_hybrid_position",false);
    r = {base.sharedAdditive,radial.sharedAdditive,positionInfo.sharedAdditive,hybrid.sharedAdditive};
    names = ["Shared WLS: no maneuver";"Shared WLS: local radial"; ...
        "Shared WLS: history position information";"Shared WLS: hybrid position"];
    ablation.noManeuver=r{1}; ablation.localRadial=r{2};
    ablation.positionInformation=r{3}; ablation.hybridPosition=r{4};
    ablation.summary=table(repmat(string(scenario),4,1),names,zeros(4,1),zeros(4,1),zeros(4,1), ...
        'VariableNames',{'scenario','caseName','positionRMSE','velocityRMSE','totalResidualRMSE'});
    for k=1:numel(r)
        e=r{k}.xhat(1:4,:,:)-repmat(r{k}.etaTrue,1,1,size(r{k}.xhat,3));
        ablation.summary.positionRMSE(k)=sqrt(mean(vecnorm(e(1:2,:,:),2,1).^2,'all'));
        ablation.summary.velocityRMSE(k)=sqrt(mean(vecnorm(e(3:4,:,:),2,1).^2,'all'));
        de=r{k}.dHat-repmat(r{k}.dTrue,1,1,size(r{k}.dHat,3));
        ablation.summary.totalResidualRMSE(k)=sqrt(mean(vecnorm(de,2,1).^2,'all'));
    end
    disp(ablation.summary);
    if makePlots
        f=figure('Name',"Position-information ablation: "+string(scenario)); tiledlayout(4,1,'TileSpacing','compact');
        for q=1:4
            nexttile; hold on;
            for k=1:numel(r)
                if q==1, e=r{k}.xhat(1:2,:,:)-repmat(r{k}.etaTrue(1:2,:),1,1,size(r{k}.xhat,3)); yl='position RMSE [m]';
                elseif q==2, e=r{k}.xhat(3:4,:,:)-repmat(r{k}.etaTrue(3:4,:),1,1,size(r{k}.xhat,3)); yl='velocity RMSE [m/s]';
                elseif q==3, e=r{k}.dHat-repmat(r{k}.dTrue,1,1,size(r{k}.dHat,3)); yl='RMSE(ahat-a)';
                else
                    plot(r{k}.time,max(mean(r{k}.positionInfoLambdaMin,2),eps),'LineWidth',1.2); continue;
                end
                plot(r{k}.time,reshape(sqrt(mean(vecnorm(e,2,1).^2,3)),1,[]),'LineWidth',1.2);
            end
            grid on; ylabel(yl); if q==1, title('position-observability maneuver ablation'); legend(names,'Location','best'); end
            if q==4, set(gca,'YScale','log'); ylabel('\lambda_{min}(I_{r|v})'); end
            if q==4, xlabel('time [s]'); end
        end
        ablation.figure=f;
        fm=figure('Name',"Position-information maneuver commands: "+string(scenario)); tiledlayout(3,1,'TileSpacing','compact');
        for q=2:4
            nexttile; hold on;
            for i=1:size(r{q}.watcherA,3)
                plot(r{q}.time,reshape(vecnorm(r{q}.watcherA(:,:,i),2,1),1,[]), ...
                    'LineWidth',1.1,'DisplayName',"watcher "+i);
            end
            grid on; ylabel('||a_{w,i}||'); title(names(q)); legend('Location','best');
            if q==3, xlabel('time [s]'); end
        end
        ablation.maneuverFigure=fm;
        % Direct truth-versus-estimate view.  RMSE alone can hide bias and
        % phase error, so also show the mean local state/residual estimate.
        ablation.overlayFigure=plot_spiral_position_information_results(ablation);
    end
end
