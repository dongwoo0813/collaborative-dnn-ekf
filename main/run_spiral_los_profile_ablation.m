function ablation = run_spiral_los_profile_ablation(seed,T,dt,scenario,makePlots)
%RUN_SPIRAL_LOS_PROFILE_ABLATION Compare validated radial and Eq.-(22) control.
    if nargin < 1 || isempty(seed), seed=101; end
    if nargin < 2 || isempty(T), T=600; end
    if nargin < 3 || isempty(dt), dt=.1; end
    if nargin < 4 || isempty(scenario), scenario="both"; end
    if nargin < 5 || isempty(makePlots), makePlots=true; end
    if string(scenario) == "both"
        ablation.nearParallel = run_spiral_los_profile_ablation(seed,T,dt,"near_parallel",makePlots);
        ablation.wellConditioned = run_spiral_los_profile_ablation(seed,T,dt,"well_conditioned",makePlots);
        ablation.summary = [ablation.nearParallel.summary;ablation.wellConditioned.summary];
        disp(ablation.summary); return;
    end
    base = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,false);
    radial = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,true,"local_radial",false);
    los = run_toy_distributed_additive_dnn_ekf(seed,T,false,scenario,dt,true,"local_los_profile",false);
    ablation.noManeuver=base.sharedAdditive; ablation.localRadial=radial.sharedAdditive; ablation.localLOSProfile=los.sharedAdditive;
    r={ablation.noManeuver,ablation.localRadial,ablation.localLOSProfile};
    names=["Shared WLS: no maneuver";"Shared WLS: local radial";"Shared WLS: local LOS profile"];
    ablation.summary=table(repmat(string(scenario),3,1),names,zeros(3,1),zeros(3,1),zeros(3,1), ...
        'VariableNames',{'scenario','caseName','positionRMSE','velocityRMSE','totalResidualRMSE'});
    for k=1:3
        e=r{k}.xhat(1:4,:,:)-repmat(r{k}.etaTrue,1,1,size(r{k}.xhat,3));
        ablation.summary.positionRMSE(k)=sqrt(mean(vecnorm(e(1:2,:,:),2,1).^2,'all'));
        ablation.summary.velocityRMSE(k)=sqrt(mean(vecnorm(e(3:4,:,:),2,1).^2,'all'));
        de=r{k}.dHat-repmat(r{k}.dTrue,1,1,size(r{k}.dHat,3));
        ablation.summary.totalResidualRMSE(k)=sqrt(mean(vecnorm(de,2,1).^2,'all'));
    end
    disp(ablation.summary);
    if makePlots
        f=figure('Name',"LOS-profile ablation: "+string(scenario)); tiledlayout(3,1,'TileSpacing','compact');
        for q=1:3
            nexttile; hold on;
            for k=1:3
                if q==1, e=r{k}.xhat(1:2,:,:)-repmat(r{k}.etaTrue(1:2,:),1,1,size(r{k}.xhat,3)); yl='position RMSE [m]';
                elseif q==2, e=r{k}.xhat(3:4,:,:)-repmat(r{k}.etaTrue(3:4,:),1,1,size(r{k}.xhat,3)); yl='velocity RMSE [m/s]';
                else, e=r{k}.dHat-repmat(r{k}.dTrue,1,1,size(r{k}.dHat,3)); yl='RMSE(ahat-a)'; end
                plot(r{k}.time,reshape(sqrt(mean(vecnorm(e,2,1).^2,3)),1,[]),'LineWidth',1.2);
            end
            grid on; ylabel(yl); if q==1, legend(names,'Location','best'); end; if q==3, xlabel('time [s]'); end
        end
        ablation.figure=f;
    end
end
