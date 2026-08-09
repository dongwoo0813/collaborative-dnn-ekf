function figs = plot_toy_acceleration_sharing_observability(out,saveFigures)
%PLOT_TOY_ACCELERATION_SHARING_OBSERVABILITY Coast versus active toy result.
    if nargin < 2, saveFigures = false; end
    f1 = figure('Name','Toy shared-acceleration estimation errors');
    tiledlayout(3,1,'TileSpacing','compact');
    names = ["position";"velocity";"acceleration"];
    for q = 1:3
        nexttile; hold on;
        plot(out.localCoast.time,rmse(out.localCoast,q),'LineWidth',1.1);
        plot(out.localActive.time,rmse(out.localActive,q),'LineWidth',1.1);
        plot(out.coast.time,rmse(out.coast,q),'LineWidth',1.1);
        plot(out.active.time,rmse(out.active,q),'LineWidth',1.2);
        title(names(q)+" estimation error"); ylabel('RMSE'); grid on;
        if q==1
            legend('local coast (no sharing)','local covariance-rollout active', ...
                'shared acceleration / coast','shared acceleration / covariance-rollout active', ...
                'Location','northwest');
        end
        if q==3, xlabel('time [s]'); end
    end
    f2 = figure('Name','Toy shared acceleration estimate');
    tiledlayout(2,1,'TileSpacing','compact');
    for q = 1:2
        nexttile; hold on;
        plot(out.active.time,out.active.etaTrue(4+q,:),'k','LineWidth',1.3);
        for i=1:size(out.active.xhat,3), plot(out.active.time,squeeze(out.active.xhat(4+q,:,i)),'LineWidth',.9); end
        title("shared acceleration component "+q); ylabel('m/s^2'); grid on;
        if q==1, legend(["truth"; "watcher "+string((1:4)')],'Location','best'); end
        if q==2, xlabel('time [s]'); end
    end
    f3 = figure('Name','Toy local observability-triggered maneuvers');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on;
    for i=1:size(out.active.xhat,3)
        ratio = out.active.localMetric(:,i)./out.active.localReference(:,i);
        valid = isfinite(ratio);
        plot(out.active.time(valid),ratio(valid),'-o','MarkerSize',3, ...
            'LineWidth',1.1);
    end
    if out.cfg.control.mode == "position_only" || out.cfg.control.mode == "rollout_position"
        title('Local normalized radial-position variance');
        ylabel('radial variance / reference');
        for i=1:size(out.active.xhat,3)
            threshold = out.active.localTriggerThreshold(:,i);
            valid = isfinite(threshold);
            plot(out.active.time(valid),threshold(valid),'--','Color',[.25 .25 .25], ...
                'HandleVisibility','off');
        end
    else
        threshold = out.cfg.control.triggerFraction;
        title('Local normalized observability metric'); ylabel('D-opt score / reference');
        yline(threshold,'k--','trigger threshold');
    end
    legend("watcher "+string((1:size(out.active.xhat,3))'),'Location','best'); grid on;
    nexttile; hold on;
    for i=1:size(out.active.xhat,3)
        stairs(out.active.time,double(out.active.triggerLog(:,i))+2*(i-1),'LineWidth',1.1);
    end
    title('Independent local maneuver triggers'); xlabel('time [s]');
    ylabel('triggered watcher index'); grid on;
    f4 = [];
    if isfield(out.active,'gammaQ')
        f4 = figure('Name','Toy adaptive process-noise covariance matching');
        tiledlayout(2,1,'TileSpacing','compact');
        nexttile; hold on;
        for i=1:size(out.active.xhat,3)
            plot(out.active.time,out.active.cmRatio(:,i),'LineWidth',1.0);
        end
        yline(1,'k--','target'); grid on;
        title('Innovation covariance matching ratio');
        ylabel('S_{empirical} / S_{model}');
        legend("watcher "+string((1:size(out.active.xhat,3))'),'Location','best');
        nexttile; hold on;
        for i=1:size(out.active.xhat,3)
            semilogy(out.active.time,out.active.gammaQ(:,i),'LineWidth',1.0);
        end
        yline(1,'k--','nominal Q'); grid on;
        title('Adaptive white-jerk process-noise multiplier');
        xlabel('time [s]'); ylabel('\gamma_Q');
    end
    % Component-wise state time histories for every local posterior in the
    % active shared-acceleration case.  Each watcher keeps its own r,v
    % state, while the acceleration traces should overlap after consensus.
    f5 = figure('Name','Toy watcher component-wise state estimates', ...
        'Position',[80 40 1550 900]);
    tiledlayout(size(out.active.xhat,3),3,'TileSpacing','compact', ...
        'Padding','compact');
    stateNames = ["position" "velocity" "acceleration"];
    stateUnits = ["m" "m/s" "m/s^2"];
    componentColors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980];
    for i=1:size(out.active.xhat,3)
        for q=1:3
            rows = (q-1)*2+(1:2);
            nexttile; hold on;
            % Truth is black; solid/dashed distinguish x/y.  Estimates use
            % colour, so a component can be followed without ambiguity.
            plot(out.active.time,out.active.etaTrue(rows(1),:),'k-', ...
                'LineWidth',1.15,'DisplayName','truth x');
            plot(out.active.time,out.active.etaTrue(rows(2),:),'k--', ...
                'LineWidth',1.15,'DisplayName','truth y');
            plot(out.active.time,squeeze(out.active.xhat(rows(1),:,i)), ...
                '-','Color',componentColors(1,:),'LineWidth',1.0, ...
                'DisplayName','estimate x');
            plot(out.active.time,squeeze(out.active.xhat(rows(2),:,i)), ...
                '-','Color',componentColors(2,:),'LineWidth',1.0, ...
                'DisplayName','estimate y');
            title("watcher "+i+": "+stateNames(q));
            ylabel(stateUnits(q)); grid on;
            if i==size(out.active.xhat,3), xlabel('time [s]'); end
            if i==1 && q==1, legend('Location','best'); end
        end
    end
    f6 = [];
    if isfield(out.active,'fusionIncrement')
        f6 = figure('Name','Toy directional inertial acceleration information');
        tiledlayout(3,1,'TileSpacing','compact');
        labels = ["a_x information";"a_y information"];
        for q=1:2
            nexttile; hold on;
            for i=1:size(out.active.xhat,3)
                value = squeeze(out.active.fusionIncrement(q,q,:,i));
                semilogy(out.active.time,max(value,1e-16),'LineWidth',1.0);
            end
            title('Watcher directional acceleration information: '+labels(q));
            ylabel('\Delta\Lambda'); grid on;
            if q==1
                legend("watcher "+string((1:size(out.active.xhat,3))'), ...
                    'Location','best');
            end
        end
        nexttile; hold on;
        lambdaMin = nan(size(out.active.time));
        for k=1:numel(out.active.time)
            Y = out.active.fusionInformation(:,:,k);
            if all(isfinite(Y),'all')
                lambdaMin(k) = min(eig(.5*(Y+Y')));
            end
        end
        semilogy(out.active.time,max(lambdaMin,1e-16),'k','LineWidth',1.2);
        title('Network weakest inertial acceleration-information direction');
        xlabel('time [s]'); ylabel('\lambda_{min}(\Lambda_{fuse})'); grid on;
    end
    figs = struct('errors',f1,'acceleration',f2,'observability',f3, ...
        'covarianceMatching',f4,'watcherStateTimeHistory',f5, ...
        'directionalFusionInformation',f6);
    if saveFigures
        if ~isfolder('results'), mkdir('results'); end
        exportgraphics(f1,fullfile('results','toy_shared_acceleration_errors.png'),'Resolution',180);
        exportgraphics(f2,fullfile('results','toy_shared_acceleration_estimate.png'),'Resolution',180);
        exportgraphics(f3,fullfile('results','toy_local_observability_trigger.png'),'Resolution',180);
        if ~isempty(f4)
            exportgraphics(f4,fullfile('results','toy_adaptive_covariance_matching.png'),'Resolution',180);
        end
        exportgraphics(f5,fullfile('results','toy_watcher_component_time_history.png'),'Resolution',180);
        if ~isempty(f6)
            exportgraphics(f6,fullfile('results','toy_directional_fusion_information.png'),'Resolution',180);
        end
    end
end

function value = rmse(res,block)
    rows = (block-1)*2+(1:2); Nw = size(res.xhat,3);
    truth = repmat(res.etaTrue(rows,:),1,1,Nw);
    e = vecnorm(res.xhat(rows,:,:)-truth,2,1);
    value = reshape(sqrt(mean(e.^2,3)),1,[]);
end
