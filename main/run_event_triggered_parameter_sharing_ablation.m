function study = run_event_triggered_parameter_sharing_ablation(seed,T,dt,scenario,makePlots,nWatchers)
%RUN_EVENT_TRIGGERED_PARAMETER_SHARING_ABLATION
% Compare per-step parameter sharing with a ground-station event trigger.
% Every watcher still adapts only its own output head from angle-only
% measurements.  In the event-triggered case, remote branches use the last
% communicated posterior rather than the newest local posterior.
%
% Example:
%   study = run_event_triggered_parameter_sharing_ablation(101,600,.1, ...
%       "well_conditioned",true,4);

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 5 || isempty(makePlots), makePlots = true; end
    if nargin < 6 || isempty(nWatchers), nWatchers = 4; end
    common = {seed,T,false,scenario,dt,true,"local_radial",true, ...
        nWatchers,"parameter_covariance","additive_vector"};
    instantaneous = runToyQuiet(common,"instantaneous");
    eventTriggered = runToyQuiet(common,"event_triggered");
    cases = {instantaneous.sharedAdditive,eventTriggered.sharedAdditive};
    labels = ["per-step sharing"; "event-triggered sharing"];
    summary = table(labels,zeros(2,1),zeros(2,1),zeros(2,1),zeros(2,1), ...
        'VariableNames',{'caseName','positionRMSE','velocityRMSE', ...
        'accelerationRMSE','parameterUploads'});
    for c = 1:2
        r = cases{c};
        e = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
        de = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
        summary.positionRMSE(c) = sqrt(mean(vecnorm(e(1:2,:,:),2,1).^2,'all'));
        summary.velocityRMSE(c) = sqrt(mean(vecnorm(e(3:4,:,:),2,1).^2,'all'));
        summary.accelerationRMSE(c) = sqrt(mean(vecnorm(de,2,1).^2,'all'));
        summary.parameterUploads(c) = nnz(r.communicationEvent);
    end
    study = struct('instantaneous',instantaneous,'eventTriggered',eventTriggered, ...
        'summary',summary,'communicationTrigger', ...
        "Mahalanobis posterior change >= 4 with a 5 s per-watcher minimum interval");
    disp(summary);

    if makePlots
        fig = figure('Name','Event-triggered parameter-sharing ablation');
        tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
        for q = 1:2
            r = cases{q}; e = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
            positionRMSE = squeeze(sqrt(mean(vecnorm(e(1:2,:,:),2,1).^2,3)));
            if q == 1, nexttile; hold on; end
            plot(r.time,positionRMSE,'LineWidth',1.2,'DisplayName',labels(q));
        end
        grid on; ylabel('position RMSE [m]'); title('State-estimation effect'); legend('Location','best');
        nexttile; hold on;
        for q = 1:2
            r = cases{q}; de = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
            plot(r.time,squeeze(sqrt(mean(vecnorm(de,2,1).^2,3))), ...
                'LineWidth',1.2,'DisplayName',labels(q));
        end
        grid on; ylabel('acceleration RMSE [m/s^2]'); title('Shared-model effect'); legend('Location','best');
        nexttile; hold on;
        r = eventTriggered;
        for i = 1:size(r.communicationEvent,2)
            stairs(r.time,.2*double(r.communicationEvent(:,i)),'LineWidth',1.0, ...
                'DisplayName',"watcher "+i);
        end
        grid on; xlabel('time [s]'); ylabel('upload indicator');
        title('Event-triggered ground-station parameter uploads'); legend('Location','best');
        study.figure = fig;
    end
end

function out = runToyQuiet(common,communicationMode)
    evalc('out = run_toy_distributed_additive_dnn_ekf(common{:},communicationMode);');
end
