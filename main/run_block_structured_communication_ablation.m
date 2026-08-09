function study = run_block_structured_communication_ablation( ...
    seed,T,dt,nWatchers,hiddenLayerCount,makePlots)
%RUN_BLOCK_STRUCTURED_COMMUNICATION_ABLATION Isolate periodic sharing.
%
% The local learning rule, truth, measurements, initial conditions, DNN
% blocks, and maneuver policy are held fixed.  Only the communication mode
% of the shared block-structured DNN is changed:
%
%   1) local independent block
%   2) global block model with no parameter communication
%   3) the same global block model with full-block broadcasts every 60 s
%
% This answers the narrow question: does periodically replacing the remote
% cached blocks improve the current learning architecture, or does the
% synchronization itself introduce harmful model jumps?
%
% Example:
%   study = run_block_structured_communication_ablation( ...
%       101,600,0.1,4,3,true);
%   study.summary

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = 0.1; end
    if nargin < 4 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end

    fprintf(['Block-structured communication ablation: seed=%d, T=%.0f s, ' ...
        'dt=%.2f s, watchers=%d, hiddenLayers=%d\n'], ...
        seed,T,dt,nWatchers,hiddenLayerCount);

    noShareRun = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount,"never");
    periodicRun = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount,"periodic_60s");

    % Both calls regenerate the common data with the same seed.  This check
    % guards against accidentally comparing different target trajectories.
    assert(isequal(noShareRun.truth,periodicRun.truth), ...
        'The two communication cases did not use identical truth data.');
    assert(isequal(noShareRun.trueAcceleration,periodicRun.trueAcceleration), ...
        'The two communication cases did not use identical acceleration data.');

    study.localOnly = noShareRun.localOnly;
    study.noSharing = noShareRun.sharedBlock;
    study.periodic60s = periodicRun.sharedBlock;
    study.noSharingRun = noShareRun;
    study.periodicRun = periodicRun;
    study.cfg = periodicRun.cfg;
    study.truth = periodicRun.truth;
    study.trueAcceleration = periodicRun.trueAcceleration;
    study.architecture = periodicRun.architecture;

    [jumpTimes,jumpMagnitudes] = synchronizationJumps( ...
        study.periodic60s,dt);
    study.synchronization.times = jumpTimes;
    study.synchronization.nextSampleAccelerationJumps = jumpMagnitudes;
    study.summary = makeCommunicationSummary(study);
    disp(study.summary);

    if makePlots
        study.figures = plotCommunicationStudy(study);
    else
        study.figures = struct;
    end
end

function summary = makeCommunicationSummary(study)
    names = ["Local independent block"; ...
        "Shared block: no communication"; ...
        "Shared block: periodic 60 s"];
    cases = {study.localOnly,study.noSharing,study.periodic60s};
    nCases = numel(cases);

    positionRMSE = zeros(nCases,1);
    velocityRMSE = zeros(nCases,1);
    accelerationRMSE = zeros(nCases,1);
    finalPositionRMSE = zeros(nCases,1);
    meanNIS = zeros(nCases,1);
    parameterUploads = zeros(nCases,1);
    meanSyncOutputJump = nan(nCases,1);
    maxSyncOutputJump = nan(nCases,1);

    for k = 1:nCases
        result = cases{k};
        positionRMSE(k) = result.positionRMSE;
        velocityRMSE(k) = result.velocityRMSE;
        accelerationRMSE(k) = result.accelerationRMSE;
        finalPositionRMSE(k) = result.finalPositionRMSE;
        meanNIS(k) = finiteMean(result.nis(2:end,:));
        parameterUploads(k) = result.parameterUploads;
    end

    jumps = study.synchronization.nextSampleAccelerationJumps;
    if ~isempty(jumps)
        meanSyncOutputJump(3) = mean(jumps);
        maxSyncOutputJump(3) = max(jumps);
    end

    summary = table(names,positionRMSE,velocityRMSE,accelerationRMSE, ...
        finalPositionRMSE,meanNIS,parameterUploads,meanSyncOutputJump, ...
        maxSyncOutputJump,'VariableNames',{'caseName','positionRMSE', ...
        'velocityRMSE','accelerationRMSE','finalPositionRMSE','meanNIS', ...
        'parameterUploads','meanSyncOutputJump','maxSyncOutputJump'});
end

function value = finiteMean(x)
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end

function [eventTimes,jumps] = synchronizationJumps(result,dt)
% Observed output change from an upload sample to its next sample.
%
% The cache is replaced after the acceleration output at the upload sample
% is stored.  Therefore the first output that uses the received block is at
% k+1.  This diagnostic includes one normal propagation step and should be
% interpreted as an observed synchronization-associated jump, not as a
% pure same-state network discontinuity.

    eventTimes = unique([result.uploadTimes{:}]);
    dMean = mean(result.dHat,3);
    validTimes = zeros(1,0);
    jumps = zeros(1,0);
    for q = 1:numel(eventTimes)
        k = round(eventTimes(q)/dt)+1;
        if k >= 1 && k < size(dMean,2)
            validTimes(end+1) = eventTimes(q); %#ok<AGROW>
            jumps(end+1) = norm(dMean(:,k+1)-dMean(:,k)); %#ok<AGROW>
        end
    end
    eventTimes = validTimes;
end

function figs = plotCommunicationStudy(study)
    t = study.cfg.time;
    cases = {study.localOnly,study.noSharing,study.periodic60s};
    labels = {'local independent block', ...
        'shared block: no communication', ...
        'shared block: periodic 60 s'};
    colors = lines(3);

    figs.errors = figure('Name','Block-DNN communication ablation');
    tiledlayout(4,1,'TileSpacing','compact');
    fields = {'positionError','velocityError','accelerationError'};
    titles = {'position estimation error','velocity estimation error', ...
        'acceleration approximation error'};
    units = {'RMSE [m]','RMSE [m/s]','RMSE [m/s^2]'};
    for row = 1:3
        nexttile; hold on; grid on;
        for c = 1:3
            plot(t,cases{c}.(fields{row}),'LineWidth',1.15, ...
                'Color',colors(c,:));
        end
        title(titles{row}); ylabel(units{row});
        if row == 1, legend(labels,'Location','best'); end
    end
    nexttile; hold on; grid on;
    smoothWindow = max(1,round(5/study.cfg.dt));
    for c = 1:3
        nisMean = mean(cases{c}.nis,2,'omitnan');
        nisMean(1) = NaN;
        plot(t,movmean(nisMean,smoothWindow,'omitnan'), ...
            'LineWidth',1.1,'Color',colors(c,:));
    end
    yline(1,'k--','NIS reference');
    title('bearing-innovation consistency (5 s moving mean)');
    ylabel('mean NIS'); xlabel('time [s]');

    figs.acceleration = figure('Name','Communication effect on acceleration');
    tiledlayout(2,1,'TileSpacing','compact');
    dLocal = mean(study.localOnly.dHat,3);
    dNoShare = mean(study.noSharing.dHat,3);
    dPeriodic = mean(study.periodic60s.dHat,3);
    for axisID = 1:2
        nexttile; hold on; grid on;
        plot(t,study.trueAcceleration(axisID,:),'k','LineWidth',1.6);
        plot(t,dLocal(axisID,:),'Color',colors(1,:),'LineWidth',0.9);
        plot(t,dNoShare(axisID,:),'Color',colors(2,:),'LineWidth',1.0);
        plot(t,dPeriodic(axisID,:),'Color',colors(3,:),'LineWidth',1.15);
        ylabel(sprintf('d_%c [m/s^2]','x'+axisID-1));
        if axisID == 1
            legend([{'truth'},labels],'Location','best');
        end
    end
    xlabel('time [s]');

    figs.synchronization = figure('Name','Periodic block synchronization');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on; grid on;
    for owner = 1:study.cfg.Nw
        ti = study.periodic60s.uploadTimes{owner};
        if ~isempty(ti)
            scatter(ti,owner*ones(size(ti)),26,'filled');
        end
    end
    title('60 s full-block uploads');
    ylabel('owner watcher');
    ylim([0.5 study.cfg.Nw+0.5]);

    nexttile; hold on; grid on;
    stem(study.synchronization.times, ...
        study.synchronization.nextSampleAccelerationJumps,'filled');
    title('observed next-sample acceleration-output change after synchronization');
    ylabel('||d_{k+1}-d_k|| [m/s^2]'); xlabel('time [s]');
end
