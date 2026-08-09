function study = run_full_dnn_depth_stability_ablation(seed,T,dt,scenario,makePlots,nWatchers,hiddenLayers)
%RUN_FULL_DNN_DEPTH_STABILITY_ABLATION Full-DNN depth and stability study.
% Compares three- and four-hidden-layer additive DNN branches under the
% same angle-only, event-triggered collaborative EKF.  Each depth is run
% with the default parameter random walk and with a conservative setting
% that reduces Q_theta and makes remote-posterior uploads less frequent.
%
% Example:
%   study = run_full_dnn_depth_stability_ablation(101,600,.1, ...
%       "well_conditioned",true,4,[3 4]);

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 5 || isempty(makePlots), makePlots = true; end
    if nargin < 6 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 7 || isempty(hiddenLayers), hiddenLayers = [3 4]; end
    validateattributes(hiddenLayers,{'numeric'}, {'vector','integer','>=',3,'<=',4});
    hiddenLayers = unique(hiddenLayers(:)');

    variants = struct( ...
        'name',{"default","conservative"}, ...
        'tuning',{struct,struct('parameterProcessScale',.10, ...
            'mahalanobisThreshold',8,'minCommunicationInterval',10)});
    nRun = numel(hiddenLayers)*numel(variants);
    runs = cell(nRun,1); row = 0;
    summary = table('Size',[nRun 9], ...
        'VariableTypes',["double","string","double","double","double", ...
         "double","double","double","double"], ...
        'VariableNames',{'hiddenLayers','variant','positionRMSE','velocityRMSE', ...
         'accelerationRMSE','finalPositionRMSE','terminalPositionRMSE', ...
         'parameterUploads','meanFinalParameterChange'});

    for depth = hiddenLayers
        for v = 1:numel(variants)
            row = row+1;
            args = {seed,T,false,scenario,dt,true,"local_information",true, ...
                nWatchers,"parameter_covariance","additive_full_dnn", ...
                "event_triggered",true,depth,variants(v).tuning};
            runs{row} = runToyQuiet(args);
            r = runs{row}.sharedAdditive;
            [pRMSE,vRMSE,aRMSE,finalRMSE,terminalRMSE] = caseMetrics(r);
            summary.hiddenLayers(row) = depth;
            summary.variant(row) = string(variants(v).name);
            summary.positionRMSE(row) = pRMSE;
            summary.velocityRMSE(row) = vRMSE;
            summary.accelerationRMSE(row) = aRMSE;
            summary.finalPositionRMSE(row) = finalRMSE;
            summary.terminalPositionRMSE(row) = terminalRMSE;
            summary.parameterUploads(row) = nnz(r.communicationEvent);
            summary.meanFinalParameterChange(row) = ...
                mean(vecnorm(r.thetaCache-r.thetaInitial,2,1));
        end
    end

    study = struct('runs',{runs},'summary',summary, ...
        'description',"Full-DNN depth comparison with conservative event-trigger tuning");
    disp(summary);

    if makePlots
        fig = figure('Name','Full-DNN depth and stability ablation');
        tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
        nexttile; hold on;
        for k = 1:nRun
            r = runs{k}.sharedAdditive;
            e = r.xhat(1:2,:,:)-repmat(r.etaTrue(1:2,:),1,1,size(r.xhat,3));
            plot(r.time,squeeze(sqrt(mean(vecnorm(e,2,1).^2,3))), ...
                'LineWidth',1.15,'DisplayName',runLabel(summary(k,:)));
        end
        grid on; xlabel('time [s]'); ylabel('position RMSE [m]');
        title('Shared full-DNN state accuracy'); legend('Location','best');

        nexttile; hold on;
        for k = 1:nRun
            r = runs{k}.sharedAdditive;
            de = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
            plot(r.time,squeeze(sqrt(mean(vecnorm(de,2,1).^2,3))), ...
                'LineWidth',1.15,'DisplayName',runLabel(summary(k,:)));
        end
        grid on; xlabel('time [s]'); ylabel('acceleration RMSE [m/s^2]');
        title('Shared residual approximation'); legend('Location','best');

        nexttile; hold on;
        for v = 1:numel(variants)
            idx = summary.variant == string(variants(v).name);
            plot(summary.hiddenLayers(idx),summary.terminalPositionRMSE(idx),'-o', ...
                'LineWidth',1.3,'DisplayName',string(variants(v).name));
        end
        grid on; xlabel('number of hidden layers'); ylabel('terminal position RMSE [m]');
        title('Terminal-window stability'); legend('Location','best');

        nexttile; hold on;
        for v = 1:numel(variants)
            idx = summary.variant == string(variants(v).name);
            plot(summary.hiddenLayers(idx),summary.parameterUploads(idx),'-o', ...
                'LineWidth',1.3,'DisplayName',string(variants(v).name));
        end
        grid on; xlabel('number of hidden layers'); ylabel('parameter uploads');
        title('Ground-station communication load'); legend('Location','best');
        study.figure = fig;
    end
end

function out = runToyQuiet(args)
    evalc('out = run_toy_distributed_additive_dnn_ekf(args{:});');
end

function [pRMSE,vRMSE,aRMSE,finalRMSE,terminalRMSE] = caseMetrics(r)
    e = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
    de = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
    positionSeries = squeeze(sqrt(mean(vecnorm(e(1:2,:,:),2,1).^2,3)));
    pRMSE = sqrt(mean(positionSeries.^2));
    velocitySeries = squeeze(sqrt(mean(vecnorm(e(3:4,:,:),2,1).^2,3)));
    vRMSE = sqrt(mean(velocitySeries.^2));
    accelerationSeries = squeeze(sqrt(mean(vecnorm(de,2,1).^2,3)));
    aRMSE = sqrt(mean(accelerationSeries.^2));
    finalRMSE = positionSeries(end);
    terminalStart = max(1,find(r.time >= r.time(end)-60,1,'first'));
    terminalRMSE = sqrt(mean(positionSeries(terminalStart:end).^2));
end

function label = runLabel(row)
    label = sprintf('%d layers, %s',row.hiddenLayers,char(row.variant));
end
