function study = run_full_dnn_watcher_scaling_ablation( ...
    seeds,T,dt,watcherCounts,scenario,makePlots,hiddenLayerCount,communicationMode)
%RUN_FULL_DNN_WATCHER_SCALING_ABLATION Validate collaborative full-DNN scaling.
%
% Runs the *same* angle-only, full-DNN EKF experiment for every seed and
% watcher count.  Within each run the toy driver uses the same truth,
% bearing-noise realization, and nominal-derived maneuver schedule for
%   (1) local full-DNN adaptation without parameter sharing, and
%   (2) scale-normalized mean ensemble, and
%   (3) raw shared additive full-DNN adaptation with event-triggered uploads.
%
% The test answers two separate questions:
%   A. Does raw additive sharing improve over both local adaptation and the
%      mean ensemble at a fixed number of watchers?
%   B. Does the shared model improve from one well-separated watcher to a
%      larger, nested well-conditioned watcher formation?
%
% A positive paired improvement is local RMSE minus shared RMSE.  The 95%%
% confidence interval is computed over seeds; its lower bound must be
% positive before claiming a statistically consistent shared benefit.

    if nargin < 1 || isempty(seeds), seeds = 101:110; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(watcherCounts), watcherCounts = 1:4; end
    if nargin < 5 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end
    if nargin < 7 || isempty(hiddenLayerCount), hiddenLayerCount = 4; end
    if nargin < 8 || isempty(communicationMode), communicationMode = "event_triggered"; end

    seeds = reshape(seeds,1,[]);
    watcherCounts = reshape(watcherCounts,1,[]);
    validateattributes(watcherCounts,{'numeric'}, {'integer','>=',1,'<=',8});
    scenario = string(scenario); communicationMode = string(communicationMode);
    assert(scenario == "well_conditioned", ...
        ['This scaling claim is intentionally restricted to the nested, ', ...
         'non-collinear well_conditioned formation.']);
    assert(any(communicationMode == ["event_triggered" "instantaneous"]), ...
        'communicationMode must be event_triggered or instantaneous.');

    nN = numel(watcherCounts); nSeed = numel(seeds);
    metricNames = ["positionRMSE" "velocityRMSE" "accelerationRMSE" ...
        "finalPositionRMSE" "terminalPositionRMSE"];
    nMetric = numel(metricNames);
    localMetric = nan(nN,nSeed,nMetric);
    meanMetric = nan(nN,nSeed,nMetric);
    sharedMetric = nan(nN,nSeed,nMetric);
    uploads = nan(nN,nSeed);
    diagnostic = struct('meanNIS',nan(nN,nSeed), ...
        'meanContributionNorm',nan(nN,nSeed), ...
        'meanParameterCovarianceTrace',nan(nN,nSeed), ...
        'finalParameterChangeNorm',nan(nN,nSeed), ...
        'meanBranchCorrelation',nan(nN,nSeed), ...
        'meanAbsoluteBranchCorrelation',nan(nN,nSeed), ...
        'branchCorrelation',{cell(nN,nSeed)});

    fprintf(['Full-DNN watcher scaling: %d seeds, N_w=%s, L=%d, ', ...
        'scenario=%s, communication=%s\n'], nSeed,mat2str(watcherCounts), ...
        hiddenLayerCount,scenario,communicationMode);
    for n = 1:nN
        for s = 1:nSeed
            args = {seeds(s),T,false,scenario,dt,true,"local_information",true, ...
                watcherCounts(n),"parameter_covariance","additive_full_dnn", ...
                communicationMode,true,hiddenLayerCount};
            out = runToyQuiet(args);
            localMetric(n,s,:) = extractMetrics(out.localOnly);
            meanMetric(n,s,:) = extractMetrics(out.meanEnsemble);
            sharedMetric(n,s,:) = extractMetrics(out.sharedAdditive);
            uploads(n,s) = nnz(out.sharedAdditive.communicationEvent);
            d = extractDiagnostics(out.sharedAdditive);
            diagnostic.meanNIS(n,s) = d.meanNIS;
            diagnostic.meanContributionNorm(n,s) = d.meanContributionNorm;
            diagnostic.meanParameterCovarianceTrace(n,s) = d.meanParameterCovarianceTrace;
            diagnostic.finalParameterChangeNorm(n,s) = d.finalParameterChangeNorm;
            diagnostic.meanBranchCorrelation(n,s) = d.meanBranchCorrelation;
            diagnostic.meanAbsoluteBranchCorrelation(n,s) = d.meanAbsoluteBranchCorrelation;
            diagnostic.branchCorrelation{n,s} = d.branchCorrelation;
        end
    end

    localMean = reshape(mean(localMetric,2,'omitnan'),nN,nMetric);
    meanEnsembleMean = reshape(mean(meanMetric,2,'omitnan'),nN,nMetric);
    sharedMean = reshape(mean(sharedMetric,2,'omitnan'),nN,nMetric);
    localStd = reshape(std(localMetric,0,2,'omitnan'),nN,nMetric);
    sharedStd = reshape(std(sharedMetric,0,2,'omitnan'),nN,nMetric);
    pairedGain = localMetric-sharedMetric;
    gainMean = reshape(mean(pairedGain,2,'omitnan'),nN,nMetric);
    gainSE = reshape(std(pairedGain,0,2,'omitnan'),nN,nMetric)/sqrt(nSeed);
    gainCI95 = cat(3,gainMean-1.96*gainSE,gainMean+1.96*gainSE);
    sharedWinRate = reshape(mean(pairedGain > 0,2,'omitnan'),nN,nMetric);
    meanGain = meanMetric-sharedMetric;
    meanGainMean = reshape(mean(meanGain,2,'omitnan'),nN,nMetric);
    meanGainSE = reshape(std(meanGain,0,2,'omitnan'),nN,nMetric)/sqrt(nSeed);
    meanGainCI95 = cat(3,meanGainMean-1.96*meanGainSE,meanGainMean+1.96*meanGainSE);
    meanWinRate = reshape(mean(meanGain > 0,2,'omitnan'),nN,nMetric);

    % Scaling uses the same seeds and compares each shared system to N_w=1.
    baseline = sharedMetric(1,:,:);
    scaleGain = repmat(baseline,nN,1,1)-sharedMetric;
    scaleMean = reshape(mean(scaleGain,2,'omitnan'),nN,nMetric);
    scaleSE = reshape(std(scaleGain,0,2,'omitnan'),nN,nMetric)/sqrt(nSeed);
    scaleCI95 = cat(3,scaleMean-1.96*scaleSE,scaleMean+1.96*scaleSE);
    scaleWinRate = reshape(mean(scaleGain > 0,2,'omitnan'),nN,nMetric);

    nParamsPerBranch = fullDnnParameterCount(hiddenLayerCount);
    summary = table(watcherCounts(:),repmat(nParamsPerBranch,nN,1), ...
        watcherCounts(:)*nParamsPerBranch, ...
        localMean(:,1),meanEnsembleMean(:,1),sharedMean(:,1), ...
        gainMean(:,1),gainCI95(:,1,1),gainCI95(:,1,2),sharedWinRate(:,1), ...
        meanGainMean(:,1),meanGainCI95(:,1,1),meanGainCI95(:,1,2),meanWinRate(:,1), ...
        localMean(:,3),sharedMean(:,3),gainMean(:,3),gainCI95(:,3,1),gainCI95(:,3,2),sharedWinRate(:,3), ...
        scaleMean(:,1),scaleCI95(:,1,1),scaleCI95(:,1,2),scaleWinRate(:,1), ...
        mean(uploads,2,'omitnan'),mean(diagnostic.meanNIS,2,'omitnan'), ...
        mean(diagnostic.meanContributionNorm,2,'omitnan'), ...
        mean(diagnostic.meanParameterCovarianceTrace,2,'omitnan'), ...
        mean(diagnostic.finalParameterChangeNorm,2,'omitnan'), ...
        mean(diagnostic.meanBranchCorrelation,2,'omitnan'), ...
        mean(diagnostic.meanAbsoluteBranchCorrelation,2,'omitnan'), ...
        'VariableNames',{'nWatchers','parametersPerBranch','globalParameterCount', ...
        'localPositionRMSE','meanPositionRMSE','sharedPositionRMSE','sharedPositionGain', ...
        'sharedPositionGainCI95Lower','sharedPositionGainCI95Upper','sharedPositionWinRate', ...
        'meanMinusSharedPositionGain','meanMinusSharedPositionGainCI95Lower', ...
        'meanMinusSharedPositionGainCI95Upper','meanMinusSharedPositionWinRate', ...
        'localAccelerationRMSE','sharedAccelerationRMSE','sharedAccelerationGain', ...
        'sharedAccelerationGainCI95Lower','sharedAccelerationGainCI95Upper','sharedAccelerationWinRate', ...
        'sharedPositionGainVsOne','sharedPositionGainVsOneCI95Lower', ...
        'sharedPositionGainVsOneCI95Upper','sharedPositionGainVsOneWinRate', ...
        'meanParameterUploads','meanNIS','meanBranchContributionNorm', ...
        'meanParameterCovarianceTrace','finalParameterChangeNorm', ...
        'meanBranchCorrelation','meanAbsoluteBranchCorrelation'});

    study = struct;
    study.seeds = seeds; study.T = T; study.dt = dt; study.scenario = scenario;
    study.watcherCounts = watcherCounts; study.hiddenLayerCount = hiddenLayerCount;
    study.communicationMode = communicationMode;
    study.metricNames = metricNames;
    study.localMetric = localMetric; study.meanMetric = meanMetric; study.sharedMetric = sharedMetric;
    study.localMean = localMean; study.localStd = localStd;
    study.meanEnsembleMean = meanEnsembleMean;
    study.sharedMean = sharedMean; study.sharedStd = sharedStd;
    study.sharedGain = pairedGain; study.sharedGainMean = gainMean;
    study.sharedGainCI95 = gainCI95; study.sharedWinRate = sharedWinRate;
    study.meanMinusSharedGain = meanGain; study.meanMinusSharedGainMean = meanGainMean;
    study.meanMinusSharedGainCI95 = meanGainCI95; study.meanMinusSharedWinRate = meanWinRate;
    study.scalingGain = scaleGain; study.scalingGainMean = scaleMean;
    study.scalingGainCI95 = scaleCI95; study.scalingWinRate = scaleWinRate;
    study.parameterUploads = uploads;
    study.diagnostic = diagnostic;
    study.summary = summary;
    study.claimRule = ['At a fixed N_w, sharing is supported only when the ', ...
        'paired 95% CIs of both local minus shared and mean minus shared ', ...
        'RMSE are entirely positive. ', ...
        'Scaling is supported only when shared N_w>1 beats shared N_w=1 ', ...
        'with an entirely positive paired 95% CI.'];

    if makePlots, study.figures = plotScaling(study); else, study.figures = struct; end
    disp(study.summary);
end

function out = runToyQuiet(arguments)
    evalc('out = run_toy_distributed_additive_dnn_ekf(arguments{:});');
end

function m = extractMetrics(r)
    stateError = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
    accelError = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
    pos = sqrt(mean(vecnorm(stateError(1:2,:,:),2,1).^2,'all'));
    vel = sqrt(mean(vecnorm(stateError(3:4,:,:),2,1).^2,'all'));
    acc = sqrt(mean(vecnorm(accelError,2,1).^2,'all'));
    final = sqrt(mean(vecnorm(stateError(1:2,round(.9*numel(r.time)):end,:),2,1).^2,'all'));
    terminal = sqrt(mean(vecnorm(stateError(1:2,max(1,end-round(60/(r.time(2)-r.time(1)))):end,:),2,1).^2,'all'));
    m = [pos vel acc final terminal];
end

function d = extractDiagnostics(r)
    d.meanNIS = mean(r.NIS,'all','omitnan');
    d.meanContributionNorm = mean(r.branchContributionNorm,'all','omitnan');
    d.meanParameterCovarianceTrace = mean(r.parameterCovarianceTrace,'all','omitnan');
    d.finalParameterChangeNorm = mean(r.parameterChangeNorm(:,end),'omitnan');
    nBranch = size(r.branchOutputDiagnostic,2);
    C = eye(nBranch);
    for i = 1:nBranch
        yi = reshape(r.branchOutputDiagnostic(:,i,:),[],1);
        for j = i+1:nBranch
            yj = reshape(r.branchOutputDiagnostic(:,j,:),[],1);
            cij = corr(yi,yj,'Rows','complete');
            C(i,j) = cij; C(j,i) = cij;
        end
    end
    d.branchCorrelation = C;
    offDiagonal = C(triu(true(nBranch),1));
    if isempty(offDiagonal)
        d.meanBranchCorrelation = nan;
        d.meanAbsoluteBranchCorrelation = nan;
    else
        d.meanBranchCorrelation = mean(offDiagonal,'omitnan');
        d.meanAbsoluteBranchCorrelation = mean(abs(offDiagonal),'omitnan');
    end
end

function n = fullDnnParameterCount(hiddenLayers)
    % Input(4)->hidden(3), hidden(3)->hidden(3), hidden(3)->output(2).
    n = (3*4+3) + (hiddenLayers-1)*(3*3+3) + (2*3+2);
end

function figs = plotScaling(study)
    c = lines(3); x = study.watcherCounts;
    figs = struct;
    figs.metrics = figure('Name','Full-DNN watcher scaling validation');
    tiledlayout(2,2,'TileSpacing','compact');
    plotMetric(1,'position RMSE [m]');
    plotMetric(2,'velocity RMSE [m/s]');
    plotMetric(3,'acceleration approximation RMSE [m/s^2]');
    plotMetric(4,'final position RMSE [m]');
    figs.gains = figure('Name','Full-DNN shared benefit validation');
    tiledlayout(2,1,'TileSpacing','compact');
    plotGain(1,'paired position-RMSE gain: local minus shared [m]');
    plotGain(3,'paired acceleration-RMSE gain: local minus shared [m/s^2]');

    function plotMetric(q,ylabelText)
        nexttile; hold on;
        errorbar(x,study.localMean(:,q),study.localStd(:,q),'-o','Color',c(1,:),'LineWidth',1.25);
        plot(x,study.meanEnsembleMean(:,q),'-o','Color',c(2,:),'LineWidth',1.25);
        errorbar(x,study.sharedMean(:,q),study.sharedStd(:,q),'-o','Color',c(3,:),'LineWidth',1.25);
        grid on; xlabel('number of watchers / sub-DNN branches'); ylabel(ylabelText);
        title(strrep(study.metricNames(q),'RMSE',' RMSE'));
        if q == 1, legend('Local full-DNN adaptation','Mean ensemble','Raw shared additive','Location','best'); end
    end

    function plotGain(q,ylabelText)
        nexttile; hold on;
        lo = study.sharedGainCI95(:,q,1); hi = study.sharedGainCI95(:,q,2);
        errorbar(x,study.sharedGainMean(:,q),study.sharedGainMean(:,q)-lo,hi-study.sharedGainMean(:,q), ...
            '-o','Color',c(3,:),'LineWidth',1.25);
        yline(0,'k--','no paired benefit'); grid on;
        xlabel('number of watchers / sub-DNN branches'); ylabel(ylabelText);
        title('sharing benefit with 95% paired confidence interval');
    end
end
