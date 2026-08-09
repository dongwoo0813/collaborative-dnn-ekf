function study = run_state_partitioned_full_dnn_scaling_ablation( ...
    seeds,T,dt,watcherCounts,scenario,makePlots,hiddenLayerCount)
%RUN_STATE_PARTITIONED_FULL_DNN_SCALING_ABLATION Test partitioned sharing.
%
% This is a paired, common-random-number experiment for the architectural
% question at the center of the collaborative DNN-EKF formulation:
%
%   1) LOCAL: watcher i propagates with its own full-DNN branch only;
%   2) MEAN: all communicated branches are evaluated and their vector
%      outputs are averaged, (1/N_w) sum_i d_i(eta); and
%   3) PARTITIONED: all communicated full-DNN branches are evaluated, but
%      the output is a state-dependent convex combination,
%           d_hat(eta) = sum_i g_i(r) d_i(eta),  sum_i g_i(r) = 1.
%
% The gates g_i are fixed soft RBF responsibilities with different spatial
% centers.  They are not LOS or covariance gates.  Therefore a branch has
% a structural state region in which its parameter changes influence the
% propagated acceleration most strongly.  This prevents the raw additive
% sum from counting the same learned residual N_w times.
%
% For every (seed,N_w), the three cases share the same true trajectory,
% bearing noise, initial conditions, maneuver schedule, and branch
% parameter updates.  Thus the paired differences isolate the fusion rule.
% Positive gain means that the state-partitioned model has lower RMSE.

    if nargin < 1 || isempty(seeds), seeds = 101:110; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(watcherCounts), watcherCounts = 1:6; end
    if nargin < 5 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end
    if nargin < 7 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end

    seeds = reshape(seeds,1,[]);
    watcherCounts = reshape(watcherCounts,1,[]);
    validateattributes(watcherCounts,{'numeric'}, {'integer','>=',1,'<=',8});
    scenario = string(scenario);
    assert(scenario == "well_conditioned", ...
        ['The scaling claim is deliberately tested only in the nested, ', ...
         'non-collinear well_conditioned formation.']);

    nN = numel(watcherCounts); nSeed = numel(seeds);
    metricNames = ["positionRMSE" "velocityRMSE" "accelerationRMSE" ...
        "finalPositionRMSE" "terminalPositionRMSE"];
    nMetric = numel(metricNames);
    localMetric = nan(nN,nSeed,nMetric);
    meanMetric = nan(nN,nSeed,nMetric);
    partitionedMetric = nan(nN,nSeed,nMetric);
    uploads = nan(nN,nSeed);

    fprintf(['State-partitioned full-DNN scaling: %d seeds, N_w=%s, ', ...
        'L=%d, scenario=%s\n'],nSeed,mat2str(watcherCounts), ...
        hiddenLayerCount,scenario);
    for n = 1:nN
        for s = 1:nSeed
            args = {seeds(s),T,false,scenario,dt,true,"local_information",true, ...
                watcherCounts(n),"parameter_covariance","partitioned_full_dnn", ...
                "event_triggered",true,hiddenLayerCount};
            out = runToyQuiet(args);

            % The first row is nominal.  The next three cases are the
            % precisely matched local, mean-output, and partitioned cases.
            localMetric(n,s,:) = extractMetrics(out.localOnly);
            meanMetric(n,s,:) = extractMetrics(out.meanEnsemble);
            partitionedMetric(n,s,:) = extractMetrics(out.sharedAdditive);
            uploads(n,s) = nnz(out.sharedAdditive.communicationEvent);
        end
    end

    localStats = summarizeMetric(localMetric);
    meanStats = summarizeMetric(meanMetric);
    partitionedStats = summarizeMetric(partitionedMetric);
    localMinusPartitioned = localMetric-partitionedMetric;
    meanMinusPartitioned = meanMetric-partitionedMetric;
    localGain = pairedStats(localMinusPartitioned);
    meanGain = pairedStats(meanMinusPartitioned);

    % A scaling claim is measured against the one-branch partitioned
    % system on exactly the same seeds, not against an unpaired run.
    oneBranch = partitionedMetric(1,:,:);
    oneMinusPartitioned = repmat(oneBranch,nN,1,1)-partitionedMetric;
    scaleGain = pairedStats(oneMinusPartitioned);

    summary = table(watcherCounts(:), ...
        localStats.mean(:,1),meanStats.mean(:,1),partitionedStats.mean(:,1), ...
        localGain.mean(:,1),localGain.ci95(:,1,1),localGain.ci95(:,1,2), ...
        meanGain.mean(:,1),meanGain.ci95(:,1,1),meanGain.ci95(:,1,2), ...
        partitionedStats.mean(:,3),localGain.mean(:,3),meanGain.mean(:,3), ...
        scaleGain.mean(:,1),scaleGain.ci95(:,1,1),scaleGain.ci95(:,1,2), ...
        mean(uploads,2,'omitnan'), ...
        'VariableNames',{'nWatchers','localPositionRMSE','meanPositionRMSE', ...
        'partitionedPositionRMSE','localMinusPartitionedPositionGain', ...
        'localMinusPartitionedPositionCI95Lower', ...
        'localMinusPartitionedPositionCI95Upper', ...
        'meanMinusPartitionedPositionGain', ...
        'meanMinusPartitionedPositionCI95Lower', ...
        'meanMinusPartitionedPositionCI95Upper', ...
        'partitionedAccelerationRMSE','localMinusPartitionedAccelerationGain', ...
        'meanMinusPartitionedAccelerationGain','oneMinusPartitionedPositionGain', ...
        'oneMinusPartitionedPositionCI95Lower', ...
        'oneMinusPartitionedPositionCI95Upper','meanParameterUploads'});

    study = struct;
    study.seeds = seeds; study.T = T; study.dt = dt; study.scenario = scenario;
    study.watcherCounts = watcherCounts; study.hiddenLayerCount = hiddenLayerCount;
    study.metricNames = metricNames;
    study.localMetric = localMetric;
    study.meanMetric = meanMetric;
    study.partitionedMetric = partitionedMetric;
    study.localStats = localStats; study.meanStats = meanStats;
    study.partitionedStats = partitionedStats;
    study.localMinusPartitioned = localMinusPartitioned;
    study.meanMinusPartitioned = meanMinusPartitioned;
    study.localGain = localGain; study.meanGain = meanGain;
    study.oneMinusPartitioned = oneMinusPartitioned;
    study.scalingGain = scaleGain;
    study.parameterUploads = uploads;
    study.summary = summary;
    study.claimRule = [ ...
        'At fixed N_w, a structural partitioning benefit requires the ', ...
        'paired 95% CI of local-minus-partitioned and mean-minus-', ...
        'partitioned RMSE to be entirely positive.  A watcher-scaling ', ...
        'benefit requires the paired 95% CI of N_w=1-minus-N_w RMSE to ', ...
        'be entirely positive.'];

    if makePlots, study.figures = plotScaling(study); else, study.figures = struct; end
    disp(study.summary);
end

function out = runToyQuiet(arguments)
    evalc('out = run_toy_distributed_additive_dnn_ekf(arguments{:});');
end

function stats = summarizeMetric(metric)
    stats.mean = squeeze(mean(metric,2,'omitnan'));
    stats.std = squeeze(std(metric,0,2,'omitnan'));
    stats.se = stats.std/sqrt(size(metric,2));
    stats.ci95 = cat(3,stats.mean-1.96*stats.se,stats.mean+1.96*stats.se);
end

function stats = pairedStats(difference)
    stats = summarizeMetric(difference);
    stats.winRate = squeeze(mean(difference > 0,2,'omitnan'));
end

function m = extractMetrics(r)
    stateError = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
    accelError = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
    pos = sqrt(mean(vecnorm(stateError(1:2,:,:),2,1).^2,'all'));
    vel = sqrt(mean(vecnorm(stateError(3:4,:,:),2,1).^2,'all'));
    acc = sqrt(mean(vecnorm(accelError,2,1).^2,'all'));
    final = sqrt(mean(vecnorm(stateError(1:2,round(.9*numel(r.time)):end,:),2,1).^2,'all'));
    terminalStart = max(1,numel(r.time)-round(60/(r.time(2)-r.time(1))));
    terminal = sqrt(mean(vecnorm(stateError(1:2,terminalStart:end,:),2,1).^2,'all'));
    m = [pos vel acc final terminal];
end

function figs = plotScaling(study)
    x = study.watcherCounts; c = lines(3); figs = struct;
    figs.metrics = figure('Name','State-partitioned full-DNN scaling');
    tiledlayout(2,2,'TileSpacing','compact');
    plotMetric(1,'position RMSE [m]');
    plotMetric(2,'velocity RMSE [m/s]');
    plotMetric(3,'acceleration RMSE [m/s^2]');
    plotMetric(4,'final position RMSE [m]');

    figs.gains = figure('Name','State-partitioned full-DNN paired gains');
    tiledlayout(3,1,'TileSpacing','compact');
    plotPair(1,study.localGain,'local minus partitioned position RMSE [m]','Local vs partitioned');
    plotPair(1,study.meanGain,'mean minus partitioned position RMSE [m]','Mean vs partitioned');
    plotPair(1,study.scalingGain,'one-branch minus partitioned position RMSE [m]', ...
        'N_w=1 vs N_w');

    function plotMetric(q,ylabelText)
        nexttile; hold on;
        errorbar(x,study.localStats.mean(:,q),study.localStats.std(:,q),'-o', ...
            'Color',c(1,:),'LineWidth',1.2);
        errorbar(x,study.meanStats.mean(:,q),study.meanStats.std(:,q),'-o', ...
            'Color',c(2,:),'LineWidth',1.2);
        errorbar(x,study.partitionedStats.mean(:,q),study.partitionedStats.std(:,q),'-o', ...
            'Color',c(3,:),'LineWidth',1.2);
        grid on; xlabel('number of watchers / sub-DNN branches'); ylabel(ylabelText);
        title(strrep(study.metricNames(q),'RMSE',' RMSE'));
        if q == 1
            legend('Local branch only','Mean output ensemble', ...
                'State-partitioned shared model','Location','best');
        end
    end

    function plotPair(q,gain,ylabelText,titleText)
        nexttile; hold on;
        lo = gain.ci95(:,q,1); hi = gain.ci95(:,q,2);
        errorbar(x,gain.mean(:,q),gain.mean(:,q)-lo,hi-gain.mean(:,q), ...
            '-o','Color',c(3,:),'LineWidth',1.2);
        yline(0,'k--','no paired benefit'); grid on;
        xlabel('number of watchers / sub-DNN branches'); ylabel(ylabelText);
        title(sprintf('%s (95%% paired CI)',titleText));
    end
end
