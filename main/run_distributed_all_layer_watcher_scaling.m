function study = run_distributed_all_layer_watcher_scaling( ...
    seeds,T,dt,watcherCounts,hiddenLayerCount,makePlots)
%RUN_DISTRIBUTED_ALL_LAYER_WATCHER_SCALING Scale the collaborative DNN-EKF.
%
% Runs paired local, output-layer information, and all-layer FOGM +
% covariance-matching cases over watcher count.  At every (seed,N_w), all
% three cases have identical truth, initial DNN parameters, and bearing
% noise realization.  A positive gain means lower RMSE than local learning.
%
% Important: this is the proposed *fixed-block* scaling experiment.  Each
% watcher owns one width-3 branch, so both sensing information and total
% global DNN parameter count grow with N_w.  It therefore demonstrates
% system-level scalability, not information-only scalability.  A separate
% fixed-total-parameter-budget study is required to isolate the latter.
%
% Example:
%   study = run_distributed_all_layer_watcher_scaling( ...
%       101:110,600,.1,1:8,3,true);

    if nargin < 1 || isempty(seeds), seeds = 101:110; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(watcherCounts), watcherCounts = 1:8; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end

    seeds = reshape(seeds,1,[]);
    watcherCounts = reshape(watcherCounts,1,[]);
    validateattributes(watcherCounts,{'numeric'}, ...
        {'vector','integer','>=',1,'<=',8});

    caseNames = ["Local independent block"; ...
        "Distributed output-layer information EKF: 60 s"; ...
        "Distributed all-layer FOGM + covariance matching EKF: 60 s"];
    metricNames = ["positionRMSE","velocityRMSE","accelerationRMSE", ...
        "finalPositionRMSE","meanNIS","parameterUploads"];
    nW = numel(watcherCounts); nSeed = numel(seeds); nCase = numel(caseNames);
    metric = nan(nW,nSeed,nCase,numel(metricNames));
    parameterPerBranch = nan(nW,1);

    fprintf('Distributed all-layer watcher scaling: %d seeds, N_w=%s, L=%d\n', ...
        nSeed,mat2str(watcherCounts),hiddenLayerCount);
    for w = 1:nW
        for s = 1:nSeed
            fprintf('  N_w=%d, seed %d (%d/%d)\n', ...
                watcherCounts(w),seeds(s),s,nSeed);
            outputRun = runQuiet(seeds(s),T,dt,watcherCounts(w), ...
                hiddenLayerCount,"collaborative_output_info_60s");
            fogmRun = runQuiet(seeds(s),T,dt,watcherCounts(w), ...
                hiddenLayerCount,"collaborative_all_fogm_cm_60s");

            localA = extractMetrics(outputRun.localOnly);
            localB = extractMetrics(fogmRun.localOnly);
            assert(max(abs(localA-localB)) < 1e-10, ...
                'Paired local baselines disagree at N_w=%d, seed=%d.', ...
                watcherCounts(w),seeds(s));
            metric(w,s,1,:) = localA;
            metric(w,s,2,:) = extractMetrics(outputRun.sharedBlock);
            metric(w,s,3,:) = extractMetrics(fogmRun.sharedBlock);
            parameterPerBranch(w) = outputRun.cfg.dnn.arch.nTheta;
        end
    end

    local = squeeze(metric(:,:,1,:));
    output = squeeze(metric(:,:,2,:));
    fogm = squeeze(metric(:,:,3,:));
    outputGain = local-output;
    fogmGain = local-fogm;
    fogmVsOutput = output-fogm;
    fogmVsOne = repmat(fogm(1,:,:),nW,1,1)-fogm;

    summary = table(watcherCounts(:),parameterPerBranch, ...
        watcherCounts(:).*parameterPerBranch, ...
        meanMetric(local,1),meanMetric(output,1),meanMetric(fogm,1), ...
        meanMetric(outputGain,1),ciLower(outputGain,1),ciUpper(outputGain,1), ...
        winRate(outputGain,1), ...
        meanMetric(fogmGain,1),ciLower(fogmGain,1),ciUpper(fogmGain,1), ...
        winRate(fogmGain,1), ...
        meanMetric(fogmVsOutput,1),ciLower(fogmVsOutput,1),ciUpper(fogmVsOutput,1), ...
        winRate(fogmVsOutput,1), ...
        meanMetric(local,3),meanMetric(output,3),meanMetric(fogm,3), ...
        meanMetric(fogm,5),meanMetric(fogm,6), ...
        meanMetric(fogmVsOne,1),ciLower(fogmVsOne,1),ciUpper(fogmVsOne,1), ...
        winRate(fogmVsOne,1), ...
        'VariableNames',{'nWatchers','parametersPerBranch','globalParameterCount', ...
        'localPositionRMSE','outputPositionRMSE','allLayerPositionRMSE', ...
        'outputGainVsLocal','outputGainCI95Lower','outputGainCI95Upper','outputWinRateVsLocal', ...
        'allLayerGainVsLocal','allLayerGainCI95Lower','allLayerGainCI95Upper','allLayerWinRateVsLocal', ...
        'allLayerGainVsOutput','allLayerGainVsOutputCI95Lower','allLayerGainVsOutputCI95Upper','allLayerWinRateVsOutput', ...
        'localAccelerationRMSE','outputAccelerationRMSE','allLayerAccelerationRMSE', ...
        'allLayerMeanNIS','allLayerMeanUploads', ...
        'allLayerGainVsOne','allLayerGainVsOneCI95Lower','allLayerGainVsOneCI95Upper','allLayerWinRateVsOne'});

    study = struct;
    study.seeds = seeds; study.T = T; study.dt = dt;
    study.watcherCounts = watcherCounts; study.hiddenLayerCount = hiddenLayerCount;
    study.caseNames = caseNames; study.metricNames = metricNames;
    study.metric = metric; study.outputGainVsLocal = outputGain;
    study.allLayerGainVsLocal = fogmGain;
    study.allLayerGainVsOutput = fogmVsOutput;
    study.allLayerGainVsOneWatcher = fogmVsOne;
    study.summary = summary;
    study.interpretation = [ ...
        "Positive allLayerGainVsLocal supports collaborative learning at fixed N_w. ", ...
        "Positive allLayerGainVsOne supports improvement over a single watcher, ", ...
        "but is confounded by the linearly growing global parameter count."];
    disp(summary);
    if makePlots, study.figures = plotScaling(study); else, study.figures = struct; end
end

function out = runQuiet(seed,T,dt,nWatchers,nHidden,mode)
    command = sprintf(['out = run_toy_block_structured_global_head_dnn_ekf(', ...
        '%d,%.17g,false,%d,%.17g,%d,"%s");'], ...
        seed,T,nWatchers,dt,nHidden,mode);
    evalc(command);
end

function m = extractMetrics(result)
    nis = result.nis(2:end,:); nis = nis(isfinite(nis));
    m = [result.positionRMSE,result.velocityRMSE,result.accelerationRMSE, ...
        result.finalPositionRMSE,mean(nis),result.parameterUploads];
end

function value = meanMetric(x,column)
    value = squeeze(mean(x(:,:,column),2,'omitnan'));
end

function value = ciLower(x,column)
    [mu,se] = pairedMeanSE(x,column);
    value = mu-1.96*se;
end

function value = ciUpper(x,column)
    [mu,se] = pairedMeanSE(x,column);
    value = mu+1.96*se;
end

function [mu,se] = pairedMeanSE(x,column)
    y = x(:,:,column);
    mu = mean(y,2,'omitnan');
    n = sum(isfinite(y),2);
    se = std(y,0,2,'omitnan')./sqrt(max(n,1));
end

function value = winRate(x,column)
    value = mean(x(:,:,column)>0,2,'omitnan');
end

function figs = plotScaling(study)
    summary = study.summary;
    figs.performance = figure('Name','Distributed all-layer watcher-count scaling');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on; grid on;
    plot(summary.nWatchers,summary.localPositionRMSE,'-o','LineWidth',1.2);
    plot(summary.nWatchers,summary.outputPositionRMSE,'-o','LineWidth',1.2);
    plot(summary.nWatchers,summary.allLayerPositionRMSE,'-o','LineWidth',1.2);
    ylabel('mean position RMSE [m]');
    legend({'local','output-layer','all-layer FOGM'},'Location','best');
    title('Fixed-block watcher scaling (global DNN size grows with N_w)');
    nexttile; hold on; grid on;
    errorbar(summary.nWatchers,summary.allLayerGainVsLocal, ...
        summary.allLayerGainVsLocal-summary.allLayerGainCI95Lower, ...
        summary.allLayerGainCI95Upper-summary.allLayerGainVsLocal,'-o','LineWidth',1.2);
    errorbar(summary.nWatchers,summary.allLayerGainVsOne, ...
        summary.allLayerGainVsOne-summary.allLayerGainVsOneCI95Lower, ...
        summary.allLayerGainVsOneCI95Upper-summary.allLayerGainVsOne,'-o','LineWidth',1.2);
    yline(0,'k--'); xlabel('number of watchers');
    ylabel('position-RMSE improvement [m]');
    legend({'all-layer minus local','all-layer minus N_w=1'},'Location','best');

    figs.consistency = figure('Name','All-layer FOGM consistency versus watcher count');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; plot(summary.nWatchers,summary.allLayerMeanNIS,'-o','LineWidth',1.2); grid on;
    yline(1,'k--'); ylabel('mean NIS');
    nexttile; yyaxis left;
    plot(summary.nWatchers,summary.allLayerAccelerationRMSE,'-o','LineWidth',1.2);
    ylabel('acceleration RMSE [m/s^2]');
    yyaxis right;
    plot(summary.nWatchers,summary.globalParameterCount,'-s','LineWidth',1.2);
    ylabel('global parameter count'); grid on; xlabel('number of watchers');
end
