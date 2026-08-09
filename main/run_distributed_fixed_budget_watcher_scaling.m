function study = run_distributed_fixed_budget_watcher_scaling( ...
    seeds,T,dt,watcherCounts,hiddenLayerCount,targetGlobalParameterCount,makePlots)
%RUN_DISTRIBUTED_FIXED_BUDGET_WATCHER_SCALING Isolate watcher-information gain.
%
% Keeps the *total* global DNN parameter count approximately fixed while
% varying N_w.  Each watcher owns one complete branch, but its hidden width
% is selected as the integer width that makes
%
%       N_w * nThetaPerBranch(width)
%
% closest to targetGlobalParameterCount.  Hence this study separates the
% benefit of added angle-only measurements / innovation packets from the
% trivial benefit of simply enlarging the global DNN.
%
% For L=3 and target=180, N_w=[1 2 4 8] gives widths [7 5 3 2] and
% global parameter counts [161 190 180 208], respectively.
%
% Example:
%   study = run_distributed_fixed_budget_watcher_scaling( ...
%       101:110,600,.1,[1 2 4 8],3,180,true);

    if nargin < 1 || isempty(seeds), seeds = 101:110; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(watcherCounts), watcherCounts = [1 2 4 8]; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(targetGlobalParameterCount)
        targetGlobalParameterCount = 4*parameterCountForWidth(3,hiddenLayerCount);
    end
    if nargin < 7 || isempty(makePlots), makePlots = true; end

    seeds = reshape(seeds,1,[]);
    watcherCounts = reshape(watcherCounts,1,[]);
    validateattributes(watcherCounts,{'numeric'}, ...
        {'vector','integer','>=',1,'<=',8});
    validateattributes(targetGlobalParameterCount,{'numeric'}, ...
        {'scalar','integer','positive'});

    widths = arrayfun(@(n) selectWidth(n,hiddenLayerCount, ...
        targetGlobalParameterCount),watcherCounts);
    perBranch = arrayfun(@(q) parameterCountForWidth(q,hiddenLayerCount),widths);
    globalCount = watcherCounts.*perBranch;
    caseNames = ["Local independent block"; ...
        "Distributed output-layer information EKF: 60 s"; ...
        "Distributed all-layer FOGM + covariance matching EKF: 60 s"];
    metricNames = ["positionRMSE","velocityRMSE","accelerationRMSE", ...
        "finalPositionRMSE","meanNIS","parameterUploads"];
    nW = numel(watcherCounts); nSeed = numel(seeds); nCase = numel(caseNames);
    metric = nan(nW,nSeed,nCase,numel(metricNames));

    fprintf(['Fixed-budget watcher scaling: %d seeds, N_w=%s, widths=%s, ', ...
        'target=%d parameters, L=%d\n'],nSeed,mat2str(watcherCounts), ...
        mat2str(widths),targetGlobalParameterCount,hiddenLayerCount);
    for w = 1:nW
        for s = 1:nSeed
            fprintf('  N_w=%d, width=%d, seed %d (%d/%d)\n', ...
                watcherCounts(w),widths(w),seeds(s),s,nSeed);
            outputRun = runQuiet(seeds(s),T,dt,watcherCounts(w), ...
                hiddenLayerCount,"collaborative_output_info_60s",widths(w));
            fogmRun = runQuiet(seeds(s),T,dt,watcherCounts(w), ...
                hiddenLayerCount,"collaborative_all_fogm_cm_60s",widths(w));
            localA = extractMetrics(outputRun.localOnly);
            localB = extractMetrics(fogmRun.localOnly);
            assert(max(abs(localA-localB)) < 1e-10, ...
                'Paired local baselines disagree at N_w=%d, seed=%d.', ...
                watcherCounts(w),seeds(s));
            metric(w,s,1,:) = localA;
            metric(w,s,2,:) = extractMetrics(outputRun.sharedBlock);
            metric(w,s,3,:) = extractMetrics(fogmRun.sharedBlock);
        end
    end

    % Keep explicit singleton dimensions so one-seed pilot runs work too.
    local = reshape(metric(:,:,1,:),nW,nSeed,numel(metricNames));
    output = reshape(metric(:,:,2,:),nW,nSeed,numel(metricNames));
    fogm = reshape(metric(:,:,3,:),nW,nSeed,numel(metricNames));
    outputGain = local-output;
    fogmGain = local-fogm;
    fogmVsOutput = output-fogm;
    fogmVsOne = repmat(fogm(1,:,:),nW,1,1)-fogm;

    summary = table(watcherCounts(:),widths(:),perBranch(:),globalCount(:), ...
        meanMetric(local,1),meanMetric(output,1),meanMetric(fogm,1), ...
        meanMetric(fogmGain,1),ciLower(fogmGain,1),ciUpper(fogmGain,1),winRate(fogmGain,1), ...
        meanMetric(fogmVsOutput,1),ciLower(fogmVsOutput,1),ciUpper(fogmVsOutput,1),winRate(fogmVsOutput,1), ...
        meanMetric(fogmVsOne,1),ciLower(fogmVsOne,1),ciUpper(fogmVsOne,1),winRate(fogmVsOne,1), ...
        meanMetric(local,3),meanMetric(output,3),meanMetric(fogm,3), ...
        meanMetric(fogm,5),meanMetric(fogm,6), ...
        'VariableNames',{'nWatchers','branchWidth','parametersPerBranch','globalParameterCount', ...
        'localPositionRMSE','outputPositionRMSE','allLayerPositionRMSE', ...
        'allLayerGainVsLocal','allLayerGainCI95Lower','allLayerGainCI95Upper','allLayerWinRateVsLocal', ...
        'allLayerGainVsOutput','allLayerGainVsOutputCI95Lower','allLayerGainVsOutputCI95Upper','allLayerWinRateVsOutput', ...
        'allLayerGainVsOne','allLayerGainVsOneCI95Lower','allLayerGainVsOneCI95Upper','allLayerWinRateVsOne', ...
        'localAccelerationRMSE','outputAccelerationRMSE','allLayerAccelerationRMSE', ...
        'allLayerMeanNIS','allLayerMeanUploads'});

    study = struct;
    study.seeds = seeds; study.T = T; study.dt = dt;
    study.watcherCounts = watcherCounts; study.branchWidths = widths;
    study.hiddenLayerCount = hiddenLayerCount;
    study.targetGlobalParameterCount = targetGlobalParameterCount;
    study.caseNames = caseNames; study.metricNames = metricNames; study.metric = metric;
    study.outputGainVsLocal = outputGain;
    study.allLayerGainVsLocal = fogmGain;
    study.allLayerGainVsOutput = fogmVsOutput;
    study.allLayerGainVsOneWatcher = fogmVsOne;
    study.summary = summary;
    study.interpretation = [ ...
        "Global DNN size is approximately fixed. Positive allLayerGainVsOne ", ...
        "therefore supports an added-watcher information/coordination benefit."];
    disp(summary);
    if makePlots, study.figures = plotScaling(study); else, study.figures = struct; end
end

function q = selectWidth(nWatchers,nHidden,target)
    candidates = 1:32;
    counts = nWatchers*arrayfun(@(x) parameterCountForWidth(x,nHidden),candidates);
    [~,idx] = min(abs(counts-target));
    q = candidates(idx);
end

function n = parameterCountForWidth(q,nHidden)
    % First hidden layer: q*(4 inputs + bias); later hidden layers: q*(q+bias).
    % Linear 2-by-q output layer has no output bias.
    n = q*(4+1) + (nHidden-1)*q*(q+1) + 2*q;
end

function out = runQuiet(seed,T,dt,nWatchers,nHidden,mode,width)
    command = sprintf(['out = run_toy_block_structured_global_head_dnn_ekf(', ...
        '%d,%.17g,false,%d,%.17g,%d,"%s",%d);'], ...
        seed,T,nWatchers,dt,nHidden,mode,width);
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
    [mu,se] = pairedMeanSE(x,column); value = mu-1.96*se;
end

function value = ciUpper(x,column)
    [mu,se] = pairedMeanSE(x,column); value = mu+1.96*se;
end

function [mu,se] = pairedMeanSE(x,column)
    y = x(:,:,column); mu = mean(y,2,'omitnan');
    se = std(y,0,2,'omitnan')./sqrt(max(sum(isfinite(y),2),1));
end

function value = winRate(x,column)
    value = mean(x(:,:,column)>0,2,'omitnan');
end

function figs = plotScaling(study)
    s = study.summary;
    figs.performance = figure('Name','Fixed-total-parameter watcher scaling');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on; grid on;
    plot(s.nWatchers,s.localPositionRMSE,'-o','LineWidth',1.2);
    plot(s.nWatchers,s.outputPositionRMSE,'-o','LineWidth',1.2);
    plot(s.nWatchers,s.allLayerPositionRMSE,'-o','LineWidth',1.2);
    ylabel('mean position RMSE [m]');
    legend({'local','output-layer','all-layer FOGM'},'Location','best');
    title('Fixed total-DNN-parameter budget');
    nexttile; grid on; hold on;
    errorbar(s.nWatchers,s.allLayerGainVsOne, ...
        s.allLayerGainVsOne-s.allLayerGainVsOneCI95Lower, ...
        s.allLayerGainVsOneCI95Upper-s.allLayerGainVsOne,'-o','LineWidth',1.2);
    yline(0,'k--'); xlabel('number of watchers');
    ylabel('all-layer gain vs N_w=1 [m]');

    figs.diagnostics = figure('Name','Fixed-budget all-layer diagnostics');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; plot(s.nWatchers,s.allLayerMeanNIS,'-o','LineWidth',1.2); grid on;
    yline(1,'k--'); ylabel('mean NIS');
    nexttile; yyaxis left;
    plot(s.nWatchers,s.allLayerAccelerationRMSE,'-o','LineWidth',1.2);
    ylabel('acceleration RMSE [m/s^2]');
    yyaxis right;
    stairs(s.nWatchers,s.branchWidth,'-s','LineWidth',1.2);
    ylabel('branch width'); xlabel('number of watchers'); grid on;
end
