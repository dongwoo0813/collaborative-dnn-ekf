function study = run_grouped_block_collaboration_ablation( ...
    seeds,T,dt,nBlocks,branchWidth,makePlots)
%RUN_GROUPED_BLOCK_COLLABORATION_ABLATION More watchers per fixed DNN block.
%
% Keeps one global DNN fixed at nBlocks branches of branchWidth.  A watcher
% sends its information packet to only its assigned block, so the 8-watcher
% two-per-block case adds measurement information without adding DNN
% parameters or asking every scalar innovation to update every block.
%
% Cases:
%   4x1: four watchers / four blocks / one packet source per block.
%   8x1: eight watchers / four blocks / only one source per block.
%   8x2: eight watchers / four blocks / two sources per block (proposed).
%
% Example:
%   study = run_grouped_block_collaboration_ablation(101:110,600,.1,4,3,true);

    if nargin < 1 || isempty(seeds), seeds = 101:110; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(nBlocks), nBlocks = 4; end
    if nargin < 5 || isempty(branchWidth), branchWidth = 3; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end
    seeds = reshape(seeds,1,[]);
    validateattributes(nBlocks,{'numeric'},{'scalar','integer','>=',1,'<=',8});
    validateattributes(branchWidth,{'numeric'},{'scalar','integer','>=',1,'<=',32});

    nFour = nBlocks; nEight = 2*nBlocks;
    assert(nEight <= 8,'This toy driver supports at most eight watchers.');
    caseNames = ["4 watchers: local independent"; ...
        "4 watchers: 1 watcher per block"; ...
        "8 watchers: local independent"; ...
        "8 watchers: 1 watcher per block"; ...
        "8 watchers: 2 watchers per block"];
    metricNames = ["positionRMSE","velocityRMSE","accelerationRMSE", ...
        "finalPositionRMSE","meanNIS","parameterUploads"];
    nSeed = numel(seeds); nCase = numel(caseNames);
    metric = nan(nSeed,nCase,numel(metricNames));
    perSeed = cell(nSeed,1);

    fprintf(['Grouped-block collaboration: %d seeds, %d blocks x width %d; ', ...
        'compare %d and %d watchers.\n'],nSeed,nBlocks,branchWidth,nFour,nEight);
    for s = 1:nSeed
        fprintf('  seed %d (%d/%d)\n',seeds(s),s,nSeed);
        four = runQuiet(seeds(s),T,dt,nFour,nBlocks,branchWidth,"assigned");
        eightOne = runQuiet(seeds(s),T,dt,nEight,nBlocks,branchWidth, ...
            "assigned_one_per_block");
        eightTwo = runQuiet(seeds(s),T,dt,nEight,nBlocks,branchWidth,"assigned");
        metric(s,1,:) = extractMetrics(four.localOnly);
        metric(s,2,:) = extractMetrics(four.sharedBlock);
        metric(s,3,:) = extractMetrics(eightTwo.localOnly);
        metric(s,4,:) = extractMetrics(eightOne.sharedBlock);
        metric(s,5,:) = extractMetrics(eightTwo.sharedBlock);
        perSeed{s} = struct('fourOne',four.sharedBlock, ...
            'eightOne',eightOne.sharedBlock,'eightTwo',eightTwo.sharedBlock);
    end

    % Preserve the metric dimension for one-seed smoke tests.
    fourOne = reshape(metric(:,2,:),nSeed,numel(metricNames));
    localEight = reshape(metric(:,3,:),nSeed,numel(metricNames));
    eightOne = reshape(metric(:,4,:),nSeed,numel(metricNames));
    eightTwo = reshape(metric(:,5,:),nSeed,numel(metricNames));
    gainTwoVsOne = eightOne-eightTwo;
    gainTwoVsFour = fourOne-eightTwo;
    gainTwoVsLocal = localEight-eightTwo;

    summary = table(caseNames,meanMetric(metric,1),stdMetric(metric,1), ...
        meanMetric(metric,2),meanMetric(metric,3),meanMetric(metric,4),meanMetric(metric,5), ...
        'VariableNames',{'caseName','positionRMSEMean','positionRMSEStd', ...
        'velocityRMSEMean','accelerationRMSEMean','finalPositionRMSEMean','meanNIS'});
    paired = table( ...
        meanGain(gainTwoVsOne,1),ciLower(gainTwoVsOne,1),ciUpper(gainTwoVsOne,1),winRate(gainTwoVsOne,1), ...
        meanGain(gainTwoVsFour,1),ciLower(gainTwoVsFour,1),ciUpper(gainTwoVsFour,1),winRate(gainTwoVsFour,1), ...
        meanGain(gainTwoVsLocal,1),ciLower(gainTwoVsLocal,1),ciUpper(gainTwoVsLocal,1),winRate(gainTwoVsLocal,1), ...
        meanGain(gainTwoVsOne,3),meanGain(gainTwoVsFour,3), ...
        'VariableNames',{'twoPerBlockGainVsOnePerBlock','twoVsOneCI95Lower','twoVsOneCI95Upper','twoVsOneWinRate', ...
        'twoPerBlockGainVsFourWatcher','twoVsFourCI95Lower','twoVsFourCI95Upper','twoVsFourWinRate', ...
        'twoPerBlockGainVsEightLocal','twoVsLocalCI95Lower','twoVsLocalCI95Upper','twoVsLocalWinRate', ...
        'twoPerBlockAccelerationGainVsOne','twoPerBlockAccelerationGainVsFour'});

    study = struct;
    study.seeds = seeds; study.T = T; study.dt = dt;
    study.nBlocks = nBlocks; study.branchWidth = branchWidth;
    study.globalParameterCount = nBlocks*parameterCount(branchWidth,3);
    study.caseNames = caseNames; study.metricNames = metricNames;
    study.metric = metric; study.perSeed = perSeed;
    study.summary = summary; study.paired = paired;
    study.claimRule = [ ...
        "The key claim is supported when twoPerBlockGainVsOnePerBlock has ", ...
        "a positive 95% CI: additional packet sources improve the same fixed DNN."];
    disp(summary); disp(paired);
    if makePlots, study.figures = plotStudy(study); else, study.figures = struct; end
end

function out = runQuiet(seed,T,dt,nWatchers,nBlocks,width,packetMode)
    command = sprintf(['out = run_toy_block_structured_global_head_dnn_ekf(', ...
        '%d,%.17g,false,%d,%.17g,3,"collaborative_all_fogm_cm_60s",%d,%d,"%s");'], ...
        seed,T,nWatchers,dt,width,nBlocks,packetMode);
    evalc(command);
end

function m = extractMetrics(result)
    nis = result.nis(2:end,:); nis = nis(isfinite(nis));
    m = [result.positionRMSE,result.velocityRMSE,result.accelerationRMSE, ...
        result.finalPositionRMSE,mean(nis),result.parameterUploads];
end

function n = parameterCount(width,nHidden)
    n = width*5 + (nHidden-1)*width*(width+1) + 2*width;
end

function value = meanMetric(x,column)
    value = squeeze(mean(x(:,:,column),1,'omitnan'))';
end

function value = stdMetric(x,column)
    value = squeeze(std(x(:,:,column),0,1,'omitnan'))';
end

function value = meanGain(x,column)
    value = mean(x(:,column),'omitnan');
end

function value = ciLower(x,column)
    [mu,se] = gainMeanSE(x,column); value = mu-1.96*se;
end

function value = ciUpper(x,column)
    [mu,se] = gainMeanSE(x,column); value = mu+1.96*se;
end

function [mu,se] = gainMeanSE(x,column)
    y = x(:,column); mu = mean(y,'omitnan');
    se = std(y,0,'omitnan')/sqrt(max(sum(isfinite(y)),1));
end

function value = winRate(x,column)
    value = mean(x(:,column)>0,'omitnan');
end

function figs = plotStudy(study)
    colors = lines(numel(study.caseNames));
    figs.metrics = figure('Name','Grouped DNN-block collaboration');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on; grid on;
    for c = 1:numel(study.caseNames)
        scatter(study.seeds,study.metric(:,c,1),36,'filled', ...
            'MarkerFaceColor',colors(c,:));
    end
    ylabel('position RMSE [m]'); legend(study.caseNames,'Location','best');
    title('Fixed 4-block, width-3 DNN: packet-source ablation');
    nexttile; hold on; grid on;
    one = squeeze(study.metric(:,4,1)); two = squeeze(study.metric(:,5,1));
    stem(study.seeds,one-two,'filled','LineWidth',1.1); yline(0,'k--');
    xlabel('seed'); ylabel('one-per-block minus two-per-block [m]');

    figs.summary = figure('Name','Grouped-block collaboration summary');
    bar(categorical(study.caseNames),squeeze(mean(study.metric(:,:,1),1)));
    grid on; ylabel('mean position RMSE [m]'); xtickangle(18);
end
