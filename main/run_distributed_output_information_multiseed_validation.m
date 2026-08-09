function study = run_distributed_output_information_multiseed_validation( ...
    seeds,T,dt,nWatchers,hiddenLayerCount,makePlots)
%RUN_DISTRIBUTED_OUTPUT_INFORMATION_MULTISEED_VALIDATION Validate Phase 1.
%
% Repeats the five-case distributed-block ablation over paired random seeds.
% The primary claim is accepted only when a collaborative information-EKF
% case improves on local independent learning with consistent innovation
% statistics across the seed set.
%
% Example:
%   study = run_distributed_output_information_multiseed_validation( ...
%       101:110,600,.1,4,3,true);

    if nargin < 1 || isempty(seeds), seeds = 101:110; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end
    seeds = reshape(seeds,1,[]);

    caseNames = ["Local independent block"; ...
        "Shared block: no communication"; ...
        "Shared block: raw periodic 60 s"; ...
        "Distributed all-parameter information EKF: 60 s"; ...
        "Distributed output-layer information EKF: 60 s"; ...
        "Distributed all-layer FOGM + covariance matching EKF: 60 s"];
    nSeeds = numel(seeds); nCases = numel(caseNames);
    metricNames = ["positionRMSE","velocityRMSE","accelerationRMSE", ...
        "finalPositionRMSE","meanNIS"];
    metrics = nan(nSeeds,nCases,numel(metricNames));
    perSeed = cell(nSeeds,1);

    for s = 1:nSeeds
        fprintf('Output-layer collaborative validation: seed %d (%d/%d)\n', ...
            seeds(s),s,nSeeds);
        run = run_distributed_block_information_ablation( ...
            seeds(s),T,dt,nWatchers,hiddenLayerCount,false);
        perSeed{s} = run;
        for c = 1:nCases
            metrics(s,c,:) = [run.summary.positionRMSE(c), ...
                run.summary.velocityRMSE(c),run.summary.accelerationRMSE(c), ...
                run.summary.finalPositionRMSE(c),run.summary.meanNIS(c)];
        end
    end

    outputCase = nCases-1;
    fogmCase = nCases;
    localCase = 1;
    positionImprovement = metrics(:,localCase,1)-metrics(:,outputCase,1);
    velocityImprovement = metrics(:,localCase,2)-metrics(:,outputCase,2);
    accelerationImprovement = metrics(:,localCase,3)-metrics(:,outputCase,3);
    finalPositionImprovement = metrics(:,localCase,4)-metrics(:,outputCase,4);
    fogmPositionImprovement = metrics(:,localCase,1)-metrics(:,fogmCase,1);
    fogmAccelerationImprovement = metrics(:,localCase,3)-metrics(:,fogmCase,3);

    summary = table(caseNames, ...
        squeeze(mean(metrics(:,:,1),1,'omitnan'))', ...
        squeeze(std(metrics(:,:,1),0,1,'omitnan'))', ...
        squeeze(mean(metrics(:,:,2),1,'omitnan'))', ...
        squeeze(mean(metrics(:,:,3),1,'omitnan'))', ...
        squeeze(mean(metrics(:,:,4),1,'omitnan'))', ...
        squeeze(mean(metrics(:,:,5),1,'omitnan'))', ...
        'VariableNames',{'caseName','positionRMSEMean','positionRMSEStd', ...
        'velocityRMSEMean','accelerationRMSEMean','finalPositionRMSEMean', ...
        'meanNIS'});

    paired = table( ...
        mean(positionImprovement,'omitnan'),std(positionImprovement,0,'omitnan'), ...
        nnz(positionImprovement>0), ...
        mean(velocityImprovement,'omitnan'), ...
        mean(accelerationImprovement,'omitnan'), ...
        mean(finalPositionImprovement,'omitnan'), ...
        mean(metrics(:,outputCase,5),'omitnan'), ...
        'VariableNames',{'meanPositionImprovement','stdPositionImprovement', ...
        'positivePositionSeeds','meanVelocityImprovement', ...
        'meanAccelerationImprovement','meanFinalPositionImprovement', ...
        'outputLayerMeanNIS'});
    pairedFOGM = table( ...
        mean(fogmPositionImprovement,'omitnan'),std(fogmPositionImprovement,0,'omitnan'), ...
        nnz(fogmPositionImprovement>0),mean(fogmAccelerationImprovement,'omitnan'), ...
        mean(metrics(:,fogmCase,5),'omitnan'), ...
        'VariableNames',{'meanPositionImprovement','stdPositionImprovement', ...
        'positivePositionSeeds','meanAccelerationImprovement','allLayerMeanNIS'});

    study.seeds = seeds;
    study.perSeed = perSeed;
    study.caseNames = caseNames;
    study.metricNames = metricNames;
    study.metrics = metrics;
    study.summary = summary;
    study.pairedOutputVsLocal = paired;
    study.pairedAllLayerFOGMVsLocal = pairedFOGM;
    disp(summary);
    disp(paired);
    disp(pairedFOGM);

    if makePlots
        study.figures = plotValidation(study,positionImprovement, ...
            accelerationImprovement,fogmPositionImprovement, ...
            fogmAccelerationImprovement);
    else
        study.figures = struct;
    end
end

function figs = plotValidation(study,positionImprovement,accelerationImprovement, ...
    fogmPositionImprovement,fogmAccelerationImprovement)
    nCases = numel(study.caseNames);
    colors = lines(nCases);
    figs.metrics = figure('Name','Multi-seed output-layer collaboration validation');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; hold on; grid on;
    for c = 1:nCases
        scatter(study.seeds,squeeze(study.metrics(:,c,1)),36, ...
            'filled','MarkerFaceColor',colors(c,:));
    end
    ylabel('position RMSE [m]');
    legend(study.caseNames,'Location','best');
    nexttile; hold on; grid on;
    stem(study.seeds,positionImprovement,'filled','LineWidth',1.1); hold on;
    stem(study.seeds,fogmPositionImprovement,'filled','LineWidth',1.1);
    yline(0,'k--');
    xlabel('seed'); ylabel('local minus output-layer position RMSE [m]');
    legend('output layer','all-layer FOGM','Location','best');
    title('Positive values favor distributed learning');

    figs.improvement = figure('Name','Output-layer collaboration paired improvements');
    tiledlayout(2,1,'TileSpacing','compact');
    nexttile; stem(study.seeds,positionImprovement,'filled'); hold on;
    stem(study.seeds,fogmPositionImprovement,'filled'); grid on; yline(0,'k--');
    ylabel('position improvement [m]');
    nexttile; stem(study.seeds,accelerationImprovement,'filled'); hold on;
    stem(study.seeds,fogmAccelerationImprovement,'filled'); grid on; yline(0,'k--');
    xlabel('seed'); ylabel('acceleration improvement [m/s^2]');
end
