function study = run_angle_only_parameter_learning_ablation(seed,T,dt,scenario,makePlots,nWatchers,hiddenLayerCount,architecture)
%RUN_ANGLE_ONLY_PARAMETER_LEARNING_ABLATION Validate online DNN learning.
% Compares three estimators under identical truth, bearing-noise realization,
% and nominal-derived maneuver schedule:
%   1) frozen branch parameters (no online parameter adaptation),
%   2) local angle-only branch adaptation without parameter communication,
%   3) local angle-only adaptation with instantaneous parameter sharing.
%
% This is deliberately not a comparison against unknown "true branch"
% parameters.  Individual additive branch allocations are non-identifiable;
% the measurable quantities are the total acceleration, target state, NIS,
% posterior parameter movement, and the number of transmitted posteriors.

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 5 || isempty(makePlots), makePlots = true; end
    if nargin < 6 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 7 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 8 || isempty(architecture), architecture = "additive_vector"; end
    architecture = string(architecture);
    assert(any(architecture == ["additive_vector" "additive_full_dnn"]), ...
        'Use additive_vector or additive_full_dnn for this ablation.');

    sharedArguments = {seed,T,false,scenario,dt,true,"local_radial",true, ...
        nWatchers,"parameter_covariance",architecture};

    frozenOut = runToyQuiet(sharedArguments,"never",false,hiddenLayerCount);
    localOut = runToyQuiet(sharedArguments,"never",true,hiddenLayerCount);
    sharedOut = runToyQuiet(sharedArguments,"instantaneous",true,hiddenLayerCount);

    results = {frozenOut.sharedAdditive,localOut.sharedAdditive,sharedOut.sharedAdditive};
    if architecture == "additive_full_dnn"
        labels = ["Frozen full DNN";"Local full-DNN angle-only adaptation"; ...
            "Shared full-DNN angle-only adaptation"];
    else
        labels = ["Frozen heads";"Local angle-only adaptation"; ...
        "Shared angle-only adaptation"];
    end
    metrics = cellfun(@learningMetrics,results,'UniformOutput',false);

    study = struct;
    study.seed = seed; study.T = T; study.dt = dt;
    study.scenario = string(scenario); study.nWatchers = nWatchers;
    study.hiddenLayerCount = hiddenLayerCount;
    study.architecture = architecture;
    study.frozen = frozenOut.sharedAdditive;
    study.local = localOut.sharedAdditive;
    study.shared = sharedOut.sharedAdditive;
    study.labels = labels;
    study.summary = table(labels, ...
        cellfun(@(m)m.positionRMSE,metrics)', ...
        cellfun(@(m)m.velocityRMSE,metrics)', ...
        cellfun(@(m)m.accelerationRMSE,metrics)', ...
        cellfun(@(m)m.finalPositionRMSE,metrics)', ...
        cellfun(@(m)m.meanParameterChange,metrics)', ...
        cellfun(@(m)m.meanNIS,metrics)', ...
        cellfun(@(m)m.uploadCount,metrics)', ...
        'VariableNames',{'caseName','positionRMSE','velocityRMSE', ...
        'accelerationRMSE','finalPositionRMSE','meanParameterChange', ...
        'meanNIS','parameterUploads'});

    if makePlots
        study.figures = plotLearningAblation(results,labels);
    else
        study.figures = struct;
    end
    disp(study.summary);
end

function out = runToyQuiet(arguments,communicationMode,adaptParameters,hiddenLayerCount)
    % Suppress repeated per-case tables; retain deterministic RNG resetting
    % inside the toy driver so all three cases receive identical data.
    evalc('out = run_toy_distributed_additive_dnn_ekf(arguments{:},communicationMode,adaptParameters,hiddenLayerCount);');
end

function m = learningMetrics(r)
    stateError = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
    accelerationError = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
    theta0 = reshape(r.thetaInitial,size(r.thetaInitial,1),1,size(r.thetaInitial,2));
    thetaChange = r.xhat(5:end,:,:)-repmat(theta0,1,numel(r.time),1);
    headNorm = squeeze(vecnorm(thetaChange,2,1));
    if isvector(headNorm), headNorm = reshape(headNorm,[],1); end
    positionSeries = squeeze(sqrt(mean(vecnorm(stateError(1:2,:,:),2,1).^2,3)));
    m.positionSeries = positionSeries(:)';
    m.accelerationSeries = squeeze(sqrt(mean(vecnorm(accelerationError,2,1).^2,3)))';
    m.parameterChangeSeries = mean(headNorm,2)';
    m.positionRMSE = sqrt(mean(vecnorm(stateError(1:2,:,:),2,1).^2,'all'));
    m.velocityRMSE = sqrt(mean(vecnorm(stateError(3:4,:,:),2,1).^2,'all'));
    m.accelerationRMSE = sqrt(mean(vecnorm(accelerationError,2,1).^2,'all'));
    m.finalPositionRMSE = sqrt(mean(vecnorm(stateError(1:2,round(.9*numel(r.time)):end,:),2,1).^2,'all'));
    m.meanParameterChange = mean(headNorm,'all');
    m.meanNIS = mean(r.NIS,'all','omitnan');
    m.uploadCount = nnz(r.communicationEvent);
end

function figs = plotLearningAblation(results,labels)
    colors = lines(numel(results));
    metrics = cellfun(@learningMetrics,results,'UniformOutput',false);
    figs = struct;
    figs.performance = figure('Name','Angle-only parameter learning ablation');
    tiledlayout(3,1,'TileSpacing','compact');
    titles = ["position estimation error" "total acceleration approximation error" ...
        "mean DNN-parameter posterior change"];
    for q=1:3
        nexttile; hold on;
        for c=1:numel(results)
            if q == 1, y = metrics{c}.positionSeries;
            elseif q == 2, y = metrics{c}.accelerationSeries;
            else, y = metrics{c}.parameterChangeSeries; end
            plot(results{c}.time,y,'LineWidth',1.25,'Color',colors(c,:));
        end
        title(titles(q)); xlabel('time [s]');
        if q == 1, ylabel('position RMSE [m]');
        elseif q == 2, ylabel('RMSE(d-hat minus d) [m/s^2]');
        else, ylabel('mean ||\theta_i-\theta_{i,0}||'); end
        grid on;
        if q == 1, legend(labels,'Location','best'); end
    end
    figs.innovation = figure('Name','Angle-only innovation consistency'); hold on;
    for c=1:numel(results)
        nis = mean(results{c}.NIS,2,'omitnan');
        plot(results{c}.time,nis,'LineWidth',1.25,'Color',colors(c,:));
    end
    yline(1,'k--','NIS reference'); grid on; xlabel('time [s]'); ylabel('mean NIS');
    title('bearing-innovation consistency'); legend(labels,'Location','best');
end
