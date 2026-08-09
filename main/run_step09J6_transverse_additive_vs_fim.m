function [out,diagOut,figures] = run_step09J6_transverse_additive_vs_fim( ...
    makePlots,desiredBaseline,simulationTime,maneuverMode,seedOverride, ...
    communicationArchitecture,uploadMode,residualMode)
%RUN_STEP09J6_TRANSVERSE_ADDITIVE_VS_FIM Compare GS residual composition.
%
% Both cases use independent bearing-only target-state filters, the same
% target truth, 20-m (default) prescribed transverse maneuver, seed,
% measurement-noise draws, DNN architecture, initial weights, and fixed
% covariance model. Only cfg.gs.compositeMode changes:
%
%   A: additive
%   F: fim_weighted_additive
%
% No measurements or kinematic-state estimates are centralized.
%
% Usage:
%   [out20,diag20,fig20] = ...
%       run_step09J6_transverse_additive_vs_fim(true,20);
%   [outLong,diagLong,figLong] = ...
%       run_step09J6_transverse_additive_vs_fim(true,100,600);
%   [outActive,diagActive,figActive] = ...
%       run_step09J6_transverse_additive_vs_fim( ...
%       true,100,600,"observability_seeking");
%   Coast/frozen baseline is selected with maneuverMode="coast".


    if nargin < 1
        makePlots = true;
    end
    if nargin < 2
        desiredBaseline = 20;
    end
    if nargin < 3
        simulationTime = 200;
    end
    if nargin < 4
        maneuverMode = "transverse";
    end
    if nargin < 5
        seedOverride = [];
    end
    if nargin < 6 || isempty(communicationArchitecture)
        communicationArchitecture = "gs";
    end
    if nargin < 7 || isempty(uploadMode)
        uploadMode = "after_measurement_update";
    end
    if nargin < 8 || isempty(residualMode)
        residualMode = "both";
    end
    maneuverMode = string(maneuverMode);
    communicationArchitecture = string(communicationArchitecture);
    uploadMode = string(uploadMode);
    residualMode = string(residualMode);
    validateattributes(desiredBaseline,{'numeric'}, ...
        {'scalar','real','finite','positive'},mfilename,'desiredBaseline');
    validateattributes(simulationTime,{'numeric'}, ...
        {'scalar','real','finite','positive'},mfilename,'simulationTime');

    addpath(genpath(pwd));
    [cfgBase,seed,meta] = config_step09J6_seed101_operational();
    if ~isempty(seedOverride)
        validateattributes(seedOverride,{'numeric'},{'scalar','finite','real'}, ...
            mfilename,"seedOverride");
        seed = double(seedOverride);
    end
    cfgBase.T = simulationTime;
    cfgBase.time = 0:cfgBase.dt:cfgBase.T;
    cfgBase.N = numel(cfgBase.time);
    cfgBase.scenario.watcherModel = "matched_velocity_coast";
    cfgBase.watchers.motionMode = "prescribed";
    cfgBase.watchers.coastVelocity = cfgBase.target.v0;
    cfgBase.control.translationMode = "none";
    cfgBase.meas.type = "bearing";
    cfgBase.meas.sigmaBearing = deg2rad(0.01);
    cfgBase.meas.R = cfgBase.meas.sigmaBearing^2;
    cfgBase.meas.availabilityMode = "always";
    cfgBase.fov.enabled = false;
    cfgBase.fov.guardUnimplementedMode = true;

    cfgBase.control.obs.enabled = true;
    cfgBase.control.obs.mode = "transverse";
    cfgBase.control.obs.startTime = 40.0;
    cfgBase.control.obs.burnDuration = 50.0;
    coastAfterBurn = cfgBase.T-(cfgBase.control.obs.startTime+ ...
        cfgBase.control.obs.burnDuration);
    if coastAfterBurn <= 0 && maneuverMode ~= "coast"
        error("simulationTime must exceed burn end time %.1f s.", ...
            cfgBase.control.obs.startTime+cfgBase.control.obs.burnDuration);
    end
    baselineFactor = 0.5*cfgBase.control.obs.burnDuration^2+ ...
        cfgBase.control.obs.burnDuration*coastAfterBurn;
    requestedAcceleration = desiredBaseline/baselineFactor;
    cfgBase.control.obs.acceleration = requestedAcceleration;
    switch maneuverMode
        case "coast"
            % Frozen/coasting baseline: no active translational maneuver.
            % The prescribed watcher model retains its initial coast
            % velocity, so this isolates the information available without
            % an intentional observability maneuver.
            cfgBase.control.obs.enabled = false;
            cfgBase.control.translationMode = "none";
            cfgBase.control.obs.acceleration = 0.0;
            cfgBase.control.obs.burnDuration = 0.0;
            coastAfterBurn = cfgBase.T-cfgBase.control.obs.startTime;
            baselineFactor = 0.0;
            requestedAcceleration = 0.0;
        case "transverse"
            % Existing prescribed maneuver.
        case "observability_seeking"
            % Use a realistic 20-mN small-spacecraft propulsion envelope
            % and keep the active-sensing controller enabled from 40 s to
            % the end of the run. Recompute the acceleration so the input
            % desiredBaseline refers to this longer maneuver interval.
            cfgBase.watchers.maxThrust = 0.02;
            cfgBase.control.obs.burnDuration = ...
                cfgBase.T-cfgBase.control.obs.startTime;
            coastAfterBurn = 0;
            baselineFactor = 0.5*cfgBase.control.obs.burnDuration^2;
            requestedAcceleration = desiredBaseline/baselineFactor;
            cfgBase.watchers.motionMode = "controlled";
            cfgBase.control.translationMode = "observability_seeking";
            cfgBase.control.obs.mode = "observability_seeking";
            accelerationLimit = min(cfgBase.watchers.maxThrust./ ...
                cfgBase.watchers.mass);
            cfgBase.control.obs.acceleration = min( ...
                requestedAcceleration,accelerationLimit);
            cfgBase.control.obs.numCandidateDirections = 8;
            cfgBase.control.obs.planningHorizon = 30.0;
            cfgBase.control.obs.planningDt = 0.5;
            cfgBase.control.obs.replanInterval = 5.0;
            cfgBase.control.obs.jointScoreEnabled = true;
            cfgBase.control.obs.geometryWeight = 0.7;
            cfgBase.control.obs.parameterWeight = 0.3;
        otherwise
            error("Unsupported maneuverMode %s.",maneuverMode);
    end
    maneuverDeltaV = cfgBase.control.obs.acceleration* ...
        cfgBase.control.obs.burnDuration;
    requiredThrust = max(cfgBase.watchers.mass)* ...
        cfgBase.control.obs.acceleration;
    straightLineEquivalentBaseline = ...
        cfgBase.control.obs.acceleration*baselineFactor;

    % Use the same innovation covariance-matching law in both cases.
    % gammaTheta and gammaEpsilon are updated after every valid bearing
    % measurement and are used by the following prediction step. Keep the
    % nominal qEpsilonC0 supplied by config_step03_local_DNN_EKF instead of
    % replacing it by zero.
    cfgBase.dnn.adaptQThetaEnabled = true;
    cfgBase.dnn.adaptQEpsilonEnabled = true;
    cfgBase.dnn.predictionResidualSource = "GS_composite";
    cfgBase.gs.enabled = true;
    cfgBase.gs.uploadMode = uploadMode;
    cfgBase.communication.architecture = communicationArchitecture;
    % The new geometry-information additive case uses a genuine cumulative
    % sum of LOS projectors. This removes the legacy EMA forgetting factor.
    % The additive baseline collects the same passive metadata, so both
    % simulations retain identical non-composite settings.
    cfgBase.gs.fimGate.accumulationMode = "cumulative_sum";

    cfgAdd = cfgBase;
    cfgAdd.step.name = "step09J6_transverse_additive_"+ ...
        string(desiredBaseline)+"m";
    cfgAdd.gs.compositeMode = "additive";

    cfgFIM = cfgBase;
    cfgFIM.step.name = "step09J6_transverse_FIM_weighted_additive_"+ ...
        string(desiredBaseline)+"m";
    cfgFIM.gs.compositeMode = "fim_weighted_additive";

    fprintf("Step 09-J.6 bearing-only additive versus FIM-weighted additive\n");
    fprintf("maneuver mode=%s\n",maneuverMode);
    fprintf("baseline=%.3f m, sigma_b=%.4f deg, T=%.1f s, dt=%.4g s\n", ...
        desiredBaseline,rad2deg(cfgBase.meas.sigmaBearing), ...
        cfgBase.T,cfgBase.dt);
    fprintf("a=%.4e m/s^2, delta-v=%.5f m/s, thrust=%.5f N\n", ...
        cfgBase.control.obs.acceleration,maneuverDeltaV,requiredThrust);
    fprintf("thrust limit=%.5f N, active interval=%.1f-%.1f s\n", ...
        cfgBase.watchers.maxThrust,cfgBase.control.obs.startTime, ...
        cfgBase.control.obs.startTime+cfgBase.control.obs.burnDuration);
    if straightLineEquivalentBaseline < desiredBaseline
        fprintf(['thrust-limited straight-line-equivalent baseline=' ...
            '%.3f m (requested %.3f m)\n'], ...
            straightLineEquivalentBaseline,desiredBaseline);
    end
    fprintf("adaptive Q enabled for theta and epsilon; qEpsilonC0=%.3e; theta0Std=%.3e\n", ...
        cfgBase.dnn.qEpsilonC0,meta.thetaInitStd);

    if residualMode == "fim"
        fprintf("\nRunning FIM-weighted-additive only...\n");
        rng(seed);
        resFIM = simulate_GS_DNN_EKF(cfgFIM);
        controllerSummaryFIM = summarizeObservabilityController(resFIM,cfgFIM);
        assert(all(isfinite(resFIM.xhat(:))) && ...
            all(isfinite(resFIM.dnnResidual(:))), ...
            "A non-finite estimate or DNN residual was produced.");
        mFIM = summarizeCase(resFIM,cfgBase);
        performanceSummary = structSummaryTable(mFIM,"FIM-weighted-additive");
        disp(performanceSummary);
        out = struct('resGSAdd',[],'resGSFIM',resFIM, ...
            'controllerSummaryAdd',[],'controllerSummaryFIM',controllerSummaryFIM, ...
            'cfgGSAdd',[],'cfgGSFIM',cfgFIM,'seed',seed, ...
            'thetaInitStd',meta.thetaInitStd,'desiredBaseline',desiredBaseline, ...
            'maneuverMode',maneuverMode,'simulationTime',simulationTime, ...
            'coastAfterBurn',coastAfterBurn, ...
            'maneuverAcceleration',cfgBase.control.obs.acceleration, ...
            'requestedAcceleration',requestedAcceleration, ...
            'straightLineEquivalentBaseline',straightLineEquivalentBaseline, ...
            'maneuverDeltaV',maneuverDeltaV,'requiredThrust',requiredThrust, ...
            'configuredMaxThrust',cfgBase.watchers.maxThrust, ...
            'performanceSummary',performanceSummary, ...
            'communicationArchitecture',communicationArchitecture, ...
            'uploadMode',uploadMode,'residualMode',residualMode);
        diagOut = [];
        figures = gobjects(0);
        return;
    end

    fprintf("\nRunning additive GS...\n");
    rng(seed);
    resAdd = simulate_GS_DNN_EKF(cfgAdd);
    fprintf("Running FIM-weighted-additive GS...\n");
    rng(seed);
    resFIM = simulate_GS_DNN_EKF(cfgFIM);
    controllerSummaryAdd = summarizeObservabilityController(resAdd,cfgAdd);
    controllerSummaryFIM = summarizeObservabilityController(resFIM,cfgFIM);
    assert(all(isfinite(resAdd.xhat(:))) && ...
        all(isfinite(resFIM.xhat(:))), ...
        "A non-finite kinematic estimate was produced.");
    assert(all(isfinite(resAdd.dnnResidual(:))) && ...
        all(isfinite(resFIM.dnnResidual(:))), ...
        "A non-finite DNN residual was produced.");

    mAdd = summarizeCase(resAdd,cfgBase);
    mFIM = summarizeCase(resFIM,cfgBase);
    caseName = ["additive";"FIM-weighted-additive"];
    positionRMSE = [mAdd.positionRMSE;mFIM.positionRMSE];
    finalPositionRMSE = [mAdd.finalPositionRMSE;mFIM.finalPositionRMSE];
    exactRangeRMSE = [mAdd.exactRangeRMSE;mFIM.exactRangeRMSE];
    finalExactRangeRMSE = [mAdd.finalExactRangeRMSE; ...
        mFIM.finalExactRangeRMSE];
    velocityRMSE = [mAdd.velocityRMSE;mFIM.velocityRMSE];
    finalVelocityRMSE = [mAdd.finalVelocityRMSE;mFIM.finalVelocityRMSE];
    residualVectorRMSE = [mAdd.residualVectorRMSE; ...
        mFIM.residualVectorRMSE];
    finalResidualVectorRMSE = [mAdd.finalResidualVectorRMSE; ...
        mFIM.finalResidualVectorRMSE];
    meanCosine = [mAdd.meanCosine;mFIM.meanCosine];
    finalCosine = [mAdd.finalCosine;mFIM.finalCosine];
    meanNormRatio = [mAdd.meanNormRatio;mFIM.meanNormRatio];
    finalNormRatio = [mAdd.finalNormRatio;mFIM.finalNormRatio];
    meanNIS = [mAdd.meanNIS;mFIM.meanNIS];
    meanThetaUpdateNorm = [mAdd.meanThetaUpdateNorm; ...
        mFIM.meanThetaUpdateNorm];
    meanPthetaEtaFro = [mAdd.meanPthetaEtaFro;mFIM.meanPthetaEtaFro];
    meanGeometryInformationMinEig = [mAdd.meanGeometryInformationMinEig; ...
        mFIM.meanGeometryInformationMinEig];
    meanGeometryInformationCondition = ...
        [mAdd.meanGeometryInformationCondition; ...
        mFIM.meanGeometryInformationCondition];
    positionConvergenceRatio = [mAdd.positionConvergenceRatio; ...
        mFIM.positionConvergenceRatio];
    rangeConvergenceRatio = [mAdd.rangeConvergenceRatio; ...
        mFIM.rangeConvergenceRatio];
    finalPositionLogSlope = [mAdd.finalPositionLogSlope; ...
        mFIM.finalPositionLogSlope];
    finalRangeLogSlope = [mAdd.finalRangeLogSlope; ...
        mFIM.finalRangeLogSlope];

    performanceSummary = table(caseName,positionRMSE, ...
        finalPositionRMSE,exactRangeRMSE,finalExactRangeRMSE, ...
        velocityRMSE,finalVelocityRMSE,residualVectorRMSE, ...
        finalResidualVectorRMSE,meanCosine,finalCosine,meanNormRatio, ...
        finalNormRatio,meanNIS,meanThetaUpdateNorm,meanPthetaEtaFro, ...
        meanGeometryInformationMinEig,meanGeometryInformationCondition, ...
        positionConvergenceRatio,rangeConvergenceRatio, ...
        finalPositionLogSlope,finalRangeLogSlope);
    disp(performanceSummary);

    out = struct('resGSAdd',resAdd,'resGSFIM',resFIM, ...
        'controllerSummaryAdd',controllerSummaryAdd, ...
        'controllerSummaryFIM',controllerSummaryFIM, ...
        'cfgGSAdd',cfgAdd,'cfgGSFIM',cfgFIM,'seed',seed, ...
        'thetaInitStd',meta.thetaInitStd, ...
        'desiredBaseline',desiredBaseline, ...
        'maneuverMode',maneuverMode, ...
        'simulationTime',simulationTime, ...
        'coastAfterBurn',coastAfterBurn, ...
        'maneuverAcceleration',cfgBase.control.obs.acceleration, ...
        'requestedAcceleration',requestedAcceleration, ...
        'straightLineEquivalentBaseline', ...
        straightLineEquivalentBaseline, ...
        'maneuverDeltaV',maneuverDeltaV, ...
        'requiredThrust',requiredThrust, ...
        'configuredMaxThrust',cfgBase.watchers.maxThrust, ...
        'performanceSummary',performanceSummary);

    diagnosticTimes = unique([0 25 50 100 0.5*simulationTime ...
        0.75*simulationTime simulationTime]);
    diagnosticTimes = diagnosticTimes(diagnosticTimes<=simulationTime);
    diagOut = run_step09J6_estimate_norm_cosine_diagnostic( ...
        out,diagnosticTimes,false);
    figures = gobjects(0);
    if makePlots
        figures = makeFigures(resAdd,resFIM,performanceSummary,cfgBase);
    end
end

function summary = structSummaryTable(m,caseName)
%STRUCTSUMMARYTABLE Build the standard one-row performance table.
summary = table(string(caseName),m.positionRMSE,m.finalPositionRMSE, ...
    m.exactRangeRMSE,m.finalExactRangeRMSE,m.velocityRMSE, ...
    m.finalVelocityRMSE,m.residualVectorRMSE,m.finalResidualVectorRMSE, ...
    m.meanCosine,m.finalCosine,m.meanNormRatio,m.finalNormRatio,m.meanNIS, ...
    m.meanThetaUpdateNorm,m.meanPthetaEtaFro,m.meanGeometryInformationMinEig, ...
    m.meanGeometryInformationCondition,m.positionConvergenceRatio, ...
    m.rangeConvergenceRatio,m.finalPositionLogSlope,m.finalRangeLogSlope, ...
    'VariableNames',{'caseName','positionRMSE','finalPositionRMSE', ...
    'exactRangeRMSE','finalExactRangeRMSE','velocityRMSE', ...
    'finalVelocityRMSE','residualVectorRMSE','finalResidualVectorRMSE', ...
    'meanCosine','finalCosine','meanNormRatio','finalNormRatio','meanNIS', ...
    'meanThetaUpdateNorm','meanPthetaEtaFro', ...
    'meanGeometryInformationMinEig','meanGeometryInformationCondition', ...
    'positionConvergenceRatio','rangeConvergenceRatio', ...
    'finalPositionLogSlope','finalRangeLogSlope'});
end

function m = summarizeCase(res,cfg)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    initialIdx = 1:max(1,round(0.1*N));
    finalIdx = max(1,round(0.9*N)):N;
    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    posError = res.xhat(1:dim,:,:)-etaTrue(1:dim,:,:);
    velIdx = dim+(1:dim);
    velError = res.xhat(velIdx,:,:)-etaTrue(velIdx,:,:);
    positionNorm = reshape(sqrt(sum(posError.^2,1)),N,Nw);
    velocityNorm = reshape(sqrt(sum(velError.^2,1)),N,Nw);

    targetPosition = repmat( ...
        reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
    trueRange = reshape(sqrt(sum( ...
        (targetPosition-res.watcherR).^2,1)),N,Nw);
    estimateRange = reshape(sqrt(sum( ...
        (res.xhat(1:dim,:,:)-res.watcherR).^2,1)),N,Nw);
    rangeError = estimateRange-trueRange;

    dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
    dHat = res.dnnResidual;
    residualError = reshape(sqrt(sum((dHat-dTrue).^2,1)),N,Nw);
    trueNorm = reshape(sqrt(sum(dTrue.^2,1)),N,Nw);
    estimateNorm = reshape(sqrt(sum(dHat.^2,1)),N,Nw);
    dotValue = reshape(sum(dHat.*dTrue,1),N,Nw);
    normFloor = max(1e-12,1e-3*median(trueNorm(:),"omitnan"));
    valid = trueNorm>normFloor & estimateNorm>normFloor;
    cosine = NaN(N,Nw);
    cosine(valid) = dotValue(valid)./(trueNorm(valid).*estimateNorm(valid));
    ratio = NaN(N,Nw);
    ratioValid = trueNorm>normFloor;
    ratio(ratioValid) = estimateNorm(ratioValid)./trueNorm(ratioValid);

    m.positionRMSE = rmsAll(positionNorm);
    m.finalPositionRMSE = rmsAll(positionNorm(finalIdx,:));
    m.exactRangeRMSE = rmsAll(rangeError);
    m.finalExactRangeRMSE = rmsAll(rangeError(finalIdx,:));
    m.velocityRMSE = rmsAll(velocityNorm);
    m.finalVelocityRMSE = rmsAll(velocityNorm(finalIdx,:));
    m.residualVectorRMSE = rmsAll(residualError);
    m.finalResidualVectorRMSE = rmsAll(residualError(finalIdx,:));
    m.meanCosine = mean(cosine(:),"omitnan");
    m.finalCosine = mean(cosine(finalIdx,:),"all","omitnan");
    m.meanNormRatio = mean(ratio(:),"omitnan");
    m.finalNormRatio = mean(ratio(finalIdx,:),"all","omitnan");
    nis = res.NIS(isfinite(res.NIS));
    m.meanNIS = mean(nis,"omitnan");
    m.meanThetaUpdateNorm = mean(res.thetaUpdateNorm(:),"omitnan");
    m.meanPthetaEtaFro = mean(res.PthetaEtaFro(:),"omitnan");
    x = res.fimGateOmegaSigmaMinEig;
    m.meanGeometryInformationMinEig = mean(x(isfinite(x)),"omitnan");
    x = res.fimGateOmegaSigmaCond;
    m.meanGeometryInformationCondition = mean(x(isfinite(x)),"omitnan");
    m.positionConvergenceRatio = rmsAll(positionNorm(finalIdx,:))/ ...
        max(rmsAll(positionNorm(initialIdx,:)),realmin);
    m.rangeConvergenceRatio = rmsAll(rangeError(finalIdx,:))/ ...
        max(rmsAll(rangeError(initialIdx,:)),realmin);
    m.finalPositionLogSlope = logSlope(res.time(finalIdx), ...
        sqrt(mean(positionNorm(finalIdx,:).^2,2,"omitnan")));
    m.finalRangeLogSlope = logSlope(res.time(finalIdx), ...
        sqrt(mean(rangeError(finalIdx,:).^2,2,"omitnan")));
end

function slope = logSlope(time,value)
    valid = isfinite(time(:)) & isfinite(value(:)) & value(:)>0;
    if nnz(valid)<2
        slope = NaN;
        return;
    end
    p = polyfit(time(valid),log(value(valid)),1);
    slope = p(1);
end

function figures = makeFigures(add,fim,summary,cfg)
    labels = summary.caseName;
    colors = lines(2);
    f1 = figure('Name','Step 09-J.6 additive versus FIM-weighted-additive summary');
    tiledlayout(2,2,'TileSpacing','compact');
    nexttile;
    bar(categorical(labels),[summary.exactRangeRMSE, ...
        summary.finalExactRangeRMSE]);
    grid on; ylabel('range RMSE [m]'); title('Exact-range estimation');
    legend('whole run','final 10%','Location','best');
    nexttile;
    bar(categorical(labels),[summary.positionRMSE, ...
        summary.finalPositionRMSE]);
    grid on; ylabel('position RMSE [m]'); title('Target position');
    legend('whole run','final 10%','Location','best');
    nexttile;
    bar(categorical(labels),[summary.residualVectorRMSE, ...
        summary.finalResidualVectorRMSE]);
    grid on; ylabel('residual RMSE [m/s^2]'); title('Residual learning');
    legend('whole run','final 10%','Location','best');
    nexttile;
    bar(categorical(labels),[summary.finalCosine,summary.finalNormRatio]);
    hold on; yline(1,'k:','ideal'); grid on;
    ylabel('dimensionless'); title('Final residual agreement');
    legend('cosine','norm ratio','Location','best');

    f2 = figure('Name','Step 09-J.6 additive versus FIM-weighted-additive histories');
    tiledlayout(2,2,'TileSpacing','compact');
    cases = {add,fim};
    for panel = 1:4
        nexttile; hold on; grid on;
        for ic = 1:2
            plot(cases{ic}.time,timeSeries(cases{ic},cfg,panel), ...
                'LineWidth',1.05,'Color',colors(ic,:));
        end
        xline(cfg.control.obs.startTime,'k:','burn start');
        xline(cfg.control.obs.startTime+cfg.control.obs.burnDuration, ...
            'k--','burn end');
        xlabel('time [s]');
        if panel == 1
            title('RMS exact-range error'); ylabel('error [m]');
            legend(labels,'Location','best');
        elseif panel == 2
            title('Mean position error'); ylabel('error [m]');
        elseif panel == 3
            title('Mean velocity error'); ylabel('error [m/s]');
        else
            title('Mean residual-vector error'); ylabel('error [m/s^2]');
        end
    end

    operationalPlots = plot_step09J6_operational_results(struct( ...
        'resGSAdd',add,'resGSFIM',fim,'cfgGSFIM', ...
        struct('gs',struct('compositeMode', ...
        "fim_weighted_additive"))));
    extraFigures = gobjects(0);
    figureFields = {'truthEstimateFigure','errorFigure', ...
        'adaptationFigure','stateAdditiveFigure','stateFIMFigure'};
    for k = 1:numel(figureFields)
        name = figureFields{k};
        if isfield(operationalPlots,name) && ...
                ~isempty(operationalPlots.(name)) && ...
                isgraphics(operationalPlots.(name),'figure')
            extraFigures(end+1,1) = operationalPlots.(name); %#ok<AGROW>
        end
    end
    figures = [f1;f2;extraFigures];
end

function y = timeSeries(res,cfg,panel)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    if panel == 1
        targetPosition = repmat( ...
            reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
        trueRange = reshape(sqrt(sum( ...
            (targetPosition-res.watcherR).^2,1)),N,Nw);
        estimateRange = reshape(sqrt(sum( ...
            (res.xhat(1:dim,:,:)-res.watcherR).^2,1)),N,Nw);
        y = sqrt(mean((estimateRange-trueRange).^2,2,"omitnan"));
    elseif panel == 2
        e = res.xhat(1:dim,:,:)-etaTrue(1:dim,:,:);
        y = mean(reshape(sqrt(sum(e.^2,1)),N,Nw),2,"omitnan");
    elseif panel == 3
        idx = dim+(1:dim);
        e = res.xhat(idx,:,:)-etaTrue(idx,:,:);
        y = mean(reshape(sqrt(sum(e.^2,1)),N,Nw),2,"omitnan");
    else
        dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
        e = reshape(sqrt(sum((res.dnnResidual-dTrue).^2,1)),N,Nw);
        y = mean(e,2,"omitnan");
    end
end

function value = rmsAll(x)
    value = sqrt(mean(x(:).^2,"omitnan"));
end
