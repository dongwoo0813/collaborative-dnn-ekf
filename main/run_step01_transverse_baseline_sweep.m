function sweep = run_step01_transverse_baseline_sweep(makePlots)
%RUN_STEP01_TRANSVERSE_BASELINE_SWEEP Sweep transverse maneuver baseline.
%
% Known-dynamics, physical-state-only, bearing EKFs are used so the effect
% of maneuver magnitude is not mixed with target residual or DNN parameter
% uncertainty. Every watcher uses only its own sequential bearings.
%
% Desired final transverse offsets from the no-burn coast are
%
%     3.375, 10, 20, and 30 m.
%
% The burn starts at 40 s, lasts 50 s, and is followed by 110 s of coast.
% For constant transverse acceleration a,
%
%     B_final = a*(0.5*Tburn^2 + Tburn*Tcoast) = 6750*a.
%
% Usage:
%   sweep01Baseline = run_step01_transverse_baseline_sweep(true);


    if nargin < 1
        makePlots = true;
    end

    addpath(genpath(pwd));
    seed = 101;
    cfgBase = config_step01_physical_EKF();
    cfgBase.T = 200.0;
    cfgBase.dt = 0.1;
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
    cfgBase.truth.useResidual = false;
    cfgBase.ekf.q_acc = 0;

    cfgBase.control.obs.enabled = true;
    cfgBase.control.obs.mode = "transverse";
    cfgBase.control.obs.startTime = 40.0;
    cfgBase.control.obs.burnDuration = 50.0;

    desiredBaseline = [3.375;10;20;30];
    coastAfterBurn = cfgBase.T-(cfgBase.control.obs.startTime+ ...
        cfgBase.control.obs.burnDuration);
    baselineFactor = 0.5*cfgBase.control.obs.burnDuration^2 + ...
        cfgBase.control.obs.burnDuration*coastAfterBurn;
    acceleration = desiredBaseline/baselineFactor;
    deltaV = acceleration*cfgBase.control.obs.burnDuration;
    requiredThrust = acceleration*max(cfgBase.watchers.mass);
    configuredThrustLimit = cfgBase.watchers.maxThrust* ...
        ones(size(desiredBaseline));

    caseNames = "B"+(0:numel(desiredBaseline)-1).'+" "+ ...
        string(desiredBaseline)+" m";
    nCases = numel(caseNames);
    results = cell(nCases,1);
    configs = cell(nCases,1);
    status = repmat("not run",nCases,1);

    fprintf("Step 01 known-dynamics transverse-baseline sweep\n");
    fprintf("independent bearing-only EKFs; no residual and no DNN\n");
    fprintf("sigma_b=%.4f deg, T=%.1f s, dt=%.4g s\n", ...
        rad2deg(cfgBase.meas.sigmaBearing),cfgBase.T,cfgBase.dt);
    fprintf("burn: start=%.1f s, duration=%.1f s; final coast=%.1f s\n", ...
        cfgBase.control.obs.startTime,cfgBase.control.obs.burnDuration, ...
        coastAfterBurn);

    for ic = 1:nCases
        cfg = cfgBase;
        cfg.control.obs.acceleration = acceleration(ic);
        cfg.step.name = "step01_transverse_baseline_" + ...
            string(desiredBaseline(ic)) + "m";
        configs{ic} = cfg;
        fprintf("\n%s: a=%.4e m/s^2, delta-v=%.4f m/s, thrust=%.4f N...\n", ...
            caseNames(ic),acceleration(ic),deltaV(ic),requiredThrust(ic));
        try
            rng(seed);
            res = simulatePhysicalEKF(cfg);
            assert(all(isfinite(res.xhat(:))), ...
                "Non-finite target-state estimate was produced.");
            results{ic} = res;
            status(ic) = "ok";
        catch ME
            status(ic) = "failed: "+string(ME.message);
            warning("%s failed: %s",caseNames(ic),ME.message);
        end
    end

    actualBaseline = NaN(nCases,1);
    positionRMSE = NaN(nCases,1);
    finalPositionRMSE = NaN(nCases,1);
    exactRangeRMSE = NaN(nCases,1);
    finalExactRangeRMSE = NaN(nCases,1);
    velocityRMSE = NaN(nCases,1);
    finalVelocityRMSE = NaN(nCases,1);
    meanNIS = NaN(nCases,1);
    meanLOSChangeSigma = NaN(nCases,1);
    maxLOSChangeSigma = NaN(nCases,1);
    minWatcherGramianLambda = NaN(nCases,1);
    medianWatcherGramianLambda = NaN(nCases,1);
    maxWatcherGramianCondition = NaN(nCases,1);
    rangeRMSEByWatcher = NaN(nCases,cfgBase.Nw);
    finalRangeRMSEByWatcher = NaN(nCases,cfgBase.Nw);
    gramianLambdaByWatcher = NaN(nCases,cfgBase.Nw);
    gramianConditionByWatcher = NaN(nCases,cfgBase.Nw);

    % Counterfactual zero-burn trajectory is deterministic and does not
    % require another EKF run.
    cfgReference = cfgBase;
    cfgReference.control.obs.enabled = false;
    cfgReference.control.obs.mode = "none";
    referenceWatcherR = zeros(cfgBase.dim,cfgBase.N,cfgBase.Nw);
    for i = 1:cfgBase.Nw
        for k = 1:cfgBase.N
            ws = watcherTrajectory(i,cfgBase.time(k),cfgReference);
            referenceWatcherR(:,k,i) = ws.r;
        end
    end

    for ic = 1:nCases
        if isempty(results{ic}), continue; end
        m = summarizeCase(results{ic},referenceWatcherR,configs{ic});
        actualBaseline(ic) = m.actualBaseline;
        positionRMSE(ic) = m.positionRMSE;
        finalPositionRMSE(ic) = m.finalPositionRMSE;
        exactRangeRMSE(ic) = m.exactRangeRMSE;
        finalExactRangeRMSE(ic) = m.finalExactRangeRMSE;
        velocityRMSE(ic) = m.velocityRMSE;
        finalVelocityRMSE(ic) = m.finalVelocityRMSE;
        meanNIS(ic) = m.meanNIS;
        meanLOSChangeSigma(ic) = m.meanLOSChangeSigma;
        maxLOSChangeSigma(ic) = m.maxLOSChangeSigma;
        rangeRMSEByWatcher(ic,:) = m.rangeRMSEByWatcher;
        finalRangeRMSEByWatcher(ic,:) = m.finalRangeRMSEByWatcher;
        gramianLambdaByWatcher(ic,:) = m.gramianLambda;
        gramianConditionByWatcher(ic,:) = m.gramianCondition;
        minWatcherGramianLambda(ic) = min(m.gramianLambda);
        medianWatcherGramianLambda(ic) = median(m.gramianLambda);
        maxWatcherGramianCondition(ic) = max(m.gramianCondition);
    end

    thrustLimitRatio = requiredThrust./configuredThrustLimit;
    summary = table(caseNames,status,desiredBaseline,actualBaseline, ...
        acceleration,deltaV,requiredThrust,configuredThrustLimit, ...
        thrustLimitRatio,positionRMSE,finalPositionRMSE,exactRangeRMSE, ...
        finalExactRangeRMSE,velocityRMSE,finalVelocityRMSE,meanNIS, ...
        meanLOSChangeSigma,maxLOSChangeSigma,minWatcherGramianLambda, ...
        medianWatcherGramianLambda,maxWatcherGramianCondition, ...
        'VariableNames',{'caseName','status','desiredBaseline', ...
        'actualBaseline','acceleration','deltaV','requiredThrust', ...
        'configuredThrustLimit','thrustLimitRatio','positionRMSE', ...
        'finalPositionRMSE','exactRangeRMSE','finalExactRangeRMSE', ...
        'velocityRMSE','finalVelocityRMSE','meanNIS', ...
        'meanLOSChangeSigma','maxLOSChangeSigma', ...
        'minWatcherGramianLambda','medianWatcherGramianLambda', ...
        'maxWatcherGramianCondition'});
    disp(summary);

    caseColumn = repelem(caseNames,cfgBase.Nw,1);
    watcherColumn = repmat((1:cfgBase.Nw).',nCases,1);
    watcherSummary = table(caseColumn,watcherColumn, ...
        reshape(rangeRMSEByWatcher.',[],1), ...
        reshape(finalRangeRMSEByWatcher.',[],1), ...
        reshape(gramianLambdaByWatcher.',[],1), ...
        reshape(gramianConditionByWatcher.',[],1), ...
        'VariableNames',{'caseName','watcher','rangeRMSE', ...
        'finalRangeRMSE','priorWhitenedGramianLambdaMin', ...
        'priorWhitenedGramianCondition'});
    disp(watcherSummary);

    figures = gobjects(0);
    if makePlots && all(~cellfun(@isempty,results))
        figures = makeFigures(results,summary,rangeRMSEByWatcher, ...
            finalRangeRMSEByWatcher,gramianLambdaByWatcher, ...
            gramianConditionByWatcher,configs);
    end

    sweep = struct;
    sweep.summary = summary;
    sweep.watcherSummary = watcherSummary;
    sweep.results = results;
    sweep.configs = configs;
    sweep.seed = seed;
    sweep.figures = figures;
end

function m = summarizeCase(res,referenceWatcherR,cfg)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    finalIdx = max(1,round(0.9*N)):N;
    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    posError = res.xhat(1:dim,:,:)-etaTrue(1:dim,:,:);
    velIdx = dim+(1:dim);
    velError = res.xhat(velIdx,:,:)-etaTrue(velIdx,:,:);
    positionNorm = reshape(sqrt(sum(posError.^2,1)),N,Nw);
    velocityNorm = reshape(sqrt(sum(velError.^2,1)),N,Nw);

    targetPosition = repmat( ...
        reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
    trueRelative = targetPosition-res.watcherR;
    trueRange = reshape(sqrt(sum(trueRelative.^2,1)),N,Nw);
    estimateRange = reshape(sqrt(sum( ...
        (res.xhat(1:dim,:,:)-res.watcherR).^2,1)),N,Nw);
    rangeError = estimateRange-trueRange;

    referenceRelative = targetPosition-referenceWatcherR;
    unitNow = trueRelative./max(sqrt(sum(trueRelative.^2,1)),realmin);
    unitReference = referenceRelative./ ...
        max(sqrt(sum(referenceRelative.^2,1)),realmin);
    losDot = reshape(sum(unitNow.*unitReference,1),N,Nw);
    losChangeSigma = acos(max(-1,min(1,losDot)))/cfg.meas.sigmaBearing;
    displacement = reshape(sqrt(sum( ...
        (res.watcherR-referenceWatcherR).^2,1)),N,Nw);
    postBurn = res.time >= cfg.control.obs.startTime+ ...
        cfg.control.obs.burnDuration;

    nis = res.NIS(isfinite(res.NIS));
    m.actualBaseline = max(displacement(:));
    m.positionRMSE = rmsAll(positionNorm);
    m.finalPositionRMSE = rmsAll(positionNorm(finalIdx,:));
    m.exactRangeRMSE = rmsAll(rangeError);
    m.finalExactRangeRMSE = rmsAll(rangeError(finalIdx,:));
    m.velocityRMSE = rmsAll(velocityNorm);
    m.finalVelocityRMSE = rmsAll(velocityNorm(finalIdx,:));
    m.meanNIS = mean(nis,"omitnan");
    m.meanLOSChangeSigma = mean( ...
        losChangeSigma(postBurn,:),"all","omitnan");
    m.maxLOSChangeSigma = max(losChangeSigma(:));
    m.rangeRMSEByWatcher = sqrt(mean(rangeError.^2,1,"omitnan"));
    m.finalRangeRMSEByWatcher = sqrt(mean( ...
        rangeError(finalIdx,:).^2,1,"omitnan"));
    [m.gramianLambda,m.gramianCondition] = ...
        nominalObservabilityGramian(res,cfg);
end

function [lambdaMin,conditionNumber] = nominalObservabilityGramian(res,cfg)
    dim = cfg.dim;
    priorScale = diag([sqrt(cfg.ekf.P0_pos)*ones(dim,1); ...
        sqrt(cfg.ekf.P0_vel)*ones(dim,1)]);
    lambdaMin = zeros(1,cfg.Nw);
    conditionNumber = zeros(1,cfg.Nw);
    for i = 1:cfg.Nw
        W = zeros(2*dim);
        for k = 1:numel(res.time)
            dt0 = res.time(k)-res.time(1);
            Phi = [eye(dim),dt0*eye(dim);zeros(dim),eye(dim)];
            watcherState.r = res.watcherR(:,k,i);
            H = measurementJacobian(res.etaTrue(:,k),watcherState,cfg);
            W = W+Phi'*(H'*(1/cfg.meas.R)*H)*Phi;
        end
        J = priorScale*W*priorScale;
        e = sort(max(real(eig(0.5*(J+J'))),0));
        lambdaMin(i) = e(1);
        if e(1) <= max(e(end),1)*1e-14
            conditionNumber(i) = Inf;
        else
            conditionNumber(i) = e(end)/e(1);
        end
    end
end

function figures = makeFigures(results,summary,rangeByWatcher, ...
        finalRangeByWatcher,lambdaByWatcher,conditionByWatcher,configs)
    labels = summary.caseName;
    colors = lines(height(summary));

    f1 = figure('Name','Step 01 transverse-baseline sweep summary');
    tiledlayout(2,2,'TileSpacing','compact');
    nexttile;
    plot(summary.actualBaseline,summary.finalExactRangeRMSE,'o-', ...
        'LineWidth',1.3); grid on;
    xlabel('final transverse baseline [m]');
    ylabel('final range RMSE [m]'); title('Range versus baseline');
    nexttile;
    plot(summary.actualBaseline,summary.finalPositionRMSE,'o-', ...
        'LineWidth',1.3); grid on;
    xlabel('final transverse baseline [m]');
    ylabel('final position RMSE [m]'); title('Position versus baseline');
    nexttile;
    semilogy(summary.actualBaseline,summary.minWatcherGramianLambda, ...
        'o-','LineWidth',1.3); grid on;
    xlabel('final transverse baseline [m]');
    ylabel('minimum watcher \lambda_{min}');
    title('Worst-watcher observability strength');
    nexttile;
    semilogy(summary.actualBaseline,summary.maxWatcherGramianCondition, ...
        'o-','LineWidth',1.3); grid on;
    xlabel('final transverse baseline [m]');
    ylabel('maximum condition number');
    title('Worst-watcher conditioning');

    f2 = figure('Name','Step 01 transverse-baseline watcher metrics');
    tiledlayout(2,2,'TileSpacing','compact');
    nexttile;
    bar(categorical(labels),rangeByWatcher); grid on;
    ylabel('range RMSE [m]'); title('Whole-run watcher range');
    legend('W1','W2','W3','W4','Location','best');
    nexttile;
    bar(categorical(labels),finalRangeByWatcher); grid on;
    ylabel('final range RMSE [m]'); title('Final watcher range');
    legend('W1','W2','W3','W4','Location','best');
    nexttile;
    bar(categorical(labels),max(lambdaByWatcher,realmin));
    set(gca,'YScale','log'); grid on;
    ylabel('prior-whitened \lambda_{min}');
    title('Watcher observability strength');
    legend('W1','W2','W3','W4','Location','best');
    nexttile;
    finiteCondition = conditionByWatcher;
    finiteCondition(~isfinite(finiteCondition)) = 1e18;
    bar(categorical(labels),finiteCondition);
    set(gca,'YScale','log'); grid on;
    ylabel('condition number'); title('Watcher conditioning');
    legend('W1','W2','W3','W4','Location','best');

    f3 = figure('Name','Step 01 transverse-baseline histories');
    tiledlayout(2,2,'TileSpacing','compact');
    for panel = 1:3
        nexttile; hold on; grid on;
        for ic = 1:numel(results)
            plot(results{ic}.time,timeSeries(results{ic},panel), ...
                'LineWidth',1.0,'Color',colors(ic,:));
        end
        xline(configs{1}.control.obs.startTime,'k:','burn start');
        xline(configs{1}.control.obs.startTime+ ...
            configs{1}.control.obs.burnDuration,'k--','burn end');
        xlabel('time [s]');
        if panel == 1
            ylabel('range error [m]'); title('RMS exact-range error');
            legend(labels,'Location','best');
        elseif panel == 2
            ylabel('position error [m]'); title('Mean position error');
        else
            ylabel('velocity error [m/s]'); title('Mean velocity error');
        end
    end
    nexttile;
    yyaxis left;
    plot(summary.actualBaseline,summary.requiredThrust,'o-', ...
        'LineWidth',1.2); hold on;
    yline(summary.configuredThrustLimit(1),'--','configured limit');
    ylabel('required thrust [N]');
    yyaxis right;
    plot(summary.actualBaseline,summary.deltaV,'s-','LineWidth',1.2);
    ylabel('\Delta v [m/s]'); grid on;
    xlabel('final transverse baseline [m]'); title('Maneuver cost');
    figures = [f1;f2;f3];
end

function y = timeSeries(res,panel)
    dim = size(res.etaTrue,1)/2;
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
    else
        idx = dim+(1:dim);
        e = res.xhat(idx,:,:)-etaTrue(idx,:,:);
        y = mean(reshape(sqrt(sum(e.^2,1)),N,Nw),2,"omitnan");
    end
end

function value = rmsAll(x)
    value = sqrt(mean(x(:).^2,"omitnan"));
end
