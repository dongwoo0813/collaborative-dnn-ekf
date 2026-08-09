function sweep = run_step09J6_transverse_residual_model_ablation( ...
    makePlots, desiredBaseline)
%RUN_STEP09J6_TRANSVERSE_RESIDUAL_MODEL_ABLATION Isolate residual modelling.
%
% The watcher geometry, E2 LOS-transverse burn, initial physical-state
% errors, random seed, and bearing-noise standard-normal draws are paired.
% Only the target residual and estimator treatment of it change:
%
%   R0 known dynamics       no truth residual; physical EKF
%   R1 ignored residual     truth residual active; physical EKF predicts 0
%   R2 oracle residual      truth residual active and known in prediction
%   R3 DNN residual         truth residual active and learned by GS DNN-EKF
%
% R0/R1 use the physical-state-only baseline EKF. R2 uses the augmented
% local simulator with oracle prediction, while R3 adds GS composition.
% Both use the same augmented DNN-EKF prediction/update core. In R2 theta
% is dynamically decoupled because the oracle residual has no theta
% derivative; avoiding the GS loop also prevents unnecessary DNN work.
%
% Usage:
%   sweep09j6Residual = ...
%       run_step09J6_transverse_residual_model_ablation(true);
%   sweep09j6Residual20m = ...
%       run_step09J6_transverse_residual_model_ablation(true,20);


    if nargin < 1
        makePlots = true;
    end
    if nargin < 2
        desiredBaseline = 3.375;
    end
    validateattributes(desiredBaseline,{'numeric'}, ...
        {'scalar','real','finite','positive'},mfilename,'desiredBaseline');

    addpath(genpath(pwd));
    [cfgCommon,seed,meta] = config_step09J6_seed101_operational();

    cfgCommon.scenario.watcherModel = "matched_velocity_coast";
    cfgCommon.watchers.motionMode = "prescribed";
    cfgCommon.watchers.coastVelocity = cfgCommon.target.v0;
    cfgCommon.control.translationMode = "none";
    cfgCommon.meas.type = "bearing";
    cfgCommon.meas.sigmaBearing = deg2rad(0.01);
    cfgCommon.meas.R = cfgCommon.meas.sigmaBearing^2;
    cfgCommon.meas.availabilityMode = "always";
    cfgCommon.fov.enabled = false;

    cfgCommon.control.obs.enabled = true;
    cfgCommon.control.obs.mode = "transverse";
    cfgCommon.control.obs.startTime = 40.0;
    cfgCommon.control.obs.burnDuration = 50.0;
    coastAfterBurn = cfgCommon.T-(cfgCommon.control.obs.startTime+ ...
        cfgCommon.control.obs.burnDuration);
    if coastAfterBurn < 0
        error("Burn must end before cfg.T.");
    end
    baselineFactor = 0.5*cfgCommon.control.obs.burnDuration^2+ ...
        cfgCommon.control.obs.burnDuration*coastAfterBurn;
    cfgCommon.control.obs.acceleration = desiredBaseline/baselineFactor;
    maneuverDeltaV = cfgCommon.control.obs.acceleration* ...
        cfgCommon.control.obs.burnDuration;
    requiredThrust = max(cfgCommon.watchers.mass)* ...
        cfgCommon.control.obs.acceleration;

    % Freeze covariance matching so it cannot become an additional varying
    % factor. Approximation-error acceleration noise is also removed in this
    % diagnostic; R3 must explain the residual through theta mobility.
    cfgCommon.dnn.adaptQThetaEnabled = false;
    cfgCommon.dnn.adaptQEpsilonEnabled = false;
    cfgCommon.dnn.qEpsilonC0 = 0;

    caseNames = [
        "R0 known dynamics"
        "R1 ignored residual"
        "R2 oracle residual"
        "R3 DNN residual"];
    residualTreatment = ["none";"ignored";"oracle";"GS_composite"];
    nCases = numel(caseNames);
    results = cell(nCases,1);
    configs = cell(nCases,1);
    status = repmat("not run",nCases,1);

    fprintf("Step 09-J.6 transverse residual-model ablation\n");
    fprintf("E2 LOS-transverse coast maneuver in every case\n");
    fprintf("sigma_b=%.4f deg, T=%.1f s, dt=%.4g s\n", ...
        rad2deg(cfgCommon.meas.sigmaBearing),cfgCommon.T,cfgCommon.dt);
    fprintf(['burn: start=%.1f s, duration=%.1f s, a=%.3e m/s^2; ' ...
        'target baseline=%.3f m\n'], ...
        cfgCommon.control.obs.startTime,cfgCommon.control.obs.burnDuration, ...
        cfgCommon.control.obs.acceleration,desiredBaseline);
    fprintf("maneuver delta-v=%.5f m/s; required 20-kg thrust=%.5f N\n", ...
        maneuverDeltaV,requiredThrust);
    fprintf("adaptive Q matching frozen; qEpsilonC0=0; theta0Std=%.3e\n", ...
        meta.thetaInitStd);

    for ic = 1:nCases
        cfg = cfgCommon;
        cfg.step.name = "step09J6_transverse_residual_" + ...
            residualTreatment(ic)+"_"+string(desiredBaseline)+"m";

        if ic <= 2
            cfg.estimator.type = "physical_EKF";
            cfg.ekf.q_acc = 0;
            cfg.truth.useResidual = (ic == 2);
        elseif ic == 3
            cfg.truth.useResidual = true;
            cfg.estimator.type = "oracle_residual_EKF";
            cfg.dnn.predictionResidualSource = "oracle";
            cfg.dnn.residualInjectionGain = 1.0;
        else
            cfg.truth.useResidual = true;
            cfg.estimator.type = "GS_DNN_EKF";
            cfg.dnn.predictionResidualSource = "GS_composite";
            cfg.dnn.residualInjectionGain = 1.0;
            cfg.gs.enabled = true;
            cfg.gs.compositeMode = "additive";
        end
        configs{ic} = cfg;

        fprintf("\n%s (%s)...\n",caseNames(ic),residualTreatment(ic));
        try
            rng(seed);
            if ic <= 2
                res = simulatePhysicalEKF(cfg);
            elseif ic == 3
                res = simulateLocalDNNEKF(cfg);
            else
                res = simulate_GS_DNN_EKF(cfg);
            end
            assert(all(isfinite(res.xhat(:))), ...
                "Non-finite target-state estimate was produced.");
            results{ic} = res;
            status(ic) = "ok";
        catch ME
            status(ic) = "failed: " + string(ME.message);
            warning("%s failed: %s",caseNames(ic),ME.message);
        end
    end

    positionRMSE = NaN(nCases,1);
    finalPositionRMSE = NaN(nCases,1);
    exactRangeRMSE = NaN(nCases,1);
    finalExactRangeRMSE = NaN(nCases,1);
    velocityRMSE = NaN(nCases,1);
    finalVelocityRMSE = NaN(nCases,1);
    residualVectorRMSE = NaN(nCases,1);
    finalResidualVectorRMSE = NaN(nCases,1);
    meanNIS = NaN(nCases,1);
    meanDiagonalNormalizedError = NaN(nCases,1);
    meanThetaUpdateNorm = NaN(nCases,1);
    meanPthetaEtaFro = NaN(nCases,1);
    minWatcherGramianLambda = NaN(nCases,1);
    medianWatcherGramianLambda = NaN(nCases,1);
    maxWatcherGramianCondition = NaN(nCases,1);
    gramianLambdaByWatcher = NaN(nCases,cfgCommon.Nw);
    gramianConditionByWatcher = NaN(nCases,cfgCommon.Nw);

    for ic = 1:nCases
        if isempty(results{ic}), continue; end
        m = summarizeResidualCase(results{ic},configs{ic}, ...
            residualTreatment(ic));
        positionRMSE(ic) = m.positionRMSE;
        finalPositionRMSE(ic) = m.finalPositionRMSE;
        exactRangeRMSE(ic) = m.exactRangeRMSE;
        finalExactRangeRMSE(ic) = m.finalExactRangeRMSE;
        velocityRMSE(ic) = m.velocityRMSE;
        finalVelocityRMSE(ic) = m.finalVelocityRMSE;
        residualVectorRMSE(ic) = m.residualVectorRMSE;
        finalResidualVectorRMSE(ic) = m.finalResidualVectorRMSE;
        meanNIS(ic) = m.meanNIS;
        meanDiagonalNormalizedError(ic) = ...
            m.meanDiagonalNormalizedError;
        meanThetaUpdateNorm(ic) = m.meanThetaUpdateNorm;
        meanPthetaEtaFro(ic) = m.meanPthetaEtaFro;
        gramianLambdaByWatcher(ic,:) = m.gramianLambda;
        gramianConditionByWatcher(ic,:) = m.gramianCondition;
        minWatcherGramianLambda(ic) = min(m.gramianLambda);
        medianWatcherGramianLambda(ic) = median(m.gramianLambda);
        maxWatcherGramianCondition(ic) = max(m.gramianCondition);
    end

    summary = table(caseNames,residualTreatment,status,positionRMSE, ...
        finalPositionRMSE,exactRangeRMSE,finalExactRangeRMSE, ...
        velocityRMSE,finalVelocityRMSE,residualVectorRMSE, ...
        finalResidualVectorRMSE,meanNIS,meanDiagonalNormalizedError, ...
        meanThetaUpdateNorm,meanPthetaEtaFro,minWatcherGramianLambda, ...
        medianWatcherGramianLambda,maxWatcherGramianCondition, ...
        'VariableNames',{'caseName','residualTreatment','status', ...
        'positionRMSE','finalPositionRMSE','exactRangeRMSE', ...
        'finalExactRangeRMSE','velocityRMSE','finalVelocityRMSE', ...
        'residualVectorRMSE','finalResidualVectorRMSE','meanNIS', ...
        'meanDiagonalNormalizedError','meanThetaUpdateNorm', ...
        'meanPthetaEtaFro','minWatcherGramianLambda', ...
        'medianWatcherGramianLambda','maxWatcherGramianCondition'});
    disp(summary);

    caseColumn = repelem(caseNames,cfgCommon.Nw,1);
    watcherColumn = repmat((1:cfgCommon.Nw).',nCases,1);
    gramianSummary = table(caseColumn,watcherColumn, ...
        reshape(gramianLambdaByWatcher.',[],1), ...
        reshape(gramianConditionByWatcher.',[],1), ...
        'VariableNames',{'caseName','watcher', ...
        'priorWhitenedGramianLambdaMin','priorWhitenedGramianCondition'});
    disp(gramianSummary);

    figures = gobjects(0);
    if makePlots && all(~cellfun(@isempty,results))
        figures = plotResidualAblation(results,summary, ...
            gramianLambdaByWatcher,gramianConditionByWatcher, ...
            configs,residualTreatment);
    end

    sweep = struct;
    sweep.summary = summary;
    sweep.gramianSummary = gramianSummary;
    sweep.results = results;
    sweep.configs = configs;
    sweep.seed = seed;
    sweep.maneuver = struct('desiredBaseline',desiredBaseline, ...
        'acceleration',cfgCommon.control.obs.acceleration, ...
        'deltaV',maneuverDeltaV,'requiredThrust',requiredThrust, ...
        'configuredThrustLimit',cfgCommon.watchers.maxThrust);
    sweep.figures = figures;
end

function m = summarizeResidualCase(res,cfg,treatment)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    finalIdx = max(1,round(0.9*N)):N;
    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    errorState = res.xhat(1:2*dim,:,:)-etaTrue;
    posError = errorState(1:dim,:,:);
    velError = errorState(dim+(1:dim),:,:);
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
    dHat = operationalResidualEstimate(res,cfg,treatment);
    residualError = reshape(sqrt(sum((dHat-dTrue).^2,1)),N,Nw);

    if isfield(res,"PdiagEta")
        PdiagEta = res.PdiagEta;
    else
        PdiagEta = res.Pdiag(1:2*dim,:,:);
    end
    normalizedDiagonalError = reshape(sum( ...
        errorState.^2./max(PdiagEta,realmin),1),N,Nw);

    nis = res.NIS(isfinite(res.NIS));
    m.positionRMSE = rmsAll(positionNorm);
    m.finalPositionRMSE = rmsAll(positionNorm(finalIdx,:));
    m.exactRangeRMSE = rmsAll(rangeError);
    m.finalExactRangeRMSE = rmsAll(rangeError(finalIdx,:));
    m.velocityRMSE = rmsAll(velocityNorm);
    m.finalVelocityRMSE = rmsAll(velocityNorm(finalIdx,:));
    m.residualVectorRMSE = rmsAll(residualError);
    m.finalResidualVectorRMSE = rmsAll(residualError(finalIdx,:));
    m.meanNIS = mean(nis,"omitnan");
    m.meanDiagonalNormalizedError = mean( ...
        normalizedDiagonalError(:),"omitnan");
    if isfield(res,"thetaUpdateNorm")
        m.meanThetaUpdateNorm = mean(res.thetaUpdateNorm(:),"omitnan");
    else
        m.meanThetaUpdateNorm = NaN;
    end
    if isfield(res,"PthetaEtaFro")
        m.meanPthetaEtaFro = mean(res.PthetaEtaFro(:),"omitnan");
    else
        m.meanPthetaEtaFro = NaN;
    end
    [m.gramianLambda,m.gramianCondition] = ...
        nominalObservabilityGramian(res,cfg);
end

function dHat = operationalResidualEstimate(res,cfg,treatment)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    if treatment == "none" || treatment == "ignored"
        dHat = zeros(dim,N,Nw);
    elseif treatment == "oracle"
        dHat = zeros(dim,N,Nw);
        for i = 1:Nw
            for k = 1:N
                dHat(:,k,i) = trueResidual( ...
                    res.time(k),res.xhat(1:2*dim,k,i),cfg);
            end
        end
    else
        dHat = res.dnnResidual;
    end
end

function [lambdaMin,conditionNumber] = nominalObservabilityGramian(res,cfg)
% Prior-whitened nominal DI Gramian. This is a geometry/conditioning
% diagnostic, not a proof for the nonlinear DNN-augmented system.
    dim = cfg.dim;
    Nw = cfg.Nw;
    Rinv = 1/cfg.meas.R;
    priorScale = diag([sqrt(cfg.ekf.P0_pos)*ones(dim,1); ...
        sqrt(cfg.ekf.P0_vel)*ones(dim,1)]);
    lambdaMin = zeros(1,Nw);
    conditionNumber = zeros(1,Nw);
    for i = 1:Nw
        W = zeros(2*dim);
        for k = 1:numel(res.time)
            dt0 = res.time(k)-res.time(1);
            Phi = [eye(dim),dt0*eye(dim);zeros(dim),eye(dim)];
            watcherState.r = res.watcherR(:,k,i);
            H = measurementJacobian(res.etaTrue(:,k),watcherState,cfg);
            W = W + Phi'*(H'*Rinv*H)*Phi;
        end
        J = priorScale*W*priorScale;
        J = 0.5*(J+J');
        e = sort(max(real(eig(J)),0));
        lambdaMin(i) = e(1);
        if e(1) <= max(e(end),1)*1e-14
            conditionNumber(i) = Inf;
        else
            conditionNumber(i) = e(end)/e(1);
        end
    end
end

function figures = plotResidualAblation(results,summary,lambdaByWatcher, ...
        conditionByWatcher,configs,treatments)
    labels = summary.caseName;
    colors = lines(height(summary));

    f1 = figure('Name','Step 09-J.6 transverse residual-model summary');
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
    grid on; ylabel('residual RMSE [m/s^2]'); title('Residual modelling');
    legend('whole run','final 10%','Location','best');
    nexttile;
    bar(categorical(labels),summary.meanNIS);
    hold on; yline(1,'k:','expected mean'); grid on;
    ylabel('mean NIS'); title('Bearing innovation consistency');

    f2 = figure('Name','Step 09-J.6 transverse residual-model histories');
    tiledlayout(2,2,'TileSpacing','compact');
    for panel = 1:4
        nexttile; hold on; grid on;
        for ic = 1:numel(results)
            y = residualAblationSeries(results{ic},configs{ic}, ...
                treatments(ic),panel);
            plot(results{ic}.time,y,'LineWidth',1.05, ...
                'Color',colors(ic,:));
        end
        xline(configs{1}.control.obs.startTime,'k:','burn start');
        xline(configs{1}.control.obs.startTime+ ...
            configs{1}.control.obs.burnDuration,'k--','burn end');
        xlabel('time [s]');
        if panel == 1
            title('Mean position error'); ylabel('error [m]');
            legend(labels,'Location','best');
        elseif panel == 2
            title('RMS exact-range error'); ylabel('error [m]');
        elseif panel == 3
            title('Mean velocity error'); ylabel('error [m/s]');
        else
            title('Mean residual-vector error');
            ylabel('error [m/s^2]');
        end
    end

    f3 = figure('Name','Step 09-J.6 nominal observability Gramian');
    tiledlayout(1,2,'TileSpacing','compact');
    nexttile;
    bar(categorical(labels),max(lambdaByWatcher,realmin));
    set(gca,'YScale','log'); grid on;
    ylabel('prior-whitened \lambda_{min}');
    title('Nominal observability strength');
    legend('W1','W2','W3','W4','Location','best');
    nexttile;
    finiteCondition = conditionByWatcher;
    finiteCondition(~isfinite(finiteCondition)) = 1e18;
    bar(categorical(labels),finiteCondition);
    set(gca,'YScale','log'); grid on;
    ylabel('condition number'); title('Nominal Gramian conditioning');
    legend('W1','W2','W3','W4','Location','best');
    figures = [f1;f2;f3];
end

function y = residualAblationSeries(res,cfg,treatment,panel)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    if panel == 1
        e = res.xhat(1:dim,:,:)-etaTrue(1:dim,:,:);
        y = mean(reshape(sqrt(sum(e.^2,1)),N,Nw),2,"omitnan");
    elseif panel == 2
        targetPosition = repmat( ...
            reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
        rhoTrue = reshape(sqrt(sum( ...
            (targetPosition-res.watcherR).^2,1)),N,Nw);
        rhoHat = reshape(sqrt(sum( ...
            (res.xhat(1:dim,:,:)-res.watcherR).^2,1)),N,Nw);
        y = sqrt(mean((rhoHat-rhoTrue).^2,2,"omitnan"));
    elseif panel == 3
        idx = dim+(1:dim);
        e = res.xhat(idx,:,:)-etaTrue(idx,:,:);
        y = mean(reshape(sqrt(sum(e.^2,1)),N,Nw),2,"omitnan");
    else
        dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
        dHat = operationalResidualEstimate(res,cfg,treatment);
        e = reshape(sqrt(sum((dHat-dTrue).^2,1)),N,Nw);
        y = mean(e,2,"omitnan");
    end
end

function value = rmsAll(x)
    value = sqrt(mean(x(:).^2,"omitnan"));
end
