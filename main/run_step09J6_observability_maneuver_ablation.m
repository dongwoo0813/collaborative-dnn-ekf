function sweep = run_step09J6_observability_maneuver_ablation( ...
    makePlots, referenceMode)
%RUN_STEP09J6_OBSERVABILITY_MANEUVER_ABLATION Bearing-only maneuver study.
%
% Four paired cases use the same target truth, random seed, initial DNN
% parameters, covariance model, measurement-noise standard-normal draws,
% and additive GS composition:
%
%   M0 current circular       - existing prescribed circular watchers
%   M1 LOS-parallel           - circle + calibrated initial-LOS burn
%   M2 LOS-transverse         - circle + calibrated transverse burn
%   M3 alternating transverse - M2 with alternating watcher signs
%
% The maneuver is analytically superposed on watcherTrajectory, so M0 is
% exactly the former prescribed trajectory. Every available bearing updates
% eta and theta. No range measurement is used.
%
% Usage:
%   sweep09j6Obs = run_step09J6_observability_maneuver_ablation(true);
%   sweep09j6Coast = run_step09J6_observability_maneuver_ablation( ...
%       true, "coast");

% Important outputs:
%   sweep.summary  - metrics for all four cases
%   sweep.results  - complete simulation results
%   sweep.configs  - exact configurations
%   sweep.figures  - generated figure handles


    if nargin < 1
        makePlots = true;
    end
    if nargin < 2
        referenceMode = "circular";
    end
    referenceMode = string(referenceMode);

    addpath(genpath(pwd));
    [cfgBase, seed, meta] = config_step09J6_seed101_operational();

    cfgBase.meas.type = "bearing";
    cfgBase.meas.sigmaBearing = deg2rad(0.01);
    if cfgBase.dim == 2
        cfgBase.meas.R = cfgBase.meas.sigmaBearing^2;
    elseif cfgBase.dim == 3
        cfgBase.meas.R = cfgBase.meas.sigmaBearing^2 * eye(2);
    else
        error("Unsupported cfg.dim=%d.", cfgBase.dim);
    end

    cfgBase.meas.availabilityMode = "always";
    cfgBase.watchers.motionMode = "prescribed";
    cfgBase.control.translationMode = "none";
    cfgBase.gs.compositeMode = "additive";
    cfgBase.dnn.predictionResidualSource = "GS_composite";

    switch referenceMode
        case "circular"
            cfgBase.scenario.watcherModel = "prescribed_orbit";
            caseNames = [
                "M0 current circular"
                "M1 LOS-parallel"
                "M2 LOS-transverse"
                "M3 alternating transverse"];

        case "coast"
            cfgBase.scenario.watcherModel = "matched_velocity_coast";
            cfgBase.watchers.coastVelocity = cfgBase.target.v0;
            caseNames = [
                "N0 matched-velocity coast"
                "N1 coast + LOS-parallel"
                "N2 coast + LOS-transverse"
                "N3 coast + alternating transverse"];

        otherwise
            error("referenceMode must be 'circular' or 'coast'.");
    end

    % A 50-s, 5e-4 m/s^2 burn produces 0.625 m displacement at burn end
    % and Delta-v=0.025 m/s. At 1 km, 0.625 m is about 3.6 sigma for a
    % 0.01-deg bearing sensor before accounting for the moving geometry.
    cfgBase.control.obs.enabled = false;
    cfgBase.control.obs.mode = "none";
    cfgBase.control.obs.startTime = 40.0;
    cfgBase.control.obs.burnDuration = 50.0;
    cfgBase.control.obs.acceleration = 5e-4;

    maneuverModes = ["none"; "parallel"; "transverse"; ...
        "transverse_alternating"];
    nCases = numel(caseNames);

    results = cell(nCases,1);
    configs = cell(nCases,1);
    status = repmat("not run", nCases, 1);

    fprintf("Step 09-J.6 bearing-only observability-maneuver ablation\n");
    fprintf("reference motion: %s\n", referenceMode);
    fprintf("sigma_b=%.4f deg, T=%.1f s, dt=%.4g s\n", ...
        rad2deg(cfgBase.meas.sigmaBearing), cfgBase.T, cfgBase.dt);
    fprintf("burn: start=%.1f s, duration=%.1f s, a=%.3e m/s^2\n", ...
        cfgBase.control.obs.startTime, ...
        cfgBase.control.obs.burnDuration, ...
        cfgBase.control.obs.acceleration);
    fprintf("theta0Std=%.3e; additive GS; every bearing updates eta/theta.\n", ...
        meta.thetaInitStd);

    for ic = 1:nCases
        cfg = cfgBase;
        cfg.control.obs.mode = maneuverModes(ic);
        cfg.control.obs.enabled = maneuverModes(ic) ~= "none";
        cfg.step.name = "step09J6_" + referenceMode + ...
            "_obs_maneuver_" + maneuverModes(ic);
        configs{ic} = cfg;

        fprintf("\n%s (%s)...\n", caseNames(ic), maneuverModes(ic));
        try
            % Common random numbers preserve the paired comparison even
            % though each maneuver creates a different noiseless bearing.
            rng(seed);
            res = simulate_GS_DNN_EKF(cfg);
            assert(all(isfinite(res.xhat(:))), ...
                "Non-finite kinematic estimate was produced.");
            assert(all(isfinite(res.dnnResidual(:))), ...
                "Non-finite DNN residual was produced.");
            results{ic} = res;
            status(ic) = "ok";
        catch ME
            status(ic) = "failed: " + string(ME.message);
            warning("%s failed: %s", caseNames(ic), ME.message);
        end
    end

    positionRMSE = NaN(nCases,1);
    finalPositionRMSE = NaN(nCases,1);
    exactRangeRMSE = NaN(nCases,1);
    finalExactRangeRMSE = NaN(nCases,1);
    crossTrackRMSE = NaN(nCases,1);
    velocityRMSE = NaN(nCases,1);
    residualVectorRMSE = NaN(nCases,1);
    finalResidualVectorRMSE = NaN(nCases,1);
    meanCosine = NaN(nCases,1);
    finalCosine = NaN(nCases,1);
    meanNormRatio = NaN(nCases,1);
    meanNIS = NaN(nCases,1);
    maxManeuverDisplacement = NaN(nCases,1);
    maxTransverseDisplacement = NaN(nCases,1);
    meanLOSProfileChangeSigma = NaN(nCases,1);
    maxLOSProfileChangeSigma = NaN(nCases,1);
    maxAdditionalThrust = NaN(nCases,1);

    if ~isempty(results{1})
        resReference = results{1};
        for ic = 1:nCases
            if isempty(results{ic})
                continue;
            end
            m = summarizeManeuverCase(results{ic}, resReference, ...
                configs{ic});
            positionRMSE(ic) = m.positionRMSE;
            finalPositionRMSE(ic) = m.finalPositionRMSE;
            exactRangeRMSE(ic) = m.exactRangeRMSE;
            finalExactRangeRMSE(ic) = m.finalExactRangeRMSE;
            crossTrackRMSE(ic) = m.crossTrackRMSE;
            velocityRMSE(ic) = m.velocityRMSE;
            residualVectorRMSE(ic) = m.residualVectorRMSE;
            finalResidualVectorRMSE(ic) = m.finalResidualVectorRMSE;
            meanCosine(ic) = m.meanCosine;
            finalCosine(ic) = m.finalCosine;
            meanNormRatio(ic) = m.meanNormRatio;
            meanNIS(ic) = m.meanNIS;
            maxManeuverDisplacement(ic) = m.maxManeuverDisplacement;
            maxTransverseDisplacement(ic) = ...
                m.maxTransverseDisplacement;
            meanLOSProfileChangeSigma(ic) = ...
                m.meanLOSProfileChangeSigma;
            maxLOSProfileChangeSigma(ic) = ...
                m.maxLOSProfileChangeSigma;
            maxAdditionalThrust(ic) = m.maxAdditionalThrust;
        end
    end

    summary = table(caseNames, maneuverModes, status, positionRMSE, ...
        finalPositionRMSE, exactRangeRMSE, finalExactRangeRMSE, ...
        crossTrackRMSE, velocityRMSE, residualVectorRMSE, ...
        finalResidualVectorRMSE, meanCosine, finalCosine, meanNormRatio, ...
        meanNIS, maxManeuverDisplacement, maxTransverseDisplacement, ...
        meanLOSProfileChangeSigma, maxLOSProfileChangeSigma, ...
        maxAdditionalThrust, ...
        'VariableNames', {'caseName','maneuverMode','status', ...
        'positionRMSE','finalPositionRMSE','exactRangeRMSE', ...
        'finalExactRangeRMSE','crossTrackRMSE','velocityRMSE', ...
        'residualVectorRMSE','finalResidualVectorRMSE','meanCosine', ...
        'finalCosine','meanNormRatio','meanNIS', ...
        'maxManeuverDisplacement','maxTransverseDisplacement', ...
        'meanLOSProfileChangeSigma','maxLOSProfileChangeSigma', ...
        'maxAdditionalThrust'});
    disp(summary);

    figures = gobjects(0);
    if makePlots && ~isempty(results{1})
        figures = plotManeuverAblation(results, summary, configs{1});
    end

    sweep = struct;
    sweep.summary = summary;
    sweep.results = results;
    sweep.configs = configs;
    sweep.seed = seed;
    sweep.thetaInitStd = meta.thetaInitStd;
    sweep.referenceMode = referenceMode;
    sweep.figures = figures;
end

function m = summarizeManeuverCase(res, resReference, cfg)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    finalIdx = max(1,round(0.9*N)):N;

    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    posError = res.xhat(1:dim,:,:) - etaTrue(1:dim,:,:);
    velIdx = dim + (1:dim);
    velError = res.xhat(velIdx,:,:) - etaTrue(velIdx,:,:);

    targetPosition = repmat( ...
        reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
    relativeTrue = targetPosition - res.watcherR;
    relativeHat = res.xhat(1:dim,:,:) - res.watcherR;
    rangeTrue = reshape(sqrt(sum(relativeTrue.^2,1)),N,Nw);
    rangeHat = reshape(sqrt(sum(relativeHat.^2,1)),N,Nw);
    rangeError = rangeHat - rangeTrue;
    losUnit = relativeTrue ./ max(reshape(rangeTrue,1,N,Nw),realmin);

    losProjection = reshape(sum(posError .* losUnit,1),N,Nw);
    positionNorm = reshape(sqrt(sum(posError.^2,1)),N,Nw);
    crossTrackNorm = sqrt(max(positionNorm.^2-losProjection.^2,0));
    velocityNorm = reshape(sqrt(sum(velError.^2,1)),N,Nw);

    dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
    dHat = res.dnnResidual;
    dError = reshape(sqrt(sum((dHat-dTrue).^2,1)),N,Nw);
    trueNorm = reshape(sqrt(sum(dTrue.^2,1)),N,Nw);
    estimateNorm = reshape(sqrt(sum(dHat.^2,1)),N,Nw);
    dotValue = reshape(sum(dHat.*dTrue,1),N,Nw);
    normFloor = max(1e-12,1e-3*median(trueNorm(:),"omitnan"));
    valid = trueNorm > normFloor & estimateNorm > normFloor;
    cosine = NaN(N,Nw);
    cosine(valid) = dotValue(valid) ./ ...
        (trueNorm(valid).*estimateNorm(valid));
    ratio = NaN(N,Nw);
    ratioValid = trueNorm > normFloor;
    ratio(ratioValid) = estimateNorm(ratioValid)./trueNorm(ratioValid);

    % Counterfactual M0 geometry: this quantifies only the extra maneuver's
    % change to the LOS measurement profile.
    relativeReference = targetPosition - resReference.watcherR;
    referenceRange = sqrt(sum(relativeReference.^2,1));
    referenceLOS = relativeReference ./ max(referenceRange,realmin);
    losDot = reshape(sum(losUnit.*referenceLOS,1),N,Nw);
    losAngle = acos(max(-1,min(1,losDot)));
    losChangeSigma = losAngle / cfg.meas.sigmaBearing;

    deltaWatcherR = res.watcherR - resReference.watcherR;
    displacementNorm = reshape(sqrt(sum(deltaWatcherR.^2,1)),N,Nw);
    parallelDisplacement = reshape( ...
        sum(deltaWatcherR.*referenceLOS,1),N,Nw);
    transverseDisplacement = sqrt(max( ...
        displacementNorm.^2-parallelDisplacement.^2,0));

    deltaThrust = res.watcherU - resReference.watcherU;
    deltaThrustNorm = reshape(sqrt(sum(deltaThrust.^2,1)),N,Nw);
    postManeuver = res.time >= cfg.control.obs.startTime;

    m.positionRMSE = rmsAll(positionNorm);
    m.finalPositionRMSE = rmsAll(positionNorm(finalIdx,:));
    m.exactRangeRMSE = rmsAll(rangeError);
    m.finalExactRangeRMSE = rmsAll(rangeError(finalIdx,:));
    m.crossTrackRMSE = rmsAll(crossTrackNorm);
    m.velocityRMSE = rmsAll(velocityNorm);
    m.residualVectorRMSE = rmsAll(dError);
    m.finalResidualVectorRMSE = rmsAll(dError(finalIdx,:));
    m.meanCosine = mean(cosine(:),"omitnan");
    m.finalCosine = mean(cosine(finalIdx,:),"all","omitnan");
    m.meanNormRatio = mean(ratio(:),"omitnan");
    nis = res.NIS(isfinite(res.NIS));
    m.meanNIS = mean(nis,"omitnan");
    m.maxManeuverDisplacement = max(displacementNorm(:));
    m.maxTransverseDisplacement = max(transverseDisplacement(:));
    m.meanLOSProfileChangeSigma = mean( ...
        losChangeSigma(postManeuver,:),"all","omitnan");
    m.maxLOSProfileChangeSigma = max(losChangeSigma(:));
    m.maxAdditionalThrust = max(deltaThrustNorm(:));
end

function figures = plotManeuverAblation(results, summary, cfg)
    valid = summary.status == "ok";
    labels = summary.caseName(valid);
    colors = lines(height(summary));

    f1 = figure('Name','Step 09-J.6 observability maneuver summary');
    tiledlayout(2,2,'TileSpacing','compact');

    nexttile;
    bar(categorical(labels), [summary.exactRangeRMSE(valid), ...
        summary.finalExactRangeRMSE(valid)]);
    grid on; ylabel('range RMSE [m]');
    title('Angle-only range estimation');
    legend('whole run','final 10%','Location','best');

    nexttile;
    bar(categorical(labels), [summary.positionRMSE(valid), ...
        summary.finalPositionRMSE(valid)]);
    grid on; ylabel('position RMSE [m]');
    title('Target position estimation');
    legend('whole run','final 10%','Location','best');

    nexttile;
    bar(categorical(labels), [summary.meanLOSProfileChangeSigma(valid), ...
        summary.maxLOSProfileChangeSigma(valid)]);
    hold on; yline(3,'k:','3 sigma'); grid on;
    ylabel('bearing-profile change / sigma_b');
    title('Maneuver detectability');
    legend('post-burn mean','maximum','Location','best');

    nexttile;
    bar(categorical(labels), [summary.residualVectorRMSE(valid), ...
        summary.finalResidualVectorRMSE(valid)]);
    grid on; ylabel('residual RMSE [m/s^2]');
    title('DNN residual learning');
    legend('whole run','final 10%','Location','best');

    f2 = figure('Name','Step 09-J.6 observability maneuver histories');
    tiledlayout(2,2,'TileSpacing','compact');
    titles = ["Target position error", "Exact range error", ...
        "DNN residual vector error", "LOS profile change"];
    ylabels = ["mean error [m]", "RMS error [m]", ...
        "mean error [m/s^2]", "mean change / sigma_b"];

    for panel = 1:4
        nexttile; hold on; grid on;
        for ic = 1:numel(results)
            if isempty(results{ic})
                continue;
            end
            series = maneuverTimeSeries( ...
                results{ic}, results{1}, cfg, panel);
            plot(results{ic}.time, series, 'LineWidth',1.05, ...
                'Color',colors(ic,:));
        end
        xline(cfg.control.obs.startTime,'k:','burn start');
        xline(cfg.control.obs.startTime+cfg.control.obs.burnDuration, ...
            'k--','burn end');
        if panel == 4
            yline(3,'k-.','3 sigma');
        end
        title(titles(panel)); ylabel(ylabels(panel)); xlabel('time [s]');
        if panel == 1
            legend(summary.caseName(summary.status=="ok"), ...
                'Location','best');
        end
    end

    figures = [f1;f2];
end

function series = maneuverTimeSeries(res, resReference, cfg, panel)
    dim = cfg.dim;
    [~,N,Nw] = size(res.xhat);
    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);

    switch panel
        case 1
            errorValue = res.xhat(1:dim,:,:) - etaTrue(1:dim,:,:);
            errorNorm = reshape(sqrt(sum(errorValue.^2,1)),N,Nw);
            series = mean(errorNorm,2,"omitnan");

        case 2
            targetPosition = repmat( ...
                reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
            trueRange = reshape(sqrt(sum( ...
                (targetPosition-res.watcherR).^2,1)),N,Nw);
            estimateRange = reshape(sqrt(sum( ...
                (res.xhat(1:dim,:,:)-res.watcherR).^2,1)),N,Nw);
            series = sqrt(mean((estimateRange-trueRange).^2,2,"omitnan"));

        case 3
            dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
            errorNorm = reshape(sqrt(sum( ...
                (res.dnnResidual-dTrue).^2,1)),N,Nw);
            series = mean(errorNorm,2,"omitnan");

        case 4
            targetPosition = repmat( ...
                reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
            relativeNow = targetPosition-res.watcherR;
            relativeRef = targetPosition-resReference.watcherR;
            unitNow = relativeNow ./ max(sqrt(sum(relativeNow.^2,1)),realmin);
            unitRef = relativeRef ./ max(sqrt(sum(relativeRef.^2,1)),realmin);
            dotValue = reshape(sum(unitNow.*unitRef,1),N,Nw);
            angle = acos(max(-1,min(1,dotValue)));
            series = mean(angle/cfg.meas.sigmaBearing,2,"omitnan");

        otherwise
            error("Unknown panel.");
    end
end

function value = rmsAll(x)
    value = sqrt(mean(x(:).^2,"omitnan"));
end
