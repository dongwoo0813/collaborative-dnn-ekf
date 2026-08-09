function sweep = run_step09J6_bearing_noise_sweep(makePlots)
%RUN_STEP09J6_BEARING_NOISE_SWEEP Isolate bearing-noise sensitivity.
%
% Cases use 1-sigma bearing noise of 0.2, 0.05, 0.02, and 0.01 deg.
% The seed, truth, initial physical state, initial DNN parameters, DNN
% covariance/process model, additive GS composition, and measurement-noise
% standard-normal draws are common to all cases. Adaptive Q matching is
% frozen so that bearing accuracy is the only experimental factor.
%
% Usage:
%   sweep09j6Noise = run_step09J6_bearing_noise_sweep();
%   sweep09j6Noise = run_step09J6_bearing_noise_sweep(false);

% Important outputs:
%   sweep.summary       - one-row-per-noise-level metric table
%   sweep.results       - complete simulation result for each case
%   sweep.configs       - exact configuration for each case
%   sweep.figures       - generated figure handles

% The LOS-direction error is the target position error projected onto the
% true watcher-to-target LOS. It is the range-like weakly observable error.

    if nargin < 1
        makePlots = true;
    end

    addpath(genpath(pwd));
    [cfgBase, seed, meta] = config_step09J6_seed101_operational();

    cfgBase.gs.compositeMode = "additive";
    cfgBase.meas.type = "bearing";
    cfgBase.dnn.adaptQThetaEnabled = false;
    cfgBase.dnn.adaptQEpsilonEnabled = false;

    sigmaBearingDeg = [0.2; 0.05; 0.02; 0.01];
    sigmaBearingArcsec = 3600 * sigmaBearingDeg;
    crossTrackNoiseAt1km = 1000 * deg2rad(sigmaBearingDeg);
    nCases = numel(sigmaBearingDeg);
    caseNames = compose("C%d %.3g deg", (0:nCases-1).', sigmaBearingDeg);

    results = cell(nCases,1);
    configs = cell(nCases,1);
    status = repmat("not run",nCases,1);

    positionRMSE = NaN(nCases,1);
    losDirectionRMSE = NaN(nCases,1);
    crossTrackPositionRMSE = NaN(nCases,1);
    velocityRMSE = NaN(nCases,1);
    residualVectorRMSE = NaN(nCases,1);
    meanCosine = NaN(nCases,1);
    finalCosine = NaN(nCases,1);
    meanNormRatio = NaN(nCases,1);
    meanThetaUpdateNorm = NaN(nCases,1);
    meanNIS = NaN(nCases,1);

    fprintf("Step 09-J.6 bearing-noise sweep (additive GS)\n");
    fprintf("theta0Std=%.3e, sqrt(Ptheta0)=%.3e, sigmaSS=%.3e\n", ...
        meta.thetaInitStd,sqrt(cfgBase.dnn.Ptheta0), ...
        cfgBase.dnn.thetaSigmaSS);
    fprintf("Adaptive Q matching is frozen in every case.\n");

    for ic = 1:nCases
        cfg = cfgBase;
        cfg.meas.sigmaBearing = deg2rad(sigmaBearingDeg(ic));
        if cfg.dim == 2
            cfg.meas.R = cfg.meas.sigmaBearing^2;
        elseif cfg.dim == 3
            cfg.meas.R = cfg.meas.sigmaBearing^2 * eye(2);
        else
            error("Unsupported cfg.dim=%d.",cfg.dim);
        end
        cfg.step.name = sprintf("step09J6_bearing_noise_%gdeg", ...
            sigmaBearingDeg(ic));
        configs{ic} = cfg;

        fprintf("\n%s: sigma_b=%.4f deg = %.1f arcsec", ...
            caseNames(ic),sigmaBearingDeg(ic),sigmaBearingArcsec(ic));
        fprintf(", 1-km transverse scale=%.3f m\n", ...
            crossTrackNoiseAt1km(ic));

        try
            % The same seed preserves the truth, initialization, DNN draw,
            % and underlying standard-normal measurement-noise sequence.
            rng(seed);
            res = simulate_GS_DNN_EKF(cfg);
            assert(all(isfinite(res.xhat(:))), ...
                "Non-finite kinematic estimate was produced.");
            assert(all(isfinite(res.dnnResidual(:))), ...
                "Non-finite DNN residual was produced.");

            results{ic} = res;
            status(ic) = "ok";
            m = summarizeNoiseCase(res,cfg.dim);
            positionRMSE(ic) = m.positionRMSE;
            losDirectionRMSE(ic) = m.losDirectionRMSE;
            crossTrackPositionRMSE(ic) = m.crossTrackPositionRMSE;
            velocityRMSE(ic) = m.velocityRMSE;
            residualVectorRMSE(ic) = m.residualVectorRMSE;
            meanCosine(ic) = m.meanCosine;
            finalCosine(ic) = m.finalCosine;
            meanNormRatio(ic) = m.meanNormRatio;
            meanThetaUpdateNorm(ic) = m.meanThetaUpdateNorm;
            meanNIS(ic) = m.meanNIS;
        catch ME
            status(ic) = "failed: " + string(ME.message);
            warning("%s failed: %s",caseNames(ic),ME.message);
        end
    end

    summary = table(caseNames,status,sigmaBearingDeg, ...
        sigmaBearingArcsec,crossTrackNoiseAt1km,positionRMSE, ...
        losDirectionRMSE,crossTrackPositionRMSE,velocityRMSE, ...
        residualVectorRMSE,meanCosine,finalCosine,meanNormRatio, ...
        meanThetaUpdateNorm,meanNIS, ...
        'VariableNames',{'caseName','status','sigmaBearingDeg', ...
        'sigmaBearingArcsec','crossTrackNoiseAt1km', ...
        'positionRMSE','losDirectionRMSE','crossTrackPositionRMSE', ...
        'velocityRMSE','residualVectorRMSE','meanCosine', ...
        'finalCosine','meanNormRatio','meanThetaUpdateNorm','meanNIS'});
    disp(summary);

    figures = gobjects(0);
    if makePlots
        figures = plotNoiseSweep(results,summary);
    end

    sweep = struct;
    sweep.summary = summary;
    sweep.results = results;
    sweep.configs = configs;
    sweep.seed = seed;
    sweep.thetaInitStd = meta.thetaInitStd;
    sweep.referenceRange = 1000;
    sweep.figures = figures;
end

function m = summarizeNoiseCase(res,dim)
    [~,N,Nw] = size(res.xhat);
    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
    posError = res.xhat(1:dim,:,:) - etaTrue(1:dim,:,:);
    velIdx = dim + (1:dim);
    velError = res.xhat(velIdx,:,:) - etaTrue(velIdx,:,:);

    targetPosition = repmat(reshape(res.etaTrue(1:dim,:),dim,N,1), ...
        1,1,Nw);
    relativePosition = targetPosition - res.watcherR;
    relativeRange = sqrt(sum(relativePosition.^2,1));
    losUnit = relativePosition ./ max(relativeRange,realmin);

    losError = reshape(sum(posError .* losUnit,1),N,Nw);
    positionErrorNorm = reshape(sqrt(sum(posError.^2,1)),N,Nw);
    crossTrackSquared = max(positionErrorNorm.^2 - losError.^2,0);
    crossTrackErrorNorm = sqrt(crossTrackSquared);
    velocityErrorNorm = reshape(sqrt(sum(velError.^2,1)),N,Nw);

    dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
    dHat = res.dnnResidual;
    residualErrorNorm = reshape(sqrt(sum((dHat-dTrue).^2,1)),N,Nw);
    trueNorm = reshape(sqrt(sum(dTrue.^2,1)),N,Nw);
    estimateNorm = reshape(sqrt(sum(dHat.^2,1)),N,Nw);
    dotProduct = reshape(sum(dHat.*dTrue,1),N,Nw);
    normFloor = max(1e-12,1e-3*median(trueNorm(:),"omitnan"));

    validCosine = trueNorm > normFloor & estimateNorm > normFloor;
    cosine = NaN(N,Nw);
    cosine(validCosine) = dotProduct(validCosine) ./ ...
        (trueNorm(validCosine).*estimateNorm(validCosine));
    normRatio = NaN(N,Nw);
    validRatio = trueNorm > normFloor;
    normRatio(validRatio) = estimateNorm(validRatio)./trueNorm(validRatio);
    finalIdx = max(1,round(0.9*N)):N;

    m.positionRMSE = sqrt(mean(positionErrorNorm(:).^2,"omitnan"));
    m.losDirectionRMSE = sqrt(mean(losError(:).^2,"omitnan"));
    m.crossTrackPositionRMSE = ...
        sqrt(mean(crossTrackErrorNorm(:).^2,"omitnan"));
    m.velocityRMSE = sqrt(mean(velocityErrorNorm(:).^2,"omitnan"));
    m.residualVectorRMSE = ...
        sqrt(mean(residualErrorNorm(:).^2,"omitnan"));
    m.meanCosine = mean(cosine(:),"omitnan");
    m.finalCosine = mean(cosine(finalIdx,:),"all","omitnan");
    m.meanNormRatio = mean(normRatio(:),"omitnan");
    m.meanThetaUpdateNorm = mean(res.thetaUpdateNorm(:),"omitnan");
    nis = res.NIS(isfinite(res.NIS));
    if isempty(nis)
        m.meanNIS = NaN;
    else
        m.meanNIS = mean(nis);
    end
end

function figures = plotNoiseSweep(results,summary)
    ok = summary.status == "ok";
    sigma = summary.sigmaBearingDeg(ok);

    f1 = figure('Name','Step 09-J.6 bearing-noise summary');
    tiledlayout(2,2,'TileSpacing','compact');

    nexttile;
    loglog(sigma,summary.positionRMSE(ok),'-o','LineWidth',1.3);
    hold on;
    loglog(sigma,summary.losDirectionRMSE(ok),'-s','LineWidth',1.3);
    loglog(sigma,summary.crossTrackPositionRMSE(ok),'-^','LineWidth',1.3);
    grid on; set(gca,'XDir','reverse');
    xlabel('\sigma_b [deg]'); ylabel('RMSE [m]');
    title('Target position error');
    legend('total','LOS/range-like','cross-track','Location','best');

    nexttile;
    loglog(sigma,summary.residualVectorRMSE(ok),'-o','LineWidth',1.3);
    grid on; set(gca,'XDir','reverse');
    xlabel('\sigma_b [deg]'); ylabel('RMSE [m/s^2]');
    title('DNN residual vector error');

    nexttile;
    semilogx(sigma,summary.meanCosine(ok),'-o','LineWidth',1.3);
    hold on;
    semilogx(sigma,summary.finalCosine(ok),'-s','LineWidth',1.3);
    yline(0,'k:'); grid on; ylim([-1 1]); set(gca,'XDir','reverse');
    xlabel('\sigma_b [deg]'); ylabel('cosine');
    title('Residual direction alignment');
    legend('whole run','final 10%','Location','best');

    nexttile;
    yyaxis left;
    loglog(sigma,summary.velocityRMSE(ok),'-o','LineWidth',1.3);
    ylabel('velocity RMSE [m/s]');
    yyaxis right;
    semilogx(sigma,summary.meanNIS(ok),'-s','LineWidth',1.3);
    ylabel('mean NIS');
    grid on; set(gca,'XDir','reverse'); xlabel('\sigma_b [deg]');
    title('Velocity accuracy and consistency');

    f2 = figure('Name','Step 09-J.6 bearing-noise time histories');
    tiledlayout(2,2,'TileSpacing','compact');
    colors = lines(height(summary));
    labels = strings(0,1);

    for panel = 1:4
        nexttile; hold on; grid on;
        for ic = 1:height(summary)
            if isempty(results{ic})
                continue;
            end
            [series,time] = noiseTimeSeries(results{ic},panel);
            plot(time,series,'LineWidth',1.05,'Color',colors(ic,:));
            if panel == 1
                labels(end+1,1) = summary.caseName(ic); %#ok<AGROW>
            end
        end
        switch panel
            case 1
                ylabel('mean position error [m]');
                title('Target position error');
                legend(labels,'Location','best');
            case 2
                ylabel('RMS LOS error [m]');
                title('Range-like position error');
            case 3
                ylabel('mean residual error [m/s^2]');
                title('DNN residual vector error');
            otherwise
                yline(0,'k:'); ylim([-1 1]);
                ylabel('mean cosine');
                title('Residual direction alignment');
        end
        xlabel('time [s]');
    end

    figures = [f1;f2];
end

function [series,time] = noiseTimeSeries(res,panel)
    time = res.time;
    dim = size(res.trueResidual,1);
    [~,N,Nw] = size(res.xhat);
    etaTrue = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);

    if panel <= 2
        posError = res.xhat(1:dim,:,:) - etaTrue(1:dim,:,:);
        if panel == 1
            value = reshape(sqrt(sum(posError.^2,1)),N,Nw);
            series = mean(value,2,"omitnan");
        else
            targetPosition = repmat( ...
                reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
            relativePosition = targetPosition - res.watcherR;
            relativeRange = sqrt(sum(relativePosition.^2,1));
            losUnit = relativePosition ./ max(relativeRange,realmin);
            value = reshape(sum(posError.*losUnit,1),N,Nw);
            series = sqrt(mean(value.^2,2,"omitnan"));
        end
        return;
    end

    dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
    dHat = res.dnnResidual;
    if panel == 3
        value = reshape(sqrt(sum((dHat-dTrue).^2,1)),N,Nw);
        series = mean(value,2,"omitnan");
    else
        trueNorm = reshape(sqrt(sum(dTrue.^2,1)),N,Nw);
        estimateNorm = reshape(sqrt(sum(dHat.^2,1)),N,Nw);
        dotProduct = reshape(sum(dHat.*dTrue,1),N,Nw);
        floorValue = max(1e-12,1e-3*median(trueNorm(:),"omitnan"));
        valid = trueNorm > floorValue & estimateNorm > floorValue;
        value = NaN(N,Nw);
        value(valid) = dotProduct(valid) ./ ...
            (trueNorm(valid).*estimateNorm(valid));
        series = mean(value,2,"omitnan");
    end
end
