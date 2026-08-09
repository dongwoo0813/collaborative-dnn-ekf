function obs = run_step09J6_measurement_observability_ablation(makePlots)
%RUN_STEP09J6_MEASUREMENT_OBSERVABILITY_ABLATION Compare measurement information.
%
% B0: bearing only, sigma_b = 0.2 deg
% B1: range + bearing, sigma_r = 1 m, sigma_b = 0.2 deg
% B2: full relative position, sigma_p = 1 m per component
% B3: direct residual oracle, sigma_d = 1e-6 m/s^2
%
% All cases use the same seed, initial DNN draw, two-hidden-layer network,
% additive GS composition, Ptheta0, and Qtheta. Adaptive Q matching is
% frozen so that measurement information is the only experimental factor.

    if nargin < 1
        makePlots = true;
    end

    addpath(genpath(pwd));
    [cfgBase, seed, meta] = config_step09J6_seed101_operational();

    cfgBase.gs.compositeMode = "additive";
    cfgBase.dnn.adaptQThetaEnabled = false;
    cfgBase.dnn.adaptQEpsilonEnabled = false;

    caseNames = ["B0 bearing", "B1 range+bearing", ...
                 "B2 relative position", "B3 direct residual oracle"];
    measTypes = ["bearing", "range_bearing", ...
                 "relative_position", "direct_residual"];
    nCases = numel(caseNames);

    results = cell(nCases,1);
    configs = cell(nCases,1);
    status = repmat("not run", nCases,1);

    residualVectorRMSE = NaN(nCases,1);
    residualRMSEAtTrueEta = NaN(nCases,1);
    meanCosine = NaN(nCases,1);
    meanCosineAtTrueEta = NaN(nCases,1);
    initialCosine = NaN(nCases,1);
    finalCosine = NaN(nCases,1);
    meanNormRatio = NaN(nCases,1);
    positionRMSE = NaN(nCases,1);
    velocityRMSE = NaN(nCases,1);
    meanThetaUpdateNorm = NaN(nCases,1);
    meanPthetaEtaFro = NaN(nCases,1);

    fprintf("Step 09-J.6 measurement-observability ablation\n");
    fprintf("theta0Std=%.3e, sqrt(Ptheta0)=%.3e, sigmaSS=%.3e\n", ...
        meta.thetaInitStd, sqrt(cfgBase.dnn.Ptheta0), ...
        cfgBase.dnn.thetaSigmaSS);
    fprintf("Adaptive Q matching is frozen in all B cases.\n");

    for ic = 1:nCases
        cfg = configureMeasurementCase(cfgBase, measTypes(ic));
        cfg.step.name = "step09J6_observability_" + ...
            replace(measTypes(ic), "_", "-");
        configs{ic} = cfg;

        fprintf("\n%s (%s)\n", caseNames(ic), measTypes(ic));
        try
            rng(seed);
            res = simulate_GS_DNN_EKF(cfg);
            assert(all(isfinite(res.dnnResidual(:))), ...
                "Non-finite DNN residual was produced.");
            results{ic} = res;
            status(ic) = "ok";

            m = summarizeObservabilityCase(res, cfg.dim);
            residualVectorRMSE(ic) = m.residualVectorRMSE;
            residualRMSEAtTrueEta(ic) = m.residualRMSEAtTrueEta;
            meanCosine(ic) = m.meanCosine;
            meanCosineAtTrueEta(ic) = m.meanCosineAtTrueEta;
            initialCosine(ic) = m.initialCosine;
            finalCosine(ic) = m.finalCosine;
            meanNormRatio(ic) = m.meanNormRatio;
            positionRMSE(ic) = m.positionRMSE;
            velocityRMSE(ic) = m.velocityRMSE;
            meanThetaUpdateNorm(ic) = m.meanThetaUpdateNorm;
            meanPthetaEtaFro(ic) = m.meanPthetaEtaFro;
        catch ME
            status(ic) = "failed: " + string(ME.message);
            warning("%s failed: %s", caseNames(ic), ME.message);
        end
    end

    summary = table(caseNames.', measTypes.', status, ...
        residualVectorRMSE, residualRMSEAtTrueEta, meanCosine, ...
        meanCosineAtTrueEta, initialCosine, finalCosine, meanNormRatio, ...
        positionRMSE, velocityRMSE, meanThetaUpdateNorm, meanPthetaEtaFro, ...
        'VariableNames', {'caseName','measurementType','status', ...
        'residualVectorRMSE','residualRMSEAtTrueEta','meanCosine', ...
        'meanCosineAtTrueEta','initialCosine','finalCosine', ...
        'meanNormRatio','positionRMSE','velocityRMSE', ...
        'meanThetaUpdateNorm','meanPthetaEtaFro'});

    disp(summary);

    figures = gobjects(0);
    if makePlots
        figures = plotObservabilitySweep(caseNames, results, summary);
    end

    obs = struct;
    obs.summary = summary;
    obs.results = results;
    obs.configs = configs;
    obs.seed = seed;
    obs.thetaInitStd = meta.thetaInitStd;
    obs.figures = figures;
end

function cfg = configureMeasurementCase(cfg, measType)
    cfg.meas.type = measType;
    switch measType
        case "bearing"
            cfg.meas.R = cfg.meas.sigmaBearing^2;
        case "range_bearing"
            cfg.meas.sigmaRange = 1.0;
            cfg.meas.R = diag([cfg.meas.sigmaRange^2, ...
                               cfg.meas.sigmaBearing^2]);
        case "relative_position"
            cfg.meas.sigmaPosition = 1.0;
            cfg.meas.R = cfg.meas.sigmaPosition^2 * eye(cfg.dim);
        case "direct_residual"
            cfg.meas.sigmaDirectResidual = 1e-6;
            cfg.meas.directResidualTargetScale = 1 / max(cfg.Nw,1);
            cfg.meas.R = cfg.meas.sigmaDirectResidual^2 * eye(cfg.dim);
        otherwise
            error("Unsupported measurement type: %s", measType);
    end
end

function m = summarizeObservabilityCase(res, dim)
    [~, N, Nw] = size(res.dnnResidual);
    dTrue = repmat(reshape(res.trueResidual, dim, N, 1), 1, 1, Nw);

    [rmseOperational, cosineOperational, ratioOperational] = ...
        residualMetrics(res.dnnResidual, dTrue);
    [rmseTrueEta, cosineTrueEta] = ...
        residualMetrics(res.dnnResidualAtTrueEta, dTrue);

    nInitial = max(1, round(0.1*N));
    idxFinal = max(1, round(0.9*N)):N;

    etaTrue = repmat(reshape(res.etaTrue, 2*dim, N, 1), 1, 1, Nw);
    posErr = reshape(sqrt(sum((res.xhat(1:dim,:,:) - ...
        etaTrue(1:dim,:,:)).^2,1)), N, Nw);
    idxV = dim + (1:dim);
    velErr = reshape(sqrt(sum((res.xhat(idxV,:,:) - ...
        etaTrue(idxV,:,:)).^2,1)), N, Nw);

    m.residualVectorRMSE = rmseOperational;
    m.residualRMSEAtTrueEta = rmseTrueEta;
    m.meanCosine = mean(cosineOperational(:), "omitnan");
    m.meanCosineAtTrueEta = mean(cosineTrueEta(:), "omitnan");
    m.initialCosine = mean(cosineOperational(1:nInitial,:), "all", "omitnan");
    m.finalCosine = mean(cosineOperational(idxFinal,:), "all", "omitnan");
    m.meanNormRatio = mean(ratioOperational(:), "omitnan");
    m.positionRMSE = sqrt(mean(posErr(:).^2, "omitnan"));
    m.velocityRMSE = sqrt(mean(velErr(:).^2, "omitnan"));
    m.meanThetaUpdateNorm = mean(res.thetaUpdateNorm(:), "omitnan");
    m.meanPthetaEtaFro = mean(res.PthetaEtaFro(:), "omitnan");
end

function [vectorRMSE, cosine, normRatio] = residualMetrics(dHat, dTrue)
    [~, N, Nw] = size(dHat);
    errorNorm = reshape(sqrt(sum((dHat-dTrue).^2,1)), N, Nw);
    trueNorm = reshape(sqrt(sum(dTrue.^2,1)), N, Nw);
    estimateNorm = reshape(sqrt(sum(dHat.^2,1)), N, Nw);
    dotProduct = reshape(sum(dHat.*dTrue,1), N, Nw);
    floorValue = max(1e-12, 1e-3*median(trueNorm(:),"omitnan"));

    valid = trueNorm > floorValue & estimateNorm > floorValue;
    cosine = NaN(N,Nw);
    cosine(valid) = dotProduct(valid) ./ ...
        (trueNorm(valid).*estimateNorm(valid));
    normRatio = NaN(N,Nw);
    validRatio = trueNorm > floorValue;
    normRatio(validRatio) = estimateNorm(validRatio)./trueNorm(validRatio);
    vectorRMSE = sqrt(mean(errorNorm(:).^2,"omitnan"));
end

function figures = plotObservabilitySweep(caseNames, results, summary)
    ok = summary.status == "ok";
    x = 1:numel(caseNames);

    f1 = figure('Name','Step 09-J.6 measurement observability summary');
    tiledlayout(2,2,'TileSpacing','compact');
    nexttile;
    semilogy(x(ok),summary.residualVectorRMSE(ok),'-o','LineWidth',1.3);
    grid on; ylabel('residual RMSE [m/s^2]'); title('Operational residual');
    nexttile;
    plot(x(ok),summary.finalCosine(ok),'-o','LineWidth',1.3);
    hold on; yline(0,'k:'); grid on; ylim([-1 1]);
    ylabel('final-window cosine'); title('Learned direction');
    nexttile;
    plot(x(ok),summary.positionRMSE(ok),'-o','LineWidth',1.3);
    grid on; ylabel('position RMSE [m]'); title('Target position');
    nexttile;
    semilogy(x(ok),summary.meanThetaUpdateNorm(ok),'-o','LineWidth',1.3);
    grid on; ylabel('mean ||Delta theta||_2'); title('Parameter correction');
    for ax = findall(f1,'Type','axes').'
        ax.XTick = x; ax.XTickLabel = caseNames; ax.XTickLabelRotation = 20;
    end

    f2 = figure('Name','Step 09-J.6 residual learning by measurement type');
    tiledlayout(2,1,'TileSpacing','compact');
    colors = lines(numel(caseNames));
    labels = strings(0,1);
    nexttile; hold on; grid on;
    for ic=1:numel(caseNames)
        if isempty(results{ic}), continue; end
        res=results{ic}; [~,N,Nw]=size(res.dnnResidual);
        dTrue=repmat(reshape(res.trueResidual,size(res.trueResidual,1),N,1),1,1,Nw);
        err=reshape(sqrt(sum((res.dnnResidual-dTrue).^2,1)),N,Nw);
        semilogy(res.time,mean(err,2,'omitnan'),'LineWidth',1.1,'Color',colors(ic,:));
        labels(end+1,1)=caseNames(ic); %#ok<AGROW>
    end
    ylabel('mean residual error norm [m/s^2]'); title('Residual error');
    legend(labels,'Location','best');
    nexttile; hold on; grid on;
    for ic=1:numel(caseNames)
        if isempty(results{ic}), continue; end
        res=results{ic}; [~,N,Nw]=size(res.dnnResidual);
        dTrue=repmat(reshape(res.trueResidual,size(res.trueResidual,1),N,1),1,1,Nw);
        [~,c]=residualMetrics(res.dnnResidual,dTrue);
        plot(res.time,mean(c,2,'omitnan'),'LineWidth',1.1,'Color',colors(ic,:));
    end
    yline(0,'k:'); ylim([-1 1]); xlabel('time [s]'); ylabel('mean cosine');
    title('Residual direction alignment');

    figures=[f1;f2];
end
