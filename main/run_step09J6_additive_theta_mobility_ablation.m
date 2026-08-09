function sweep = run_step09J6_additive_theta_mobility_ablation(makePlots)
%RUN_STEP09J6_ADDITIVE_THETA_MOBILITY_ABLATION Separate Ptheta0 and Qtheta effects.
%
% The five cases use the same random seed, truth, measurements, initial DNN
% parameter draw, architecture, and additive GS composition.  Only the
% initial parameter covariance and FOGM parameter mobility are changed.
%
% Usage:
%   sweep = run_step09J6_additive_theta_mobility_ablation();
%   sweep = run_step09J6_additive_theta_mobility_ablation(false);

    if nargin < 1
        makePlots = true;
    end

    addpath(genpath(pwd));
    [cfgBase, seed, meta] = config_step09J6_seed101_operational();

    % Keep the actual random parameter draw fixed in every case.  PthetaStd
    % is only a readable way to specify sqrt(Ptheta0), not theta0Std.
    caseNames = ["A0 baseline", "A1 P only", "A2 Q only", ...
                 "A3 P and Q", "A4 aggressive"];
    PthetaStd = [7.5e-5, 1e-3, 7.5e-5, 1e-3, 1e-2];
    thetaSigmaSS = [7.5e-5, 7.5e-5, 1e-3, 1e-3, 1e-2];
    thetaTau = [10000, 10000, 1000, 1000, 1000];

    nCases = numel(caseNames);
    caseResults = cell(nCases, 1);
    caseConfigs = cell(nCases, 1);
    status = repmat("not run", nCases, 1);

    residualVectorRMSE = NaN(nCases,1);
    meanCosine = NaN(nCases,1);
    meanNormRatio = NaN(nCases,1);
    positionRMSE = NaN(nCases,1);
    velocityRMSE = NaN(nCases,1);
    meanThetaUpdateNorm = NaN(nCases,1);
    meanPthetaEtaFro = NaN(nCases,1);
    finalTracePtheta = NaN(nCases,1);
    finalGammaTheta = NaN(nCases,1);

    fprintf("Step 09-J.6 additive theta-mobility ablation\n");
    fprintf("theta0Std is fixed at %.3e in all cases.\n", meta.thetaInitStd);

    for ic = 1:nCases
        cfg = cfgBase;
        cfg.step.name = "step09J6_additive_theta_mobility_" + ...
            replace(caseNames(ic), " ", "_");
        cfg.gs.compositeMode = "additive";
        cfg.dnn.Ptheta0 = PthetaStd(ic)^2;
        cfg.dnn.thetaSigmaSS = thetaSigmaSS(ic);
        cfg.dnn.thetaTau = thetaTau(ic);

        caseConfigs{ic} = cfg;
        fprintf("\n%s: sqrt(Ptheta0)=%.3e, sigmaSS=%.3e, tau=%.1f s\n", ...
            caseNames(ic), PthetaStd(ic), thetaSigmaSS(ic), thetaTau(ic));

        try
            % Resetting the seed makes the initial states, DNN draw, and
            % measurement-noise realization common across all cases.
            rng(seed);
            res = simulate_GS_DNN_EKF(cfg);
            assert(all(isfinite(res.dnnResidual(:))), ...
                "Non-finite DNN residual was produced.");

            caseResults{ic} = res;
            status(ic) = "ok";

            m = summarizeCase(res, cfg.dim);
            residualVectorRMSE(ic) = m.residualVectorRMSE;
            meanCosine(ic) = m.meanCosine;
            meanNormRatio(ic) = m.meanNormRatio;
            positionRMSE(ic) = m.positionRMSE;
            velocityRMSE(ic) = m.velocityRMSE;
            meanThetaUpdateNorm(ic) = m.meanThetaUpdateNorm;
            meanPthetaEtaFro(ic) = m.meanPthetaEtaFro;
            finalTracePtheta(ic) = m.finalTracePtheta;
            finalGammaTheta(ic) = m.finalGammaTheta;
        catch ME
            status(ic) = "failed: " + string(ME.message);
            warning("%s failed: %s", caseNames(ic), ME.message);
        end
    end

    summary = table(caseNames.', status, PthetaStd.', thetaSigmaSS.', ...
        thetaTau.', residualVectorRMSE, meanCosine, meanNormRatio, ...
        positionRMSE, velocityRMSE, meanThetaUpdateNorm, ...
        meanPthetaEtaFro, finalTracePtheta, finalGammaTheta, ...
        'VariableNames', {'caseName', 'status', 'PthetaStd', ...
        'thetaSigmaSS', 'thetaTau', 'residualVectorRMSE', ...
        'meanCosine', 'meanNormRatio', 'positionRMSE', ...
        'velocityRMSE', 'meanThetaUpdateNorm', 'meanPthetaEtaFro', ...
        'finalTracePtheta', 'finalGammaTheta'});

    disp(summary);

    figures = gobjects(0);
    if makePlots
        figures = plotSweep(caseNames, caseResults, summary);
    end

    sweep = struct;
    sweep.summary = summary;
    sweep.results = caseResults;
    sweep.configs = caseConfigs;
    sweep.seed = seed;
    sweep.thetaInitStd = meta.thetaInitStd;
    sweep.figures = figures;
end

function m = summarizeCase(res, dim)
    dHat = res.dnnResidual;
    [~, N, Nw] = size(dHat);
    dTrue = repmat(reshape(res.trueResidual, dim, N, 1), 1, 1, Nw);

    residualErrorNorm = reshape(sqrt(sum((dHat - dTrue).^2, 1)), N, Nw);
    trueNorm = reshape(sqrt(sum(dTrue.^2, 1)), N, Nw);
    estimateNorm = reshape(sqrt(sum(dHat.^2, 1)), N, Nw);
    dotProduct = reshape(sum(dHat .* dTrue, 1), N, Nw);

    normFloor = max(1e-12, 1e-3 * median(trueNorm(:), "omitnan"));
    valid = trueNorm > normFloor & estimateNorm > normFloor;

    cosine = NaN(N, Nw);
    cosine(valid) = dotProduct(valid) ./ ...
        (trueNorm(valid) .* estimateNorm(valid));
    normRatio = NaN(N, Nw);
    normRatio(trueNorm > normFloor) = estimateNorm(trueNorm > normFloor) ./ ...
        trueNorm(trueNorm > normFloor);

    etaTrue = repmat(reshape(res.etaTrue, 2*dim, N, 1), 1, 1, Nw);
    positionErrorNorm = reshape(sqrt(sum( ...
        (res.xhat(1:dim,:,:) - etaTrue(1:dim,:,:)).^2, 1)), N, Nw);
    idxV = dim + (1:dim);
    velocityErrorNorm = reshape(sqrt(sum( ...
        (res.xhat(idxV,:,:) - etaTrue(idxV,:,:)).^2, 1)), N, Nw);

    m.residualVectorRMSE = sqrt(mean(residualErrorNorm(:).^2, "omitnan"));
    m.meanCosine = mean(cosine(:), "omitnan");
    m.meanNormRatio = mean(normRatio(:), "omitnan");
    m.positionRMSE = sqrt(mean(positionErrorNorm(:).^2, "omitnan"));
    m.velocityRMSE = sqrt(mean(velocityErrorNorm(:).^2, "omitnan"));
    m.meanThetaUpdateNorm = mean(res.thetaUpdateNorm(:), "omitnan");
    m.meanPthetaEtaFro = mean(res.PthetaEtaFro(:), "omitnan");
    m.finalTracePtheta = mean(res.tracePtheta(end,:), "omitnan");
    m.finalGammaTheta = mean(res.gammaTheta(end,:), "omitnan");
end

function figures = plotSweep(caseNames, caseResults, summary)
    ok = summary.status == "ok";
    x = 1:numel(caseNames);

    f1 = figure('Name', 'Step 09-J.6 theta mobility ablation summary');
    tiledlayout(2,2, 'TileSpacing', 'compact');

    nexttile;
    semilogy(x(ok), summary.residualVectorRMSE(ok), '-o', 'LineWidth', 1.3);
    grid on; ylabel('residual vector RMSE [m/s^2]');
    title('Residual error');

    nexttile;
    plot(x(ok), summary.meanCosine(ok), '-o', 'LineWidth', 1.3);
    grid on; ylim([-1 1]); ylabel('mean cosine');
    title('Residual direction');

    nexttile;
    plot(x(ok), summary.meanNormRatio(ok), '-o', 'LineWidth', 1.3);
    hold on; yline(1, 'k:'); grid on; ylabel('mean norm ratio');
    title('Residual magnitude');

    nexttile;
    yyaxis left;
    plot(x(ok), summary.positionRMSE(ok), '-o', 'LineWidth', 1.3);
    ylabel('position RMSE [m]');
    yyaxis right;
    plot(x(ok), summary.velocityRMSE(ok), '-s', 'LineWidth', 1.3);
    ylabel('velocity RMSE [m/s]'); grid on;
    title('Kinematic-state accuracy');

    for ax = findall(f1, 'Type', 'axes').'
        ax.XTick = x;
        ax.XTickLabel = caseNames;
        ax.XTickLabelRotation = 20;
    end

    f2 = figure('Name', 'Step 09-J.6 theta learning authority');
    tiledlayout(3,1, 'TileSpacing', 'compact');
    colors = lines(numel(caseNames));

    labels = strings(0,1);
    for panel = 1:3
        nexttile; hold on; grid on;
        for ic = 1:numel(caseNames)
            if isempty(caseResults{ic})
                continue;
            end
            res = caseResults{ic};
            switch panel
                case 1
                    y = mean(res.thetaUpdateNorm, 2, "omitnan");
                case 2
                    y = mean(res.PthetaEtaFro, 2, "omitnan");
                otherwise
                    y = mean(res.tracePtheta, 2, "omitnan");
            end
            semilogy(res.time, max(y, realmin), 'LineWidth', 1.1, ...
                'Color', colors(ic,:));
            if panel == 1
                labels(end+1,1) = caseNames(ic); %#ok<AGROW>
            end
        end
        if panel == 1
            ylabel('mean ||Delta theta||_2');
            title('Measurement-update parameter correction');
            legend(labels, 'Location', 'best');
        elseif panel == 2
            ylabel('mean ||P_{theta eta}||_F');
            title('State-parameter cross covariance');
        else
            ylabel('mean trace(P_{theta theta})');
            xlabel('time [s]');
            title('Parameter covariance');
        end
    end

    figures = [f1; f2];
end
