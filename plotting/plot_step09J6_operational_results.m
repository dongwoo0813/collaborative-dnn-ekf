function plots = plot_step09J6_operational_results(out09j5)
%PLOT_STEP09J6_OPERATIONAL_RESULTS Plot truth, estimates, and residual errors.
%   plots = plot_step09J6_operational_results(out09j5)
%   uses only saved simulation outputs and does not rerun the simulation.

    arguments
        out09j5 (1,1) struct
    end

    add = out09j5.resGSAdd;
    fim = out09j5.resGSFIM;
    weightedLabel = getWeightedLabel(out09j5);
    required = ["time", "trueResidual", "dnnResidual"];
    for f = required
        assert(isfield(add, f) && isfield(fim, f), ...
            "Both J6 results must contain %s.", f);
    end

    t = add.time(:);
    N = numel(t);
    dTrue = orientTruth(add.trueResidual, N);
    dim = size(dTrue,1);
    dAdd = orientEstimate(add.dnnResidual, dim, N);
    dFim = orientEstimate(fim.dnnResidual, dim, N);

    true3 = reshape(dTrue, dim, N, 1);
    errAdd = dAdd - true3;
    errFim = dFim - true3;
    errNormAdd = permute(vecnorm(errAdd,2,1), [2 3 1]);
    errNormFim = permute(vecnorm(errFim,2,1), [2 3 1]);
    meanErrAdd = mean(errNormAdd, 2, "omitnan");
    meanErrFim = mean(errNormFim, 2, "omitnan");

    meanAdd = mean(dAdd, 3, "omitnan");
    meanFim = mean(dFim, 3, "omitnan");
    trueNorm = vecnorm(dTrue,2,1).';
    meanNormAdd = mean(permute(vecnorm(dAdd,2,1),[2 3 1]),2,"omitnan");
    meanNormFim = mean(permute(vecnorm(dFim,2,1),[2 3 1]),2,"omitnan");
    denom = max(trueNorm, max(1e-12,1e-3*median(trueNorm,"omitnan")));

    plots = struct();
    plots.meanResidualErrorAdd = meanErrAdd;
    plots.meanResidualErrorFIM = meanErrFim;

    plots.truthEstimateFigure = figure("Name", ...
        "Step 09-J.6 residual truth and estimates", "Color", "w");
    tiledlayout(dim,1,"TileSpacing","compact","Padding","compact");
    for j = 1:dim
        nexttile;
        plot(t,dTrue(j,:),"k","LineWidth",1.5); hold on;
        plot(t,meanAdd(j,:),"LineWidth",1.1);
        plot(t,meanFim(j,:),"LineWidth",1.1);
        grid on;
        ylabel(sprintf("d_%d [m/s^2]",j));
        if j == 1
            title("True residual and watcher-mean operational estimates");
            legend("truth","additive",weightedLabel,"Location","best");
        end
    end
    xlabel("time [s]");

    plots.errorFigure = figure("Name", ...
        "Step 09-J.6 residual error", "Color", "w");
    tiledlayout(3,1,"TileSpacing","compact","Padding","compact");
    nexttile;
    plot(t,meanErrAdd,"LineWidth",1.2); hold on;
    plot(t,meanErrFim,"LineWidth",1.2); grid on;
    ylabel("mean ||dHat-dTrue|| [m/s^2]");
    title("Operational residual vector error, mean over watchers");
    legend("additive",weightedLabel,"Location","best");
    nexttile;
    plot(t,meanErrAdd./denom,"LineWidth",1.2); hold on;
    plot(t,meanErrFim./denom,"LineWidth",1.2); grid on;
    ylabel("relative error"); yline(1,"k:");
    legend("additive",weightedLabel,"Location","best");
    nexttile;
    plot(t,trueNorm,"k","LineWidth",1.4); hold on;
    plot(t,meanNormAdd,"LineWidth",1.1);
    plot(t,meanNormFim,"LineWidth",1.1); grid on;
    xlabel("time [s]"); ylabel("residual norm [m/s^2]");
    legend("truth","additive",weightedLabel,"Location","best");

    plots.adaptationFigure = [];
    if isfield(add,"gammaTheta") && isfield(add,"gammaEpsilon") && ...
            isfield(fim,"gammaTheta") && isfield(fim,"gammaEpsilon")
        plots.adaptationFigure = figure("Name", ...
            "Step 09-J.6 adaptive process noise", "Color", "w");
        tiledlayout(2,1,"TileSpacing","compact","Padding","compact");
        nexttile;
        plot(t,mean(add.gammaTheta,2,"omitnan"),"LineWidth",1.2); hold on;
        plot(t,mean(fim.gammaTheta,2,"omitnan"),"LineWidth",1.2); grid on;
        ylabel("gammaTheta"); legend("additive",weightedLabel,"Location","best");
        nexttile;
        plot(t,mean(add.gammaEpsilon,2,"omitnan"),"LineWidth",1.2); hold on;
        plot(t,mean(fim.gammaEpsilon,2,"omitnan"),"LineWidth",1.2); grid on;
        xlabel("time [s]"); ylabel("gammaEpsilon");
        legend("additive",weightedLabel,"Location","best");
    end

    % Kinematic-state consistency figures.  The error and 3-sigma curves
    % are kept watcher-specific; averaging covariances across watchers would
    % not be a statistically valid bound for the watcher-mean estimate.
    stateFields = ["etaTrue", "xhat", "PdiagEta"];
    if all(isfield(add,stateFields)) && all(isfield(fim,stateFields))
        [plots.stateAdditiveFigure, plots.stateSummaryAdditive] = ...
            plotKinematicStateCase(add, "additive");
        [plots.stateFIMFigure, plots.stateSummaryFIM] = ...
            plotKinematicStateCase(fim, weightedLabel);
    else
        plots.stateAdditiveFigure = [];
        plots.stateFIMFigure = [];
        plots.stateSummaryAdditive = table();
        plots.stateSummaryFIM = table();
        warning("Kinematic-state fields etaTrue/xhat/PdiagEta are missing.");
    end
end

function [fig,T] = plotKinematicStateCase(res,caseName)
    t = res.time(:);
    etaTrue = res.etaTrue;
    etaHat = res.xhat;
    Pdiag = res.PdiagEta;
    [nEta,N,Nw] = size(etaHat);

    assert(size(etaTrue,1)==nEta && size(etaTrue,2)==N, ...
        "etaTrue and xhat dimensions are inconsistent.");
    assert(isequal(size(Pdiag),size(etaHat)), ...
        "PdiagEta and xhat dimensions are inconsistent.");

    dim = nEta/2;
    assert(dim==round(dim),"Kinematic state must have [r;v] structure.");
    truth3 = reshape(etaTrue,nEta,N,1);
    stateError = etaHat-truth3;
    sigma = sqrt(max(Pdiag,0));
    colors = lines(Nw);

    rmse = squeeze(sqrt(mean(stateError.^2,2,"omitnan")));
    insideFraction = squeeze(mean(abs(stateError)<=3*sigma,2,"omitnan"));
    if Nw==1
        rmse = reshape(rmse,nEta,1);
        insideFraction = reshape(insideFraction,nEta,1);
    end

    labels = strings(nEta,1);
    units = strings(nEta,1);
    for j=1:nEta
        if j<=dim
            labels(j) = sprintf("r_%d",j);
            units(j) = "m";
        else
            labels(j) = sprintf("v_%d",j-dim);
            units(j) = "m/s";
        end
    end

    fig = figure("Name","Step 09-J.6 kinematic state - "+caseName, ...
        "Color","w");
    tiledlayout(nEta,2,"TileSpacing","compact","Padding","compact");
    for j=1:nEta
        nexttile;
        plot(t,etaTrue(j,:),"k","LineWidth",1.5); hold on;
        for iw=1:Nw
            plot(t,reshape(etaHat(j,:,iw),[],1), ...
                "Color",colors(iw,:),"LineWidth",0.8);
        end
        grid on; ylabel(labels(j)+" ["+units(j)+"]");
        if j==1
            legend(["truth",compose("watcher %d",1:Nw)],"Location","best");
            title("Truth and watcher estimates");
        end

        nexttile;
        for iw=1:Nw
            e = reshape(stateError(j,:,iw),[],1);
            b = 3*reshape(sigma(j,:,iw),[],1);
            plot(t,e,"Color",colors(iw,:),"LineWidth",0.8); hold on;
            plot(t,b,"--","Color",colors(iw,:),"LineWidth",0.65, ...
                "HandleVisibility","off");
            plot(t,-b,"--","Color",colors(iw,:),"LineWidth",0.65, ...
                "HandleVisibility","off");
        end
        yline(0,"k:","HandleVisibility","off"); grid on;
        ylabel("error ["+units(j)+"]");
        if j==1
            legend(compose("watcher %d error",1:Nw),"Location","best");
            title("Estimation error with watcher-specific +/-3 sigma");
        end
    end
    xlabel("time [s]");
    sgtitle("Target kinematic-state consistency: "+caseName);

    T = table(labels,units,mean(rmse,2,"omitnan"), ...
        mean(insideFraction,2,"omitnan"),min(insideFraction,[],2), ...
        'VariableNames',{'state','unit','meanWatcherRMSE', ...
        'meanFractionInside3Sigma','minimumWatcherFractionInside3Sigma'});
end

function X = orientTruth(X,N)
    if size(X,2) == N
        return;
    elseif size(X,1) == N
        X = X.';
    else
        error("trueResidual does not match the time dimension.");
    end
end

function X = orientEstimate(X,dim,N)
    if size(X,1) == dim && size(X,2) == N
        return;
    elseif size(X,1) == N && size(X,2) == dim
        X = permute(X,[2 1 3]);
    else
        error("dnnResidual does not match truth/time dimensions.");
    end
end

function label = getWeightedLabel(out)
    label="bearing-FIM-gated";
    if isfield(out,"cfgGSFIM") && isfield(out.cfgGSFIM,"gs") && ...
            isfield(out.cfgGSFIM.gs,"compositeMode") && ...
            string(out.cfgGSFIM.gs.compositeMode)== ...
            "output_information_fusion"
        label="output-information-fusion";
    elseif isfield(out,"cfgGSFIM") && isfield(out.cfgGSFIM,"gs") && ...
            isfield(out.cfgGSFIM.gs,"compositeMode") && ...
            string(out.cfgGSFIM.gs.compositeMode)== ...
            "fim_weighted_additive"
        label="FIM-weighted-additive";
    end
end
