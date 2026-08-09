function report = analyzeStep10C1CovarianceMatching(out,makePlots)
%ANALYZESTEP10C1COVARIANCEMATCHING Assess adaptive Qtheta/Qepsilon behavior.
%
% This is a passive post-run diagnostic.  A healthy matcher should have a
% finite trace ratio near one after burn-in, should not persist at gamma
% clamps, and should not show a systematically inconsistent NIS trend.

    if nargin < 2 || isempty(makePlots), makePlots = true; end
    assert(isfield(out,"resOutputInformation") && ...
        isfield(out,"cfgOutputInformation"), ...
        "Input must be a Step 10-C.1 output structure.");
    res = out.resOutputInformation;
    cfg = out.cfgOutputInformation;
    required = ["gammaTheta","gammaEpsilon","cmRatio", ...
        "cmTraceEmp","cmTraceModel","NIS","time"];
    assert(all(isfield(res,required)), ...
        "Result lacks covariance-matching diagnostics.");

    N = numel(res.time);
    burnIn = getField(cfg.dnn,"cmBurnInMeas",0);
    % With always-available measurements, sample index is a conservative
    % approximation to available-measurement count.  The exact logs are
    % still plotted rather than silently discarded.
    first = min(N,max(1,round(burnIn)+1));
    idx = first:N;
    ratio = res.cmRatio(idx,:);
    gammaTheta = res.gammaTheta(idx,:);
    gammaEpsilon = res.gammaEpsilon(idx,:);
    nis = res.NIS(idx,:);
    validRatio = ratio(isfinite(ratio) & ratio > 0);

    ratioMedian = median(validRatio,"omitnan");
    ratioLogRMSE = sqrt(mean(log(validRatio).^2,"omitnan"));
    gammaThetaMin = getField(cfg.dnn,"gammaThetaMin",NaN);
    gammaThetaMax = getField(cfg.dnn,"gammaThetaMax",NaN);
    gammaEpsilonMin = getField(cfg.dnn,"gammaEpsilonMin",NaN);
    gammaEpsilonMax = getField(cfg.dnn,"gammaEpsilonMax",NaN);
    thetaAtBound = boundFraction(gammaTheta,gammaThetaMin,gammaThetaMax);
    epsilonAtBound = boundFraction(gammaEpsilon,gammaEpsilonMin,gammaEpsilonMax);

    report = struct();
    report.startIndex = first;
    report.startTime = res.time(first);
    report.ratioMedian = ratioMedian;
    report.ratioLogRMSE = ratioLogRMSE;
    report.ratioFiniteFraction = nnz(isfinite(ratio))/numel(ratio);
    report.gammaThetaAtBoundFraction = thetaAtBound;
    report.gammaEpsilonAtBoundFraction = epsilonAtBound;
    report.gammaThetaFinal = res.gammaTheta(end,:);
    report.gammaEpsilonFinal = res.gammaEpsilon(end,:);
    report.meanNISPostBurnIn = mean(nis(:),"omitnan");
    report.meanTraceEmpPostBurnIn = mean(res.cmTraceEmp(idx,:),"all","omitnan");
    report.meanTraceModelPostBurnIn = mean(res.cmTraceModel(idx,:),"all","omitnan");

    fprintf("Adaptive covariance-matching diagnostic (t >= %.1f s)\n", ...
        report.startTime);
    fprintf("  median trace ratio: %.3g (target 1)\n",report.ratioMedian);
    fprintf("  log-ratio RMSE: %.3g (target 0)\n",report.ratioLogRMSE);
    fprintf("  gammaTheta at bounds: %.1f %%\n",100*thetaAtBound);
    fprintf("  gammaEpsilon at bounds: %.1f %%\n",100*epsilonAtBound);
    fprintf("  post-burn-in mean NIS: %.3g\n",report.meanNISPostBurnIn);

    if makePlots
        f = figure('Name','Step 10-C.1 adaptive covariance matching');
        tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
        nexttile;
        semilogy(res.time,res.cmRatio,'LineWidth',1.0); hold on;
        yline(1,'k--','target'); grid on;
        title('Innovation trace ratio'); xlabel('time [s]');
        ylabel('trace(S_{emp}) / trace(S_{model})');

        nexttile;
        semilogy(res.time,res.gammaTheta,'LineWidth',1.0); hold on;
        yline(gammaThetaMin,'k:'); yline(gammaThetaMax,'k:'); grid on;
        title('Adaptive parameter-noise multiplier'); xlabel('time [s]');
        ylabel('\gamma_\theta');

        nexttile;
        semilogy(res.time,res.gammaEpsilon,'LineWidth',1.0); hold on;
        yline(gammaEpsilonMin,'k:'); yline(gammaEpsilonMax,'k:'); grid on;
        title('Adaptive residual-noise multiplier'); xlabel('time [s]');
        ylabel('\gamma_\epsilon');

        nexttile;
        plot(res.time,res.NIS,'LineWidth',0.85); grid on;
        title('Bearing NIS'); xlabel('time [s]'); ylabel('NIS');
        report.figure = f;
    end
end

function value = getField(s,name,defaultValue)
    if isfield(s,name), value = double(s.(name)); else, value = defaultValue; end
end

function value = boundFraction(x,lower,upper)
    valid = isfinite(x);
    if ~any(valid(:)) || ~isfinite(lower) || ~isfinite(upper)
        value = NaN;
        return;
    end
    tolerance = 1e-8*max(1,abs([lower,upper]));
    atBound = abs(x-lower) <= tolerance(1) | abs(x-upper) <= tolerance(2);
    value = nnz(atBound & valid)/nnz(valid);
end
