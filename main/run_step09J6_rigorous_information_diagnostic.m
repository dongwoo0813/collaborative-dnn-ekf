function diagInfo = run_step09J6_rigorous_information_diagnostic( ...
    out, windowSeconds, makePlots)
%RUN_STEP09J6_RIGOROUS_INFORMATION_DIAGNOSTIC Model-based FIM diagnostics.
%
% This function deliberately does NOT form a residual mean gate.  The
% current DNN architecture defines the GS residual as a sum of branch
% components.  Multiplying those components by matrices whose sum is I is
% not an unbiased fusion rule for that model.
%
% Instead, this diagnostic does two things using an already completed run:
%
% 1. Test the two competing branch semantics directly:
%       component model : d = sum_j d_j
%       expert model    : d = mean_j d_j
%
% 2. Compute a finite-horizon, measurement-likelihood FIM for a locally
%    constant unknown residual acceleration.  Initial position and velocity
%    at the beginning of each window are nuisance parameters and are removed
%    with a Schur complement.
%
% For sample l in a window beginning at t0,
%
%   r(t_l) = r0 + tau_l v0 + 0.5 tau_l^2 d,
%   J_l    = H_r,l [I, tau_l I, 0.5 tau_l^2 I],
%   I      = sum_l J_l' R_l^{-1} J_l.
%
% Partitioning I with x0=[r0;v0] and d gives
%
%   I_d|x = I_dd - I_dx pinv(I_xx) I_xd.
%
% No exponential moving average, arbitrary forgetting factor, branch-count
% multiplier, or output blending is used.
%
% Usage after the 20-m experiment:
%   info20 = run_step09J6_rigorous_information_diagnostic(out20,50,true);

    if nargin < 2 || isempty(windowSeconds)
        windowSeconds = 50;
    end
    if nargin < 3
        makePlots = true;
    end
    validateattributes(windowSeconds,{'numeric'}, ...
        {'scalar','real','finite','positive'},mfilename,'windowSeconds');

    [resAdd,cfgAdd] = getAdditiveCase(out);
    branchSummary = evaluateBranchSemantics(resAdd,cfgAdd);

    caseNames = "additive";
    results = {resAdd};
    configs = {cfgAdd};
    if isstruct(out) && isfield(out,'resGSFIM') && ...
            isfield(out,'cfgGSFIM')
        caseNames(end+1,1) = "legacy bearing-FIM-gated";
        results{end+1,1} = out.resGSFIM;
        configs{end+1,1} = out.cfgGSFIM;
    end

    fim = cell(numel(results),1);
    fimSummary = table();
    for ic = 1:numel(results)
        fim{ic} = finiteHorizonAccelerationFIM( ...
            results{ic},configs{ic},windowSeconds);
        one = summarizeFIM(fim{ic},caseNames(ic));
        fimSummary = [fimSummary;one]; %#ok<AGROW>
    end

    fprintf('\nStep 09-J.6 rigorous information diagnostic\n');
    fprintf(['branch semantics are evaluated from saved outputs; ', ...
        'no estimator is rerun.\n']);
    fprintf(['finite-horizon model: locally constant residual acceleration, ', ...
        'window = %.1f s\n\n'],windowSeconds);
    disp(branchSummary);
    disp(fimSummary);

    figures = gobjects(0);
    if makePlots
        figures = makeDiagnosticFigures(branchSummary,fim,fimSummary, ...
            caseNames,windowSeconds);
    end

    diagInfo = struct('branchSummary',branchSummary, ...
        'fimSummary',fimSummary,'fim',{fim}, ...
        'caseNames',caseNames,'windowSeconds',windowSeconds, ...
        'figures',figures, ...
        'interpretation',[ ...
        "Use additive composition if the sum row is best.", ...
        "BLUE/information mean fusion is admissible only if the expert-mean model is supported.", ...
        "The finite-window FIM is a quality/observability diagnostic, not a mean gate."]);
end

function [res,cfg] = getAdditiveCase(out)
    if isfield(out,'resGSAdd')
        res = out.resGSAdd;
    else
        error('Input must contain out.resGSAdd.');
    end
    if isfield(out,'cfgGSAdd')
        cfg = out.cfgGSAdd;
    else
        error('Input must contain out.cfgGSAdd.');
    end
end

function summary = evaluateBranchSemantics(res,cfg)
    C = res.dnnResidualBranchContrib;
    if ndims(C) ~= 4
        error('Expected dnnResidualBranchContrib to be dim-by-Nw-by-N-by-Nw.');
    end
    [dim,Nw,N,Nreceiver] = size(C);
    dTrue = repmat(reshape(res.trueResidual,dim,1,N,1),1,1,1,Nreceiver);

    active = squeeze(any(abs(C)>0,1)); % Nw-by-N-by-Nreceiver
    nActive = sum(active,1);
    dSum = sum(C,2);
    dMean = dSum ./ reshape(max(nActive,1),1,1,N,Nreceiver);

    labels = ["component sum";"equal expert mean"; ...
        "branch "+string((1:Nw).')];
    nRows = numel(labels);
    vectorRMSE = NaN(nRows,1);
    meanCosine = NaN(nRows,1);
    meanNormRatio = NaN(nRows,1);

    candidates = cell(nRows,1);
    candidates{1} = dSum;
    candidates{2} = dMean;
    for j = 1:Nw
        candidates{j+2} = C(:,j,:,:);
    end

    trueNorm = sqrt(sum(dTrue.^2,1));
    floorValue = max(1e-12,1e-3*median(trueNorm(:),'omitnan'));
    for q = 1:nRows
        dhat = candidates{q};
        err = sqrt(sum((dhat-dTrue).^2,1));
        vectorRMSE(q) = sqrt(mean(err(:).^2,'omitnan'));
        estNorm = sqrt(sum(dhat.^2,1));
        dotValue = sum(dhat.*dTrue,1);
        valid = trueNorm>floorValue & estNorm>floorValue;
        c = dotValue(valid)./(trueNorm(valid).*estNorm(valid));
        meanCosine(q) = mean(c,'omitnan');
        validRatio = trueNorm>floorValue;
        ratio = estNorm(validRatio)./trueNorm(validRatio);
        meanNormRatio(q) = mean(ratio,'omitnan');
    end
    summary = table(labels,vectorRMSE,meanCosine,meanNormRatio, ...
        'VariableNames',{'compositionHypothesis','residualVectorRMSE', ...
        'meanCosine','meanNormRatio'});
    summary.Properties.UserData.branchModel = string(cfg.dnn.branchModel);
end

function out = finiteHorizonAccelerationFIM(res,cfg,windowSeconds)
    t = res.time(:).';
    N = numel(t);
    dim = cfg.dim;
    Nw = size(res.xhat,3);
    if dim ~= 2 && dim ~= 3
        error('Only 2-D and 3-D bearing models are supported.');
    end

    % Evaluate at at most 2 Hz.  This changes only diagnostic output times;
    % every measurement inside each batch window still contributes to FIM.
    dt = median(diff(t));
    stride = max(1,round(0.5/dt));
    evalIndex = unique([1:stride:N,N]);
    Ne = numel(evalIndex);
    minEig = NaN(Ne,Nw);
    maxEig = NaN(Ne,Nw);
    conditionNumber = Inf(Ne,Nw);
    worstSigma = Inf(Ne,Nw);
    effectiveRank = zeros(Ne,Nw);
    infoMatrix = NaN(dim,dim,Ne,Nw);

    R = bearingNoiseCovariance(cfg);
    Rinv = pinv(R);
    for iw = 1:Nw
        for ie = 1:Ne
            k = evalIndex(ie);
            idx0 = find(t >= t(k)-windowSeconds,1,'first');
            validIndex = idx0:k;
            if isfield(res,'measAvail') && ~isempty(res.measAvail)
                avail = squeeze(res.measAvail(validIndex,iw));
                validIndex = validIndex(logical(avail(:).'));
            end
            if numel(validIndex) < 2*dim+1
                continue;
            end
            t0 = t(validIndex(1));
            F = zeros(3*dim);
            for l = validIndex
                eta = res.xhat(:,l,iw);
                watcherState = struct('r',res.watcherR(:,l,iw));
                H = measurementJacobian(eta,watcherState,cfg);
                Hbearing = selectBearingRows(H,cfg);
                Hr = Hbearing(:,1:dim);
                tau = t(l)-t0;
                J = [Hr,tau*Hr,0.5*tau^2*Hr];
                F = F + J.'*Rinv*J;
            end
            F = 0.5*(F+F.');
            ix = 1:2*dim;
            ia = 2*dim+(1:dim);
            Fxx = F(ix,ix);
            Facc = F(ia,ia)-F(ia,ix)*pinv(Fxx)*F(ix,ia);
            Facc = projectPSD(Facc);
            e = sort(real(eig(Facc)),'ascend');
            infoMatrix(:,:,ie,iw) = Facc;
            minEig(ie,iw) = e(1);
            maxEig(ie,iw) = e(end);
            tol = max(e(end),1)*1e-10;
            effectiveRank(ie,iw) = sum(e>tol);
            if e(1)>tol
                conditionNumber(ie,iw) = e(end)/e(1);
                worstSigma(ie,iw) = 1/sqrt(e(1));
            end
        end
    end
    out = struct('time',t(evalIndex),'evalIndex',evalIndex, ...
        'windowSeconds',windowSeconds,'informationMatrix',infoMatrix, ...
        'minEig',minEig,'maxEig',maxEig, ...
        'conditionNumber',conditionNumber,'worstAccelerationSigma',worstSigma, ...
        'effectiveRank',effectiveRank);
end

function R = bearingNoiseCovariance(cfg)
    nz = max(cfg.dim-1,1);
    if isscalar(cfg.meas.R)
        R = cfg.meas.R*eye(nz);
    else
        if string(cfg.meas.type)=="range_bearing"
            R = cfg.meas.R(2:end,2:end);
        else
            R = cfg.meas.R;
        end
    end
end

function Hb = selectBearingRows(H,cfg)
    if string(cfg.meas.type)=="range_bearing"
        Hb = H(2:end,:);
    else
        Hb = H;
    end
end

function A = projectPSD(A)
    A = 0.5*(A+A.');
    [V,D] = eig(A);
    d = real(diag(D));
    scale = max(max(abs(d)),1);
    d(d<scale*1e-12) = 0;
    A = V*diag(d)*V.';
    A = 0.5*(A+A.');
end

function T = summarizeFIM(fim,caseName)
    fullRankFraction = mean(fim.effectiveRank(:)==size(fim.informationMatrix,1), ...
        'omitnan');
    finiteSigma = fim.worstAccelerationSigma(isfinite(fim.worstAccelerationSigma));
    if isempty(finiteSigma)
        medianWorstAccelerationSigma = Inf;
        finalWorstAccelerationSigma = Inf;
    else
        medianWorstAccelerationSigma = median(finiteSigma,'omitnan');
        finalWorstAccelerationSigma = max(fim.worstAccelerationSigma(end,:));
    end
    finiteCond = fim.conditionNumber(isfinite(fim.conditionNumber));
    if isempty(finiteCond)
        medianConditionNumber = Inf;
    else
        medianConditionNumber = median(finiteCond,'omitnan');
    end
    T = table(caseName,fullRankFraction,medianWorstAccelerationSigma, ...
        finalWorstAccelerationSigma,medianConditionNumber);
end

function figs = makeDiagnosticFigures(branchSummary,fim,fimSummary, ...
        caseNames,windowSeconds)
    f1 = figure('Name','Rigorous branch-semantics diagnostic');
    tiledlayout(1,3,'TileSpacing','compact');
    nexttile; bar(categorical(branchSummary.compositionHypothesis), ...
        branchSummary.residualVectorRMSE); grid on;
    ylabel('RMSE [m/s^2]'); title('Residual-vector error');
    nexttile; bar(categorical(branchSummary.compositionHypothesis), ...
        branchSummary.meanCosine); hold on; yline(1,'k:'); grid on;
    ylabel('cosine'); title('Direction agreement');
    nexttile; bar(categorical(branchSummary.compositionHypothesis), ...
        branchSummary.meanNormRatio); hold on; yline(1,'k:'); grid on;
    ylabel('norm ratio'); title('Magnitude agreement');

    f2 = figure('Name','Finite-horizon residual-acceleration FIM');
    tiledlayout(2,2,'TileSpacing','compact');
    colors = lines(numel(fim));
    nexttile; hold on; grid on;
    for i=1:numel(fim)
        semilogy(fim{i}.time,medianPositive(fim{i}.minEig), ...
            'LineWidth',1.2,'Color',colors(i,:));
    end
    ylabel('median \lambda_{min}(I_{d|x})');
    title(sprintf('Acceleration information, %.1f-s batch',windowSeconds));
    legend(caseNames,'Location','best');
    nexttile; hold on; grid on;
    for i=1:numel(fim)
        semilogy(fim{i}.time,medianFinite(fim{i}.conditionNumber), ...
            'LineWidth',1.2,'Color',colors(i,:));
    end
    ylabel('median condition number'); title('Information conditioning');
    nexttile; hold on; grid on;
    for i=1:numel(fim)
        semilogy(fim{i}.time,medianFinite(fim{i}.worstAccelerationSigma), ...
            'LineWidth',1.2,'Color',colors(i,:));
    end
    xlabel('time [s]'); ylabel('worst-axis sigma [m/s^2]');
    title('Measurement-only acceleration CRLB');
    nexttile;
    bar(categorical(fimSummary.caseName),fimSummary.fullRankFraction);
    ylim([0 1]); grid on; ylabel('fraction');
    title('Full-rank window fraction');
    figs = [f1;f2];
end

function y = medianPositive(x)
    x(~isfinite(x) | x<=0) = NaN;
    y = median(x,2,'omitnan');
end

function y = medianFinite(x)
    x(~isfinite(x)) = NaN;
    y = median(x,2,'omitnan');
end
