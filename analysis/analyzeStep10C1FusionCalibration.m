function diag = analyzeStep10C1FusionCalibration(out,makePlots)
%ANALYZESTEP10C1FUSIONCALIBRATION Diagnose GS output-fusion assumptions.
%
% Uses the replay result at a common truth-state input to reconstruct each
% watcher's raw local full-residual expert.  It reports empirical expert
% error, pairwise error correlation, and parameter-covariance traces.  The
% covariance trace is not a calibrated output covariance; it is shown only
% to reveal potential EKF overconfidence.  This analysis never feeds truth
% back into a controller or estimator.

    if nargin < 2 || isempty(makePlots), makePlots = true; end
    assert(isfield(out,"resOutputInformation") && ...
        isfield(out,"cfgOutputInformation"), ...
        "Input must be a Step 10-C.1 output structure.");
    res = out.resOutputInformation;
    cfg = out.cfgOutputInformation;
    dim = cfg.dim; Nw = cfg.Nw; N = numel(res.time);
    assert(isfield(res,"thetaHat") && isfield(res,"dnnResidualAtTrueEta"), ...
        "Output-information result lacks theta/residual diagnostic logs.");

    raw = zeros(dim,N,Nw);
    for j = 1:Nw
        for k = 1:N
            raw(:,k,j) = evaluateBranchResidualModel(j, ...
                res.etaTrue(:,k),res.thetaHat(:,k,j),cfg);
        end
    end
    dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
    errorRaw = raw-dTrue;
    errorNorm = squeeze(vecnorm(errorRaw,2,1));
    if Nw == 1, errorNorm = errorNorm(:); end
    rawRMSE = sqrt(mean(errorNorm.^2,1,"omitnan")).';

    % Correlate the signed per-component error time histories.  High
    % positive correlations violate the independent-expert assumption used
    % by precision addition in output-information fusion.
    corrByComponent = NaN(Nw,Nw,dim);
    for q = 1:dim
        E = squeeze(errorRaw(q,:,:));
        if Nw == 1, E = E(:); end
        corrByComponent(:,:,q) = corrcoef(E,"Rows","complete");
    end
    meanErrorCorrelation = mean(corrByComponent,3,"omitnan");
    meanOffDiagonalCorrelation = mean( ...
        meanErrorCorrelation(~eye(Nw)),"omitnan");

    fusedError = res.dnnResidualAtTrueEta-dTrue;
    fusedErrorNorm = squeeze(vecnorm(fusedError,2,1));
    fusedRMSE = sqrt(mean(fusedErrorNorm(:).^2,"omitnan"));
    if isfield(res,"tracePtheta")
        tracePtheta = res.tracePtheta;
    else
        tracePtheta = squeeze(sum(res.PdiagTheta,1));
    end

    diag = struct();
    diag.time = res.time(:);
    diag.rawExpertResidualAtTrueEta = raw;
    diag.rawExpertError = errorRaw;
    diag.rawExpertErrorNorm = errorNorm;
    diag.rawExpertRMSE = rawRMSE;
    diag.fusedResidualAtTrueEta = res.dnnResidualAtTrueEta;
    diag.fusedRMSE = fusedRMSE;
    diag.errorCorrelationByComponent = corrByComponent;
    diag.meanErrorCorrelation = meanErrorCorrelation;
    diag.meanOffDiagonalErrorCorrelation = meanOffDiagonalCorrelation;
    diag.tracePtheta = tracePtheta;

    fprintf("Output-fusion empirical diagnostic\n");
    fprintf("  raw expert RMSE [m/s^2]: %s\n",num2str(rawRMSE.',"%.3e "));
    fprintf("  fused RMSE [m/s^2]: %.3e\n",fusedRMSE);
    fprintf("  mean off-diagonal expert-error correlation: %.3f\n", ...
        meanOffDiagonalCorrelation);

    if makePlots
        f = figure('Name','Step 10-C.1 output-fusion calibration diagnostic');
        tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
        nexttile;
        plot(res.time,errorNorm,'LineWidth',1.0); grid on;
        title('Raw local expert error at true state');
        xlabel('time [s]'); ylabel('||d_j-d_{true}|| [m/s^2]');
        legend(compose('expert %d',1:Nw),'Location','best');

        nexttile;
        bar([rawRMSE;fusedRMSE]); grid on;
        xticklabels([compose('expert %d',1:Nw),"fused"]);
        xtickangle(25); ylabel('RMSE [m/s^2]');
        title('Raw-expert versus fused residual error');

        nexttile;
        imagesc(meanErrorCorrelation,[-1 1]); axis image; colorbar;
        xticks(1:Nw); yticks(1:Nw);
        xlabel('expert j'); ylabel('expert i');
        title('Mean signed-error correlation');

        nexttile;
        semilogy(res.time,max(tracePtheta,realmin),'LineWidth',1.0); grid on;
        xlabel('time [s]'); ylabel('trace(P_{theta})');
        title('EKF parameter covariance trace (not actual error)');
        legend(compose('expert %d',1:Nw),'Location','best');
        diag.figure = f;
    end
end
