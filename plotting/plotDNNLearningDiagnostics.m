function dnnStats = plotDNNLearningDiagnostics(results, cfg)
%{
Function:
    plotDNNLearningDiagnostics.m

Purpose:
    Plot basic learning diagnostics for Step 03 local DNN-EKF.

    This function checks whether the local DNN-EKF is actually learning by
    plotting:

        1. ||theta_i|| for each watcher.
        2. Estimated DNN residual acceleration dHat_i(t).
        3. True residual acceleration dTrue(t), if available.
        4. RMSE between dHat_i(t) and dTrue(t), for diagnostic purposes.

    Important:
        In the current local-only architecture, each watcher uses only its
        own local branch

            dHat_i(eta) = W_i phi_i(eta).

        If the truth residual is a composite branchwise model,

            dTrue(eta) = sum_j d_j(eta),

        then dHat_i is not guaranteed to match the full dTrue. Therefore,
        the residual RMSE printed here should be interpreted as a diagnostic,
        not necessarily as the final learning objective.

Inputs:
    results - simulation results structure
    cfg     - simulation configuration structure

Outputs:
    dnnStats - diagnostic statistics structure
%}

    if ~isfield(results, "thetaHat")
        error("results.thetaHat does not exist. Run simulateLocalDNNEKF first.");
    end

    if ~isfield(results, "dnnResidual")
        error("results.dnnResidual does not exist. Run simulateLocalDNNEKF with DNN residual logging.");
    end

    time = results.time(:);
    N = numel(time);
    Nw = cfg.Nw;
    dim = cfg.dim;

    thetaNorm = zeros(N, Nw);

    for i = 1:Nw
        theta_i = squeeze(results.thetaHat(:,:,i));   % nTheta x N
        thetaNorm(:,i) = sqrt(sum(theta_i.^2, 1)).';
    end

    % ---------------------------------------------------------------------
    % Figure 1: theta norm
    % ---------------------------------------------------------------------
    figure;
    hold on;
    grid on;

    for i = 1:Nw
        plot(time, thetaNorm(:,i), "LineWidth", 1.2, ...
            "DisplayName", "Watcher " + string(i));
    end

    xlabel("Time [s]");
    ylabel("||theta_i||");
    title("Local DNN-EKF Parameter Norm");
    legend("Location", "best");

    % ---------------------------------------------------------------------
    % Figure 2: DNN residual estimate versus true residual
    % ---------------------------------------------------------------------
    figure;
    tiledlayout(dim,1);

    residualRMSE = NaN(Nw,1);

    for ell = 1:dim

        nexttile;
        hold on;
        grid on;

        if isfield(results, "trueResidual")
            dTrue_ell = results.trueResidual(ell,:).';
            plot(time, dTrue_ell, "k--", "LineWidth", 1.5, ...
                "DisplayName", "True residual");
        else
            dTrue_ell = NaN(N,1);
        end

        for i = 1:Nw
            dHat_i = squeeze(results.dnnResidual(ell,:,i)).';
            plot(time, dHat_i, "LineWidth", 1.1, ...
                "DisplayName", "Watcher " + string(i));
        end

        xlabel("Time [s]");
        ylabel("a_{res," + string(ell) + "}");
        title("Residual Acceleration Component " + string(ell));
        legend("Location", "best");

    end

    % ---------------------------------------------------------------------
    % Residual RMSE diagnostic
    % ---------------------------------------------------------------------
    if isfield(results, "trueResidual")

        dTrue = results.trueResidual;   % dim x N

        for i = 1:Nw
            dHat_i = squeeze(results.dnnResidual(:,:,i));  % dim x N
            err_i = dHat_i - dTrue;
            residualRMSE(i) = sqrt(mean(sum(err_i.^2, 1)));
        end

        fprintf("\n=== DNN Residual Learning Diagnostics ===\n");
        fprintf("%10s %16s %16s\n", ...
            "Watcher", "ThetaNormFinal", "Residual RMSE");

        for i = 1:Nw
            fprintf("%10d %16.6e %16.6e\n", ...
                i, thetaNorm(end,i), residualRMSE(i));
        end

        fprintf("=========================================\n\n");

    else

        fprintf("\n=== DNN Residual Learning Diagnostics ===\n");
        fprintf("results.trueResidual does not exist, so residual RMSE was not computed.\n");
        fprintf("=========================================\n\n");

    end

    % ---------------------------------------------------------------------
    % Output statistics
    % ---------------------------------------------------------------------
    dnnStats.thetaNorm = thetaNorm;
    dnnStats.thetaNormFinal = thetaNorm(end,:).';
    dnnStats.residualRMSE = residualRMSE;

end