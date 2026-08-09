function plotCovMatchingDiagnostics(results, cfg)
%{
Function:
    plotCovMatchingDiagnostics.m

Purpose:
    Plot adaptive covariance matching diagnostics for the Step 03 local
    DNN-EKF.

    This function visualizes the adaptive process-noise multiplier

        Q_theta,k = gammaTheta_i,k * Q_theta,base

    and the innovation covariance matching ratio

        ratio_i,k = trace(S_hat_i,k) / trace(S_model_i,k).

    The adaptation follows the same basic idea as the covariance matching
    code in the single DNN-MEKF script: estimate empirical innovation
    covariance with EWMA, compare it with the model innovation covariance,
    and multiplicatively update a Q gain.

Inputs:
    results - simulation results structure
    cfg     - simulation configuration structure

Required fields:
    results.time
    results.gammaTheta
    results.cmRatio

Optional fields:
    results.cmTraceEmp
    results.cmTraceModel
%}

    if ~isfield(results, "gammaTheta")
        warning("results.gammaTheta does not exist. Run simulateLocalDNNEKF with covariance matching logging first.");
        return;
    end

    if ~isfield(results, "cmRatio")
        warning("results.cmRatio does not exist. Run simulateLocalDNNEKF with covariance matching logging first.");
        return;
    end

    time = results.time(:);
    Nw = cfg.Nw;

    % ---------------------------------------------------------------------
    % Figure 1: gammaTheta
    % ---------------------------------------------------------------------
    figure;
    hold on;
    grid on;

    for i = 1:Nw
        gamma_i = results.gammaTheta(:,i);
        plot(time, gamma_i, "LineWidth", 1.3, ...
            "DisplayName", "Watcher " + string(i));
    end

    xlabel("Time [s]");
    ylabel("\gamma_\theta");
    title("Adaptive DNN Parameter Process-Noise Scale");
    legend("Location", "best");

    % ---------------------------------------------------------------------
    % Figure 2: covariance matching ratio
    % ---------------------------------------------------------------------
    figure;
    hold on;
    grid on;

    for i = 1:Nw
        ratio_i = results.cmRatio(:,i);
        plot(time, ratio_i, "LineWidth", 1.3, ...
            "DisplayName", "Watcher " + string(i));
    end

    yline(1.0, "--", "matched", "LineWidth", 1.2, ...
        "DisplayName", "matched");

    xlabel("Time [s]");
    ylabel("trace(hat{S}) / trace(S)");
    title("Innovation Covariance Matching Ratio");
    legend("Location", "best");

    % ---------------------------------------------------------------------
    % Figure 3: empirical vs model innovation covariance trace
    % ---------------------------------------------------------------------
    if isfield(results, "cmTraceEmp") && isfield(results, "cmTraceModel")

        figure;
        tiledlayout(Nw, 1);

        for i = 1:Nw

            nexttile;
            hold on;
            grid on;

            traceEmp_i = results.cmTraceEmp(:,i);
            traceModel_i = results.cmTraceModel(:,i);

            plot(time, traceEmp_i, "LineWidth", 1.2, ...
                "DisplayName", "trace(\hat{S})");

            plot(time, traceModel_i, "--", "LineWidth", 1.2, ...
                "DisplayName", "trace(S)");

            xlabel("Time [s]");
            ylabel("Innovation covariance trace");
            title("Watcher " + string(i));
            legend("Location", "best");

        end

    end

    % ---------------------------------------------------------------------
    % Print final values
    % ---------------------------------------------------------------------
    fprintf("\n=== Covariance Matching Diagnostics ===\n");
    fprintf("%10s %16s %16s %16s %16s\n", ...
        "Watcher", "gammaFinal", "ratioFinal", "traceEmp", "traceModel");

    for i = 1:Nw

        gammaFinal = lastFinite(results.gammaTheta(:,i));
        ratioFinal = lastFinite(results.cmRatio(:,i));

        if isfield(results, "cmTraceEmp")
            traceEmpFinal = lastFinite(results.cmTraceEmp(:,i));
        else
            traceEmpFinal = NaN;
        end

        if isfield(results, "cmTraceModel")
            traceModelFinal = lastFinite(results.cmTraceModel(:,i));
        else
            traceModelFinal = NaN;
        end

        fprintf("%10d %16.6e %16.6e %16.6e %16.6e\n", ...
            i, gammaFinal, ratioFinal, traceEmpFinal, traceModelFinal);

    end

    fprintf("=======================================\n\n");

end

function value = lastFinite(x)
% Return the last finite value in a vector.

    x = x(:);
    idx = find(isfinite(x), 1, "last");

    if isempty(idx)
        value = NaN;
    else
        value = x(idx);
    end

end