function plotTrueResidual(results, cfg)
%{
Function:
    plotTrueResidual.m

Purpose:
    Plot the true unknown residual acceleration used in the target truth
    dynamics.

    This function is mainly used in Step 02 to verify that the truth model
    contains a nonzero residual acceleration

        d_unk(eta,t),

    while the physical EKF prediction model does not know this residual.

Inputs:
    results - Simulation result structure.
              Required fields:
                  results.time         - time vector, size 1 x N or N x 1
                  results.trueResidual - true residual acceleration log,
                                         size cfg.dim x N

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.dim

Outputs:
    None.
    This function creates MATLAB figures.

Main equations:
    The target truth dynamics are

        dot r_t = v_t,

        dot v_t = a_nom(r_t,v_t,t) + d_unk(eta,t).

    This function plots

        d_unk(eta(t),t).

Notes:
    - In Step 01, this plot should be identically zero.
    - In Step 02, this plot should be nonzero if
      cfg.truth.useResidual = true.
    - The physical EKF prediction model should not use this residual.
%}

    if ~isfield(results, "trueResidual")
        warning("results.trueResidual is missing. Skipping true residual plot.");
        return;
    end

    time = results.time(:);
    dim = cfg.dim;
    trueResidual = results.trueResidual;

    N = numel(time);

    if size(trueResidual,1) ~= dim || size(trueResidual,2) ~= N
        error("results.trueResidual must have size cfg.dim x N.");
    end

    figure;
    tiledlayout(dim,1);

    for d = 1:dim
        nexttile;
        hold on; grid on;

        plot(time, trueResidual(d,:), "LineWidth", 1.2);

        xlabel("Time [s]");
        ylabel("Residual acceleration");
        title("True Residual Acceleration Component " + string(d));
    end

    residualNorm = zeros(N,1);

    for k = 1:N
        residualNorm(k) = norm(trueResidual(:,k));
    end

    figure;
    hold on; grid on;

    plot(time, residualNorm, "LineWidth", 1.5);

    xlabel("Time [s]");
    ylabel("Residual acceleration norm");
    title("True Residual Acceleration Norm");

end