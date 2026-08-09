function plotNIS(results, cfg)
%{
Function:
    plotNIS.m

Purpose:
    Plot the Normalized Innovation Squared (NIS) for each watcher.

    NIS is defined as

        NIS_k = nu_k' S_k^{-1} nu_k,

    where

        nu_k = z_k - zhat_k
        S_k  = H_k P_k^- H_k' + R_k

    For a consistent EKF, NIS should roughly behave like a chi-square
    random variable with nz degrees of freedom.

Inputs:
    results - simulation results structure
    cfg     - simulation configuration structure

Required fields:
    results.time
    results.NIS

Optional fields:
    results.measAvail
%}

    if ~isfield(results, "NIS")
        warning("results.NIS does not exist. Run simulateLocalDNNEKF with NIS logging first.");
        return;
    end

    time = results.time;
    Nw = cfg.Nw;

    figure;
    hold on;
    grid on;

    for i = 1:Nw

        nis_i = results.NIS(:,i);

        if isfield(results, "measAvail")
            avail_i = results.measAvail(:,i);
            nis_i(~avail_i) = NaN;
        end

        plot(time, nis_i, "LineWidth", 1.2, ...
            "DisplayName", "Watcher " + string(i));

    end

    % ---------------------------------------------------------------------
    % Chi-square 95% reference line
    % ---------------------------------------------------------------------
    if cfg.dim == 2
        % Bearing-only in 2D: one angular measurement.
        nz = 1;
        chi2_95 = 3.8415;
    elseif cfg.dim == 3
        % Bearing-only in 3D is usually azimuth/elevation or unit-vector
        % equivalent with two independent angular degrees of freedom.
        nz = 2;
        chi2_95 = 5.9915;
    else
        nz = NaN;
        chi2_95 = NaN;
    end

    if ~isnan(chi2_95)
        yline(chi2_95, "--", ...
            "chi^2 95%, dof = " + string(nz), ...
            "LineWidth", 1.2, ...
            "DisplayName", "95% reference");
    end

    xlabel("Time [s]");
    ylabel("NIS");
    title("Normalized Innovation Squared");
    legend("Location", "best");

end