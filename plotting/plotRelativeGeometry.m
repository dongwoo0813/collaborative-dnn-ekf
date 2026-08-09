function plotRelativeGeometry(results, cfg)
%{
Function:
    plotRelativeGeometry.m

Purpose:
    Plot relative target-watcher geometry for all watchers.

    This function visualizes:
        1. Relative range ||rho_i(t)||.
        2. Bearing angle for cfg.dim = 2.
        3. Azimuth/elevation for cfg.dim = 3.

Inputs:
    results - Simulation result structure.
              Required fields:
                  results.time
                  results.etaTrue
                  results.watcherR

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.dim
                  cfg.Nw

Outputs:
    None.
    This function creates MATLAB figures.

Main equations:
    Relative position:

        rho_i(t) = r_t(t) - r_{w,i}(t).

    Relative range:

        rho_norm_i(t) = ||rho_i(t)||.

    For cfg.dim = 2, line-of-sight bearing:

        beta_i(t) = atan2(rho_{i,y}, rho_{i,x}).

    For cfg.dim = 3:

        az_i(t) = atan2(rho_{i,y}, rho_{i,x}),

        el_i(t) = atan2(rho_{i,z}, sqrt(rho_{i,x}^2 + rho_{i,y}^2)).

Notes:
    - This is useful for diagnosing bearing-only observability.
    - If the relative geometry barely changes, bearing-only estimation can
      become weak, especially in the radial direction.
%}

    dim = cfg.dim;
    Nw = cfg.Nw;
    time = results.time;
    N = numel(time);

    rTrue = results.etaTrue(1:dim,:);
    watcherR = results.watcherR;

    rangeLog = zeros(N, Nw);

    if dim == 2
        bearingLog = zeros(N, Nw);
    elseif dim == 3
        azLog = zeros(N, Nw);
        elLog = zeros(N, Nw);
    else
        error("Unsupported cfg.dim. Use cfg.dim = 2 or cfg.dim = 3.");
    end

    for i = 1:Nw
        for k = 1:N
            rho = rTrue(:,k) - watcherR(:,k,i);

            rangeLog(k,i) = norm(rho);

            if dim == 2
                bearingLog(k,i) = atan2(rho(2), rho(1));
            else
                azLog(k,i) = atan2(rho(2), rho(1));
                elLog(k,i) = atan2(rho(3), sqrt(rho(1)^2 + rho(2)^2));
            end
        end
    end

    figure;
    hold on; grid on;

    for i = 1:Nw
        plot(time, rangeLog(:,i), "LineWidth", 1.2);
    end

    xlabel("Time [s]");
    ylabel("Relative range");
    title("Target-Watcher Relative Range");
    legend("Watcher " + string(1:Nw), "Location", "best");

    if dim == 2
        figure;
        hold on; grid on;

        for i = 1:Nw
            plot(time, unwrap(bearingLog(:,i)), "LineWidth", 1.2);
        end

        xlabel("Time [s]");
        ylabel("Bearing angle [rad]");
        title("Target-Watcher Bearing Angle");
        legend("Watcher " + string(1:Nw), "Location", "best");

    elseif dim == 3
        figure;
        hold on; grid on;

        for i = 1:Nw
            plot(time, unwrap(azLog(:,i)), "LineWidth", 1.2);
        end

        xlabel("Time [s]");
        ylabel("Azimuth [rad]");
        title("Target-Watcher Azimuth");
        legend("Watcher " + string(1:Nw), "Location", "best");

        figure;
        hold on; grid on;

        for i = 1:Nw
            plot(time, elLog(:,i), "LineWidth", 1.2);
        end

        xlabel("Time [s]");
        ylabel("Elevation [rad]");
        title("Target-Watcher Elevation");
        legend("Watcher " + string(1:Nw), "Location", "best");
    end

end