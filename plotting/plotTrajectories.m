function plotTrajectories(results, cfg)
%{
Function:
    plotTrajectories.m

Purpose:
    Plot the true target trajectory and watcher trajectories used in the
    simulation.

    This function is used to visually verify that:
        1. The target trajectory is generated correctly.
        2. Watcher trajectories are placed correctly.
        3. The target-watcher relative geometry is reasonable for
           bearing-only estimation.

Inputs:
    results - Simulation result structure.
              Required fields:
                  results.time     - time vector, size 1 x N or N x 1
                  results.etaTrue  - true target state history,
                                     size 2*cfg.dim x N
                  results.watcherR - watcher position history,
                                     size cfg.dim x N x cfg.Nw

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.dim
                  cfg.Nw

Outputs:
    None.
    This function creates a MATLAB figure.

Main equations:
    Target physical state:

        eta(t) = [r_t(t); v_t(t)].

    The plotted target trajectory is

        r_t(t).

    The watcher trajectories are read directly from the simulation log:

        results.watcherR(:,k,i) = r_{w,i}(t_k).

    The target-watcher relative position used by the measurement model is

        rho_i(t) = r_t(t) - r_{w,i}(t).

Notes:
    - This function does not regenerate watcher trajectories.
    - It plots the watcher motion actually used during the simulation.
    - For cfg.dim = 2, this function uses plot().
    - For cfg.dim = 3, this function uses plot3().
    - This function assumes simulatePhysicalEKF.m stores results.watcherR.
%}

    dim = cfg.dim;
    Nw = cfg.Nw;

    etaTrue = results.etaTrue;
    rTrue = etaTrue(1:dim, :);

    if ~isfield(results, "watcherR")
        error("results.watcherR is missing. Add watcherR logging in simulatePhysicalEKF.m first.");
    end

    watcherR = results.watcherR;

    figure;
    hold on; grid on; axis equal;

    if dim == 2

        plot(rTrue(1,:), rTrue(2,:), "k-", "LineWidth", 2);

        for i = 1:Nw
            rWLog = squeeze(watcherR(:,:,i));

            plot(rWLog(1,:), rWLog(2,:), "--", "LineWidth", 1.2);
        end

        xlabel("x");
        ylabel("y");
        title("Target and Watcher Trajectories");

        legendEntries = strings(1, Nw+1);
        legendEntries(1) = "Target";

        for i = 1:Nw
            legendEntries(i+1) = "Watcher " + string(i);
        end

        legend(legendEntries, "Location", "best");

    elseif dim == 3

        plot3(rTrue(1,:), rTrue(2,:), rTrue(3,:), "k-", "LineWidth", 2);

        for i = 1:Nw
            rWLog = squeeze(watcherR(:,:,i));

            plot3(rWLog(1,:), rWLog(2,:), rWLog(3,:), "--", "LineWidth", 1.2);
        end

        xlabel("x");
        ylabel("y");
        zlabel("z");
        title("Target and Watcher Trajectories");
        view(3);

        legendEntries = strings(1, Nw+1);
        legendEntries(1) = "Target";

        for i = 1:Nw
            legendEntries(i+1) = "Watcher " + string(i);
        end

        legend(legendEntries, "Location", "best");

    else
        error("Unsupported cfg.dim. Use cfg.dim = 2 or cfg.dim = 3.");
    end

end