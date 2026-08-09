function plotWatcherMotion(results, cfg)
%{
Function:
    plotWatcherMotion.m

Purpose:
    Plot the prescribed translational motion of all watcher spacecraft.

    This function plots:
        1. Watcher position components r_{w,i}(t).
        2. Watcher velocity components v_{w,i}(t).

Inputs:
    results - Simulation result structure.
              Required fields:
                  results.time
                  results.watcherR
                  results.watcherV

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.dim
                  cfg.Nw

Outputs:
    None.
    This function creates MATLAB figures.

Main equations:
    Watcher motion is stored as

        results.watcherR(:,k,i) = r_{w,i}(t_k),

        results.watcherV(:,k,i) = v_{w,i}(t_k).

Notes:
    - In Step 01, watcher motion is prescribed, not controlled.
    - Later, this same plotting function can be reused for controlled watcher
      trajectories if results.watcherR and results.watcherV are logged.
%}

    dim = cfg.dim;
    Nw = cfg.Nw;
    time = results.time;

    watcherR = results.watcherR;
    watcherV = results.watcherV;

    figure;
    tiledlayout(dim,1);

    for d = 1:dim
        nexttile;
        hold on; grid on;

        for i = 1:Nw
            plot(time, squeeze(watcherR(d,:,i)), "LineWidth", 1.2);
        end

        xlabel("Time [s]");
        ylabel("Position");
        title("Watcher Position Component " + string(d));
        legend("Watcher " + string(1:Nw), "Location", "best");
    end

    figure;
    tiledlayout(dim,1);

    for d = 1:dim
        nexttile;
        hold on; grid on;

        for i = 1:Nw
            plot(time, squeeze(watcherV(d,:,i)), "LineWidth", 1.2);
        end

        xlabel("Time [s]");
        ylabel("Velocity");
        title("Watcher Velocity Component " + string(d));
        legend("Watcher " + string(1:Nw), "Location", "best");
    end

end