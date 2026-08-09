function plotWatcherControl(results, cfg)
%{
Function:
    plotWatcherControl.m

Purpose:
    Plot watcher translational and attitude control histories.

    This function visualizes:
        1. Translational thrust command u_i(t).
        2. Translational thrust norm ||u_i(t)||.
        3. Attitude torque command tau_i(t).
        4. Attitude torque norm ||tau_i(t)||.

    In the current prescribed-motion baseline, these commands may be zero.
    The function is included now so that later thruster control and attitude
    control can be added without changing the plotting architecture.

Inputs:
    results - Simulation result structure.
              Required fields:
                  results.time       - time vector, size 1 x N or N x 1
                  results.watcherU   - thrust command log,
                                       size cfg.dim x N x cfg.Nw
                  results.watcherTau - torque command log,
                                       size 3 x N x cfg.Nw

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.dim
                  cfg.Nw

Outputs:
    None.
    This function creates MATLAB figures.

Main equations:
    Translational watcher dynamics, later controlled mode:

        dot r_{w,i} = v_{w,i},

        dot v_{w,i} = u_i / m_i.

    Attitude dynamics, later controlled mode:

        J_i dot omega_i = tau_i - omega_i x J_i omega_i.

    Control norms:

        ||u_i(t)|| = sqrt(u_i(t)^T u_i(t)),

        ||tau_i(t)|| = sqrt(tau_i(t)^T tau_i(t)).

Notes:
    - In prescribed watcher-motion mode, u_i and tau_i are expected to be
      zero or placeholder commands.
    - This function does not modify results.
    - This plot should remain separate from plotTrajectories.m. Trajectory
      plots show geometry; control plots show actuator effort.
%}

    time = results.time(:);
    dim = cfg.dim;
    Nw = cfg.Nw;

    if ~isfield(results, "watcherU")
        warning("results.watcherU is missing. Skipping thrust-command plots.");
        return;
    end

    if ~isfield(results, "watcherTau")
        warning("results.watcherTau is missing. Skipping torque-command plots.");
        return;
    end

    watcherU = results.watcherU;
    watcherTau = results.watcherTau;

    N = numel(time);

    if size(watcherU,1) ~= dim || size(watcherU,2) ~= N || size(watcherU,3) ~= Nw
        error("results.watcherU must have size cfg.dim x N x cfg.Nw.");
    end

    if size(watcherTau,1) ~= 3 || size(watcherTau,2) ~= N || size(watcherTau,3) ~= Nw
        error("results.watcherTau must have size 3 x N x cfg.Nw.");
    end

    thrustNorm = zeros(N, Nw);
    torqueNorm = zeros(N, Nw);

    for i = 1:Nw
        for k = 1:N
            thrustNorm(k,i) = norm(watcherU(:,k,i));
            torqueNorm(k,i) = norm(watcherTau(:,k,i));
        end
    end

    % Plot thrust components.
    figure;
    tiledlayout(dim,1);

    for d = 1:dim
        nexttile;
        hold on; grid on;

        for i = 1:Nw
            plot(time, squeeze(watcherU(d,:,i)), "LineWidth", 1.2);
        end

        xlabel("Time [s]");
        ylabel("Thrust");
        title("Watcher Thrust Component " + string(d));
        legend("Watcher " + string(1:Nw), "Location", "best");
    end

    % Plot thrust norm.
    figure;
    hold on; grid on;

    for i = 1:Nw
        plot(time, thrustNorm(:,i), "LineWidth", 1.2);
    end

    xlabel("Time [s]");
    ylabel("Thrust norm");
    title("Watcher Translational Control Effort");
    legend("Watcher " + string(1:Nw), "Location", "best");

    % Plot attitude torque components.
    figure;
    tiledlayout(3,1);

    for d = 1:3
        nexttile;
        hold on; grid on;

        for i = 1:Nw
            plot(time, squeeze(watcherTau(d,:,i)), "LineWidth", 1.2);
        end

        xlabel("Time [s]");
        ylabel("Torque");
        title("Watcher Torque Component " + string(d));
        legend("Watcher " + string(1:Nw), "Location", "best");
    end

    % Plot torque norm.
    figure;
    hold on; grid on;

    for i = 1:Nw
        plot(time, torqueNorm(:,i), "LineWidth", 1.2);
    end

    xlabel("Time [s]");
    ylabel("Torque norm");
    title("Watcher Attitude Control Effort");
    legend("Watcher " + string(1:Nw), "Location", "best");

end