function plotTrackingErrors(results, cfg)
%{
Function:
    plotTrackingErrors.m

Purpose:
    Plot watcher-local EKF tracking errors for the target physical state.

Inputs:
    results - Simulation result structure.
              Required fields:
                  results.time
                  results.etaTrue
                  results.xhat

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.dim
                  cfg.Nw

Outputs:
    None.
    This function creates MATLAB figures.

Main equations:
    For watcher i, the physical tracking error is

        e_{eta,i}(k) = eta_hat_i(k) - eta_true(k).

    The position and velocity errors are

        e_{r,i}(k) = r_hat_i(k) - r_true(k),

        e_{v,i}(k) = v_hat_i(k) - v_true(k).

Notes:
    - This function plots norm errors, not component-wise errors.
    - Component-wise plots can be added later if needed.
%}

    dim = cfg.dim;
    time = results.time;
    etaTrue = results.etaTrue;
    xhat = results.xhat;
    
    Nw = cfg.Nw;
    N = numel(time);
    
    posErrNorm = zeros(N, Nw);
    velErrNorm = zeros(N, Nw);
    
    for i = 1:Nw
        for k = 1:N
            e = xhat(:,k,i) - etaTrue(:,k);
    
            er = e(1:dim);
            ev = e(dim+1:2*dim);
    
            posErrNorm(k,i) = norm(er);
            velErrNorm(k,i) = norm(ev);
        end
    end
    
    figure;
    hold on; grid on;
    
    for i = 1:Nw
        plot(time, posErrNorm(:,i), "LineWidth", 1.2);
    end
    
    xlabel("Time [s]");
    ylabel("Position error norm");
    title("Physical EKF Position Error");
    legend("Watcher " + string(1:Nw), "Location", "best");
    
    figure;
    hold on; grid on;
    
    for i = 1:Nw
        plot(time, velErrNorm(:,i), "LineWidth", 1.2);
    end
    
    xlabel("Time [s]");
    ylabel("Velocity error norm");
    title("Physical EKF Velocity Error");
    legend("Watcher " + string(1:Nw), "Location", "best");

end