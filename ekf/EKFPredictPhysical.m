function watcher = EKFPredictPhysical(watcher, t, cfg)
%{
Function:
    ekfPredictPhysical.m

Purpose:
    Perform the EKF prediction/time-update step for the physical target
    state estimate of one watcher.

    This function propagates only the physical target state

        eta = [r_t; v_t],

    without DNN residual parameters. It is intended as the baseline
    physical EKF prediction step before adding the augmented DNN-EKF state

        X_i = [eta_i; theta_i].

Inputs:
    watcher - Watcher filter structure.
              Required fields:
                  watcher.xhat - current physical state estimate,
                                  size 2*cfg.dim x 1
                  watcher.P    - current physical covariance,
                                  size 2*cfg.dim x 2*cfg.dim

    t       - Current simulation time.
              Type: scalar.
              Units: seconds.
              Note:
                  This input is included for interface consistency.
                  In the current double-integrator prediction model, t is
                  not explicitly used. It may be used later for time-varying
                  dynamics, orbit dynamics, or control-dependent dynamics.

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.dt
                  cfg.dim
                  cfg.ekf.q_acc

Outputs:
    watcher - Updated watcher filter structure after prediction.
              Updated fields:
                  watcher.xhat - predicted physical state estimate
                  watcher.P    - predicted physical covariance

Main equations:
    The physical target state is

        eta = [r_t; v_t].

    The baseline prediction model is the discrete-time double integrator

        r_{k+1} = r_k + dt v_k,

        v_{k+1} = v_k.

    Therefore,

        eta_{k+1}^- = F eta_k^+,

    where

        F = [ I   dt I
              0     I  ].

    The covariance prediction is

        P_{k+1}^- = F P_k^+ F^T + Q.

    The process-noise covariance assumes continuous acceleration noise with
    intensity q = cfg.ekf.q_acc, discretized as

        Q = q [ dt^4/4 I    dt^3/2 I
                dt^3/2 I    dt^2   I ].

Notes:
    - This function is dimension-generic. For cfg.dim = 2, eta has size 4.
      For cfg.dim = 3, eta has size 6.
    - This baseline model does not include unknown residual acceleration.
      The residual/DNN model should be added later in ekfPredictLocalDnn.m.
    - The covariance is symmetrized after prediction to reduce numerical
      asymmetry caused by finite precision.
%}

    dt = cfg.dt;
    dim = cfg.dim;
    
    x = watcher.xhat;
    P = watcher.P;
    
    F = [eye(dim), dt*eye(dim);
         zeros(dim), eye(dim)];
    
    xPred = F * x;
    
    q = cfg.ekf.q_acc;
    
    Q = q * [dt^4/4 * eye(dim), dt^3/2 * eye(dim);
             dt^3/2 * eye(dim), dt^2   * eye(dim)];
    
    Ppred = F * P * F' + Q;
    Ppred = 0.5 * (Ppred + Ppred');
    
    watcher.xhat = xPred;
    watcher.P = Ppred;

end