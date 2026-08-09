function watcher = initPhysicalEKF(i, etaTrue0, cfg)
%{
Function:
    initPhysicalEkf.m

Purpose:
    Initialize the physical EKF for watcher i.

    This baseline EKF estimates only the physical target state

        eta = [r_t; v_t],

    where r_t is target position and v_t is target velocity.

Inputs:
    i         - Watcher index.
                Type: positive integer.
                Range: 1 <= i <= cfg.Nw.

    etaTrue0  - Initial true target physical state.
                Size: 2*cfg.dim x 1.
                Definition:
                    etaTrue0 = [r_t(0); v_t(0)].

    cfg       - Simulation configuration structure.
                Required fields:
                    cfg.dim
                    cfg.ekf.r0_err_std
                    cfg.ekf.v0_err_std
                    cfg.ekf.P0_pos
                    cfg.ekf.P0_vel

Outputs:
    watcher   - Watcher EKF structure.
                Fields:
                    watcher.id   - watcher index
                    watcher.xhat - initial physical state estimate
                    watcher.P    - initial physical covariance
                    watcher.lastInnovation
                    watcher.lastS

Main equations:
    The initial estimate is generated as

        eta_hat_i(0) = eta_true(0) + e_i,

    where

        e_i = [e_r; e_v],

    with position and velocity perturbations.

    The initial covariance is

        P_i(0) = diag(P0_pos I_dim, P0_vel I_dim).

Notes:
    - This function is dimension-generic.
    - For cfg.dim = 2, eta has size 4.
    - For cfg.dim = 3, eta has size 6.
    - DNN parameters theta_i are not included in this baseline step.
%}

    dim = cfg.dim;
    
    posErr = cfg.ekf.r0_err_std * randn(dim,1);
    velErr = cfg.ekf.v0_err_std * randn(dim,1);
    
    xhat0 = etaTrue0 + [posErr; velErr];
    
    P0 = blkdiag(cfg.ekf.P0_pos * eye(dim), ...
                 cfg.ekf.P0_vel * eye(dim));
    
    watcher.id = i;
    watcher.xhat = xhat0;
    watcher.P = P0;
    
    watcher.lastInnovation = [];
    watcher.lastS = [];

end