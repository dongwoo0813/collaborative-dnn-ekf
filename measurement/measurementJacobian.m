function H = measurementJacobian(eta, watcherState, cfg)
%{
Function:
    measurementJacobian.m

Purpose:
    Compute the Jacobian of the angle-only measurement model with respect
    to the physical target state eta = [r_t; v_t].

Inputs:
    eta          - Target physical state estimate.
                   Size: 2*cfg.dim x 1.
                   Definition:
                       eta = [r_t; v_t].

    watcherState - Watcher state structure.
                   Required fields:
                       watcherState.r - watcher position, cfg.dim x 1.

    cfg          - Simulation configuration structure.
                   Required fields:
                       cfg.dim

Outputs:
    H            - Measurement Jacobian with respect to eta.
                   If cfg.dim = 2:
                       H size is 1 x 4.
                   If cfg.dim = 3:
                       H size is 2 x 6.

Main equations:
    The relative position is

        rho = r_t - r_w.

    For 2D bearing,

        h(eta) = atan2(rho_y, rho_x),

    and

        dh/dr = [ -rho_y / ||rho||^2,  rho_x / ||rho||^2 ].

    Since the bearing measurement depends only on position,

        dh/dv = 0.

    For 3D azimuth/elevation,

        az = atan2(y,x),

        el = atan2(z, sqrt(x^2+y^2)).

Notes:
    - This is the physical-state Jacobian H_eta.
    - For the augmented DNN-EKF with state X_i = [eta_i; theta_i],
      the full measurement Jacobian should be

          H_X = [H_eta, zeros(nz, nTheta_i)].

    - The DNN parameter block theta_i is not directly measured.
%}
    
    dim = cfg.dim;
    
    rT = eta(1:dim);
    rW = watcherState.r;
    rho = rT - rW;
    
    measType = string(cfg.meas.type);

    if measType == "relative_position"
        H = [eye(dim), zeros(dim, dim)];
        return;
    end

    if measType == "direct_residual"
        error("measurementJacobian:DirectResidualNeedsAugmentedState", ...
            "direct_residual Jacobian is handled by DNN_EKF_Update_Local.");
    end

    if dim == 2
        x = rho(1);
        y = rho(2);
    
        q = x^2 + y^2;
    
        Hr = [-y/q, x/q];
        Hv = zeros(1, dim);
    
        if measType == "bearing"
            H = [Hr, Hv];
        elseif measType == "range_bearing"
            rangeValue = sqrt(q);
            Hrange = [x/rangeValue, y/rangeValue];
            H = [[Hrange; Hr], zeros(2, dim)];
        else
            error("measurementJacobian:UnsupportedType", ...
                "Unsupported measurement type: %s", measType);
        end
    
    elseif dim == 3
        x = rho(1);
        y = rho(2);
        z = rho(3);
    
        rho2_xy = x^2 + y^2;
        rho_xy = sqrt(rho2_xy);
        rho2 = rho2_xy + z^2;
    
        Haz = [-y/rho2_xy, x/rho2_xy, 0];
    
        Hel = [ ...
            -x*z/(rho2*rho_xy), ...
            -y*z/(rho2*rho_xy), ...
             rho_xy/rho2 ];
    
        Hr = [Haz; Hel];
        Hv = zeros(2, dim);
    
        if measType == "bearing"
            H = [Hr, Hv];
        elseif measType == "range_bearing"
            Hrange = [x, y, z] / sqrt(rho2);
            H = [[Hrange; Hr], zeros(3, dim)];
        else
            error("measurementJacobian:UnsupportedType", ...
                "Unsupported measurement type: %s", measType);
        end
    
    else
        error("Unsupported cfg.dim. Use cfg.dim = 2 or cfg.dim = 3.");
    end

end
