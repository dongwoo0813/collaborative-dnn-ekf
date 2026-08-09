function zhat = measurementPrediction(eta, watcherState, cfg)
%{
Function:
    measurementPrediction.m

Purpose:
    Compute the deterministic measurement prediction

        zhat = h(eta)

    for the current target state estimate and watcher state.

    This function does not add measurement noise. It is used inside the EKF
    update to compute the predicted measurement.

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
                       cfg.meas.type

Outputs:
    zhat         - Predicted measurement.
                   If cfg.dim = 2:
                       zhat is scalar bearing angle.
                   If cfg.dim = 3:
                       zhat = [azimuth; elevation], size 2 x 1.

Main equations:
    Define the relative position from watcher to target:

        rho = r_t - r_w.

    For cfg.dim = 2:

        h(eta) = atan2(rho_y, rho_x).

    For cfg.dim = 3:

        h(eta) = [az; el],

    where

        az = atan2(rho_y, rho_x),

        el = atan2(rho_z, sqrt(rho_x^2 + rho_y^2)).

Notes:
    - This function computes only the noise-free predicted measurement.
    - No random noise should be added here.
    - No FOV or measurement availability logic should be added here.
      Availability belongs in measurementModel.m or fovAvailable.m.
    - The measurement depends only on target position r_t, not velocity v_t.
%}

switch cfg.meas.type

    case "bearing"
        zhat = bearingPrediction(eta, watcherState, cfg);

    case "range_bearing"
        rho = eta(1:cfg.dim) - watcherState.r(:);
        zhat = [norm(rho); bearingPrediction(eta, watcherState, cfg)];

    case "relative_position"
        zhat = eta(1:cfg.dim) - watcherState.r(:);

    case "direct_residual"
        error("measurementPrediction:DirectResidualNeedsAugmentedState", ...
            "direct_residual prediction is handled by DNN_EKF_Update_Local.");

    otherwise
        error("Unknown cfg.meas.type in measurementPrediction.m.");
end

end

function zhat = bearingPrediction(eta, watcherState, cfg)
%{
Function:
    bearingPrediction

Purpose:
    Compute the deterministic bearing-type measurement prediction.

Inputs:
    eta          - Target physical state estimate, size 2*cfg.dim x 1.
    watcherState - Watcher state structure with watcherState.r.
    cfg          - Configuration structure with cfg.dim.

Outputs:
    zhat         - Predicted bearing measurement.
                   Scalar for 2D.
                   Two-dimensional [azimuth; elevation] for 3D.
%}
    
    dim = cfg.dim;
    
    rT = eta(1:dim);
    rW = watcherState.r;
    
    rho = rT - rW;
    
    if dim == 2
        zhat = atan2(rho(2), rho(1));
    
    elseif dim == 3
        x = rho(1);
        y = rho(2);
        z = rho(3);
    
        az = atan2(y, x);
        el = atan2(z, sqrt(x^2 + y^2));
    
        zhat = [az; el];
    
    else
        error("Unsupported cfg.dim. Use cfg.dim = 2 or cfg.dim = 3.");
    end

end
