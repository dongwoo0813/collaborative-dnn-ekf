function watcher = EKFUpdatePhysical(watcher, z, watcherState, cfg)
%{
Function:
    EKFUpdatePhysical.m

Purpose:
    Perform the EKF measurement-update step for one watcher using an
    angle-only measurement.

    This function updates the physical target state estimate

        eta = [r_t; v_t],

    where r_t is the target position and v_t is the target velocity.

    This is the baseline physical EKF update. It does not include DNN
    parameters. Later, for the augmented DNN-EKF state

        X_i = [eta_i; theta_i],

    the same measurement model can be used, but the full measurement
    Jacobian becomes

        H_X = [H_eta, zeros(nz, nTheta_i)].

Inputs:
    watcher      - Watcher EKF structure.
                   Required fields:
                       watcher.xhat - predicted physical state estimate,
                                       size 2*cfg.dim x 1
                       watcher.P    - predicted physical covariance,
                                       size 2*cfg.dim x 2*cfg.dim

    z            - Measurement from the watcher.
                   If cfg.dim = 2:
                       z is scalar bearing angle.
                   If cfg.dim = 3:
                       z = [azimuth; elevation], size 2 x 1.

    watcherState - Watcher state structure.
                   Required fields:
                       watcherState.r - watcher position, cfg.dim x 1.

    cfg          - Simulation configuration structure.
                   Required fields:
                       cfg.dim
                       cfg.meas.R
                       cfg.meas.type

Outputs:
    watcher      - Updated watcher EKF structure.
                   Updated fields:
                       watcher.xhat           - posterior state estimate
                       watcher.P              - posterior covariance
                       watcher.lastInnovation - innovation residual
                       watcher.lastS          - innovation covariance

Main equations:
    Measurement model:

        z_k = h(eta_k) + v_k,

    where

        v_k ~ N(0, R).

    Innovation:

        nu_k = z_k - h(eta_hat_k^-).

    Innovation covariance:

        S_k = H_k P_k^- H_k^T + R.

    Kalman gain:

        K_k = P_k^- H_k^T S_k^{-1}.

    State update:

        eta_hat_k^+ = eta_hat_k^- + K_k nu_k.

    Joseph-form covariance update:

        P_k^+ = (I - K_k H_k) P_k^- (I - K_k H_k)^T
                + K_k R K_k^T.

Notes:
    - The measurement prediction h(eta_hat) is computed by
      measurementPrediction.m rather than hard-coded in this EKF update.
    - The innovation is wrapped because bearing, azimuth, and elevation are
      angular measurements.
    - Joseph-form covariance update is used to improve numerical robustness.
    - The covariance is symmetrized after update to reduce finite-precision
      asymmetry.
%}
    
    x = watcher.xhat;
    P = watcher.P;
    
    % Measurement Jacobian H_eta = dh/deta.
    H = measurementJacobian(x, watcherState, cfg);
    
    % Deterministic predicted measurement zhat = h(x).
    zhat = measurementPrediction(x, watcherState, cfg);
    
    % Angular innovation residual.
    nu = wrapAngleResidual(z - zhat);
    
    R = cfg.meas.R;
    S = H * P * H' + R;
    
    K = P * H' / S;
    
    xPlus = x + K * nu;
    
    I = eye(size(P));
    PPlus = (I - K*H) * P * (I - K*H)' + K * R * K';
    PPlus = 0.5 * (PPlus + PPlus');
    
    watcher.xhat = xPlus;
    watcher.P = PPlus;
    watcher.lastInnovation = nu;
    watcher.lastS = S;

end

function a = wrapAngleResidual(a)
%{
Function:
    wrapAngleResidual

Purpose:
    Wrap angular innovation residuals to the interval [-pi, pi).

Inputs:
    a - Angular residual.
        Can be scalar or vector.

Outputs:
    a - Wrapped angular residual with the same size as input.

Main equation:
    a_wrapped = mod(a + pi, 2*pi) - pi.
%}

a = mod(a + pi, 2*pi) - pi;

end