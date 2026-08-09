function watcher = DNN_EKF_Update_Local(watcher, z, watcherState, cfg)
%{
Function:
    DNN_EKF_Update_Local.m

Purpose:
    Perform the EKF measurement-update step for one local augmented
    DNN-EKF watcher.

    The local augmented state is

        X_i = [eta_i; theta_i],

    where

        eta_i = [r_t; v_t]

    is the physical target state estimate and theta_i is the local DNN
    branch parameter vector estimated by watcher i.

    The bearing-only measurement depends only on eta_i, not directly on
    theta_i. Therefore, the augmented measurement Jacobian is

        H_X = [H_eta, zeros(nz,nTheta_i)].

    Because the theta block of H_X is zero, theta_i is updated only through
    the predicted cross-covariance P_{theta eta}.

Inputs:
    watcher      - Local DNN-EKF watcher structure.
                   Required fields:
                       watcher.xhat     - predicted augmented estimate
                                          X_i^-, size nX x 1
                       watcher.P        - predicted augmented covariance
                                          P_i^-, size nX x nX
                       watcher.idxEta   - indices of eta_i in xhat
                       watcher.idxTheta - indices of theta_i in xhat
                       watcher.nX       - augmented state dimension
                       watcher.nTheta   - local parameter dimension

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

                   Optional fields:
                       cfg.dnn.minCovDiag

Outputs:
    watcher      - Updated local DNN-EKF watcher structure.
                   Updated fields:
                       watcher.xhat           - posterior augmented estimate
                       watcher.P              - posterior augmented covariance
                       watcher.lastInnovation - angular innovation residual
                       watcher.lastS          - innovation covariance

Main equations:
    Measurement model:

        z_k = h(eta_k) + v_k,

    where

        v_k ~ N(0,R).

    Augmented Jacobian:

        H_X = [H_eta, 0].

    Innovation:

        nu_k = z_k - h(eta_hat_k^-).

    Innovation covariance:

        S_k = H_X P_k^- H_X^T + R.

    Kalman gain:

        K_k = P_k^- H_X^T S_k^{-1}.

    Augmented state update:

        X_hat_k^+ = X_hat_k^- + K_k nu_k.

    Joseph-form covariance update:

        P_k^+ = (I - K_k H_X) P_k^- (I - K_k H_X)^T
                + K_k R K_k^T.

Notes:
    - This is the augmented-state version of EKFUpdatePhysical.m.
    - The measurement prediction h(eta) and H_eta are computed using only
      the eta block of the augmented state.
    - The theta block is not directly measured.
    - If P_{theta eta} is zero, the parameter estimate will not change in
      this update. Prediction must generate P_{theta eta} through the DNN
      coupling for theta_i to learn from measurements.
    - Joseph-form covariance update is used for numerical robustness.
    - The covariance is symmetrized after the update to reduce
      finite-precision asymmetry.
%}

    x = watcher.xhat;
    P = watcher.P;

    idxEta = watcher.idxEta;

    eta = x(idxEta);

    % Oracle learning control. Unlike physical measurements, this
    % pseudo-measurement observes the local DNN branch directly and
    % therefore has a nonzero theta Jacobian.
    if string(cfg.meas.type) == "direct_residual"
        watcher = updateDirectResidualMeasurement(watcher, z, cfg);
        return;
    end

    % Measurement Jacobian with respect to the physical state only.
    H_eta = measurementJacobian(eta, watcherState, cfg);

    % Deterministic predicted measurement zhat = h(eta).
    zhat = measurementPrediction(eta, watcherState, cfg);

    z = z(:);
    zhat = zhat(:);

    if isempty(z)
        error("DNNEKFUpdateLocal received an empty measurement z. Call this function only when measurement is available.");
    end

    nz = numel(zhat);

    if numel(z) ~= nz
        error("Measurement dimension mismatch in DNNEKFUpdateLocal. numel(z) = %d, numel(zhat) = %d.", ...
            numel(z), nz);
    end

    if size(H_eta,1) ~= nz
        error("Heta row dimension does not match measurement dimension in DNNEKFUpdateLocal.");
    end

    % ---------------------------------------------------------------------
    % Sparse/block measurement update using H_X = [H_eta, 0].
    %
    % The measurement depends only on the physical state eta, not directly
    % on theta. Therefore, avoid constructing the full augmented Jacobian
    %
    %     H_X = [H_eta, zeros(nz,nTheta)]
    %
    % and avoid the dense Joseph multiplication with the full nX-by-nX
    % identity matrix.
    %
    % Innovation covariance:
    %
    %     S = H_eta P_eta_eta H_eta' + R
    %
    % Kalman gain:
    %
    %     K = P H_X' S^{-1}
    %       = P(:,idxEta) H_eta' S^{-1}
    %
    % Covariance update:
    %
    %     P^+ = P^- - K S K'
    %
    % This is algebraically equivalent to the Joseph form when
    % K = P^- H_X' S^{-1} and S = H_X P^- H_X' + R, but it avoids large
    % dense products. Since nz is 1 or 2, this is a low-rank update.
    % ---------------------------------------------------------------------

    % Only angular components are wrapped. Range and Cartesian residuals
    % retain their linear units.
    nu = measurementInnovation(z, zhat, cfg);

    R = cfg.meas.R;

    if isscalar(R)
        R = R * eye(nz);
    end

    if any(size(R) ~= [nz, nz])
        error("Measurement covariance dimension mismatch in DNNEKFUpdateLocal.");
    end

    % Extract only the physical covariance block needed by the measurement.
    Petaeta = P(idxEta, idxEta);

    % Innovation covariance. This uses only P_eta_eta.
    S = H_eta * Petaeta * H_eta' + R;
    S = 0.5 * (S + S');

    % Cross-covariance between full augmented state and measurement.
    %
    % Equivalent to:
    %     PHt = P * H_X'
    % but avoids forming H_X.
    PHt = P(:, idxEta) * H_eta';

    % Kalman gain.
    K = PHt / S;

    % Augmented state update.
    xPlus = x + K * nu;

    % Low-rank covariance update.
    %
    % Equivalent to Joseph form under the Kalman gain above:
    %     PPlus = P - K*S*K'
    %
    % This avoids the expensive full product
    %     (I-KH)P(I-KH)'.
    PPlus = P - K * S * K';
    PPlus = 0.5 * (PPlus + PPlus');

    % Optional numerical diagonal floor, consistent with DNN_EKF_Predict_Local.m.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "minCovDiag")
        minCovDiag = cfg.dnn.minCovDiag;
    else
        minCovDiag = 0;
    end

    if minCovDiag > 0
        d = diag(PPlus);
        d = max(d, minCovDiag);
        PPlus(1:size(PPlus,1)+1:end) = d;
        PPlus = 0.5 * (PPlus + PPlus');
    end

    watcher.xhat = xPlus;
    watcher.P = PPlus;
    watcher.lastInnovation = nu;
    watcher.lastS = S;

    % ---------------------------------------------------------------------
    % Adaptive covariance matching for process noise
    %
    % By default, the empirical/model innovation covariance ratio adapts
    % Qepsilon,c after every measurement. Qtheta remains fixed so that one
    % scalar innovation statistic does not adapt two indistinguishable
    % process-noise sources simultaneously.
    %
    % The updated gammaTheta will be used in the next prediction step through
    %
    %     Q_theta,k = gammaTheta_i,k * Q_theta,base.
    % ---------------------------------------------------------------------
    watcher = Adaptive_Q_Cov_Matching(watcher, nu, S);

end

function watcher = updateDirectResidualMeasurement(watcher, z, cfg)
% Directly observe one local branch's share of the true residual.

    x = watcher.xhat;
    P = watcher.P;
    idxEta = watcher.idxEta;
    idxTheta = watcher.idxTheta;

    eta = x(idxEta);
    theta = x(idxTheta);
    branchID = watcher.localBranchID;

    [dHat, Jeta, Jtheta] = evaluateBranchResidualModel( ...
        branchID, eta, theta, cfg);

    z = z(:);
    dHat = dHat(:);
    nz = numel(dHat);
    if numel(z) ~= nz
        error("DNN_EKF_Update_Local:DirectResidualDimensionMismatch", ...
            "Direct residual z has length %d; expected %d.", numel(z), nz);
    end

    H = zeros(nz, watcher.nX);
    H(:,idxEta) = Jeta;
    H(:,idxTheta) = Jtheta;
    nu = z - dHat;

    R = cfg.meas.R;
    if isscalar(R)
        R = R * eye(nz);
    end
    if any(size(R) ~= [nz,nz])
        error("DNN_EKF_Update_Local:DirectResidualRDimensionMismatch", ...
            "Direct residual R must be %d-by-%d.", nz, nz);
    end

    S = H * P * H' + R;
    S = 0.5 * (S + S');
    K = (P * H') / S;

    xPlus = x + K * nu;
    PPlus = P - K * S * K';
    PPlus = 0.5 * (PPlus + PPlus');

    minCovDiag = 0;
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "minCovDiag")
        minCovDiag = cfg.dnn.minCovDiag;
    end
    if minCovDiag > 0
        d = max(diag(PPlus), minCovDiag);
        PPlus(1:size(PPlus,1)+1:end) = d;
        PPlus = 0.5 * (PPlus + PPlus');
    end

    watcher.xhat = xPlus;
    watcher.P = PPlus;
    watcher.lastInnovation = nu;
    watcher.lastS = S;
    watcher = Adaptive_Q_Cov_Matching(watcher, nu, S);
end

function nu = measurementInnovation(z, zhat, cfg)
% Apply wrapping only to bearing entries.

    nu = z - zhat;
    switch string(cfg.meas.type)
        case "bearing"
            nu = wrapAngleResidual(nu);
        case "range_bearing"
            nu(2:end) = wrapAngleResidual(nu(2:end));
        case {"relative_position", "direct_residual"}
            % Linear residual: no wrapping.
        otherwise
            error("DNN_EKF_Update_Local:UnsupportedMeasurementType", ...
                "Unsupported measurement type: %s", string(cfg.meas.type));
    end
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


function watcher = Adaptive_Q_Cov_Matching(watcher, nu, Smodel)
%{
Function:
    Adaptive_Q_Cov_Matching

Purpose:
    Adapt the DNN parameter process-noise scale gammaTheta using
    innovation covariance matching.

    The empirical innovation covariance is updated by EWMA:

        mu_nu,k = (1-alpha_mu) mu_nu,k-1 + alpha_mu nu_k

        S_hat,k = (1-alpha_S) S_hat,k-1
                  + alpha_S (nu_k-mu_nu,k)(nu_k-mu_nu,k)'

    Then the trace ratio

        ratio = trace(S_hat,k) / trace(Smodel,k)

    is used to update

        gammaTheta <- gammaTheta * exp(gainTheta * log(ratio)).

    The update is delayed by burn-in and applied every adaptInt
    measurement updates.

Inputs:
    watcher - local DNN-EKF watcher structure
    nu      - innovation vector
    Smodel  - model innovation covariance H P H' + R

Outputs:
    watcher - watcher structure with updated watcher.cm.gammaTheta
%}

    % If covariance matching was not initialized, do nothing.
    if ~isfield(watcher, "cm")
        return;
    end

    cm = watcher.cm;

    if ~isfield(cm, "enabled") || ~cm.enabled
        watcher.cm = cm;
        return;
    end

    nu = nu(:);
    nz = numel(nu);

    Smodel = 0.5 * (Smodel + Smodel.');

    % ------------------------------------------------------------------
    % Initialize missing fields defensively
    % ------------------------------------------------------------------
    if ~isfield(cm, "measCount")
        cm.measCount = 0;
    end

    if ~isfield(cm, "firstMeas")
        cm.firstMeas = true;
    end

    if ~isfield(cm, "muNu") || numel(cm.muNu) ~= nz
        cm.muNu = zeros(nz,1);
    end

    if ~isfield(cm, "Shat") || any(size(cm.Shat) ~= [nz, nz])
        cm.Shat = Smodel;
    end

    if ~isfield(cm, "alphaS")
        cm.alphaS = 0.01;
    end

    if ~isfield(cm, "alphaMu")
        cm.alphaMu = 0.01;
    end

    if ~isfield(cm, "burnInMeas")
        cm.burnInMeas = 20;
    end

    if ~isfield(cm, "adaptInt")
        cm.adaptInt = 5;
    end

    if ~isfield(cm, "epsLog")
        cm.epsLog = 0.05;
    end

    if ~isfield(cm, "gainTheta")
        cm.gainTheta = 1e-2;
    end

    if ~isfield(cm, "ratioMin")
        cm.ratioMin = 0.1;
    end

    if ~isfield(cm, "ratioMax")
        cm.ratioMax = 10.0;
    end

    if ~isfield(cm, "gammaTheta")
        cm.gammaTheta = 1.0;
    end

    if ~isfield(cm, "gammaThetaMin")
        cm.gammaThetaMin = 0.05;
    end

    if ~isfield(cm, "gammaThetaMax")
        cm.gammaThetaMax = 50.0;
    end

    if ~isfield(cm, "adaptThetaEnabled")
        cm.adaptThetaEnabled = true;
    end

    if ~isfield(cm, "adaptEpsilonEnabled")
        cm.adaptEpsilonEnabled = false;
    end

    if ~isfield(cm, "gammaEpsilon")
        cm.gammaEpsilon = 1.0;
    end

    if ~isfield(cm, "gammaEpsilonMin")
        cm.gammaEpsilonMin = 1e-2;
    end

    if ~isfield(cm, "gammaEpsilonMax")
        cm.gammaEpsilonMax = 1e2;
    end

    if ~isfield(cm, "gainEpsilon")
        cm.gainEpsilon = 2e-2;
    end

    % ------------------------------------------------------------------
    % Measurement counter
    % ------------------------------------------------------------------
    cm.measCount = cm.measCount + 1;

    % ------------------------------------------------------------------
    % EWMA empirical innovation covariance
    % ------------------------------------------------------------------
    if cm.firstMeas
        cm.muNu = nu;
        nuCentered = nu - cm.muNu;

        cm.Shat = (1 - cm.alphaS) * cm.Shat ...
                + cm.alphaS * (nuCentered * nuCentered.');

        cm.firstMeas = false;
    else
        cm.muNu = (1 - cm.alphaMu) * cm.muNu + cm.alphaMu * nu;
        nuCentered = nu - cm.muNu;

        cm.Shat = (1 - cm.alphaS) * cm.Shat ...
                + cm.alphaS * (nuCentered * nuCentered.');
    end

    cm.Shat = 0.5 * (cm.Shat + cm.Shat.');

    % ------------------------------------------------------------------
    % Trace-ratio covariance matching
    % ------------------------------------------------------------------
    traceEmp = trace(cm.Shat);
    traceModel = trace(Smodel);

    ratio = traceEmp / max(traceModel, 1e-12);

    % Clamp ratio to avoid violent adaptation.
    ratio = min(max(ratio, cm.ratioMin), cm.ratioMax);

    logRatio = log(ratio);

    % Deadband.
    if abs(logRatio) < cm.epsLog
        logRatioUsed = 0;
    else
        logRatioUsed = logRatio;
    end

    % ------------------------------------------------------------------
    % Adapt gammaTheta after every valid measurement update.
    % ------------------------------------------------------------------
    if cm.adaptThetaEnabled

        cm.gammaTheta = cm.gammaTheta * exp(cm.gainTheta * logRatioUsed);

        cm.gammaTheta = min(max(cm.gammaTheta, cm.gammaThetaMin), ...
                                  cm.gammaThetaMax);
    end


    % Adapt the continuous-time DNN approximation-error intensity after
    % every valid measurement update. The EWMA innovation covariance and
    % bounded multiplicative gain make this update gradual.
    if cm.adaptEpsilonEnabled
        cm.gammaEpsilon = cm.gammaEpsilon * ...
            exp(cm.gainEpsilon * logRatioUsed);

        cm.gammaEpsilon = min(max(cm.gammaEpsilon, cm.gammaEpsilonMin), ...
                                    cm.gammaEpsilonMax);
    end

    % ------------------------------------------------------------------
    % Store diagnostics
    % ------------------------------------------------------------------
    cm.lastRatio = ratio;
    cm.lastLogRatio = logRatio;
    cm.lastTraceEmp = traceEmp;
    cm.lastTraceModel = traceModel;

    watcher.cm = cm;

end
