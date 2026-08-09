function Ppred = predict_Cov_Block_DNN_EKF(F, P, Q, watcher, cfg)
%{
Function:
    predict_Cov_Block_DNN_EKF.m

Purpose:
    Compute the local DNN-EKF covariance prediction using the block upper
    triangular transition structure

        F = [ F_eta_eta   F_eta_theta
              0           F_theta_theta ],

    where the theta-theta transition is diagonal for FOGM or identity-like
    for random-walk parameter dynamics.

    This function is algebraically equivalent to

        Ppred = F * P * F' + Q,

    under the structural assumptions above, but avoids the full dense
    augmented matrix multiplication.

Inputs:
    F       - Discrete-time augmented transition matrix.
              Size: nX x nX.

    P       - Prior/posterior covariance before prediction.
              Size: nX x nX.

    Q       - Discrete-time process/model covariance.
              Size: nX x nX.

    watcher - Local watcher structure.
              Required fields:
                  watcher.idxEta
                  watcher.idxTheta
                  watcher.nEta
                  watcher.nTheta
                  watcher.nX

    cfg     - Simulation configuration.
              Currently used only for optional tolerances.

Outputs:
    Ppred   - Predicted covariance.
              Size: nX x nX.

Main equations:
    Let

        P = [ P_ee   P_et
              P_te   P_tt ],

        F = [ F_ee   F_et
              0      D_t  ],

    where D_t = diag(dTheta). Then

        L_e = F_ee P_ee + F_et P_te,

        L_t = F_ee P_et + F_et P_tt,

        P_ee^- = L_e F_ee' + L_t F_et' + Q_ee,

        P_et^- = L_t D_t' + Q_et,

        P_tt^- = D_t P_tt D_t' + Q_tt.

    Since D_t is diagonal,

        D_t P_tt D_t' = (dTheta dTheta') .* P_tt,

    and

        L_t D_t' = columnwise scaling of L_t by dTheta.

Notes:
    - This helper does not change the EKF model.
    - It is intended as a computational replacement for F*P*F' + Q.
    - It assumes F(theta, eta) is zero. If this is not true, the function
      throws an error because the block formula would no longer match the
      dense propagation.
    - It assumes F(theta, theta) is diagonal. If a later parameter model
      uses dense theta dynamics, use the dense prediction instead.
%}

    idxEta = watcher.idxEta;
    idxTheta = watcher.idxTheta;

    nX = watcher.nX;
    nEta = watcher.nEta;
    nTheta = watcher.nTheta;

    if any(size(F) ~= [nX, nX])
        error("F has wrong size in predict_Cov_Block_DNN_EKF.");
    end

    if any(size(P) ~= [nX, nX])
        error("P has wrong size in predict_Cov_Block_DNN_EKF.");
    end

    if any(size(Q) ~= [nX, nX])
        error("Q has wrong size in predict_Cov_Block_DNN_EKF.");
    end

    % ---------------------------------------------------------------------
    % Optional numerical tolerance for structural checks.
    % ---------------------------------------------------------------------
    tol = 1e-12;

    if isfield(cfg, "ekf") && isfield(cfg.ekf, "blockPredictionTol")
        tol = cfg.ekf.blockPredictionTol;
    end

    % ---------------------------------------------------------------------
    % Extract transition blocks.
    % ---------------------------------------------------------------------
    F_ee = F(idxEta, idxEta);
    F_et = F(idxEta, idxTheta);
    F_te = F(idxTheta, idxEta);
    F_tt = F(idxTheta, idxTheta);

    % Check block upper-triangular structure.
    if max(abs(F_te(:))) > tol
        error("predict_Cov_Block_DNN_EKF assumes F(theta,eta)=0, but nonzero terms were found.");
    end

    % Check diagonal theta transition.
    F_tt_diag = diag(diag(F_tt));
    if max(abs(F_tt(:) - F_tt_diag(:))) > tol
        error("predict_Cov_Block_DNN_EKF assumes diagonal F(theta,theta), but off-diagonal terms were found.");
    end

    dTheta = diag(F_tt);

    if numel(dTheta) ~= nTheta
        error("Internal error: dTheta length mismatch.");
    end

    % ---------------------------------------------------------------------
    % Extract covariance blocks.
    % ---------------------------------------------------------------------
    P_ee = P(idxEta, idxEta);
    P_et = P(idxEta, idxTheta);
    P_te = P(idxTheta, idxEta);
    P_tt = P(idxTheta, idxTheta);

    Q_ee = Q(idxEta, idxEta);
    Q_et = Q(idxEta, idxTheta);
    Q_tt = Q(idxTheta, idxTheta);

    % ---------------------------------------------------------------------
    % Block product:
    %
    %   [F_ee F_et] * P
    %
    % Only the upper block row is needed for P_ee^- and P_et^-.
    % ---------------------------------------------------------------------
    L_e = F_ee * P_ee + F_et * P_te;
    L_t = F_ee * P_et + F_et * P_tt;

    % ---------------------------------------------------------------------
    % P_eta_eta prediction.
    % ---------------------------------------------------------------------
    P_ee_pred = L_e * F_ee' + L_t * F_et' + Q_ee;

    % ---------------------------------------------------------------------
    % P_eta_theta prediction.
    %
    % Since F_tt = diag(dTheta),
    %
    %   L_t * F_tt' = L_t * diag(dTheta)
    %
    % which is column-wise scaling.
    % ---------------------------------------------------------------------
    P_et_pred = L_t .* reshape(dTheta.', 1, nTheta);
    P_et_pred = P_et_pred + Q_et;

    % ---------------------------------------------------------------------
    % P_theta_theta prediction.
    %
    % Since F_tt = diag(dTheta),
    %
    %   F_tt P_tt F_tt' = (dTheta dTheta') .* P_tt.
    % ---------------------------------------------------------------------
    P_tt_pred = (dTheta * dTheta.') .* P_tt + Q_tt;

    % ---------------------------------------------------------------------
    % Assemble and symmetrize.
    % ---------------------------------------------------------------------
    Ppred = zeros(nX, nX);

    Ppred(idxEta, idxEta) = P_ee_pred;
    Ppred(idxEta, idxTheta) = P_et_pred;
    Ppred(idxTheta, idxEta) = P_et_pred.';
    Ppred(idxTheta, idxTheta) = P_tt_pred;

    Ppred = 0.5 * (Ppred + Ppred.');

end