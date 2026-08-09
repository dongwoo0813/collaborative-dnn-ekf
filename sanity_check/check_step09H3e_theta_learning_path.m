function check_step09H3e_theta_learning_path()
%{
File:
    checks/check_step09H3e_theta_learning_path.m

Purpose:
    Verify that the indirect theta-learning path is alive for both:

        1. fixed_feature_lip
        2. mlp_general

Measurement model:
    z = h(eta) + noise

Because h does not directly depend on theta,

    H_X = [H_eta, 0].

Therefore theta can only update through the predicted cross covariance:

    P_{theta eta}^-.

This check verifies:
    - prediction generates P_{eta theta}
    - update produces a nonzero theta correction when the diagnostic
      covariance/innovation are large enough
%}

    addpath(genpath(pwd));
    rehash;

    rng(101);

    cfg0 = config_step04_GS_DNN_EKF();

    dim = cfg0.dim;

    if dim == 2
        etaTrue0 = [
            800.0;
           -150.0;
              0.08;
             -0.04
        ];
    elseif dim == 3
        etaTrue0 = [
            800.0;
           -150.0;
            120.0;
              0.08;
             -0.04;
              0.02
        ];
    else
        error("check_step09H3e:UnsupportedDim", ...
            "This check supports cfg.dim = 2 or 3.");
    end

    watcherState = struct();
    watcherState.r = zeros(dim, 1);

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.3e check: theta learning path\n");
    fprintf("============================================================\n");

    cfg = cfg0;
    cfg.dnn.branchModel = "fixed_feature_lip";
    cfg.dnn.predictionResidualSource = "local_DNN";
    cfg.dnn.useResidualInPrediction = true;
    cfg.dnn.residualInjectionGain = 1.0;

    % Diagnostic-only inflation so the one-step theta update is visible.
    cfg.dnn.Ptheta0 = max(cfg.dnn.Ptheta0, 1e-1);

    runOneThetaPathCase_step09h3e("fixed_feature_lip", etaTrue0, watcherState, cfg);

    cfg = cfg0;
    cfg.dnn.branchModel = "mlp_general";
    cfg.dnn.predictionResidualSource = "local_DNN";
    cfg.dnn.useResidualInPrediction = true;
    cfg.dnn.residualInjectionGain = 1.0;

    cfg.dnn.Ptheta0 = max(cfg.dnn.Ptheta0, 1e-1);

    cfg.dnn.mlp.hiddenSizes = [12 8 6];
    cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
    cfg.dnn.mlp.inputMode = "eta_phase";
    cfg.dnn.mlp.rScale = 1000.0;
    cfg.dnn.mlp.vScale = 0.1;

    cfg.dnn.mlp.thetaInitMode = "random_hidden_zero_output";
    cfg.dnn.mlp.hiddenWeightStd = 0.1;
    cfg.dnn.mlp.hiddenBiasStd = 0.0;
    cfg.dnn.mlp.outputWeightStd = 0.0;
    cfg.dnn.mlp.outputBiasStd = 0.0;
    cfg.dnn.mlp.thetaInitSeed = 9100;

    runOneThetaPathCase_step09h3e("mlp_general", etaTrue0, watcherState, cfg);

    fprintf("\nPASS: Step 09-H.3e theta learning path check passed.\n");

end

function runOneThetaPathCase_step09h3e(caseName, etaTrue0, watcherState, cfg)

    watcher0 = initLocalDNNEKF(1, etaTrue0, cfg);

    % One prediction step usually creates P_{v theta}, because theta enters
    % acceleration. Bearing-only measurement directly sees position, not
    % velocity, so K_theta may still be zero after only one step.
    %
    % Run several prediction steps so the kinematic coupling
    %
    %     dot r = v
    %
    % transfers P_{v theta} into P_{r theta}.
    watcherPred = watcher0;

    nWarmupPred = 20;

    for kPred = 1:nWarmupPred
        tPred = (kPred - 1) * cfg.dt;
        watcherPred = DNN_EKF_Predict_Local(watcherPred, tPred, cfg);
    end

    idxEta = watcherPred.idxEta;
    idxTheta = watcherPred.idxTheta;

    Ppred = watcherPred.P;

    PetaTheta = Ppred(idxEta, idxTheta);
    PthetaEta = Ppred(idxTheta, idxEta);

    dim = cfg.dim;

    idxRlocal = 1:dim;
    idxVlocal = dim + (1:dim);

    PrTheta = Ppred(idxEta(idxRlocal), idxTheta);
    PvTheta = Ppred(idxEta(idxVlocal), idxTheta);

    PthetaR = Ppred(idxTheta, idxEta(idxRlocal));
    PthetaV = Ppred(idxTheta, idxEta(idxVlocal));

    etaPred = watcherPred.xhat(idxEta);

    Heta = measurementJacobian(etaPred, watcherState, cfg);
    zhat = measurementPrediction(etaPred, watcherState, cfg);
    zhat = zhat(:);

    nz = numel(zhat);

    R = cfg.meas.R;
    if isscalar(R)
        R = R * eye(nz);
    end

    S = Heta * Ppred(idxEta, idxEta) * Heta' + R;
    S = 0.5 * (S + S');

    PHt = Ppred(:, idxEta) * Heta';
    K = PHt / S;

    Ktheta = K(idxTheta, :);

    % Use a slightly larger artificial innovation than Step 09-H.3d so the
    % theta correction is visible in one diagnostic update.
    if nz == 1
        dz = 5e-2;
    else
        dz = zeros(nz, 1);
        dz(1) = 5e-2;
        dz(2) = -2e-2;
    end

    z = zhat + dz;

    thetaPred = watcherPred.xhat(idxTheta);

    watcherUpd = DNN_EKF_Update_Local(watcherPred, z, watcherState, cfg);

    thetaUpd = watcherUpd.xhat(idxTheta);

    dtheta = thetaUpd - thetaPred;

    fprintf("\nCase: %s\n", caseName);
    fprintf("    nTheta                 = %d\n", watcherPred.nTheta);
    fprintf("    ||P_eta_theta^-||_F    = %.12e\n", norm(PetaTheta, "fro"));
    fprintf("    ||P_theta_eta^-||_F    = %.12e\n", norm(PthetaEta, "fro"));
    fprintf("    ||K_theta||_F          = %.12e\n", norm(Ktheta, "fro"));
    fprintf("    ||innovation||         = %.12e\n", norm(watcherUpd.lastInnovation));
    fprintf("    ||dtheta update||      = %.12e\n", norm(dtheta));
    fprintf("    trace(Ptheta pred)     = %.12e\n", trace(Ppred(idxTheta, idxTheta)));
    fprintf("    trace(Ptheta update)   = %.12e\n", trace(watcherUpd.P(idxTheta, idxTheta)));


    fprintf("    ||P_r_theta^-||_F      = %.12e\n", norm(PrTheta, "fro"));
    fprintf("    ||P_v_theta^-||_F      = %.12e\n", norm(PvTheta, "fro"));
    fprintf("    ||P_theta_r^-||_F      = %.12e\n", norm(PthetaR, "fro"));
    fprintf("    ||P_theta_v^-||_F      = %.12e\n", norm(PthetaV, "fro"));

    tol = 1e-14;

    assert(norm(PthetaEta, "fro") > tol, ...
        "%s: prediction did not generate P_theta_eta.", caseName);

    assert(norm(PthetaR, "fro") > tol, ...
        "%s: prediction generated P_theta_eta, but not P_theta_r. Increase nWarmupPred.", caseName);

    assert(norm(Ktheta, "fro") > tol, ...
        "%s: theta Kalman gain is still zero. Bearing measurement may be orthogonal to P_theta_r.", caseName);

    assert(norm(dtheta) > tol, ...
        "%s: theta update is still zero.", caseName);
end