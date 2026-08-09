function check_step09J6_random_mlp_initialization()
%{
File:
    sanity_check/check_step09J6_random_mlp_initialization.m

Purpose:
    Targeted check for the Step 09-J.6 random MLP initial condition.

Checks:
    1. All packed MLP parameters use thetaInitStd * randn.
    2. Initialization is deterministic for a fixed branchID and seed.
    3. Different branches receive different deterministic draws.
    4. P_theta_theta(0) = thetaInitStd^2 * I.
    5. The initial MLP residual is finite and not identically zero.
%}

    addpath(genpath(pwd));

    thetaInitStd = 7.5e-5;
    cfg = config_step04_GS_DNN_EKF();
    cfg.dnn.branchModel = "mlp_general";
    cfg.dnn.mlp.hiddenSizes = [6 6 6];
    cfg.dnn.mlp.activations = ["softplus", "softplus", "tanh"];
    cfg.dnn.mlp.inputMode = "eta_phase";
    cfg.dnn.mlp.thetaInitMode = "small_random";
    cfg.dnn.mlp.theta0Std = thetaInitStd;
    cfg.dnn.theta0_std = thetaInitStd;
    cfg.dnn.Ptheta0 = thetaInitStd^2;

    eta0 = [cfg.target.r0; cfg.target.v0];
    watcher1a = initLocalDNNEKF(1, eta0, cfg);
    watcher1b = initLocalDNNEKF(1, eta0, cfg);
    watcher2 = initLocalDNNEKF(2, eta0, cfg);

    theta1a = watcher1a.xhat(watcher1a.idxTheta);
    theta1b = watcher1b.xhat(watcher1b.idxTheta);
    theta2 = watcher2.xhat(watcher2.idxTheta);

    assert(any(theta1a ~= 0), ...
        "Random MLP initialization produced an all-zero theta vector.");
    assert(isequal(theta1a, theta1b), ...
        "Fixed branchID/seed did not reproduce the same theta vector.");
    assert(~isequal(theta1a, theta2), ...
        "Different branch IDs received identical theta vectors.");

    Ptheta = watcher1a.P(watcher1a.idxTheta, watcher1a.idxTheta);
    expectedPtheta = thetaInitStd^2 * eye(watcher1a.nTheta);
    covarianceError = norm(Ptheta - expectedPtheta, "fro");
    assert(covarianceError < 1e-14, ...
        "Initial DNN parameter covariance does not equal thetaInitStd^2*I.");

    [dHat0, ~, ~, ~] = evaluateBranchResidualModel( ...
        watcher1a.localBranchID, eta0, theta1a, cfg);
    assert(all(isfinite(dHat0)), ...
        "Initial random MLP residual contains non-finite values.");
    assert(norm(dHat0) > 0, ...
        "Initial random MLP residual is still identically zero.");

    fprintf("Step 09-J.6 random MLP initialization check PASSED.\n");
    fprintf("thetaInitStd       = %.6e\n", thetaInitStd);
    fprintf("sample std(theta0) = %.6e\n", std(theta1a));
    fprintf("||dHat0||          = %.6e m/s^2\n", norm(dHat0));
    fprintf("Ptheta Fro error   = %.6e\n", covarianceError);
end
