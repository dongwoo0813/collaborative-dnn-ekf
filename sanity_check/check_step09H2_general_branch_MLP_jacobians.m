function check_step09H2_general_branch_MLP_jacobians()
%{
File:
    checks/check_step09H2_general_branch_MLP_jacobians.m

Purpose:
    Check the generalized MLP residual branch Jacobians.

What this verifies:
    1. Different hidden layer dimensions work.
    2. Different hidden layer counts work.
    3. Different activation lists work.
    4. Analytic Jeta matches finite-difference Jeta.
    5. Analytic Jtheta matches finite-difference Jtheta.

This is not an EKF simulation.
This only checks the branch DNN forward/Jacobian engine.
%}

    addpath(genpath(pwd));
    rehash;

    rng(101);

    cfg0 = config_step04_GS_DNN_EKF();

    testCases = {
        struct( ...
            "name", "one_hidden_tanh", ...
            "hiddenSizes", [10], ...
            "activations", ["tanh"], ...
            "inputMode", "eta_phase")

        struct( ...
            "name", "two_hidden_softplus_tanh", ...
            "hiddenSizes", [8 6], ...
            "activations", ["softplus", "tanh"], ...
            "inputMode", "eta_phase")

        struct( ...
            "name", "three_hidden_softplus_tanh_tanh", ...
            "hiddenSizes", [12 8 6], ...
            "activations", ["softplus", "tanh", "tanh"], ...
            "inputMode", "eta_phase")

        struct( ...
            "name", "eta_only_two_hidden", ...
            "hiddenSizes", [8 5], ...
            "activations", ["softplus", "tanh"], ...
            "inputMode", "eta_only")
    };

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.2 general check: branch MLP Jacobians\n");
    fprintf("============================================================\n");

    for iCase = 1:numel(testCases)

        tc = testCases{iCase};

        cfg = cfg0;

        cfg.dnn.branchModel = "mlp_general";
        cfg.dnn.mlp.hiddenSizes = tc.hiddenSizes;
        cfg.dnn.mlp.activations = tc.activations;
        cfg.dnn.mlp.inputMode = tc.inputMode;

        cfg.dnn.mlp.rScale = 1000.0;
        cfg.dnn.mlp.vScale = 0.1;

        runOneCase_step09h2g(tc.name, cfg);

    end

    fprintf("\nPASS: all Step 09-H.2 general MLP Jacobian checks passed.\n");

end

function runOneCase_step09h2g(caseName, cfg)

    dim = cfg.dim;
    nEta = 2*dim;

    branchID = 2;

    [nTheta, arch] = branchMLPThetaNumel(cfg);

    theta = 0.05 * randn(nTheta, 1);

    if dim == 2
        eta = [
            800.0;
           -150.0;
              0.08;
             -0.04
        ];
    elseif dim == 3
        eta = [
            800.0;
           -150.0;
            120.0;
              0.08;
             -0.04;
              0.02
        ];
    else
        error("check_step09H2_general:UnsupportedDim", ...
            "This check supports cfg.dim = 2 or 3.");
    end

    [dHat, Jeta, Jtheta, cache] = branchMLPForwardJacobians( ...
        branchID, eta, theta, cfg);

    assert(all(isfinite(dHat)), "dHat contains non-finite values.");
    assert(all(isfinite(Jeta(:))), "Jeta contains non-finite values.");
    assert(all(isfinite(Jtheta(:))), "Jtheta contains non-finite values.");

    assert(isequal(size(dHat), [dim 1]), "Bad dHat size.");
    assert(isequal(size(Jeta), [dim nEta]), "Bad Jeta size.");
    assert(isequal(size(Jtheta), [dim nTheta]), "Bad Jtheta size.");

    JetaFD = finiteDifferenceEta_step09h2g( ...
        branchID, eta, theta, cfg, dim, nEta);

    JthetaFD = finiteDifferenceTheta_step09h2g( ...
        branchID, eta, theta, cfg, dim, nTheta);

    absErrEta = norm(Jeta - JetaFD, "fro");
    relErrEta = absErrEta / max(1.0, norm(JetaFD, "fro"));

    absErrTheta = norm(Jtheta - JthetaFD, "fro");
    relErrTheta = absErrTheta / max(1.0, norm(JthetaFD, "fro"));

    fprintf("\nCase: %s\n", caseName);
    fprintf("    inputMode       = %s\n", arch.inputMode);
    fprintf("    layerSizes      = %s\n", mat2str(arch.layerSizes));
    fprintf("    activations     = %s\n", strjoin(string({arch.layers.activation}), ", "));
    fprintf("    nTheta          = %d\n", nTheta);
    fprintf("    ||dHat||        = %.6e\n", norm(dHat));
    fprintf("    ||Jeta-JetaFD|| = %.6e, rel = %.6e\n", absErrEta, relErrEta);
    fprintf("    ||Jth-JthFD||   = %.6e, rel = %.6e\n", absErrTheta, relErrTheta);

    % Finite-difference can be slightly noisy if the network output is tiny.
    assert(absErrEta < 1e-7 || relErrEta < 1e-5, ...
        "Jeta finite-difference check failed for %s.", caseName);

    assert(absErrTheta < 1e-7 || relErrTheta < 1e-5, ...
        "Jtheta finite-difference check failed for %s.", caseName);

    % This also confirms that the input cache is internally consistent.
    assert(numel(cache.xi) == arch.inputDim, ...
        "Cached xi has wrong length.");

end

function JetaFD = finiteDifferenceEta_step09h2g( ...
    branchID, eta, theta, cfg, dim, nEta)

    JetaFD = zeros(dim, nEta);

    for k = 1:nEta

        h = 1e-6 * max(1.0, abs(eta(k)));

        etaPlus = eta;
        etaMinus = eta;

        etaPlus(k) = etaPlus(k) + h;
        etaMinus(k) = etaMinus(k) - h;

        dPlus = branchMLPForwardOnly_step09h2g( ...
            branchID, etaPlus, theta, cfg);

        dMinus = branchMLPForwardOnly_step09h2g( ...
            branchID, etaMinus, theta, cfg);

        JetaFD(:, k) = (dPlus - dMinus) / (2*h);

    end

end

function JthetaFD = finiteDifferenceTheta_step09h2g( ...
    branchID, eta, theta, cfg, dim, nTheta)

    JthetaFD = zeros(dim, nTheta);

    for k = 1:nTheta

        h = 1e-6 * max(1.0, abs(theta(k)));

        thetaPlus = theta;
        thetaMinus = theta;

        thetaPlus(k) = thetaPlus(k) + h;
        thetaMinus(k) = thetaMinus(k) - h;

        dPlus = branchMLPForwardOnly_step09h2g( ...
            branchID, eta, thetaPlus, cfg);

        dMinus = branchMLPForwardOnly_step09h2g( ...
            branchID, eta, thetaMinus, cfg);

        JthetaFD(:, k) = (dPlus - dMinus) / (2*h);

    end

end

function dHat = branchMLPForwardOnly_step09h2g(branchID, eta, theta, cfg)
%BRANCHMLPFORWARDONLY_STEP09H2G Convenience wrapper for finite differences.

    [dHat, ~, ~] = branchMLPForwardJacobians(branchID, eta, theta, cfg);

end