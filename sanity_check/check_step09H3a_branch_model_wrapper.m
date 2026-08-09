function check_step09H3a_branch_model_wrapper()
%{
File:
    checks/check_step09H3a_branch_model_wrapper.m

Purpose:
    Check the unified branch residual wrapper:

        evaluateBranchResidualModel(...)

What this verifies:
    1. The old fixed-feature branch still works through the wrapper.
    2. The new mlp_general branch works through the wrapper.
    3. Both return finite dHat, Jeta, Jtheta with consistent dimensions.

This does not connect the MLP to the EKF yet.
%}

    addpath(genpath(pwd));
    rehash;

    rng(101);

    cfg0 = config_step04_GS_DNN_EKF();

    dim = cfg0.dim;
    nEta = 2*dim;
    branchID = 2;

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
        error("check_step09H3a:UnsupportedDim", ...
            "This check supports cfg.dim = 2 or 3.");
    end

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-H.3a check: branch model wrapper\n");
    fprintf("============================================================\n");

    % ---------------------------------------------------------------------
    % Case 1: old fixed-feature branch
    % ---------------------------------------------------------------------
    cfg = cfg0;
    cfg.dnn.branchModel = "fixed_feature_lip";

    nThetaFixed = getFixedFeatureThetaNumel_step09h3a(cfg);
    
    thetaFixed = 0.05 * randn(nThetaFixed, 1);

    [dFixed, JetaFixed, JthetaFixed] = evaluateBranchResidualModel( ...
        branchID, eta, thetaFixed, cfg);

    assertBranchOutput_step09h3a( ...
        "fixed_feature_lip", dFixed, JetaFixed, JthetaFixed, ...
        dim, nEta, nThetaFixed);

    fprintf("\nCase: fixed_feature_lip\n");
    fprintf("    nTheta        = %d\n", nThetaFixed);
    fprintf("    ||dHat||      = %.6e\n", norm(dFixed));
    fprintf("    size(Jeta)    = %s\n", mat2str(size(JetaFixed)));
    fprintf("    size(Jtheta)  = %s\n", mat2str(size(JthetaFixed)));

    % ---------------------------------------------------------------------
    % Case 2: new general MLP branch
    % ---------------------------------------------------------------------
    cfg = cfg0;
    cfg.dnn.branchModel = "mlp_general";
    cfg.dnn.mlp.hiddenSizes = [12 8 6];
    cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
    cfg.dnn.mlp.inputMode = "eta_phase";
    cfg.dnn.mlp.rScale = 1000.0;
    cfg.dnn.mlp.vScale = 0.1;

    [nThetaMLP, arch] = branchMLPThetaNumel(cfg);
    thetaMLP = 0.05 * randn(nThetaMLP, 1);

    [dMLP, JetaMLP, JthetaMLP] = evaluateBranchResidualModel( ...
        branchID, eta, thetaMLP, cfg);

    assertBranchOutput_step09h3a( ...
        "mlp_general", dMLP, JetaMLP, JthetaMLP, ...
        dim, nEta, nThetaMLP);

    fprintf("\nCase: mlp_general\n");
    fprintf("    layerSizes    = %s\n", mat2str(arch.layerSizes));
    fprintf("    nTheta        = %d\n", nThetaMLP);
    fprintf("    ||dHat||      = %.6e\n", norm(dMLP));
    fprintf("    size(Jeta)    = %s\n", mat2str(size(JetaMLP)));
    fprintf("    size(Jtheta)  = %s\n", mat2str(size(JthetaMLP)));

    fprintf("\nPASS: Step 09-H.3a branch model wrapper check passed.\n");

end

function assertBranchOutput_step09h3a( ...
    caseName, dHat, Jeta, Jtheta, dim, nEta, nTheta)

    assert(all(isfinite(dHat)), ...
        "%s dHat contains non-finite values.", caseName);

    assert(all(isfinite(Jeta(:))), ...
        "%s Jeta contains non-finite values.", caseName);

    assert(all(isfinite(Jtheta(:))), ...
        "%s Jtheta contains non-finite values.", caseName);

    assert(isequal(size(dHat), [dim 1]), ...
        "%s dHat has wrong size.", caseName);

    assert(isequal(size(Jeta), [dim nEta]), ...
        "%s Jeta has wrong size.", caseName);

    assert(isequal(size(Jtheta), [dim nTheta]), ...
        "%s Jtheta has wrong size.", caseName);

end


function nTheta = getFixedFeatureThetaNumel_step09h3a(cfg)
%GETFIXEDFEATURETHETANUMEL_STEP09H3A Number of fixed-feature branch params.
%
% Current fixed-feature model:
%     d_i = W_i phi_i(eta)
%
% with
%     W_i in R^{dim x nPhi}
%     theta_i = vec(W_i)
%
% Therefore:
%     nTheta = dim*nPhi
%
% Prefer cfg.dnn.nThetaBranch when available. Otherwise fall back to the
% known current feature size nPhi = 2*dim + 5.
    
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "nThetaBranch")
        nTheta = cfg.dnn.nThetaBranch;
        return;
    end
    
    dim = cfg.dim;
    
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "nFeatureBranch")
        nPhi = cfg.dnn.nFeatureBranch;
    else
        nPhi = 2*dim + 5;
    end
    
    nTheta = dim * nPhi;

end