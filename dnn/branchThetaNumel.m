function [nTheta, info] = branchThetaNumel(cfg)
%{
File:
    dnn/branchThetaNumel.m

Purpose:
    Return the number of trainable parameters in one residual branch.

Why this helper is needed:
    The old fixed-feature LIP branch and the new MLP branch have different
    parameter dimensions.

        fixed_feature_lip : theta = vec(W), usually 18 in 2-D
        mlp_general       : theta = vec(all DNN weights/biases), e.g. 256

    EKF initialization, covariance allocation, GS cache allocation, and
    prediction Jacobians should all use this helper instead of hard-coding
    cfg.dnn.nThetaBranch.

Supported branch models:
    "fixed_feature_lip"
    "mlp_general"
%}

branchModel = "fixed_feature_lip";

if isfield(cfg, "dnn") && isfield(cfg.dnn, "branchModel")
    branchModel = string(cfg.dnn.branchModel);
end

% Backward-compatible aliases.
if branchModel == "fixed_feature" || branchModel == "lip"
    branchModel = "fixed_feature_lip";
elseif branchModel == "mlp"
    branchModel = "mlp_general";
end

switch branchModel

    case "fixed_feature_lip"

        dim = cfg.dim;

        if isfield(cfg, "dnn") && isfield(cfg.dnn, "nFeatureBranch")
            nPhi = cfg.dnn.nFeatureBranch;
        else
            nPhi = 2*dim + 5;
        end

        nTheta = dim * nPhi;

        info = struct();
        info.branchModel = branchModel;
        info.nPhi = nPhi;
        info.nTheta = nTheta;

    case "mlp_general"

        [nTheta, arch] = branchMLPThetaNumel(cfg);

        info = struct();
        info.branchModel = branchModel;
        info.arch = arch;
        info.nTheta = nTheta;

    otherwise

        error("branchThetaNumel:UnsupportedBranchModel", ...
            "Unsupported cfg.dnn.branchModel = %s.", branchModel);

end

end