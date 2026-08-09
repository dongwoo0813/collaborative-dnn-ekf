function [branchTruth, thetaStarMat] = truthBranchWeights(cfg)
%{
Function:
    truthBranchWeights.m

Purpose:
    Generate deterministic hidden truth weights for the branch-wise unknown
    residual model.

    Later, the true unknown residual acceleration can be written as

        d_unk(eta)
        =
        sum_{j=1}^{Nw} W_j^star phi_j(eta),

    where each branch has the same fixed-feature structure used by the
    DNN-EKF model,

        d_j(eta; theta_j) = W_j phi_j(eta),

    and

        theta_j = vec(W_j).

Inputs:
    cfg          - Simulation configuration structure.
                   Required fields:
                       cfg.dim
                       cfg.Nw
                       cfg.target.r0
                       cfg.target.v0

                   Optional fields:
                       cfg.truth.residualAmp
                       cfg.truth.branchWeightScale

Outputs:
    branchTruth  - Struct array containing truth branch records.
                   Size: cfg.Nw x 1.

                   Fields:
                       branchTruth(j).id
                       branchTruth(j).theta
                       branchTruth(j).W
                       branchTruth(j).active

    thetaStarMat - Matrix of true branch parameters.
                   Size: nTheta x cfg.Nw.
                   Column j is theta_j^star.

Main equations:
    For branch j,

        d_j^star(eta) = W_j^star phi_j(eta),

    where

        theta_j^star = vec(W_j^star).

    The deterministic entries of W_j^star are generated using smooth
    sine/cosine patterns so that every run gives the same hidden residual.

Notes:
    - This function does not use random numbers.
    - Therefore it does not affect MATLAB's global random seed.
    - The generated weights are hidden truth parameters. The EKF should not
      be initialized with these values.
    - The output branchTruth is compatible with compositeResidual.m because
      it contains branchTruth(j).theta.
%}

    dim = cfg.dim;
    Nw = cfg.Nw;

    % Use the initial target state only to infer the feature dimension.
    etaRef = [cfg.target.r0; cfg.target.v0];

    phiRef = featureBlock(1, etaRef, cfg);
    nPhi = numel(phiRef);

    nTheta = dim * nPhi;

    if isfield(cfg, "truth") && isfield(cfg.truth, "branchWeightScale")
        weightScale = cfg.truth.branchWeightScale;
    elseif isfield(cfg, "truth") && isfield(cfg.truth, "residualAmp")
        weightScale = cfg.truth.residualAmp / (Nw * sqrt(nPhi));
    else
        weightScale = 1e-4 / (Nw * sqrt(nPhi));
    end

    template.id = [];
    template.theta = [];
    template.W = [];
    template.active = true;

    branchTruth = repmat(template, Nw, 1);
    thetaStarMat = zeros(nTheta, Nw);

    for j = 1:Nw

        Wj = zeros(dim, nPhi);

        for d = 1:dim
            for ell = 1:nPhi

                phase1 = 0.7*j + 0.3*d + 0.5*ell;
                phase2 = 0.2*j - 0.4*d + 0.8*ell;

                Wj(d, ell) = weightScale * ...
                    (0.6*sin(phase1) + 0.4*cos(phase2));

            end
        end

        theta_j = Wj(:);

        branchTruth(j).id = j;
        branchTruth(j).theta = theta_j;
        branchTruth(j).W = Wj;
        branchTruth(j).active = true;

        thetaStarMat(:,j) = theta_j;

    end

end