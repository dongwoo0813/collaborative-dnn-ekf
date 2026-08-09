function [dHat, Jeta, Jtheta, cache] = evaluateBranchResidualModel( ...
    branchID, eta, theta, cfg)
%{
File:
    dnn/evaluateBranchResidualModel.m

Purpose:
    Unified residual-branch evaluation wrapper.

Current Step:
    Step 09-H.3a

Why this wrapper is needed:
    The previous estimator branch was a fixed-feature, linear-in-parameters
    model:

        d_i(eta) = W_i phi_i(eta)

    The new branch model is a configurable MLP:

        xi -> hidden layers -> linear output

    Rather than editing every EKF/GS function separately, this wrapper gives
    the rest of the code one common interface:

        [dHat, Jeta, Jtheta] = evaluateBranchResidualModel(...)

Supported branch models:
    cfg.dnn.branchModel = "fixed_feature_lip"
        Existing fixed-feature output-layer branch.

    cfg.dnn.branchModel = "mlp_general"
        New configurable softplus/tanh/etc. MLP branch.

Outputs:
    dHat   : dim x 1 residual acceleration estimate
    Jeta   : dim x nEta Jacobian, partial dHat / partial eta
    Jtheta : dim x nTheta Jacobian, partial dHat / partial theta

Notes:
    - This function does not apply GS additive/gating logic.
    - It only evaluates one branch.
    - GS/local composition should happen outside this function.
%}

    branchModel = getBranchModel_step09h3a(cfg);

    switch branchModel

        case "fixed_feature_lip"
            [dHat, Jeta, Jtheta, cache] = evaluateFixedFeatureBranch_step09h3a( ...
                branchID, eta, theta, cfg);

        case "mlp_general"
            [dHat, Jeta, Jtheta, cache] = branchMLPForwardJacobians( ...
                branchID, eta, theta, cfg);

        otherwise
            error("evaluateBranchResidualModel:UnsupportedBranchModel", ...
                "Unsupported cfg.dnn.branchModel = %s.", branchModel);

    end

end

function branchModel = getBranchModel_step09h3a(cfg)
%GETBRANCHMODEL_STEP09H3A Resolve branch model with backward-compatible default.

    branchModel = "fixed_feature_lip";

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "branchModel")
        branchModel = string(cfg.dnn.branchModel);
    end

    % Backward-compatible aliases.
    if branchModel == "fixed_feature"
        branchModel = "fixed_feature_lip";
    end

    if branchModel == "lip"
        branchModel = "fixed_feature_lip";
    end

    if branchModel == "mlp"
        branchModel = "mlp_general";
    end

end

function [dHat, Jeta, Jtheta, cache] = evaluateFixedFeatureBranch_step09h3a( ...
    branchID, eta, theta, cfg)
%EVALUATEFIXEDFEATUREBRANCH_STEP09H3A Self-contained fixed-feature branch.
%
% Model:
%     d_i(eta) = W_i phi_i(eta)
%
% where:
%     theta_i = vec(W_i)
%     W_i     = dim x nPhi
%
% Current feature map:
%     eta = [r; v]
%
%     rBar = r / rScale
%     vBar = v / vScale
%
%     psi_i = 2*pi*(branchID-1)/Nw
%
%     phi_i =
%       [
%         1;
%         rBar;
%         vBar;
%         rBar' rBar;
%         sin(psi_i + sum(rBar));
%         cos(psi_i + sum(vBar));
%         rBar' vBar
%       ]
%
% For dim = 2:
%     nPhi = 2*dim + 5 = 9
%
% Jacobians:
%     Jeta   = W_i * dphi/deta
%     Jtheta = phi_i^T kron I_dim
%
% This function intentionally does not call branchResidual(), because that
% helper does not exist in the current project.

    dim = cfg.dim;
    nEta = 2*dim;
    
    if numel(eta) < nEta
        error("evaluateBranchResidualModel:BadEta", ...
            "eta must contain [r; v] with length at least 2*cfg.dim.");
    end
    
    eta = eta(1:nEta);
    theta = theta(:);
    
    [phi, JphiEta, featureCache] = fixedFeatureBranchFeatures_step09h3a( ...
        branchID, eta, cfg);
    
    nPhi = numel(phi);
    nThetaExpected = dim * nPhi;
    
    if numel(theta) ~= nThetaExpected
        error("evaluateBranchResidualModel:BadFixedThetaLength", ...
            "fixed_feature_lip expected theta length %d, got %d.", ...
            nThetaExpected, numel(theta));
    end
    
    W = reshape(theta, dim, nPhi);
    
    dHat = W * phi;
    
    Jeta = W * JphiEta;
    
    % MATLAB column-major packing:
    %     theta = vec(W)
    %
    % For d = W phi:
    %     partial d / partial vec(W) = phi^T kron I_dim
    Jtheta = kron(phi.', eye(dim));
    
    cache = struct();
    cache.branchModel = "fixed_feature_lip";
    cache.branchID = branchID;
    cache.phi = phi;
    cache.JphiEta = JphiEta;
    cache.W = W;
    cache.feature = featureCache;

end



function [phi, JphiEta, cache] = fixedFeatureBranchFeatures_step09h3a( ...
    branchID, eta, cfg)
%FIXEDFEATUREBRANCHFEATURES_STEP09H3A Fixed features and analytic Jacobian.
%
% Feature map:
%     phi =
%       [
%         1;
%         rBar;
%         vBar;
%         rBar' rBar;
%         sin(psi + sum(rBar));
%         cos(psi + sum(vBar));
%         rBar' vBar
%       ]
%
% Output:
%     phi     : nPhi x 1
%     JphiEta : nPhi x 2*dim

    dim = cfg.dim;
    nEta = 2*dim;

    r = eta(1:dim);
    v = eta(dim+1:nEta);

    rScale = getFirstAvailableScalar_step09h3a(cfg, ...
        { ...
            {"dnn","feature","rScale"}, ...
            {"dnn","feature","positionScale"}, ...
            {"dnn","rScale"}, ...
            {"dnn","positionScale"} ...
        }, ...
        1000.0);

    vScale = getFirstAvailableScalar_step09h3a(cfg, ...
        { ...
            {"dnn","feature","vScale"}, ...
            {"dnn","feature","velocityScale"}, ...
            {"dnn","vScale"}, ...
            {"dnn","velocityScale"} ...
        }, ...
        1.0);

    rScale = max(rScale, eps);
    vScale = max(vScale, eps);

    rBar = r / rScale;
    vBar = v / vScale;

    if isfield(cfg, "Nw")
        Nw = cfg.Nw;
    else
        Nw = 1;
    end

    psi = 2*pi*(branchID - 1) / max(Nw, 1);

    sumR = sum(rBar);
    sumV = sum(vBar);

    rr = rBar.' * rBar;
    rv = rBar.' * vBar;

    sR = sin(psi + sumR);
    cV = cos(psi + sumV);

    phi = [
        1.0;
        rBar;
        vBar;
        rr;
        sR;
        cV;
        rv
    ];

    nPhi = numel(phi);

    JphiEta = zeros(nPhi, nEta);

    row = 1;

    % phi_1 = 1
    row = row + 1;

    % phi = rBar
    JphiEta(row:row+dim-1, 1:dim) = eye(dim) / rScale;
    row = row + dim;

    % phi = vBar
    JphiEta(row:row+dim-1, dim+1:nEta) = eye(dim) / vScale;
    row = row + dim;

    % phi = rBar' rBar
    JphiEta(row, 1:dim) = (2.0 * rBar).' / rScale;
    row = row + 1;

    % phi = sin(psi + sum(rBar))
    JphiEta(row, 1:dim) = cos(psi + sumR) * ones(1, dim) / rScale;
    row = row + 1;

    % phi = cos(psi + sum(vBar))
    JphiEta(row, dim+1:nEta) = -sin(psi + sumV) * ones(1, dim) / vScale;
    row = row + 1;

    % phi = rBar' vBar
    JphiEta(row, 1:dim) = vBar.' / rScale;
    JphiEta(row, dim+1:nEta) = rBar.' / vScale;
    row = row + 1;

    if row ~= nPhi + 1
        error("evaluateBranchResidualModel:FeatureIndexError", ...
            "Feature Jacobian row indexing ended at %d, expected %d.", ...
            row, nPhi + 1);
    end

    cache = struct();
    cache.rScale = rScale;
    cache.vScale = vScale;
    cache.Nw = Nw;
    cache.psi = psi;
    cache.rBar = rBar;
    cache.vBar = vBar;
    cache.nPhi = nPhi;

end

function value = getFirstAvailableScalar_step09h3a(cfg, pathList, defaultValue)
%GETFIRSTAVAILABLESCALAR_STEP09H3A Read first available nested scalar field.

    value = defaultValue;

    for iPath = 1:numel(pathList)

        candidate = getNestedFieldOrDefault_step09h3a( ...
            cfg, pathList{iPath}, []);

        if ~isempty(candidate) && isscalar(candidate) && isfinite(candidate)
            value = candidate;
            return;
        end

    end

end

function value = getNestedFieldOrDefault_step09h3a(s, path, defaultValue)
%GETNESTEDFIELDORDEFAULT_STEP09H3A Read nested struct field with default.

    value = defaultValue;
    current = s;

    for k = 1:numel(path)
        name = path{k};

        if ~isstruct(current) || ~isfield(current, name)
            return;
        end

        current = current.(name);
    end

    if ~isempty(current)
        value = current;
    end

end