function [dHat, Jeta, Jtheta, cache] = branchMLPForwardJacobians( ...
    branchID, eta, theta, cfg)
%{
File:
    dnn/branchMLPForwardJacobians.m

Purpose:
    Evaluate one configurable MLP residual branch and analytic Jacobians.

Current Step:
    Step 09-H.2a-general

Network:
    z0 = xi

    for ell = 1,...,L-1:
        a_ell = W_ell z_{ell-1} + b_ell
        z_ell = sigma_ell(a_ell)

    output:
        dHat = W_L z_{L-1} + b_L

Supported hidden activations:
    "softplus"
    "tanh"
    "sigmoid"
    "linear"
    "relu"

Outputs:
    dHat   : dim x 1
    Jeta   : dim x nEta
    Jtheta : dim x nTheta

Why this version:
    This does not assume a fixed number of layers.
    It supports arbitrary hiddenSizes and activation lists.

Parameter packing:
    theta = [
        W1(:); b1;
        W2(:); b2;
        ...
        WL(:); bL
    ]

Jacobian method:
    Uses standard backpropagation sensitivities.

    S_ell = partial dHat / partial a_ell

    Then

        partial dHat / partial W_ell(p,q)
            = S_ell(:,p) * z_{ell-1}(q)

        partial dHat / partial b_ell(p)
            = S_ell(:,p)

    This is computationally fine for the branch dimensions we are using and
    is more general than manually writing kron expressions for each layer.
%}

    dim = cfg.dim;
    nEta = 2*dim;

    if numel(eta) < nEta
        error("branchMLPForwardJacobians:BadEta", ...
            "eta must contain [r; v] with length at least 2*cfg.dim.");
    end

    eta = eta(1:nEta);

    arch = branchMLPArchitecture(cfg);

    if numel(theta) ~= arch.nTheta
        error("branchMLPForwardJacobians:BadThetaLength", ...
            "Expected theta length %d, got %d.", ...
            arch.nTheta, numel(theta));
    end

    theta = theta(:);

    [W, b] = unpackTheta_step09h2g(theta, arch);

    [xi, DxiDeta, inputCache] = buildInput_step09h2g( ...
        branchID, eta, cfg, arch);

    L = arch.nLayers;

    a = cell(L, 1);
    z = cell(L+1, 1);
    sigmaPrime = cell(L, 1);

    z{1} = xi;

    % ---------------------------------------------------------------------
    % Forward pass
    % ---------------------------------------------------------------------
    for ell = 1:L

        a{ell} = W{ell} * z{ell} + b{ell};

        actName = arch.layers(ell).activation;

        [z{ell+1}, sigmaPrime{ell}] = activationForward_step09h2g( ...
            a{ell}, actName);

    end

    dHat = z{L+1};

    % Output must be dim x 1.
    if ~isequal(size(dHat), [dim 1])
        error("branchMLPForwardJacobians:BadOutputSize", ...
            "dHat size is %s, expected [%d 1].", mat2str(size(dHat)), dim);
    end

    % ---------------------------------------------------------------------
    % Backprop sensitivities
    % ---------------------------------------------------------------------
    % S{ell} = partial dHat / partial a{ell}, size dim x nOut_ell.
    S = cell(L, 1);

    % Output layer is linear, so dHat = a_L.
    S{L} = eye(dim);

    for ell = L:-1:2

        % partial dHat / partial z_{ell-1}
        SdZprev = S{ell} * W{ell};  % dim x nIn_ell

        % z_{ell-1} = sigma_{ell-1}(a_{ell-1})
        % Multiply each column by sigmaPrime{ell-1}.
        S{ell-1} = SdZprev .* (sigmaPrime{ell-1}(:).');

    end

    % ---------------------------------------------------------------------
    % Jacobian wrt eta
    % ---------------------------------------------------------------------
    % partial dHat / partial xi = S1 * W1
    Jxi = S{1} * W{1};

    Jeta = Jxi * DxiDeta;

    % ---------------------------------------------------------------------
    % Jacobian wrt theta
    % ---------------------------------------------------------------------
    Jtheta = zeros(dim, arch.nTheta);

    idx = 1;

    for ell = 1:L

        zPrev = z{ell};

        nOut = arch.layers(ell).nOut;
        nIn  = arch.layers(ell).nIn;

        nW = nOut * nIn;

        % Weight block:
        % d dHat / d vec(W_ell) = zPrev^T \otimes S_ell
        Jtheta(:, idx:idx+nW-1) = kron(zPrev.', S{ell});
        idx = idx + nW;

        % Bias block:
        % d dHat / d b_ell = S_ell
        Jtheta(:, idx:idx+nOut-1) = S{ell};
        idx = idx + nOut;

    end

    if idx ~= arch.nTheta + 1
        error("branchMLPForwardJacobians:InternalIndexError", ...
            "Jtheta packing ended at idx=%d, expected %d.", ...
            idx, arch.nTheta + 1);
    end

    % ---------------------------------------------------------------------
    % Debug cache
    % ---------------------------------------------------------------------
    cache = struct();

    cache.arch = arch;
    cache.input = inputCache;

    cache.xi = xi;
    cache.DxiDeta = DxiDeta;

    cache.W = W;
    cache.b = b;

    cache.a = a;
    cache.z = z;
    cache.sigmaPrime = sigmaPrime;
    cache.S = S;

    cache.Jxi = Jxi;

end

function [W, b] = unpackTheta_step09h2g(theta, arch)
%UNPACKTHETA_STEP09H2G Unpack theta into cell arrays W{ell}, b{ell}.

    L = arch.nLayers;

    W = cell(L, 1);
    b = cell(L, 1);

    for ell = 1:L

        nOut = arch.layers(ell).nOut;
        nIn = arch.layers(ell).nIn;

        idxW = arch.layers(ell).idxW;
        idxb = arch.layers(ell).idxb;

        W{ell} = reshape(theta(idxW), nOut, nIn);
        b{ell} = theta(idxb);

    end

end

function [xi, DxiDeta, inputCache] = buildInput_step09h2g( ...
    branchID, eta, cfg, arch)
%BUILDINPUT_STEP09H2G Build branch MLP input and its eta Jacobian.
%
% Supported input modes:
%     "eta_only":
%         xi = [r/rScale; v/vScale]
%
%     "eta_phase":
%         xi = [r/rScale; v/vScale; sin(psi_i); cos(psi_i)]
%
% Later we can add:
%     "watcher_geometry"
%         xi = [eta; watcher-relative geometry; LOS; phase]
%
% without changing the network/Jacobian code.

    dim = cfg.dim;
    nEta = 2*dim;

    r = eta(1:dim);
    v = eta(dim+1:nEta);

    rScale = getNestedFieldOrDefault_step09h2g( ...
        cfg, {"dnn","mlp","rScale"}, 1000.0);

    vScale = getNestedFieldOrDefault_step09h2g( ...
        cfg, {"dnn","mlp","vScale"}, 0.1);

    inputMode = arch.inputMode;

    DxiDeta = zeros(arch.inputDim, nEta);

    switch inputMode

        case "eta_only"

            xi = [
                r / max(rScale, eps);
                v / max(vScale, eps)
            ];

            DxiDeta(1:dim, 1:dim) = eye(dim) / max(rScale, eps);
            DxiDeta(dim+1:2*dim, dim+1:2*dim) = eye(dim) / max(vScale, eps);

            psi = NaN;

        case "eta_phase"

            if isfield(cfg, "Nw")
                Nw = cfg.Nw;
            else
                Nw = 1;
            end

            psi = 2*pi*(branchID - 1) / max(Nw, 1);

            xi = [
                r / max(rScale, eps);
                v / max(vScale, eps);
                sin(psi);
                cos(psi)
            ];

            DxiDeta(1:dim, 1:dim) = eye(dim) / max(rScale, eps);
            DxiDeta(dim+1:2*dim, dim+1:2*dim) = eye(dim) / max(vScale, eps);

        otherwise
            error("branchMLPForwardJacobians:UnsupportedInputMode", ...
                "Unsupported input mode: %s.", inputMode);

    end

    if numel(xi) ~= arch.inputDim
        error("branchMLPForwardJacobians:BadInputDim", ...
            "Built xi has length %d, expected %d.", ...
            numel(xi), arch.inputDim);
    end

    inputCache = struct();

    inputCache.inputMode = inputMode;
    inputCache.branchID = branchID;
    inputCache.psi = psi;
    inputCache.rScale = rScale;
    inputCache.vScale = vScale;

end

function [y, dy] = activationForward_step09h2g(x, actName)
%ACTIVATIONFORWARD_STEP09H2G Activation and derivative.

    actName = lower(string(actName));

    switch actName

        case {"linear", "identity"}
            y = x;
            dy = ones(size(x));

        case "tanh"
            y = tanh(x);
            dy = 1.0 - y.^2;

        case "softplus"
            y = log1p(exp(-abs(x))) + max(x, 0);
            dy = sigmoid_step09h2g(x);

        case "sigmoid"
            y = sigmoid_step09h2g(x);
            dy = y .* (1.0 - y);

        case "relu"
            y = max(x, 0);
            dy = double(x > 0);

        otherwise
            error("branchMLPForwardJacobians:UnsupportedActivation", ...
                "Unsupported activation: %s.", actName);

    end

end

function y = sigmoid_step09h2g(x)
%SIGMOID_STEP09H2G Numerically stable sigmoid.

    y = zeros(size(x));

    pos = x >= 0;
    neg = ~pos;

    y(pos) = 1 ./ (1 + exp(-x(pos)));

    ex = exp(x(neg));
    y(neg) = ex ./ (1 + ex);

end

function value = getNestedFieldOrDefault_step09h2g(s, path, defaultValue)
%GETNESTEDFIELDORDEFAULT_STEP09H2G Read nested struct field with default.

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