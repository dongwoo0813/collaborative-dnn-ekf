function arch = branchMLPArchitecture(cfg)
%{
File:
    dnn/branchMLPArchitecture.m

Purpose:
    Build a general configurable MLP residual-branch architecture.

Current Step:
    Step 09-H.2a-general

Supported model:
    xi -> hidden layers -> linear output

Example:
    cfg.dnn.mlp.hiddenSizes = [12 8 6];
    cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];

    This creates:
        z1 = softplus(W1 xi + b1)
        z2 = tanh(W2 z1 + b2)
        z3 = tanh(W3 z2 + b3)
        d  = W4 z3 + b4

Parameter packing:
    theta = [
        W1(:); b1;
        W2(:); b2;
        ...
        WL(:); bL
    ]

    MATLAB column-major vec convention is used.

Notes:
    - Output layer is linear by default.
    - Hidden layer count and dimensions are arbitrary.
    - This helper is used by branchMLPThetaNumel and
      branchMLPForwardJacobians.
%}

    dim = cfg.dim;

    inputMode = string(getNestedFieldOrDefault_step09h2g( ...
        cfg, {"dnn","mlp","inputMode"}, "eta_phase"));

    switch inputMode

        case "eta_only"
            inputDim = 2*dim;

        case "eta_phase"
            % xi = [r/rScale; v/vScale; sin(psi_i); cos(psi_i)]
            inputDim = 2*dim + 2;

        otherwise
            error("branchMLPArchitecture:UnsupportedInputMode", ...
                "Unsupported cfg.dnn.mlp.inputMode = %s.", inputMode);
    end

    hiddenSizes = getNestedFieldOrDefault_step09h2g( ...
        cfg, {"dnn","mlp","hiddenSizes"}, [8 6]);

    hiddenSizes = double(hiddenSizes(:).');

    if any(hiddenSizes <= 0) || any(hiddenSizes ~= round(hiddenSizes))
        error("branchMLPArchitecture:BadHiddenSizes", ...
            "cfg.dnn.mlp.hiddenSizes must contain positive integers.");
    end

    nHidden = numel(hiddenSizes);

    rawActivations = getNestedFieldOrDefault_step09h2g( ...
        cfg, {"dnn","mlp","activations"}, []);

    if isempty(rawActivations)

        if nHidden == 0
            hiddenActivations = strings(1, 0);

        elseif nHidden == 1
            hiddenActivations = "tanh";

        else
            % Default for user's DNN-MEKF style:
            % first layer softplus, following layers tanh.
            hiddenActivations = ["softplus", repmat("tanh", 1, nHidden-1)];
        end

    else
        hiddenActivations = string(rawActivations);
        hiddenActivations = hiddenActivations(:).';

        if isscalar(hiddenActivations) && nHidden > 1
            hiddenActivations = repmat(hiddenActivations, 1, nHidden);
        end

        if numel(hiddenActivations) ~= nHidden
            error("branchMLPArchitecture:BadActivationCount", ...
                "Number of hidden activations must equal number of hidden layers.");
        end
    end

    outputDim = dim;

    layerSizes = [inputDim, hiddenSizes, outputDim];
    nLayers = numel(layerSizes) - 1;

    layers = repmat(struct( ...
        "nIn", [], ...
        "nOut", [], ...
        "activation", "", ...
        "idxW", [], ...
        "idxb", []), nLayers, 1);

    idx = 1;

    for ell = 1:nLayers

        nIn = layerSizes(ell);
        nOut = layerSizes(ell+1);

        nW = nOut * nIn;

        layers(ell).nIn = nIn;
        layers(ell).nOut = nOut;

        if ell <= nHidden
            layers(ell).activation = hiddenActivations(ell);
        else
            layers(ell).activation = "linear";
        end

        layers(ell).idxW = idx:(idx+nW-1);
        idx = idx + nW;

        layers(ell).idxb = idx:(idx+nOut-1);
        idx = idx + nOut;

    end

    arch = struct();

    arch.inputMode = inputMode;
    arch.inputDim = inputDim;
    arch.outputDim = outputDim;

    arch.hiddenSizes = hiddenSizes;
    arch.hiddenActivations = hiddenActivations;

    arch.layerSizes = layerSizes;
    arch.nHidden = nHidden;
    arch.nLayers = nLayers;
    arch.layers = layers;

    arch.nTheta = idx - 1;

    arch.outputActivation = "linear";

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