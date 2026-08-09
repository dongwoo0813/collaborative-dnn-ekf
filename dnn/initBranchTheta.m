function [theta0, info] = initBranchTheta(branchID, cfg)
%{
File:
    dnn/initBranchTheta.m

Purpose:
    Initialize one residual branch parameter vector theta_i.

Current Step:
    Step 09-H.3b

Why this helper is needed:
    fixed_feature_lip and mlp_general need different initialization logic.

    fixed_feature_lip:
        theta_i = vec(W_i)
        zero initialization is okay and preserves fair comparisons.

    mlp_general:
        theta_i contains hidden-layer weights and output-layer weights.
        If all parameters are initialized to zero, the MLP can get stuck
        with only output bias learning at first, especially with tanh hidden
        layers. Therefore the default MLP initialization is:

            hidden layers : small deterministic random weights
            output layer  : zero weights and zero bias

        This gives initial dHat = 0 while keeping nonzero hidden features.

Important:
    The MLP random initialization uses a local RandStream so it does not
    consume MATLAB's global random stream. This preserves fair comparison
    behavior for physical initial-condition randomness.
%}

    [nTheta, branchInfo] = branchThetaNumel(cfg);

    branchModel = string(branchInfo.branchModel);

    switch branchModel

        case "fixed_feature_lip"

            theta0Std = getNestedFieldOrDefault_step09h3b( ...
                cfg, {"dnn","theta0_std"}, 0.0);

            if theta0Std == 0
                theta0 = zeros(nTheta, 1);
            else
                theta0 = theta0Std * randn(nTheta, 1);
            end

            info = struct();
            info.branchModel = branchModel;
            info.nTheta = nTheta;
            info.theta0Std = theta0Std;
            info.initMode = "fixed_feature_standard";

        case "mlp_general"

            arch = branchInfo.arch;

            initMode = string(getNestedFieldOrDefault_step09h3b( ...
                cfg, {"dnn","mlp","thetaInitMode"}, ...
                "random_hidden_zero_output"));

            baseSeed = getNestedFieldOrDefault_step09h3b( ...
                cfg, {"dnn","mlp","thetaInitSeed"}, 9100);

            seed = double(baseSeed) + 1009 * double(branchID);

            rs = RandStream("mt19937ar", "Seed", seed);

            switch initMode

                case "zeros"

                    theta0 = zeros(nTheta, 1);

                case "small_random"

                    theta0Std = getNestedFieldOrDefault_step09h3b( ...
                        cfg, {"dnn","mlp","theta0Std"}, 1e-3);

                    theta0 = theta0Std * randn(rs, nTheta, 1);

                case "random_hidden_zero_output"

                    theta0 = zeros(nTheta, 1);

                    hiddenWeightStd = getNestedFieldOrDefault_step09h3b( ...
                        cfg, {"dnn","mlp","hiddenWeightStd"}, 0.1);

                    hiddenBiasStd = getNestedFieldOrDefault_step09h3b( ...
                        cfg, {"dnn","mlp","hiddenBiasStd"}, 0.0);

                    outputWeightStd = getNestedFieldOrDefault_step09h3b( ...
                        cfg, {"dnn","mlp","outputWeightStd"}, 0.0);

                    outputBiasStd = getNestedFieldOrDefault_step09h3b( ...
                        cfg, {"dnn","mlp","outputBiasStd"}, 0.0);

                    for ell = 1:arch.nLayers

                        nIn = arch.layers(ell).nIn;
                        nOut = arch.layers(ell).nOut;

                        idxW = arch.layers(ell).idxW;
                        idxb = arch.layers(ell).idxb;

                        if ell < arch.nLayers

                            % Xavier-like scaling keeps hidden preactivations
                            % moderate as layer sizes change.
                            Wstd = hiddenWeightStd / sqrt(max(nIn, 1));
                            bstd = hiddenBiasStd;

                        else

                            % Default output layer is zero, so initial dHat=0.
                            % This avoids injecting a large untrained residual
                            % acceleration at t=0.
                            Wstd = outputWeightStd;
                            bstd = outputBiasStd;

                        end

                        theta0(idxW) = Wstd * randn(rs, nOut*nIn, 1);
                        theta0(idxb) = bstd * randn(rs, nOut, 1);

                    end

                otherwise

                    error("initBranchTheta:UnsupportedMLPInitMode", ...
                        "Unsupported cfg.dnn.mlp.thetaInitMode = %s.", initMode);

            end

            info = struct();
            info.branchModel = branchModel;
            info.nTheta = nTheta;
            info.arch = arch;
            info.initMode = initMode;
            info.seed = seed;

        otherwise

            error("initBranchTheta:UnsupportedBranchModel", ...
                "Unsupported branch model: %s.", branchModel);

    end

end

function value = getNestedFieldOrDefault_step09h3b(s, path, defaultValue)
%GETNESTEDFIELDORDEFAULT_STEP09H3B Read nested struct field with default.

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