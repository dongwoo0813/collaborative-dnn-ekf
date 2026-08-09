function watcher = DNN_EKF_Predict_Local(watcher, t, cfg)
%{
Function:
    DNN_EKF_Predict_Local.m

Purpose:
    Perform the EKF prediction step for one watcher using a local
    fixed-feature DNN residual model.

    The local augmented DNN-EKF state is

        X_i = [eta_i; theta_i],

    where

        eta_i = [r_t; v_t]

    is the physical target state estimate and theta_i is the local DNN
    branch parameter vector estimated by watcher i.

    In this Step 03 local DNN-EKF prediction, watcher i uses only its own
    branch residual model

        d_i(eta_i; theta_i) = W_i phi_i(eta_i).

    No ground-station branch sharing and no peer-to-peer branch sharing are
    used in this step.

Inputs:
    watcher - Local DNN-EKF watcher structure.
              Required fields:
                  watcher.xhat
                  watcher.P
                  watcher.idxEta
                  watcher.idxTheta
                  watcher.nEta
                  watcher.nTheta
                  watcher.nX
                  watcher.localBranchID

    t       - Current simulation time t_k.
              Type: scalar.

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.dt
                  cfg.dim
                  cfg.ekf.q_acc
                  cfg.dnn.qTheta

Outputs:
    watcher - Updated watcher structure after prediction.
              Updated fields:
                  watcher.xhat
                  watcher.P

Main equations:
    Augmented local DNN-EKF dynamics:

        dot r_t = v_t,

        dot v_t = a_nom(r_t,v_t,t) + W_i phi_i(eta_i),

        dot theta_i = 0.

    The covariance prediction uses the local linearization

        P_{k+1}^- = F_k P_k^+ F_k^T + Q_k,

    where

        F_k = I + dt A_k,

    and A_k is the continuous-time Jacobian of the augmented dynamics.

    The important parameter coupling is

        partial dot v_t / partial theta_i
            = partial d_i / partial theta_i
            = kron(phi_i^T, I_dim).

Notes:
    - The state is propagated using RK4.
    - The covariance is propagated using a first-order discrete
      approximation F = I + dt A.
    - This is sufficient for the current debugging stage.
    - Later, GS/P2P versions can add nonlocal branch copies to the
      prediction model.
%}

    dt = cfg.dt;

    x = watcher.xhat;
    P = watcher.P;

    % Evaluate the residual model and all analytical Jacobians once at the
    % covariance-linearization point (t_k, x_k). The same calculation is
    % needed by RK4 stage 1, A_k, the local theta coupling, FIM gates, and
    % nonlocal covariance injection. Keep one cache instead of repeating
    % identical MLP forward/backprop passes.
    baseResidualCache = buildBaseResidualCache_step09j7( ...
        t, x, watcher, cfg);

    % Propagate augmented state using RK4.
    f = @(tt, xx) localDNNAugmentedDynamics( ...
        tt, xx, watcher, cfg, baseResidualCache);

    xPred = propagateRK4(f, t, x, dt);

    % Continuous-time local linearization.
    A = localDNNContinuousJacobian( ...
        t, x, watcher, cfg, baseResidualCache);

    % Discretize the augmented transition while preserving its block upper-
    % triangular structure. Step 09-J.8 supports the previous Euler form and
    % a more accurate second-order block Taylor form. FOGM theta decay is
    % exact in either mode.
    F = discretizeAugmentedTransitionBlock(A, dt, watcher, cfg);

    % ---------------------------------------------------------------------
    % Process noise / model-uncertainty covariance
    % ---------------------------------------------------------------------
    % Qbase contains:
    %   1. nominal physical acceleration process noise
    %   2. local theta_m process noise
    %
    % In Step 04b, when GS_composite prediction is used and
    % cfg.gs.useNonlocalBranchCovariance = true, we additionally inject
    % nonlocal GS-branch uncertainty:
    %
    %   Qnonlocal = M_{k,m} S_{d,-m,k} M_{k,m}'.
    %
    % This term accounts for the uncertainty of nonlocal branch copies
    % theta_j, j ~= m, which are used in the mean prediction but are not
    % appended to watcher m's EKF state.
    % ---------------------------------------------------------------------
    Qbase = local_DNN_Process_Noise(cfg, watcher);

    Qnonlocal = zeros(watcher.nX, watcher.nX);
    SdNonlocal = zeros(cfg.dim, cfg.dim);
    nonlocalCovDiag = struct();
    nonlocalCovDiag.enabled = false;
    nonlocalCovDiag.branchIDs = [];
    nonlocalCovDiag.numActiveNonlocal = 0;
    nonlocalCovDiag.traceSdNonlocal = 0;
    nonlocalCovDiag.traceQnonlocal = 0;

    if shouldUseNonlocalBranchCovariance(cfg)

        residualSource = getPredictionResidualSource(cfg);

        if residualSource == "GS_composite"

            % First-order consistency:
            % The current DNN_EKF_Predict_Local.m uses A(t,x_k) and
            % F = I + dt*A(t,x_k). Therefore, evaluate the nonlocal
            % covariance injection at the same physical state x_k.
            etaForNonlocalCov = x(watcher.idxEta);

            [Qnonlocal, SdNonlocal, nonlocalCovDiag] = ...
                computeNonlocalBranchCovarianceInjection( ...
                    watcher, etaForNonlocalCov, cfg, baseResidualCache);

        end

    end

    Q = Qbase + Qnonlocal;
    Q = 0.5 * (Q + Q');

    % ---------------------------------------------------------------------
    % Covariance prediction.
    %
    % Dense form:
    %     Ppred = F * P * F' + Q
    %
    % Block-structured form:
    %     Uses F = [F_eta_eta, F_eta_theta; 0, F_theta_theta],
    %     where F_theta_theta is diagonal for FOGM/random-walk theta
    %     dynamics.
    %
    % The block form is algebraically equivalent to the dense form under the
    % current local DNN-EKF structure, but avoids a full dense augmented
    % matrix multiplication.
    % ---------------------------------------------------------------------
    if useBlockCovariancePrediction(cfg)
        Ppred = predict_Cov_Block_DNN_EKF(F, P, Q, watcher, cfg);
    else
        Ppred = F * P * F' + Q;
        Ppred = 0.5 * (Ppred + Ppred');
    end

    % Optional numerical diagonal floor.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "minCovDiag")
        minCovDiag = cfg.dnn.minCovDiag;
    else
        minCovDiag = 0;
    end

    if minCovDiag > 0
        d = diag(Ppred);
        d = max(d, minCovDiag);
        Ppred(1:size(Ppred,1)+1:end) = d;
        Ppred = 0.5 * (Ppred + Ppred');
    end

    watcher.xhat    = xPred;
    watcher.P       = Ppred;

    % Step 04b nonlocal covariance-injection diagnostics.
    watcher.lastNonlocalCovInjection = nonlocalCovDiag;
    watcher.lastNonlocalCovInjection.SdNonlocal = SdNonlocal;
    watcher.lastNonlocalCovInjection.QnonlocalDiag = diag(Qnonlocal);

end

function xdot = localDNNAugmentedDynamics( ...
    t, x, watcher, cfg, baseResidualCache)
%{
Function:
    localDNNAugmentedDynamics

Purpose:
    Compute the continuous-time augmented dynamics for the local DNN-EKF.

Main equations:
    X_i = [eta_i; theta_i],

    dot eta_i =
        [ v_i;
          a_nom(eta_i,t) + d_i(eta_i;theta_i) ],

    dot theta_i = 0.
%}

    dim = cfg.dim;

    eta = x(watcher.idxEta);
    theta = x(watcher.idxTheta);

    if nargin < 5
        baseResidualCache = struct("valid", false);
    end

    r = eta(1:dim);
    v = eta(dim+1:2*dim);

    aNom = targetNominalDynamics(t, r, v, cfg);


    if isfield(cfg, "dnn") && isfield(cfg.dnn, "useResidualInPrediction")
        useResidualInPrediction = cfg.dnn.useResidualInPrediction;
    else
        useResidualInPrediction = true;
    end
    

    if useResidualInPrediction
    
        betaDNN = getDNNResidualInjectionGain(cfg);
        residualSource = getPredictionResidualSource(cfg);
    
        useCachedResidual = isBaseResidualCacheMatch_step09j7( ...
            baseResidualCache, t, eta, theta, residualSource);

        if useCachedResidual
            aResidual = baseResidualCache.dResidual;
        else
        switch residualSource
    
            case "local_DNN"
                branchID = watcher.localBranchID;

                % Branch-model-aware local residual.
                %
                % fixed_feature_lip:
                %     d_i = W_i phi_i(eta)
                %
                % mlp_general:
                %     d_i = MLP_i(xi_i(eta); theta_i)
                %
                % RK4 consistency:
                %     eta and theta are taken from the current RK4 stage xx.
                [aResidual, ~, ~, ~] = evaluateBranchResidualModel( ...
                    branchID, eta, theta, cfg);

            case "GS_composite"
                % Step 04 GS-assisted composite residual:
                %
                %   d_comp,m(eta)
                %       = d_m(eta; theta_m^local)
                %         + sum_{j ~= m} d_j(eta; theta_{j|m}^{GS}).
                %
                % The local branch theta_m is taken from the current RK4
                % state xx, not from watcher.xhat, so intermediate RK4
                % stages are handled consistently.
                %
                % Nonlocal branches are fixed GS cache copies stored in
                % watcher.gsBranches(j). They are not appended to the EKF
                % state and are not updated by this watcher.
                [aResidual, ~, ~, ~] = evaluateWatcherCompositeResidual( ...
                    watcher, eta, theta, cfg);
    
            case "oracle"
                % Oracle residual:
                %   Use the hidden true residual function inside the estimator
                %   prediction.
                %
                % This is NOT a realizable estimator. It is only an upper-bound
                % sanity check to answer:
                %
                %   If the residual were known perfectly, would tracking improve?
                aResidual = trueResidual(t, eta, cfg);
    
            otherwise
                error("Unsupported cfg.dnn.predictionResidualSource: %s", residualSource);
    
        end
        end

        % The DNN residual is injected with a trust/scaling factor.
        %
        %     dot v = a_nom + betaDNN * d_i(eta;theta_i)
        %
        % betaDNN = 1 recovers the previous implementation.
        % betaDNN = 0 recovers the physical prediction model.
        aDNN = betaDNN * aResidual;
    
    else
        aDNN = zeros(dim, 1);
    end
    
    etaDot = [v; aNom + aDNN];





    % -------------------------------------------------------------
    % Local DNN parameter dynamics
    %
    % random_walk:
    %     dot theta = 0
    %
    % FOGM:
    %     dot theta = -(1/tauTheta)(theta - thetaMean)
    % -------------------------------------------------------------
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "thetaDynamics")
        thetaDynamics = string(cfg.dnn.thetaDynamics);
    else
        thetaDynamics = "random_walk";
    end
    
    switch thetaDynamics
        case "random_walk"
            thetaDot = zeros(watcher.nTheta, 1);
    
        case "FOGM"
            tauTheta = cfg.dnn.thetaTau;
    
            if isfield(cfg.dnn, "thetaMean")
                if isscalar(cfg.dnn.thetaMean)
                    thetaMean = cfg.dnn.thetaMean * ones(watcher.nTheta, 1);
                else
                    thetaMean = cfg.dnn.thetaMean(:);
                end
            else
                thetaMean = zeros(watcher.nTheta, 1);
            end
    
            thetaDot = -(1/tauTheta) * (theta - thetaMean);
    
        otherwise
            error("Unsupported cfg.dnn.thetaDynamics: %s", thetaDynamics);
    end
    xdot = zeros(watcher.nX, 1);
    xdot(watcher.idxEta) = etaDot;
    xdot(watcher.idxTheta) = thetaDot;

end

function A = localDNNContinuousJacobian( ...
    t, x, watcher, cfg, baseResidualCache)
%{
Function:
    localDNNContinuousJacobian

Purpose:
    Compute the continuous-time Jacobian A for the local augmented DNN-EKF
    prediction model.

Main equations:
    The augmented dynamics are

        dot r = v,

        dot v = a_nom(eta,t) + d_i(eta;theta_i),

        dot theta_i = 0.

    Therefore,

        partial dot r / partial v = I,

        partial dot v / partial eta = numerical finite difference,

        partial dot v / partial theta_i = branchJacobianTheta.
%}

    dim = cfg.dim;

    A = zeros(watcher.nX, watcher.nX);

    idxEta = watcher.idxEta;
    idxTheta = watcher.idxTheta;

    idxR_local = 1:dim;
    idxV_local = dim + (1:dim);

    idxR = idxEta(idxR_local);
    idxV = idxEta(idxV_local);

    eta = x(idxEta);
    theta = x(idxTheta);

    if nargin < 5
        baseResidualCache = struct("valid", false);
    end

    % dot r = v
    A(idxR, idxV) = eye(dim);

    % dot v derivative with respect to eta.
    %
    % a(eta,theta) = a_nom(eta,t) + W_i phi_i(eta)
    %
    % partial a / partial eta
    %     = partial a_nom / partial eta
    %       + W_i * partial phi_i / partial eta.
    A_acc_Eta = analyticalAccelerationEtaJacobian( ...
        t, eta, theta, watcher, cfg, baseResidualCache);
    A(idxV, idxEta) = A_acc_Eta;

    % dot v derivative with respect to theta_i.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "useResidualInPrediction")
        useResidualInPrediction = cfg.dnn.useResidualInPrediction;
    else
        useResidualInPrediction = true;
    end

    if useResidualInPrediction
    
        residualSource = getPredictionResidualSource(cfg);
    
        switch residualSource
    
            case "local_DNN"
                branchID = watcher.localBranchID;
                betaDNN = getDNNResidualInjectionGain(cfg);

                % Branch-model-aware theta sensitivity.
                %
                % fixed_feature_lip:
                %     Jtheta = phi_i(eta)^T kron I_dim
                %
                % mlp_general:
                %     Jtheta = [z0^T kron S1, S1, ..., z_{L-1}^T kron SL, SL]
                %
                % Consistent with
                %
                %     dot v = a_nom + betaDNN * d_i(eta;theta_i).
                if isBaseResidualCacheMatch_step09j7( ...
                        baseResidualCache, t, eta, theta, residualSource)
                    JthetaLocal = baseResidualCache.JthetaLocalRaw;
                else
                    [~, ~, JthetaLocal, ~] = evaluateBranchResidualModel( ...
                        branchID, eta, theta, cfg);
                end

                Btheta = betaDNN * JthetaLocal;
                A(idxV, idxTheta) = Btheta;

            case "GS_composite"
                branchID = watcher.localBranchID;
                betaDNN = getDNNResidualInjectionGain(cfg);


                % Branch-model-aware local theta sensitivity inside the GS
                % composite model. Nonlocal GS branch parameters are not
                % local EKF states, so only the local branch theta_i appears
                % in A(idxV,idxTheta).
                useBaseCache = isBaseResidualCacheMatch_step09j7( ...
                    baseResidualCache, t, eta, theta, residualSource);

                if useBaseCache
                    BthetaRaw = baseResidualCache.JthetaLocalRaw;
                else
                    [~, ~, BthetaRaw, ~] = evaluateBranchResidualModel( ...
                        branchID, eta, theta, cfg);
                end

                compositeMode = "additive";

                if isfield(cfg, "gs") && isfield(cfg.gs, "compositeMode")
                    compositeMode = string(cfg.gs.compositeMode);
                end

                switch compositeMode

                    case "additive"

                        % Original additive mode:
                        %
                        %     d_i^local = d_i(eta; theta_i).
                        %
                        % Therefore
                        %
                        %     partial dot{v} / partial theta_i
                        %       = beta_DNN * partial d_i / partial theta_i.
                        Btheta = betaDNN * BthetaRaw;

                    case "gated_additive"

                        % Fully gated mode:
                        %
                        %     d_i^local = B_i(eta) d_i(eta; theta_i).
                        %
                        % Since B_i depends on eta but not theta_i,
                        %
                        %     partial dot{v} / partial theta_i
                        %       = beta_DNN * B_i(eta) * partial d_i / partial theta_i.
                        BgateLocal = branchGateMatrix(branchID, eta, cfg);
                        Btheta = betaDNN * (BgateLocal * BthetaRaw);

                    case "local_full_plus_gated_nonlocal"

                        % Hybrid mode:
                        %
                        %     d^GS_i(eta)
                        %       = d_i(eta; theta_i)
                        %         + sum_{j neq i} B_j(eta) d_j(eta; theta_j).
                        %
                        % The local EKF state contains only theta_i. The nonlocal theta_j
                        % are not local EKF states and are handled by the nonlocal covariance
                        % injection path.
                        %
                        % Because the local branch is intentionally kept full, the local
                        % theta sensitivity must also remain full/raw:
                        %
                        %     partial dot{v} / partial theta_i
                        %       = beta_DNN * partial d_i / partial theta_i.
                        Btheta = betaDNN * BthetaRaw;


                    case {"bearing_fim_gated", "output_information_fusion", ...
                            "fim_weighted_additive"}

                        % Step 09-J.3 bearing-FIM-gated mode:
                        %
                        %     d_FIM = sum_j B_{j|m} d_j(eta; theta_j).
                        %
                        % Only theta_i is part of watcher m's EKF state, so
                        %
                        %     partial dot{v} / partial theta_i
                        %       = beta_DNN * B_{i|m} * partial d_i / partial theta_i.
                        %
                        % B_{i|m} is computed from the current available OmegaBar set and
                        % is treated as fixed metadata within this prediction call.
                        if useBaseCache
                            gateDiag = baseResidualCache.gateDiag;
                        else
                            [~, ~, ~, ~, gateDiag] = ...
                                evaluateWatcherCompositeResidual( ...
                                    watcher, eta, theta, cfg);
                        end

                        if ~isfield(gateDiag, "B") || ...
                                size(gateDiag.B, 3) < branchID
                            error("DNN_EKF_Predict_Local:MissingMatrixGate", ...
                                "%s mode did not return a local B weight.", ...
                                compositeMode);
                        end

                        BgateLocal = gateDiag.B(:, :, branchID);
                        Btheta = betaDNN * (BgateLocal * BthetaRaw);


                    otherwise

                        error("DNN_EKF_Predict_Local:UnsupportedCompositeMode", ...
                            "Unsupported cfg.gs.compositeMode = %s.", compositeMode);

                end

                A(idxV, idxTheta) = Btheta;

            case "oracle"
                % Oracle residual is treated as a known function of eta.
                % It does not depend on theta_i, so there is no theta coupling.
                A(idxV, idxTheta) = zeros(dim, watcher.nTheta);
    
            otherwise
                error("Unsupported cfg.dnn.predictionResidualSource: %s", residualSource);
    
        end
    
    end

    % -------------------------------------------------------------
    % theta dynamics Jacobian
    %
    % random_walk:
    %     partial dot theta / partial theta = 0
    %
    % FOGM:
    %     partial dot theta / partial theta = -(1/tauTheta) I
    % -------------------------------------------------------------
    thetaDynamics = getThetaDynamicsMode(cfg);
    
    switch thetaDynamics
        case "random_walk"
            % Nothing to add.
    
        case "FOGM"
            tauTheta = cfg.dnn.thetaTau;
            A(idxTheta, idxTheta) = -(1/tauTheta) * eye(watcher.nTheta);
    
        otherwise
            error("Unsupported cfg.dnn.thetaDynamics: %s", thetaDynamics);
    end
    
end

function A_acc_Eta = analyticalAccelerationEtaJacobian( ...
    t, eta, theta, watcher, cfg, baseResidualCache)
%{
Function:
    analyticalAccelerationEtaJacobian

Purpose:
    Compute the analytical Jacobian of the local DNN-EKF acceleration model
    with respect to the physical target state eta.

    The local prediction acceleration is

        a(eta, theta_i, t)
        =
        a_nom(eta,t) + d_i(eta;theta_i),

    where

        d_i(eta;theta_i) = W_i phi_i(eta).

Inputs:
    t       - Current simulation time.
    eta     - Physical target state eta = [r_t; v_t].
              Size: 2*cfg.dim x 1.

    theta   - Local DNN branch parameter vector theta_i.
              Size: cfg.dim*nPhi x 1.

    watcher - Local DNN-EKF watcher structure.
              Required fields:
                  watcher.localBranchID

    cfg     - Simulation configuration structure.

Outputs:
    A_acc_Eta - Acceleration Jacobian with respect to eta.
              Size: cfg.dim x 2*cfg.dim.

Main equations:
    The acceleration model is

        a(eta,theta_i,t)
        =
        a_nom(eta,t)
        +
        W_i phi_i(eta).

    Therefore,

        partial a / partial eta
        =
        partial a_nom / partial eta
        +
        W_i partial phi_i / partial eta.

    For the current Step 03 double-integrator nominal model,

        partial a_nom / partial eta = 0.

Notes:
    - This replaces finite-difference differentiation.
    - If targetNominalDynamics.m is later changed to orbital dynamics, then
      nominalAccelerationJacobianEta.m or the local helper below should be
      updated accordingly.
%}

    dim = cfg.dim;

    branchID = watcher.localBranchID;

    if nargin < 6
        baseResidualCache = struct("valid", false);
    end

    % Branch-model-aware prediction must not infer theta length from
    % featureBlock. For mlp_general, theta contains all MLP weights/biases.
    AnomEta = nominalAccelerationJacobianEta(t, eta, cfg);
    
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "useResidualInPrediction")
        useResidualInPrediction = cfg.dnn.useResidualInPrediction;
    else
        useResidualInPrediction = true;
    end
    
    if useResidualInPrediction
    
        betaDNN = getDNNResidualInjectionGain(cfg);
        residualSource = getPredictionResidualSource(cfg);
        useBaseCache = isBaseResidualCacheMatch_step09j7( ...
            baseResidualCache, t, eta, theta, residualSource);
    
        switch residualSource
    
            case "local_DNN"
                % Branch-model-aware eta sensitivity.
                %
                % fixed_feature_lip:
                %     Jeta = W_i * dphi_i/deta
                %
                % mlp_general:
                %     Jeta = W_L D_{L-1} W_{L-1} ... D_1 W_1 dxi/deta
                if useBaseCache
                    JetaLocal = baseResidualCache.JetaResidual;
                else
                    [~, JetaLocal, ~, ~] = evaluateBranchResidualModel( ...
                        branchID, eta, theta, cfg);
                end

                A_acc_Eta = AnomEta + betaDNN * JetaLocal;

            case "GS_composite"
                % The eta derivative of the composite residual includes
                % both the local branch and all active nonlocal GS branch
                % copies:
                %
                %   partial d_comp,m / partial eta
                %       =
                %       partial d_m / partial eta
                %       + sum_{j ~= m} partial d_j / partial eta.
                %
                % The helper returns only the residual Jacobian. Add the
                % nominal acceleration Jacobian here.
                if useBaseCache
                    JetaComp = baseResidualCache.JetaResidual;
                else
                    [~, JetaComp, ~, ~] = ...
                        evaluateWatcherCompositeResidual( ...
                            watcher, eta, theta, cfg);
                end

                A_acc_Eta = AnomEta + betaDNN * JetaComp;
    
            case "oracle"
                % Finite-difference Jacobian of the hidden true residual.
                %
                % This keeps the EKF covariance prediction consistent with
                %
                %   a_pred = a_nom + betaDNN * trueResidual(t,eta,cfg).
                JoracleEta = oracleResidualJacobianEta(t, eta, cfg);
    
                A_acc_Eta = AnomEta + betaDNN * JoracleEta;
    
            otherwise
                error("Unsupported cfg.dnn.predictionResidualSource: %s", residualSource);
    
        end    
    else
        A_acc_Eta = AnomEta;
    end



end



function a = local_Acceleration(t, eta, theta, watcher, cfg)
%{
Function:
    local_Acceleration

Purpose:
    Compute the acceleration used by the local DNN-EKF prediction model.

Main equation:
    a = a_nom(r,v,t) + d_i(eta;theta_i).
%}

    dim = cfg.dim;

    r = eta(1:dim);
    v = eta(dim+1:2*dim);

    aNom = targetNominalDynamics(t, r, v, cfg);

    betaDNN = getDNNResidualInjectionGain(cfg);
    residualSource = getPredictionResidualSource(cfg);
    
    switch residualSource
    
        case "local_DNN"
            branchID = watcher.localBranchID;

            [aResidual, ~, ~, ~] = evaluateBranchResidualModel( ...
                branchID, eta, theta, cfg);

        case "GS_composite"
            [aResidual, ~, ~, ~] = evaluateWatcherCompositeResidual( ...
                watcher, eta, theta, cfg);
    
        case "oracle"
            aResidual = trueResidual(t, eta, cfg);
    
        otherwise
            error("Unsupported cfg.dnn.predictionResidualSource: %s", residualSource);
    
    end
    
    aDNN = betaDNN * aResidual;
    
    a = aNom + aDNN;

end


function A_nom_eta = nominalAccelerationJacobianEta(t, eta, cfg)
%{
Function:
    nominalAccelerationJacobianEta

Purpose:
    Compute the Jacobian of the nominal target acceleration with respect to
    eta = [r_t; v_t].

Inputs:
    t       - Current simulation time.
    eta     - Physical target state.
    cfg     - Simulation configuration.

Outputs:
    A_nom_eta - Nominal acceleration Jacobian.
              Size: cfg.dim x 2*cfg.dim.

Main equations:
    In the current Step 03 baseline,

        a_nom = 0.

    Therefore,

        partial a_nom / partial eta = 0.

Notes:
    - Later, if targetNominalDynamics.m is changed to two-body orbital
      dynamics, this function should be replaced by the analytical orbital
      acceleration Jacobian.
%}

    dim = cfg.dim;

    A_nom_eta = zeros(dim, 2*dim);

end



function Q = local_DNN_Process_Noise(cfg, watcher)
%{
Function:
    local_DNN_Process_Noise

Purpose:
    Build the discrete-time process-noise covariance for the local DNN-EKF
    prediction.

Main equations:
    Physical acceleration-noise block:

        Q_eta = q_acc [ dt^4/4 I   dt^3/2 I
                        dt^3/2 I   dt^2   I ]

    Parameter random-walk block:

        Q_theta = qTheta * dt * I.

Notes:
    - qTheta controls how quickly theta_i is allowed to adapt.
    - q_acc covers unmodeled physical acceleration and numerical mismatch.
%}

    dt = cfg.dt;
    dim = cfg.dim;

    Q = zeros(watcher.nX, watcher.nX);

    if ~isfield(cfg, "dnn") || ~isfield(cfg.dnn, "qEpsilonC0")
        error("cfg.dnn.qEpsilonC0 is required for DNN-EKF prediction.");
    end

    QepsilonC0 = cfg.dnn.qEpsilonC0;
    if isscalar(QepsilonC0)
        QepsilonC0 = QepsilonC0 * eye(dim);
    end
    if any(size(QepsilonC0) ~= [dim, dim])
        error("cfg.dnn.qEpsilonC0 must be scalar or %d-by-%d.", dim, dim);
    end

    QepsilonC0 = 0.5 * (QepsilonC0 + QepsilonC0.');
    QepsilonC = getEpsilonQScale(watcher) * QepsilonC0;

    % Exact double-integrator discretization of continuous white
    % approximation-error acceleration noise.
    Qeta = [ ...
        dt^3/3 * QepsilonC, dt^2/2 * QepsilonC;
        dt^2/2 * QepsilonC, dt     * QepsilonC];
    
    % -------------------------------------------------------------
    % Parameter process-noise covariance
    % -------------------------------------------------------------
    thetaDynamics = getThetaDynamicsMode(cfg);
    
    switch thetaDynamics
        case "random_walk"
    
            if isfield(cfg, "dnn") && isfield(cfg.dnn, "qTheta")
                qTheta = cfg.dnn.qTheta;
            else
                qTheta = 1e-10;
            end
    
            % Random-walk parameter model:
            %
            %     theta_{k+1} = theta_k + w_k
            %
            % with discrete covariance qTheta*dt.
            Qtheta = qTheta * dt * eye(watcher.nTheta);
    
        case "FOGM"
    
            if ~isfield(cfg.dnn, "thetaTau")
                error("cfg.dnn.thetaTau is required for FOGM theta dynamics.");
            end
    
            if ~isfield(cfg.dnn, "thetaSigmaSS")
                error("cfg.dnn.thetaSigmaSS is required for FOGM theta dynamics.");
            end
    
            tauTheta = cfg.dnn.thetaTau;
            sigmaThetaSS = cfg.dnn.thetaSigmaSS;
    
            alphaTheta = exp(-dt / tauTheta);
    
            % Discrete FOGM process noise chosen so that the scalar stationary
            % variance is sigmaThetaSS^2:
            %
            %     Qtheta = sigmaThetaSS^2 (1 - alphaTheta^2) I.
            %
            % This makes thetaSigmaSS an interpretable covariance hyperparameter.
            Qtheta = sigmaThetaSS^2 * (1 - alphaTheta^2) * eye(watcher.nTheta);
    
        otherwise
            error("Unsupported cfg.dnn.thetaDynamics: %s", thetaDynamics);
    end


    % -------------------------------------------------------------
    % Adaptive covariance-matching scale for DNN parameter process noise
    %
    % The base parameter process noise is first computed from either:
    %
    %   random_walk:
    %       Qtheta_base = qTheta * dt * I
    %
    %   FOGM:
    %       Qtheta_base = sigmaThetaSS^2 * (1-alphaTheta^2) * I
    %
    % Then covariance matching applies
    %
    %       Qtheta = gammaTheta * Qtheta_base.
    %
    % gammaTheta is updated in DNN_EKF_Update_Local.m from innovation
    % covariance matching. If watcher.cm does not exist yet, use gammaTheta = 1.
    % -------------------------------------------------------------
    gammaTheta = getThetaQScale(watcher);
    
    Qtheta = gammaTheta * Qtheta;



    Q(watcher.idxEta, watcher.idxEta) = Qeta;
    Q(watcher.idxTheta, watcher.idxTheta) = Qtheta;

    Q = 0.5 * (Q + Q');

end


function thetaDynamics = getThetaDynamicsMode(cfg)
%{
Function:
    getThetaDynamicsMode

Purpose:
    Return the DNN parameter dynamics mode.

Supported modes:
    "random_walk"
    "FOGM"
%}

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "thetaDynamics")
        thetaDynamics = string(cfg.dnn.thetaDynamics);
    else
        thetaDynamics = "random_walk";
    end

end


function thetaMean = getThetaMeanVector(cfg, nTheta)
%{
Function:
    getThetaMeanVector

Purpose:
    Return thetaMean as an nTheta-by-1 vector.

Notes:
    - If cfg.dnn.thetaMean is scalar, expand it to all parameters.
    - If cfg.dnn.thetaMean is already a vector, check its size.
    - If not provided, use zero mean.
%}

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "thetaMean")

        thetaMeanRaw = cfg.dnn.thetaMean;

        if isscalar(thetaMeanRaw)
            thetaMean = thetaMeanRaw * ones(nTheta, 1);
        else
            thetaMean = thetaMeanRaw(:);

            if numel(thetaMean) ~= nTheta
                error("cfg.dnn.thetaMean must be scalar or have length nTheta = %d.", nTheta);
            end
        end

    else
        thetaMean = zeros(nTheta, 1);
    end

end

function tf = shouldUseNonlocalBranchCovariance(cfg)
%{
Function:
    shouldUseNonlocalBranchCovariance

Purpose:
    Return true if Step 04b nonlocal GS-branch covariance injection is
    enabled.

Main condition:
    cfg.gs.useNonlocalBranchCovariance = true

Notes:
    This helper only decides whether the extra covariance term is allowed.
    The caller should still check that the prediction residual source is
    "GS_composite".
%}

    tf = false;

    if ~isfield(cfg, "gs")
        return;
    end

    if ~isfield(cfg.gs, "useNonlocalBranchCovariance")
        return;
    end

    tf = logical(cfg.gs.useNonlocalBranchCovariance);

end


function gammaTheta = getThetaQScale(watcher)
%{
Function:
    getThetaQScale

Purpose:
    Return the adaptive covariance-matching multiplier for the local DNN
    parameter process-noise covariance.

    The prediction step uses

        Q_theta,k = gammaTheta_i,k * Q_theta,base.

    If adaptive covariance matching has not been initialized, this function
    safely returns gammaTheta = 1.
%}

    gammaTheta = 1.0;

    if isfield(watcher, "cm")
        if isfield(watcher.cm, "gammaTheta")
            gammaTheta = watcher.cm.gammaTheta;
        end
    end

    if ~isfinite(gammaTheta) || gammaTheta <= 0
        gammaTheta = 1.0;
    end

end

function residualSource = getPredictionResidualSource(cfg)
%{
Function:
    getPredictionResidualSource

Purpose:
    Select which residual model is injected into the prediction dynamics.

Options:
    "local_DNN":
        Use the local learned branch residual

            d_i(eta;theta_i) = W_i phi_i(eta).

    "oracle":
        Use the hidden true residual

            d_true(eta,t) = trueResidual(t,eta,cfg).

        This is not a realizable estimator. It is only an upper-bound
        diagnostic test.

    "GS_composite":
        Use the Step 04 GS-assisted composite residual

            d_comp,m(eta)
                = d_m(eta;theta_m)
                  + sum_{j ~= m} d_j(eta;theta_{j|m}^{GS}).

        The local branch theta_m is part of the local EKF state.
        Nonlocal branches are GS-provided cached copies and are not
        appended to the local EKF state.
%}

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "predictionResidualSource")
        residualSource = string(cfg.dnn.predictionResidualSource);
    else
        residualSource = "local_DNN";
    end

end




function betaDNN = getDNNResidualInjectionGain(cfg)
%{
Function:
    getDNNResidualInjectionGain

Purpose:
    Return the scalar trust factor used to inject a residual acceleration
    into the physical target prediction model.

Main equation:
    The prediction acceleration is

        a_pred = a_nom + betaDNN * d_residual.

    betaDNN = 1:
        Full residual injection.

    betaDNN = 0:
        Physical prediction only.

    0 < betaDNN < 1:
        Conservative residual injection.
%}

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "residualInjectionGain")
        betaDNN = cfg.dnn.residualInjectionGain;
    else
        betaDNN = 1.0;
    end

    if ~isfinite(betaDNN)
        error("cfg.dnn.residualInjectionGain must be finite.");
    end

    if betaDNN < 0 || betaDNN > 1
        error("cfg.dnn.residualInjectionGain must be in [0,1].");
    end

end


function J = oracleResidualJacobianEta(t, eta, cfg)
%{
Function:
    oracleResidualJacobianEta

Purpose:
    Compute a finite-difference Jacobian of the hidden true residual with
    respect to the physical target state eta.

Main equation:
    J(:,j) =
        [d_true(t, eta + eps e_j) - d_true(t, eta - eps e_j)] / (2 eps).

Inputs:
    t   - current time
    eta - physical state [r_t; v_t]
    cfg - simulation configuration

Outputs:
    J   - Jacobian of trueResidual with respect to eta.
          Size: cfg.dim x 2*cfg.dim.

Notes:
    This is used only for the oracle residual diagnostic. It should not be
    used in the realizable DNN-EKF algorithm.
%}

    eta = eta(:);

    dim = cfg.dim;
    nEta = numel(eta);

    J = zeros(dim, nEta);

    % State-dependent finite-difference step.
    baseStep = 1e-6;

    for j = 1:nEta

        h = baseStep * max(1, abs(eta(j)));

        etaPlus = eta;
        etaMinus = eta;

        etaPlus(j) = etaPlus(j) + h;
        etaMinus(j) = etaMinus(j) - h;

        fPlus = trueResidual(t, etaPlus, cfg);
        fMinus = trueResidual(t, etaMinus, cfg);

        J(:,j) = (fPlus - fMinus) / (2*h);

    end

end

function gammaEpsilon = getEpsilonQScale(watcher)
%GETEPSILONQSCALE Return the adaptive Qepsilon,c multiplier.
    gammaEpsilon = 1.0;
    if isfield(watcher, "cm") && isfield(watcher.cm, "gammaEpsilon")
        gammaEpsilon = watcher.cm.gammaEpsilon;
    end
    if ~isfinite(gammaEpsilon) || gammaEpsilon <= 0
        gammaEpsilon = 1.0;
    end
end

function cache = buildBaseResidualCache_step09j7(t, x, watcher, cfg)
%BUILDBASERESIDUALCACHE_STEP09J7 One base-point DNN evaluation per prediction.

    cache = struct();
    cache.valid = false;
    cache.t = t;
    cache.eta = x(watcher.idxEta);
    cache.thetaLocal = x(watcher.idxTheta);
    cache.residualSource = getPredictionResidualSource(cfg);
    cache.dResidual = [];
    cache.JetaResidual = [];
    cache.JthetaLocalRaw = [];
    cache.branchUsed = false(cfg.Nw, 1);
    cache.gateDiag = struct();

    if isfield(cfg, "dnn") && ...
            isfield(cfg.dnn, "reusePredictionResidualCache") && ...
            ~logical(cfg.dnn.reusePredictionResidualCache)
        return;
    end

    useResidual = true;
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "useResidualInPrediction")
        useResidual = logical(cfg.dnn.useResidualInPrediction);
    end
    if ~useResidual
        return;
    end

    switch cache.residualSource
        case "local_DNN"
            branchID = watcher.localBranchID;
            [cache.dResidual, cache.JetaResidual, ...
                cache.JthetaLocalRaw, ~] = evaluateBranchResidualModel( ...
                    branchID, cache.eta, cache.thetaLocal, cfg);
            cache.branchUsed(branchID) = true;
            cache.valid = true;

        case "GS_composite"
            [cache.dResidual, cache.JetaResidual, ~, ...
                cache.branchUsed, cache.gateDiag] = ...
                evaluateWatcherCompositeResidual( ...
                    watcher, cache.eta, cache.thetaLocal, cfg);

            if ~isfield(cache.gateDiag, "JthetaRawAll")
                error("DNN_EKF_Predict_Local:MissingCachedJtheta", ...
                    "Composite residual did not return branch Jtheta cache.");
            end
            cache.JthetaLocalRaw = cache.gateDiag.JthetaRawAll( ...
                :, :, watcher.localBranchID);
            cache.valid = true;

        case "oracle"
            % Oracle is a diagnostic path, not the expensive realizable DNN
            % path targeted by this cache patch.
            return;

        otherwise
            error("DNN_EKF_Predict_Local:UnknownResidualSource", ...
                "Unsupported prediction residual source: %s", ...
                cache.residualSource);
    end
end

function tf = isBaseResidualCacheMatch_step09j7( ...
    cache, t, eta, thetaLocal, residualSource)
%ISBASERESIDUALCACHEMATCH_STEP09J7 Guard cache use by exact base-point state.

    tf = isstruct(cache) && isfield(cache, "valid") && cache.valid;
    if ~tf
        return;
    end

    tf = cache.t == t && ...
         string(cache.residualSource) == string(residualSource) && ...
         isequal(cache.eta, eta(:)) && ...
         isequal(cache.thetaLocal, thetaLocal(:));
end

function tf = useBlockCovariancePrediction(cfg)
%{
Function:
    useBlockCovariancePrediction

Purpose:
    Return true if the local DNN-EKF should use block-structured covariance
    prediction instead of dense F*P*F' + Q.

Default:
    true

Notes:
    Set

        cfg.ekf.useBlockCovPrediction = false

    to temporarily recover the dense propagation for debugging.
%}

    tf = true;

    if isfield(cfg, "ekf") && isfield(cfg.ekf, "useBlockCovPrediction")
        tf = logical(cfg.ekf.useBlockCovPrediction);
    end

end
