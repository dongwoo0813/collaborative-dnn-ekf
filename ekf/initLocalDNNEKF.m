function watcher = initLocalDNNEKF(i, etaTrue0, cfg)
%{
Function:
    initLocalDNNEKF.m

Purpose:
    Initialize the local augmented DNN-EKF for watcher i.

    In the local DNN-EKF, watcher i estimates the augmented state

        X_i = [eta_i; theta_i],

    where

        eta_i = [r_t; v_t]

    is the physical target state and theta_i is the local DNN branch
    parameter vector.

    Each watcher estimates only its own local branch parameter theta_i.
    Nonlocal branch parameters from other watchers will be added later
    through GS-assisted or P2P communication.

Inputs:
    i         - Watcher index.
                Type: positive integer.
                Range:
                    1 <= i <= cfg.Nw.

    etaTrue0  - Initial true target physical state.
                Size: 2*cfg.dim x 1.
                Definition:
                    etaTrue0 = [r_t(0); v_t(0)].

    cfg       - Simulation configuration structure.
                Required fields:
                    cfg.dim
                    cfg.Nw
                    cfg.ekf.r0_err_std
                    cfg.ekf.v0_err_std
                    cfg.ekf.P0_pos
                    cfg.ekf.P0_vel
                    cfg.dnn.nPhi
                    cfg.dnn.nThetaPerBranch
                    cfg.dnn.theta0_std
                    cfg.dnn.Ptheta0

Outputs:
    watcher   - Local DNN-EKF watcher structure.
                Fields:
                    watcher.id
                    watcher.xhat
                    watcher.P
                    watcher.idxEta
                    watcher.idxTheta
                    watcher.nEta
                    watcher.nTheta
                    watcher.nX
                    watcher.localBranchID
                    watcher.lastInnovation
                    watcher.lastS

Main equations:
    The augmented state is

        X_i = [eta_i; theta_i].

    The initial physical estimate is

        eta_hat_i(0) = eta_true(0) + e_i,

    where

        e_i = [e_r; e_v].

    The initial DNN parameter estimate is close to zero:

        theta_hat_i(0) ~ N(0, theta0_std^2 I).

    The initial augmented covariance is block diagonal:

        P_i(0)
        =
        blkdiag(P_etaeta_i(0), P_thetatheta_i(0)).

    where

        P_etaeta_i(0) = blkdiag(P0_pos I_dim, P0_vel I_dim),

        P_thetatheta_i(0) = Ptheta0 I.

Notes:
    - The filter is not initialized with the hidden truth parameter
      theta_i^star.

    - The initial cross-covariance P_eta_theta is zero.
    
    - Cross-covariance between eta_i and theta_i will be generated during
      DNN-EKF prediction because the physical acceleration depends on
      theta_i through W_i phi_i(eta).

    - This function initializes empty GS/P2P communication-cache fields so
      MATLAB struct-array assignment remains safe in later Step 04/05 code.
    
    - The actual GS branch copies are populated later by
      broadcastGSRepositoryToWatcher.m.
%}

    dim = cfg.dim;

    nEta = 2 * dim;

    % ---------------------------------------------------------------------
    % Branch-model-aware DNN parameter dimension
    % ---------------------------------------------------------------------
    % Old fixed-feature branch:
    %     theta_i = vec(W_i), usually nTheta = 18 in 2-D.
    %
    % New MLP branch:
    %     theta_i = vec(all MLP weights/biases), e.g., nTheta = 256 for
    %     layerSizes = [6 12 8 6 2].
    %
    % Do not infer nTheta from featureBlock here, because mlp_general does
    % not use featureBlock.
    [nTheta, branchInfo] = branchThetaNumel(cfg);

    branchModel = string(branchInfo.branchModel);

    % Optional backward-compatibility check for the old fixed-feature model.
    if branchModel == "fixed_feature_lip" && exist("featureBlock", "file") == 2

        etaRef = etaTrue0;
        phi_i = featureBlock(i, etaRef, cfg);
        nPhiActual = numel(phi_i);
        nThetaActual = dim * nPhiActual;

        if nThetaActual ~= nTheta
            error("initLocalDNNEKF:FixedFeatureThetaMismatch", ...
                "branchThetaNumel gives %d, but featureBlock implies %d.", ...
                nTheta, nThetaActual);
        end

    end

    nX = nEta + nTheta;

    idxEta = 1:nEta;
    idxTheta = nEta + (1:nTheta);


    % ---------------------------------------------------------------------
    % Initial physical target-state estimate eta_hat_i(0)
    % ---------------------------------------------------------------------
    posErr = cfg.ekf.r0_err_std * randn(dim,1);
    velErr = cfg.ekf.v0_err_std * randn(dim,1);

    etaHat0 = etaTrue0 + [posErr; velErr];

    Peta0 = blkdiag(cfg.ekf.P0_pos * eye(dim), ...
                    cfg.ekf.P0_vel * eye(dim));


    % ---------------------------------------------------------------------
    % Initial local DNN branch parameter estimate theta_hat_i(0)
    % ---------------------------------------------------------------------
    % fixed_feature_lip:
    %     Preserves the old behavior. If theta0_std = 0, no randn call is
    %     made, preserving fair physical-EKF comparison streams.
    %
    % mlp_general:
    %     Uses branch-model-aware initialization. By default, hidden layers
    %     are initialized with deterministic small random weights using a
    %     local RandStream, while the output layer is zero. Therefore the
    %     initial learned residual is zero, but hidden features are nonzero.
    [thetaHat0, thetaInitInfo] = initBranchTheta(i, cfg);

    if numel(thetaHat0) ~= nTheta
        error("initLocalDNNEKF:ThetaInitLengthMismatch", ...
            "initBranchTheta returned length %d, expected %d.", ...
            numel(thetaHat0), nTheta);
    end

    Ptheta0 = cfg.dnn.Ptheta0 * eye(nTheta);


    % ---------------------------------------------------------------------
    % Augmented state and covariance
    % ---------------------------------------------------------------------
    xhat0 = zeros(nX,1);
    xhat0(idxEta) = etaHat0;
    xhat0(idxTheta) = thetaHat0;

    P0 = zeros(nX,nX);
    P0(idxEta, idxEta) = Peta0;
    P0(idxTheta, idxTheta) = Ptheta0;

    P0 = 0.5 * (P0 + P0');

    % ---------------------------------------------------------------------
    % Store watcher DNN-EKF structure
    % ---------------------------------------------------------------------
    watcher.id = i;

    watcher.xhat = xhat0;
    watcher.P = P0;

    watcher.idxEta = idxEta;
    watcher.idxTheta = idxTheta;

    watcher.nEta = nEta;
    watcher.nTheta = nTheta;
    watcher.nX = nX;

    watcher.localBranchID = i;

    watcher.branchModel = branchModel;
    watcher.branchInfo = branchInfo;
    watcher.thetaInitInfo = thetaInitInfo;

    % ---------------------------------------------------------------------
    % Step 09-J.1 local bearing-geometry support metadata
    % ---------------------------------------------------------------------
    % OmegaBar stores the cumulative direction-only bearing-FIM support
    % associated with this watcher's own local branch. It is passive metadata
    % at this step: it does not change xhat, P, GS upload, or the composite
    % residual until the later bearing_fim_gated mode is added.
    watcher.OmegaBar = zeros(dim, dim);
    watcher.numOmegaUpdates = 0;
    watcher.lastLOSUnit = NaN(dim, 1);
    watcher.lastOmegaUpdateTime = NaN;

    watcher.lastOmegaUpdate = struct();
    watcher.lastOmegaUpdate.updated = false;
    watcher.lastOmegaUpdate.reason = "not_initialized";
    watcher.lastOmegaUpdate.traceOmegaBar = 0;
    watcher.lastOmegaUpdate.minEigOmegaBar = 0;
    watcher.lastOmegaUpdate.nullResidual = NaN;

    % ---------------------------------------------------------------------
    % Step 04 communication-cache placeholders
    % ---------------------------------------------------------------------
    % These fields are intentionally initialized here, even though Step 03
    % does not use GS communication. The reason is MATLAB struct-array
    % consistency: later, Step 04 will update watchers(i).gsBranches after
    % a GS broadcast. If only one watcher receives this new field, assigning
    % it back into the watchers array can fail because different struct
    % elements have different field sets.
    %
    % The actual nonlocal branch records are populated by
    % gs/broadcastGSRepositoryToWatcher.m.
    %
    % Important: gsBranches contains only cached nonlocal branch copies from
    % the GS. It must not overwrite watcher.xhat(watcher.idxTheta), which is
    % this watcher's own local branch posterior.
    watcher.gsBranches = [];

    watcher.lastGSBroadcast = struct();
    watcher.lastGSBroadcast.time = NaN;
    watcher.lastGSBroadcast.numIncluded = 0;
    watcher.lastGSBroadcast.includedBranchIDs = [];

    watcher.lastGSUpload = struct();
    watcher.lastGSUpload.time = NaN;
    watcher.lastGSUpload.accepted = false;
    watcher.lastGSUpload.branchID = i;


    % ---------------------------------------------------------------------
    % Step 04b non-local covariance-injection diagnostics placeholder
    % ---------------------------------------------------------------------
    % This field must be initialized for every watcher before the watchers
    % struct array is formed. Otherwise, DNN_EKF_Predict_Local.m may add this
    % field to only one watcher during prediction, causing MATLAB struct-array
    % assignment errors:
    %
    %     Subscripted assignment between dissimilar structures.
    %
    % The actual values are overwritten inside DNN_EKF_Predict_Local.m.
    watcher.lastNonlocalCovInjection = struct();

    watcher.lastNonlocalCovInjection.enabled = false;
    watcher.lastNonlocalCovInjection.branchIDs = [];
    watcher.lastNonlocalCovInjection.youngCoefficients = [];
    watcher.lastNonlocalCovInjection.traceSj = [];
    watcher.lastNonlocalCovInjection.traceSdNonlocal = 0;
    watcher.lastNonlocalCovInjection.traceQnonlocal = 0;
    watcher.lastNonlocalCovInjection.numActiveNonlocal = 0;

    watcher.lastNonlocalCovInjection.SdNonlocal = zeros(cfg.dim, cfg.dim);
    watcher.lastNonlocalCovInjection.QnonlocalDiag = zeros(nX, 1);


    
    % ---------------------------------------------------------------------
    % Last measurement-update diagnostics
    % ---------------------------------------------------------------------
    watcher.lastInnovation = [];
    watcher.lastS = [];
    
    % ---------------------------------------------------------------------
    % Adaptive covariance matching state for DNN parameter process noise
    %
    % The goal is to adapt the local scalar multiplier gammaTheta such that
    %
    %     Q_theta,k = gammaTheta_i,k * Q_theta,base.
    %
    % The covariance matching uses an EWMA empirical innovation covariance
    %
    %     S_hat,k = (1-alpha_S) S_hat,k-1 + alpha_S (nu_k-mu_k)(nu_k-mu_k)'
    %
    % and compares trace(S_hat,k) against trace(S_model,k), where
    %
    %     S_model,k = H_X P^- H_X' + R.
    %
    % This is the local Step 03 analogue of the covariance-matching Q gain
    % idea from the single DNN-MEKF script.
    % ---------------------------------------------------------------------
    
    switch string(cfg.meas.type)
        case "bearing"
            nz = cfg.dim - 1;
        case "range_bearing"
            nz = cfg.dim;
        case {"relative_position", "direct_residual"}
            nz = cfg.dim;
        otherwise
            error("initLocalDNNEKF:UnsupportedMeasurementType", ...
                "Unsupported measurement type: %s", string(cfg.meas.type));
    end
    
    R0 = cfg.meas.R;
    if isscalar(R0)
        R0 = R0 * eye(nz);
    end
    
    if any(size(R0) ~= [nz, nz])
        error("cfg.meas.R has incompatible size in initLocalDNNEKF.");
    end
    
    cm = struct();
    
    % Enable/disable adaptive covariance matching.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "adaptQThetaEnabled")
        cm.adaptThetaEnabled = cfg.dnn.adaptQThetaEnabled;
    else
        cm.adaptThetaEnabled = true;
    end

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "adaptQEpsilonEnabled")
        cm.adaptEpsilonEnabled = cfg.dnn.adaptQEpsilonEnabled;
    else
        cm.adaptEpsilonEnabled = false;
    end

    cm.enabled = cm.adaptThetaEnabled || cm.adaptEpsilonEnabled;
    
    % EWMA settings for empirical innovation covariance.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "cmAlphaS")
        cm.alphaS = cfg.dnn.cmAlphaS;
    else
        cm.alphaS = 0.01;
    end
    
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "cmAlphaMu")
        cm.alphaMu = cfg.dnn.cmAlphaMu;
    else
        cm.alphaMu = 0.01;
    end
    
    % Adaptation scheduling.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "cmBurnInMeas")
        cm.burnInMeas = cfg.dnn.cmBurnInMeas;
    else
        cm.burnInMeas = 20;
    end
    
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "cmAdaptInt")
        cm.adaptInt = cfg.dnn.cmAdaptInt;
    else
        cm.adaptInt = 5;
    end
    
    % Deadband on log ratio.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "cmEpsLog")
        cm.epsLog = cfg.dnn.cmEpsLog;
    else
        cm.epsLog = 0.05;
    end
    
    % Multiplicative adaptation gain.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "cmGainTheta")
        cm.gainTheta = cfg.dnn.cmGainTheta;
    else
        cm.gainTheta = 1e-2;
    end
    
    % Clamp empirical/model covariance ratio before taking log.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "cmRatioMin")
        cm.ratioMin = cfg.dnn.cmRatioMin;
    else
        cm.ratioMin = 0.1;
    end
    
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "cmRatioMax")
        cm.ratioMax = cfg.dnn.cmRatioMax;
    else
        cm.ratioMax = 10.0;
    end
    
    % Clamp gammaTheta.
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "gammaTheta0")
        cm.gammaTheta = cfg.dnn.gammaTheta0;
    else
        cm.gammaTheta = 1.0;
    end
    
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "gammaThetaMin")
        cm.gammaThetaMin = cfg.dnn.gammaThetaMin;
    else
        cm.gammaThetaMin = 0.05;
    end
    
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "gammaThetaMax")
        cm.gammaThetaMax = cfg.dnn.gammaThetaMax;
    else
        cm.gammaThetaMax = 50.0;
    end

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "gammaEpsilon0")
        cm.gammaEpsilon = cfg.dnn.gammaEpsilon0;
    else
        cm.gammaEpsilon = 1.0;
    end

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "gammaEpsilonMin")
        cm.gammaEpsilonMin = cfg.dnn.gammaEpsilonMin;
    else
        cm.gammaEpsilonMin = 1e-2;
    end

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "gammaEpsilonMax")
        cm.gammaEpsilonMax = cfg.dnn.gammaEpsilonMax;
    else
        cm.gammaEpsilonMax = 1e2;
    end

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "cmGainEpsilon")
        cm.gainEpsilon = cfg.dnn.cmGainEpsilon;
    else
        cm.gainEpsilon = 2e-2;
    end
    
    % Internal EWMA state.
    cm.measCount = 0;
    cm.firstMeas = true;
    cm.muNu = zeros(nz,1);
    cm.Shat = R0;
    
    % Last adaptation diagnostics.
    cm.lastRatio = NaN;
    cm.lastLogRatio = NaN;
    cm.lastTraceEmp = NaN;
    cm.lastTraceModel = NaN;
    
    watcher.cm = cm;
end
