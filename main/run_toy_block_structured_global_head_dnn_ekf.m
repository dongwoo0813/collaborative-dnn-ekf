function out = run_toy_block_structured_global_head_dnn_ekf( ...
    seed,T,makePlots,nWatchers,dt,hiddenLayerCount,communicationMode,branchWidth,nParameterBlocks,packetAssignmentMode,informationTriggerThreshold,centralizedPeriod)
%RUN_TOY_BLOCK_STRUCTURED_GLOBAL_HEAD_DNN_EKF Separate architecture test.
%
% This file intentionally does not modify or call the older additive/WLS
% toy.  It implements one block-structured global DNN:
%
%   h_i^1 = tanh(W_i^1 eta_n + b_i^1)
%   h_i^l = tanh(W_i^l h_i^(l-1) + b_i^l)
%   d_hat  = [W_1^out ... W_N^out] [h_1; ...; h_N]
%          = sum_i W_i^out h_i.
%
% Watcher i owns {W_i^l,b_i^l,W_i^out}.  W_i^out h_i is not trained as an
% independent full-residual expert in the shared case: every bearing
% innovation is formed using the complete cached global model, and the
% local EKF updates only owner i's coordinate block.  At a communication
% instant, the entire local block posterior is uploaded and broadcast.  Every watcher
% evaluates all cached blocks at its own local state during propagation.
%
% The final layer is linear.  A separate output bias is deliberately
% omitted so N identical bias terms cannot create a trivial ambiguity.
%
% Comparisons returned:
%   nominal    : constant-velocity EKF, no learned residual
%   localOnly  : one block used as a complete local residual model
%   sharedBlock: proposed global block-structured DNN
%
% Example:
%   out = run_toy_block_structured_global_head_dnn_ekf( ...
%       101,600,true,4,0.1,3,"periodic_60s");
%
% Optional argument 12 overrides the GS synchronization period for every
% periodic mode.  For example, use T=5.1 and period=5 to exercise one
% full-joint GS correction in a short hybrid sanity run.

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(makePlots), makePlots = true; end
    if nargin < 4 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 5 || isempty(dt), dt = 0.1; end
    if nargin < 6 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 7 || isempty(communicationMode)
        communicationMode = "periodic_60s";
    end
    if nargin < 8 || isempty(branchWidth), branchWidth = 3; end
    if nargin < 9 || isempty(nParameterBlocks), nParameterBlocks = nWatchers; end
    if nargin < 10 || isempty(packetAssignmentMode), packetAssignmentMode = "all_to_all"; end
    if nargin < 11, informationTriggerThreshold = []; end
    if nargin < 12, centralizedPeriod = []; end

    validateattributes(nWatchers,{'numeric'}, ...
        {'scalar','integer','>=',1,'<=',8});
    validateattributes(hiddenLayerCount,{'numeric'}, ...
        {'scalar','integer','>=',2,'<=',5});
    validateattributes(branchWidth,{'numeric'}, ...
        {'scalar','integer','>=',1,'<=',32});
    validateattributes(nParameterBlocks,{'numeric'}, ...
        {'scalar','integer','>=',1,'<=',nWatchers});
    packetAssignmentMode = string(packetAssignmentMode);
    assert(any(packetAssignmentMode == ["all_to_all" "assigned" "assigned_one_per_block"]), ...
        'packetAssignmentMode must be all_to_all, assigned, or assigned_one_per_block.');
    communicationMode = string(communicationMode);
    assert(any(communicationMode == ...
        ["periodic_60s" "event_triggered" "instantaneous" "never" ...
         "collaborative_info_60s" "collaborative_output_info_60s" ...
         "collaborative_all_fogm_cm_60s" ...
         "hybrid_owner_ekf_joint_gs_60s" ...
         "hybrid_canonical_factor_joint_gs_60s" ...
         "hybrid_canonical_factor_joint_gs_position_aided_60s" ...
         "hybrid_canonical_factor_joint_gs_observability_aware" ...
         "centralized_common_dnn_step" "centralized_common_dnn_periodic" ...
         "centralized_common_dnn_periodic_fixed_q" ...
         "centralized_common_dnn_linearized_one_step_parity" ...
         "centralized_common_dnn_factor_packet_60s" ...
         "centralized_common_dnn_iterated_factor_packet_60s" ...
         "centralized_common_dnn_damped_factor_packet_5s" ...
         "centralized_common_dnn_full_trajectory_fg_5s" ...
         "centralized_window_static_dnn_fg_5s" ...
         "centralized_multirate_dnn_fg_5s" ...
         "block_owned_full_trajectory_fg_5s" ...
         "block_owned_exact_schur_fg_5s" ...
         "block_owned_structured_schur_fg_5s" ...
         "block_owned_packet_structured_schur_fg_5s" ...
         "block_owned_packet_structured_schur_diagnostic_5s" ...
         "block_owned_watcher_packet_structured_schur_fg_5s" ...
         "block_owned_watcher_multirate_packet_fg_5s" ...
         "block_owned_watcher_multirate_packet_fg_5s" ...
         "block_owned_time_ordered_sqrt_fg_5s" ...
         "block_owned_matrix_free_fg_5s"]), ...
        ['communicationMode must be periodic_60s, event_triggered, ' ...
         'instantaneous, never, collaborative_info_60s, or ' ...
         'collaborative_output_info_60s, collaborative_all_fogm_cm_60s, or ' ...
         'hybrid_owner_ekf_joint_gs_60s, hybrid_canonical_factor_joint_gs_60s, or ' ...
         'hybrid_canonical_factor_joint_gs_position_aided_60s, or ' ...
         'hybrid_canonical_factor_joint_gs_observability_aware, or ' ...
         'centralized_common_dnn_step, centralized_common_dnn_periodic, or ' ...
         'centralized_common_dnn_factor_packet_60s, or ' ...
         'centralized_common_dnn_iterated_factor_packet_60s.']);

    cfg = makeConfig(seed,T,dt,nWatchers,hiddenLayerCount,communicationMode,branchWidth,nParameterBlocks,packetAssignmentMode,informationTriggerThreshold);
    if communicationMode == "centralized_common_dnn_damped_factor_packet_5s"
        cfg.communication.period = 5;
    end
    if any(communicationMode == ["centralized_common_dnn_full_trajectory_fg_5s" "centralized_window_static_dnn_fg_5s" "centralized_multirate_dnn_fg_5s" "block_owned_full_trajectory_fg_5s" "block_owned_exact_schur_fg_5s" "block_owned_structured_schur_fg_5s" "block_owned_packet_structured_schur_fg_5s" "block_owned_packet_structured_schur_diagnostic_5s" "block_owned_watcher_packet_structured_schur_fg_5s" "block_owned_watcher_multirate_packet_fg_5s" "block_owned_time_ordered_sqrt_fg_5s" "block_owned_matrix_free_fg_5s"]) && isempty(centralizedPeriod)
        cfg.communication.period = 5;
    end
    if ~isempty(centralizedPeriod)
        validateattributes(centralizedPeriod,{'numeric'}, {'scalar','positive','finite'});
        % Keep the optional synchronization-period override consistent across
        % both centralized replay and GS packet/joint-update modes.  This is
        % particularly important for short sanity tests: without this, a
        % 5-s run of a *_60s hybrid mode never exercises its GS update.
        cfg.communication.centralizedPeriod = centralizedPeriod;
        cfg.communication.period = centralizedPeriod;
    end
    fprintf(['Block-structured global-head DNN EKF: seed=%d, T=%.0f s, ' ...
        'dt=%.2f s, watchers=%d, hiddenLayers=%d, width=%d, communication=%s\n'], ...
        seed,T,dt,nWatchers,hiddenLayerCount,cfg.dnn.width,communicationMode);

    data = makeCommonData(cfg);
    initial = makeInitialModels(cfg);

    rng(seed+1000); out.nominal = simulateCase(cfg,data,initial,"nominal");
    rng(seed+1000); out.localOnly = simulateCase(cfg,data,initial,"local_only");
    if communicationMode == "collaborative_info_60s"
        rng(seed+1000);
        out.sharedBlock = simulateCollaborativeBlockInformationCase( ...
            cfg,data,initial);
    elseif communicationMode == "collaborative_output_info_60s"
        rng(seed+1000);
        out.sharedBlock = simulateCollaborativeOutputInformationCase( ...
            cfg,data,initial);
    elseif communicationMode == "collaborative_all_fogm_cm_60s"
        rng(seed+1000);
        out.sharedBlock = simulateCollaborativeSelectedInformationCase( ...
            cfg,data,initial,1:cfg.dnn.arch.nTheta, ...
            "collaborative_all_fogm_covariance_matching",true);
    elseif communicationMode == "hybrid_owner_ekf_joint_gs_60s"
        rng(seed+1000);
        out.sharedBlock = simulateHybridOwnerEKFJointGSCase(cfg,data,initial);
    elseif communicationMode == "hybrid_canonical_factor_joint_gs_60s" || ...
            communicationMode == "hybrid_canonical_factor_joint_gs_position_aided_60s" || ...
            communicationMode == "hybrid_canonical_factor_joint_gs_observability_aware"
        rng(seed+1000);
        out.sharedBlock = simulateCanonicalFactorHybridCase(cfg,data,initial);
    elseif communicationMode == "centralized_common_dnn_step"
        rng(seed+1000);
        out.sharedBlock = simulateCentralizedCommonDNNCase(cfg,data,initial);
    elseif communicationMode == "centralized_common_dnn_periodic"
        rng(seed+1000);
        out.sharedBlock = simulateCentralizedPeriodicCommonDNNCase(cfg,data,initial);
    elseif communicationMode == "centralized_common_dnn_periodic_fixed_q"
        rng(seed+1000);
        out.sharedBlock = simulateCentralizedPeriodicCommonDNNCase(cfg,data,initial);
    elseif communicationMode == "centralized_common_dnn_factor_packet_60s"
        rng(seed+1000);
        out.sharedBlock = simulateCentralizedFactorPacketCase(cfg,data,initial);
    elseif communicationMode == "centralized_common_dnn_iterated_factor_packet_60s"
        rng(seed+1000);
        out.sharedBlock = simulateCentralizedIteratedFactorPacketCase(cfg,data,initial);
    elseif communicationMode == "centralized_common_dnn_damped_factor_packet_5s"
        rng(seed+1000);
        out.sharedBlock = simulateCentralizedIteratedFactorPacketCase(cfg,data,initial);
    elseif communicationMode == "centralized_common_dnn_full_trajectory_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateFullTrajectoryFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "centralized_window_static_dnn_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateWindowStaticThetaFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "centralized_multirate_dnn_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateMultiRateThetaFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "block_owned_full_trajectory_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateFullTrajectoryFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "block_owned_exact_schur_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateFullTrajectoryFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "block_owned_structured_schur_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateFullTrajectoryFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "block_owned_packet_structured_schur_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateFullTrajectoryFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "block_owned_packet_structured_schur_diagnostic_5s"
        rng(seed+1000);
        out.sharedBlock = simulateFullTrajectoryFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "block_owned_watcher_packet_structured_schur_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateFullTrajectoryFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "block_owned_watcher_multirate_packet_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateMultiRateOwnerWatcherPacketCase(cfg,data,initial);
    elseif communicationMode == "block_owned_time_ordered_sqrt_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateFullTrajectoryFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "block_owned_matrix_free_fg_5s"
        rng(seed+1000);
        out.sharedBlock = simulateFullTrajectoryFactorGraphCase(cfg,data,initial);
    elseif communicationMode == "centralized_common_dnn_linearized_one_step_parity"
        rng(seed+1000);
        out.sharedBlock = simulateLinearizedOneStepParityCase(cfg,data,initial);
    else
        rng(seed+1000); out.sharedBlock = simulateCase(cfg,data,initial,"shared_block");
    end

    out.cfg = cfg;
    out.truth = data.truth;
    out.trueAcceleration = data.dTrue;
    out.architecture = architectureDescription(cfg);
    out.summary = makeSummary(out);
    disp(out.summary);
    if makePlots
        out.figures = plotResults(out);
    else
        out.figures = struct;
    end
end

function cfg = makeConfig(seed,T,dt,nWatchers,nHidden,communicationMode,branchWidth,nParameterBlocks,packetAssignmentMode,informationTriggerThreshold)
    cfg.seed = seed;
    cfg.T = T;
    cfg.dt = dt;
    cfg.time = 0:dt:T;
    cfg.N = numel(cfg.time);
    cfg.Nw = nWatchers;
    cfg.Nblocks = nParameterBlocks;

    cfg.dnn.width = branchWidth;
    cfg.dnn.hiddenLayers = nHidden;
    cfg.dnn.inputScale = [100;100;0.8;0.8];
    cfg.dnn.ridge = 1e-5;
    cfg.dnn.initialWeightScale = 0.65;
    cfg.dnn.initialOutputWeightScale = 1e-3;
    cfg.dnn.parameterDeviationLimit = 0.75;
    cfg.dnn.parameterStepLimit = 0.08;
    cfg.dnn.arch = blockArchitecture(cfg);

    cfg.spiral.radiusGoal = 100;
    cfg.spiral.radialRate = 0.30;
    cfg.spiral.angularRate = 0.012;
    cfg.spiral.velocityGain = 0.035;
    cfg.target.r0 = [5;0];
    cfg.target.v0 = desiredVelocity(cfg.target.r0,cfg);

    angles = 2*pi*(0:nWatchers-1)/max(nWatchers,1);
    if nWatchers == 1, angles = 0; end
    cfg.watchers.r = 1000*[cos(angles);sin(angles)];
    cfg.measurement.sigma = deg2rad(0.01);
    cfg.measurement.positionEnabled = contains(communicationMode,"position_aided");
    cfg.measurement.positionSigma = 20;

    cfg.ekf.Peta0 = diag([30^2 30^2 0.08^2 0.08^2]);
    cfg.ekf.Ptheta0 = 2e-4;
    cfg.ekf.qAcceleration = (2e-5)^2;
    cfg.ekf.qTheta = 2e-10;
    cfg.ekf.remoteCovarianceScale = 0.15;

    cfg.communication.mode = communicationMode;
    cfg.communication.period = 60;
    cfg.communication.centralizedPeriod = dt;
    cfg.communication.minInterval = 15;
    cfg.communication.maxSilence = 90;
    cfg.communication.normalizedMahalanobisThreshold = 0.75;
    cfg.communication.varianceFloor = 1e-8;

    % Distributed block-information EKF safeguards.  The packet covariance
    % includes uncertainty from every parameter block; otherwise the same
    % angle innovation would make independent block covariances collapse.
    cfg.collaborative.infoStepLimit = 0.01;
    cfg.collaborative.parameterCovarianceFloor = 1e-6;
    cfg.collaborative.thetaTau = 1000;
    cfg.collaborative.covMatchRate = 0.10;
    cfg.collaborative.covMatchBounds = [0.05 20];
    cfg.collaborative.stateCovMatchRate = 0.05;
    cfg.collaborative.stateCovMatchBounds = [0.25 10];
    % The fixed-Q periodic mode is the fair comparison reference for the
    % full-trajectory factor graph, which presently keeps Q fixed.
    cfg.collaborative.enableCovarianceMatching = ...
        communicationMode ~= "centralized_common_dnn_periodic_fixed_q";
    cfg.collaborative.enableParameterClipping = ~any(communicationMode == ...
        ["centralized_common_dnn_periodic_fixed_q" ...
         "centralized_common_dnn_full_trajectory_fg_5s"]);
    cfg.collaborative.resetGain = 0.25;
    cfg.collaborative.syncStateCovarianceInflation = 1.05;
    cfg.collaborative.lmDampingSchedule = [0 1e-6 1e-4 1e-2 1];
    cfg.collaborative.lineSearchScales = [1 0.5 0.25 0.125 0.0625];
    cfg.collaborative.observabilityAware = contains(communicationMode,"observability_aware");
    % Prior-normalized average information per parameter.  The minimum
    % across blocks is used so every DNN block must receive useful data.
    cfg.collaborative.informationTriggerThreshold = 1.00;
    if ~isempty(informationTriggerThreshold)
        validateattributes(informationTriggerThreshold,{'numeric'}, ...
            {'scalar','positive','finite'});
        cfg.collaborative.informationTriggerThreshold = informationTriggerThreshold;
    end
    cfg.collaborative.packetAssignmentMode = packetAssignmentMode;
    cfg.collaborative.observerOwner = mod(0:nWatchers-1,nParameterBlocks)+1;
    cfg.collaborative.packetEnabled = true(1,nWatchers);
    if packetAssignmentMode == "assigned_one_per_block"
        cfg.collaborative.packetEnabled = false(1,nWatchers);
        for block = 1:nParameterBlocks
            firstObserver = find(cfg.collaborative.observerOwner == block,1,'first');
            cfg.collaborative.packetEnabled(firstObserver) = true;
        end
    end

    % Fixed-lag nonlinear solver settings.  The 5-s reference commonly
    % converges within five iterations, while longer windows may need more.
    cfg.factorGraph.maxIterations = 10;
    cfg.factorGraph.relativeCostTolerance = 1e-3;
    cfg.factorGraph.thetaKnotInterval = 1.0; % multi-rate benchmark default [s]
    cfg.factorGraph.blockOwnedPackets = any(communicationMode == ...
        ["block_owned_full_trajectory_fg_5s" "block_owned_exact_schur_fg_5s" ...
         "block_owned_structured_schur_fg_5s" "block_owned_packet_structured_schur_fg_5s" ...
         "block_owned_packet_structured_schur_diagnostic_5s" ...
         "block_owned_watcher_packet_structured_schur_fg_5s" ...
         "block_owned_time_ordered_sqrt_fg_5s"]);
    cfg.factorGraph.matrixFreeSolver = communicationMode == "block_owned_matrix_free_fg_5s";
    cfg.factorGraph.exactSchurSolver = communicationMode == "block_owned_exact_schur_fg_5s";
    cfg.factorGraph.structuredSchurSolver = communicationMode == "block_owned_structured_schur_fg_5s";
    cfg.factorGraph.packetStructuredSchurSolver = any(communicationMode == ...
        ["block_owned_packet_structured_schur_fg_5s" "block_owned_packet_structured_schur_diagnostic_5s" ...
         "block_owned_watcher_packet_structured_schur_fg_5s"]);
    cfg.factorGraph.packetParityDiagnostic = communicationMode == "block_owned_packet_structured_schur_diagnostic_5s";
    cfg.factorGraph.distributedOwnerWatcherPackets = communicationMode == ...
        "block_owned_watcher_packet_structured_schur_fg_5s";
    cfg.factorGraph.distributedMultiRatePackets = communicationMode == ...
        "block_owned_watcher_multirate_packet_fg_5s";
    cfg.factorGraph.timeOrderedSquareRootSolver = communicationMode == "block_owned_time_ordered_sqrt_fg_5s";
    % Experimental fixed-lag boundary elimination is disabled by default:
    % it changes the nonlinear batch problem unless past factors are kept
    % available for relinearization.
    cfg.factorGraph.timeOrderedBoundaryElimination = false;
    if cfg.factorGraph.matrixFreeSolver, cfg.factorGraph.blockOwnedPackets = true; end
    % Inexact-GN inner tolerance.  A converged 1e-6 PCG solve is preferable
    % to accepting a 1e-8 request that hits the iteration cap.
    cfg.factorGraph.linearSolveTolerance = 1e-6;
    cfg.factorGraph.linearSolveMaxIterations = 2000;
    cfg.factorGraph.cacheOwnerPackets = true;

    cfg.initial.positionSigma = 20;
    cfg.initial.velocitySigma = 0.04;
    cfg.training.nWarm = 900;
end

function arch = blockArchitecture(cfg)
    q = cfg.dnn.width;
    L = cfg.dnn.hiddenLayers;
    inputDim = 4;
    idx = 1;
    hidden = repmat(struct('nIn',0,'nOut',0,'idxW',[],'idxb',[]),L,1);
    for ell = 1:L
        nIn = inputDim;
        if ell > 1, nIn = q; end
        nW = q*nIn;
        hidden(ell).nIn = nIn;
        hidden(ell).nOut = q;
        hidden(ell).idxW = idx:(idx+nW-1); idx = idx+nW;
        hidden(ell).idxb = idx:(idx+q-1); idx = idx+q;
    end
    arch.idxOutputW = idx:(idx+2*q-1);
    arch.nTheta = idx+2*q-1;
    arch.hidden = hidden;
    arch.nHidden = L;
    arch.width = q;
    arch.inputDim = inputDim;
end

function data = makeCommonData(cfg)
    rng(cfg.seed);
    N = cfg.N;
    truth = zeros(4,N);
    truth(:,1) = [cfg.target.r0;cfg.target.v0];
    dTrue = zeros(2,N);
    for k = 1:N-1
        dTrue(:,k) = trueResidual(truth(:,k),cfg);
        truth(:,k+1) = physicalStep(truth(:,k),dTrue(:,k),cfg.dt);
    end
    dTrue(:,N) = trueResidual(truth(:,N),cfg);

    bearingNoise = cfg.measurement.sigma*randn(cfg.Nw,N);
    bearings = zeros(cfg.Nw,N);
    for i = 1:cfg.Nw
        dr = truth(1:2,:)-cfg.watchers.r(:,i);
        bearings(i,:) = atan2(dr(2,:),dr(1,:))+bearingNoise(i,:);
    end
    % Generated after bearing data so existing bearing-only cases retain
    % their exact historical truth and bearing-noise sequences.
    positionMeasurements = truth(1:2,:) + ...
        cfg.measurement.positionSigma*randn(2,N,cfg.Nw);

    initialError = [cfg.initial.positionSigma*randn(2,cfg.Nw); ...
        cfg.initial.velocitySigma*randn(2,cfg.Nw)];
    data.truth = truth;
    data.dTrue = dTrue;
    data.bearings = bearings;
    data.positionMeasurements = positionMeasurements;
    data.initialEta = truth(:,1)+initialError;
end

function initial = makeInitialModels(cfg)
%MAKEINITIALMODELS Strictly random small initialization for every case.
%
% No current-mission truth or acceleration label is used here.  All cases
% start from exactly the same random hidden layers and small random output
% heads, so performance differences come from online learning and
% communication rather than supervised warm-start information.

    arch = cfg.dnn.arch;
    thetaBase = zeros(arch.nTheta,cfg.Nblocks);
    for i = 1:cfg.Nblocks
        rng(cfg.seed+7919*i);
        thetaBase(:,i) = initializeHiddenBlock(cfg,i);
        Wout0 = cfg.dnn.initialOutputWeightScale/sqrt(cfg.Nblocks) * ...
            randn(2,arch.width);
        thetaBase(arch.idxOutputW,i) = Wout0(:);
    end

    initial.shared = thetaBase;
    % Local learning remains one independent branch per physical watcher.
    % It is deliberately separate from the shared global block count.
    thetaLocal = zeros(arch.nTheta,cfg.Nw);
    for i = 1:cfg.Nw
        rng(cfg.seed+7919*i);
        thetaLocal(:,i) = initializeHiddenBlock(cfg,i);
        Wout0 = cfg.dnn.initialOutputWeightScale/sqrt(cfg.Nblocks)*randn(2,arch.width);
        thetaLocal(arch.idxOutputW,i) = Wout0(:);
    end
    initial.local = thetaLocal;
end

function theta = initializeHiddenBlock(cfg,blockID)
    arch = cfg.dnn.arch;
    theta = zeros(arch.nTheta,1);
    for ell = 1:arch.nHidden
        nIn = arch.hidden(ell).nIn;
        W = cfg.dnn.initialWeightScale/sqrt(nIn)*randn(arch.width,nIn);
        % A deterministic phase offset helps different owners start with
        % complementary features without a learned gate.
        b = 0.20*sin(blockID+(1:arch.width)'*(ell+0.5));
        theta(arch.hidden(ell).idxW) = W(:);
        theta(arch.hidden(ell).idxb) = b;
    end
end

function result = simulateCase(cfg,data,initial,mode)
    Nw = cfg.Nw; N = cfg.N; p = cfg.dnn.arch.nTheta;
    isNominal = mode == "nominal";
    isShared = mode == "shared_block";

    if isNominal
        nX = 4;
        thetaInit = zeros(p,Nw);
    elseif isShared
        nX = 4+p;
        thetaInit = initial.shared;
    else
        nX = 4+p;
        thetaInit = initial.local;
    end

    x = zeros(nX,Nw);
    P = zeros(nX,nX,Nw);
    thetaReference = thetaInit;
    for i = 1:Nw
        x(1:4,i) = data.initialEta(:,i);
        P(1:4,1:4,i) = cfg.ekf.Peta0;
        if ~isNominal
            x(5:end,i) = thetaInit(:,i);
            P(5:end,5:end,i) = cfg.ekf.Ptheta0*eye(p);
        end
    end

    cacheTheta = thetaInit;
    cacheP = repmat(cfg.ekf.Ptheta0*eye(p),1,1,Nw);
    uploadedTheta = thetaInit;
    uploadedP = cacheP;
    lastUpload = zeros(1,Nw);
    uploadTimes = cell(1,Nw);

    etaHistory = zeros(4,N,Nw);
    dHistory = zeros(2,N,Nw);
    thetaChange = zeros(N,Nw);
    nis = zeros(N,Nw);
    etaHistory(:,1,:) = reshape(x(1:4,:),4,1,Nw);

    for k = 1:N-1
        t = cfg.time(k);
        for i = 1:Nw
            if isNominal
                [xPred,FPred,QPred,dHat] = predictNominal(x(:,i),cfg);
            else
                thetaOwner = x(5:end,i);
                localCache = cacheTheta;
                localCache(:,i) = thetaOwner;
                localPCache = cacheP;
                localPCache(:,:,i) = P(5:end,5:end,i);
                [xPred,FPred,QPred,dHat] = predictLearned( ...
                    x(:,i),i,localCache,localPCache,cfg,mode);
            end
            PPred = FPred*P(:,:,i)*FPred'+QPred;
            PPred = symmetrizePSD(PPred,1e-12);

            [innovation,Heta,R] = bearingInnovation( ...
                data.bearings(i,k+1),xPred(1:4),cfg.watchers.r(:,i),cfg);
            % The bearing has no direct dependence on the parameters; they
            % are corrected through the propagated eta-theta cross
            % covariance created by the process Jacobian.
            H = zeros(1,nX);
            H(1:4) = Heta;
            S = H*PPred*H'+R;
            K = (PPred*H')/S;
            xNew = xPred+K*innovation;
            I = eye(nX);
            PNew = (I-K*H)*PPred*(I-K*H)'+K*R*K';

            if ~isNominal
                oldTheta = xPred(5:end);
                proposed = xNew(5:end);
                step = proposed-oldTheta;
                stepNorm = norm(step);
                if stepNorm > cfg.dnn.parameterStepLimit
                    proposed = oldTheta+cfg.dnn.parameterStepLimit*step/stepNorm;
                end
                delta0 = proposed-thetaReference(:,i);
                delta0 = max(min(delta0,cfg.dnn.parameterDeviationLimit), ...
                    -cfg.dnn.parameterDeviationLimit);
                xNew(5:end) = thetaReference(:,i)+delta0;
                thetaChange(k+1,i) = norm(xNew(5:end)-thetaReference(:,i));
            end

            x(:,i) = xNew;
            P(:,:,i) = symmetrizePSD(PNew,1e-12);
            etaHistory(:,k+1,i) = xNew(1:4);
            dHistory(:,k,i) = dHat;
            nis(k+1,i) = innovation^2/S;
        end

        % Coordinate sharing happens after all local angle-only updates.
        if isShared && cfg.communication.mode ~= "never"
            for owner = 1:Nw
                thetaNow = x(5:end,owner);
                Pnow = P(5:end,5:end,owner);
                doUpload = cfg.communication.mode == "instantaneous";
                if cfg.communication.mode == "periodic_60s"
                    age = t-lastUpload(owner);
                    doUpload = age >= cfg.communication.period-0.5*cfg.dt;
                end
                if cfg.communication.mode == "event_triggered"
                    age = t-lastUpload(owner);
                    if age >= cfg.communication.minInterval
                        variance = diag(Pnow)+diag(uploadedP(:,:,owner)) + ...
                            cfg.communication.varianceFloor;
                        score = mean((thetaNow-uploadedTheta(:,owner)).^2./variance);
                        doUpload = score >= cfg.communication.normalizedMahalanobisThreshold ...
                            || age >= cfg.communication.maxSilence;
                    end
                end
                if doUpload
                    cacheTheta(:,owner) = thetaNow;
                    cacheP(:,:,owner) = Pnow;
                    uploadedTheta(:,owner) = thetaNow;
                    uploadedP(:,:,owner) = Pnow;
                    lastUpload(owner) = t;
                    uploadTimes{owner}(end+1) = t;
                end
            end
        end
    end

    % Evaluate the terminal learned acceleration at every local estimate.
    for i = 1:Nw
        if isNominal
            dHistory(:,N,i) = 0;
        elseif isShared
            finalCache = cacheTheta;
            finalCache(:,i) = x(5:end,i);
            dHistory(:,N,i) = evaluateGlobal( ...
                x(1:4,i),finalCache,cfg,0,"shared_block");
        else
            dHistory(:,N,i) = blockOutput( ...
                x(1:4,i),x(5:end,i),cfg);
        end
    end

    result.mode = mode;
    result.time = cfg.time;
    result.etaHat = etaHistory;
    result.dHat = dHistory;
    result.thetaChange = thetaChange;
    result.nis = nis;
    result.uploadTimes = uploadTimes;
    result.parameterUploads = sum(cellfun(@numel,uploadTimes));
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue);
    result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2));
    result.accelerationRMSE = sqrt(mean(result.accelerationError.^2));
    result.finalPositionRMSE = result.positionError(end);
end

function result = simulateCollaborativeBlockInformationCase(cfg,data,initial)
%SIMULATECOLLABORATIVEBLOCKINFORMATIONCASE Distributed parameter learning.
%
% Each watcher maintains only a state EKF.  The DNN parameter vector is
% partitioned by owner: owner j stores theta_j and P_j, never a full global
% covariance.  Watcher i propagates the sensitivity of its state estimate
% to every parameter block and accumulates a 60-s information packet for
% each owner.  At a synchronization epoch owner j applies the sum of the
% packets from all watchers to theta_j only.  This is a block-diagonal
% approximation of a global parameter information filter, not raw copying
% of independently estimated parameters.

    Nw = cfg.Nw; N = cfg.N; p = cfg.dnn.arch.nTheta;
    nTheta = p*Nw;
    theta = initial.shared;
    thetaReference = theta;
    Ptheta = repmat(cfg.ekf.Ptheta0*eye(p),1,1,Nw);

    eta = data.initialEta;
    Peta = repmat(cfg.ekf.Peta0,1,1,Nw);
    % G_i = d eta_i / d [theta_1; ...; theta_Nw].  This is linear in the
    % total parameter count, unlike the omitted full parameter covariance.
    G = zeros(4,nTheta,Nw);
    packetLambda = zeros(p,p,Nw,Nw); % (observer, owner)
    packetq = zeros(p,Nw,Nw);        % (parameter, observer, owner)
    uploadTimes = cell(1,Nw);
    lastSync = 0;

    etaHistory = zeros(4,N,Nw);
    dHistory = zeros(2,N,Nw);
    thetaChange = zeros(N,Nw);
    nis = zeros(N,Nw);
    etaHistory(:,1,:) = reshape(eta,4,1,Nw);

    for k = 1:N-1
        t = cfg.time(k);
        % The parameter prior is propagated once per physical time step by
        % its owners.  The covariance remains block diagonal by design.
        for owner = 1:Nw
            Ptheta(:,:,owner) = symmetrizePSD(Ptheta(:,:,owner) + ...
                cfg.ekf.qTheta*eye(p),1e-12);
        end

        for observer = 1:Nw
            [dHat,Jeta,Jall,parameterQd] = evaluateGlobalAllJacobians( ...
                eta(:,observer),theta,Ptheta,cfg);
            [etaPred,Feta,Qeta] = predictCollaborativeState( ...
                eta(:,observer),dHat,Jeta,parameterQd,cfg);
            Gpred = Feta*G(:,:,observer) + ...
                [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)]*Jall;
            Ppred = symmetrizePSD(Feta*Peta(:,:,observer)*Feta' + Qeta,1e-12);

            [innovation,Heta,R] = bearingInnovation( ...
                data.bearings(observer,k+1),etaPred,cfg.watchers.r(:,observer),cfg);
            Sstate = Heta*Ppred*Heta' + R;
            Keta = (Ppred*Heta')/Sstate;
            etaNew = etaPred + Keta*innovation;
            Ieta = eye(4);
            Pnew = (Ieta-Keta*Heta)*Ppred*(Ieta-Keta*Heta)' + Keta*R*Keta';
            Gnew = (Ieta-Keta*Heta)*Gpred;

            % Htheta is the predicted-bearing sensitivity.  Its sign is
            % chosen for the residual equation z-h ~= Htheta*dtheta+v,
            % so the information vector uses +Htheta'*nu/S.
            Htheta = Heta*Gpred;
            % State filtering uses Sstate.  Parameter learning must also
            % regard all DNN blocks as uncertain.  Omitting this term makes
            % every block falsely treat one scalar innovation as a highly
            % precise independent measurement and causes covariance collapse.
            Sinfo = Sstate;
            for uncertainOwner = 1:Nw
                cols = (uncertainOwner-1)*p+(1:p);
                Hblock = Htheta(:,cols);
                Sinfo = Sinfo + Hblock*Ptheta(:,:,uncertainOwner)*Hblock';
            end
            for owner = 1:Nw
                cols = (owner-1)*p+(1:p);
                Hij = Htheta(:,cols);
                packetLambda(:,:,observer,owner) = ...
                    packetLambda(:,:,observer,owner) + (Hij'*Hij)/Sinfo;
                packetq(:,observer,owner) = packetq(:,observer,owner) + ...
                    Hij'*(innovation/Sinfo);
            end

            eta(:,observer) = etaNew;
            Peta(:,:,observer) = symmetrizePSD(Pnew,1e-12);
            G(:,:,observer) = Gnew;
            etaHistory(:,k+1,observer) = etaNew;
            dHistory(:,k,observer) = dHat;
            nis(k+1,observer) = innovation^2/Sstate;
        end

        % All owners update at the same epoch against packets evaluated at
        % the common pre-update global model.  This preserves a canonical
        % model and avoids the raw-copy cache mismatch of the old case.
        if t-lastSync >= cfg.communication.period-0.5*cfg.dt
            for owner = 1:Nw
                Lambda = zeros(p);
                q = zeros(p,1);
                for observer = 1:Nw
                    Lambda = Lambda + packetLambda(:,:,observer,owner);
                    q = q + packetq(:,observer,owner);
                end
                informationPosterior = (Ptheta(:,:,owner) \ eye(p)) + Lambda;
                Pnew = symmetrizePSD(informationPosterior \ eye(p), ...
                    cfg.collaborative.parameterCovarianceFloor);
                delta = Pnew*q;
                deltaNorm = norm(delta);
                if deltaNorm > cfg.collaborative.infoStepLimit
                    delta = cfg.collaborative.infoStepLimit*delta/deltaNorm;
                end
                proposal = theta(:,owner) + delta;
                deviation = proposal-thetaReference(:,owner);
                deviation = max(min(deviation,cfg.dnn.parameterDeviationLimit), ...
                    -cfg.dnn.parameterDeviationLimit);
                theta(:,owner) = thetaReference(:,owner) + deviation;
                Ptheta(:,:,owner) = Pnew;
                thetaChange(k+1,owner) = norm(theta(:,owner)-thetaReference(:,owner));
                uploadTimes{owner}(end+1) = t;
            end
            packetLambda(:) = 0;
            packetq(:) = 0;
            lastSync = t;
        end
    end

    for observer = 1:Nw
        dHistory(:,N,observer) = evaluateGlobal( ...
            eta(:,observer),theta,cfg,0,"shared_block");
    end

    result.mode = "collaborative_block_information";
    result.time = cfg.time;
    result.etaHat = etaHistory;
    result.dHat = dHistory;
    result.thetaChange = thetaChange;
    result.nis = nis;
    result.uploadTimes = uploadTimes;
    result.parameterUploads = sum(cellfun(@numel,uploadTimes));
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue);
    result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2));
    result.accelerationRMSE = sqrt(mean(result.accelerationError.^2));
    result.finalPositionRMSE = result.positionError(end);
    result.parameterCovariance = Ptheta;
end

function result = simulateCollaborativeOutputInformationCase(cfg,data,initial)
    result = simulateCollaborativeSelectedInformationCase( ...
        cfg,data,initial,cfg.dnn.arch.idxOutputW, ...
        "collaborative_output_information",false);
end

function result = simulateCollaborativeSelectedInformationCase( ...
    cfg,data,initial,learnIdx,resultMode,useFOGM)
%SIMULATECOLLABORATIVEOUTPUTINFORMATIONCASE Phase-1 distributed learning.
%
% Hidden layers remain fixed random features.  Each owner updates only its
% final linear output-layer weight W_i^out with information contributed by
% every watcher's angle innovation.  This is the identifiable first stage
% before allowing the nonlinear hidden weights to adapt.

    Nw = cfg.Nw; Nb = cfg.Nblocks; N = cfg.N;
    pLearn = numel(learnIdx); nLearn = pLearn*Nb;
    theta = initial.shared;
    thetaReference = theta;
    Pout = repmat(cfg.ekf.Ptheta0*eye(pLearn),1,1,Nb);
    eta = data.initialEta;
    Peta = repmat(cfg.ekf.Peta0,1,1,Nw);
    G = zeros(4,nLearn,Nw);
    PetaOut = zeros(4,nLearn,Nw);
    packetLambda = zeros(pLearn,pLearn,Nw,Nb);
    packetq = zeros(pLearn,Nw,Nb);
    uploadTimes = cell(1,Nb);
    lastSync = 0;
    gammaTheta = ones(1,Nb);
    nisWindowSum = 0; nisWindowCount = 0;

    etaHistory = zeros(4,N,Nw);
    dHistory = zeros(2,N,Nw);
    thetaChange = zeros(N,Nb);
    nis = zeros(N,Nw);
    etaHistory(:,1,:) = reshape(eta,4,1,Nw);

    for k = 1:N-1
        t = cfg.time(k);
        for owner = 1:Nb
            if useFOGM
                alpha = exp(-cfg.dt/cfg.collaborative.thetaTau);
                theta(learnIdx,owner) = thetaReference(learnIdx,owner) + ...
                    alpha*(theta(learnIdx,owner)-thetaReference(learnIdx,owner));
                sigmaSS2 = cfg.ekf.Ptheta0;
                Qtheta = gammaTheta(owner)*sigmaSS2*(1-alpha^2)*eye(pLearn);
                Pout(:,:,owner) = symmetrizePSD(alpha^2*Pout(:,:,owner) + Qtheta,1e-12);
            else
                Pout(:,:,owner) = symmetrizePSD(Pout(:,:,owner) + ...
                    cfg.ekf.qTheta*eye(pLearn),1e-12);
            end
        end

        for observer = 1:Nw
            [dHat,Jeta,Jall,~] = evaluateGlobalSelectedJacobians( ...
                eta(:,observer),theta,Pout,learnIdx,cfg);
            Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
            etaPred = physicalStep(eta(:,observer),dHat,cfg.dt);
            Feta = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)] + Lmap*Jeta;
            Fout = Lmap*Jall;
            PoutGlobal = blockDiagonalCovariance(Pout);
            Gpred = Feta*G(:,:,observer) + Fout;
            PcrossPred = Feta*PetaOut(:,:,observer) + Fout*PoutGlobal;
            Qphysical = Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap' + 1e-12*eye(4);
            Ppred = Feta*Peta(:,:,observer)*Feta' + ...
                Feta*PetaOut(:,:,observer)*Fout' + ...
                Fout*PetaOut(:,:,observer)'*Feta' + ...
                Fout*PoutGlobal*Fout' + Qphysical;
            Ppred = symmetrizePSD(Ppred,1e-12);
            [innovation,Heta,R] = bearingInnovation( ...
                data.bearings(observer,k+1),etaPred,cfg.watchers.r(:,observer),cfg);
            Sstate = Heta*Ppred*Heta' + R;
            Keta = (Ppred*Heta')/Sstate;
            etaNew = etaPred + Keta*innovation;
            Ieta = eye(4);
            Pnew = (Ieta-Keta*Heta)*Ppred*(Ieta-Keta*Heta)' + Keta*R*Keta';
            Gnew = (Ieta-Keta*Heta)*Gpred;
            PcrossNew = (Ieta-Keta*Heta)*PcrossPred;

            Hout = Heta*Gpred;
            % Ppred already contains Fout*Pout*Fout', so adding parameter
            % uncertainty here again would double count it.
            Sinfo = Sstate;
            if cfg.collaborative.packetEnabled(observer)
                if cfg.collaborative.packetAssignmentMode == "all_to_all"
                    ownersToUpdate = 1:Nb;
                else
                    ownersToUpdate = cfg.collaborative.observerOwner(observer);
                end
                for owner = ownersToUpdate
                    cols = (owner-1)*pLearn+(1:pLearn);
                    Hij = Hout(:,cols);
                    packetLambda(:,:,observer,owner) = ...
                        packetLambda(:,:,observer,owner) + (Hij'*Hij)/Sinfo;
                    packetq(:,observer,owner) = packetq(:,observer,owner) + ...
                        Hij'*(innovation/Sinfo);
                end
            end

            eta(:,observer) = etaNew;
            Peta(:,:,observer) = symmetrizePSD(Pnew,1e-12);
            G(:,:,observer) = Gnew;
            PetaOut(:,:,observer) = PcrossNew;
            etaHistory(:,k+1,observer) = etaNew;
            dHistory(:,k,observer) = dHat;
            nis(k+1,observer) = innovation^2/Sstate;
            if cfg.collaborative.packetEnabled(observer)
                nisWindowSum = nisWindowSum + nis(k+1,observer);
                nisWindowCount = nisWindowCount + 1;
            end
        end

        if t-lastSync >= cfg.communication.period-0.5*cfg.dt
            if useFOGM && nisWindowCount > 0
                matchedNIS = max(nisWindowSum/nisWindowCount,1e-3);
                gammaTheta = min(max(gammaTheta*matchedNIS^cfg.collaborative.covMatchRate, ...
                    cfg.collaborative.covMatchBounds(1)),cfg.collaborative.covMatchBounds(2));
            end
            PoutPrior = blockDiagonalCovariance(Pout);
            deltaGlobal = zeros(nLearn,1);
            for owner = 1:Nb
                Lambda = zeros(pLearn);
                q = zeros(pLearn,1);
                for observer = 1:Nw
                    Lambda = Lambda + packetLambda(:,:,observer,owner);
                    q = q + packetq(:,observer,owner);
                end
                informationPosterior = (Pout(:,:,owner) \ eye(pLearn)) + Lambda;
                Pnew = symmetrizePSD(informationPosterior \ eye(pLearn), ...
                    cfg.collaborative.parameterCovarianceFloor);
                delta = Pnew*q;
                deltaNorm = norm(delta);
                if deltaNorm > cfg.collaborative.infoStepLimit
                    delta = cfg.collaborative.infoStepLimit*delta/deltaNorm;
                end
                thetaOld = theta(learnIdx,owner);
                proposal = theta(learnIdx,owner) + delta;
                deviation = proposal-thetaReference(learnIdx,owner);
                deviation = max(min(deviation,cfg.dnn.parameterDeviationLimit), ...
                    -cfg.dnn.parameterDeviationLimit);
                theta(learnIdx,owner) = thetaReference(learnIdx,owner) + deviation;
                Pout(:,:,owner) = Pnew;
                cols = (owner-1)*pLearn+(1:pLearn);
                deltaGlobal(cols) = theta(learnIdx,owner) - thetaOld;
                thetaChange(k+1,owner) = norm(theta(learnIdx,owner) - ...
                    thetaReference(learnIdx,owner));
                uploadTimes{owner}(end+1) = t;
            end
            % Apply the delayed parameter posterior consistently to every
            % local state posterior.  This retains only 4-by-p cross blocks,
            % not a full global DNN covariance at any watcher.
            PoutPost = blockDiagonalCovariance(Pout);
            for observer = 1:Nw
                PcrossOld = PetaOut(:,:,observer);
                Greset = PcrossOld / PoutPrior;
                eta(:,observer) = eta(:,observer) + Greset*deltaGlobal;
                Peta(:,:,observer) = symmetrizePSD(Peta(:,:,observer) + ...
                    Greset*(PoutPost-PoutPrior)*Greset',1e-12);
                PetaOut(:,:,observer) = Greset*PoutPost;
                etaHistory(:,k+1,observer) = eta(:,observer);
            end
            packetLambda(:) = 0;
            packetq(:) = 0;
            nisWindowSum = 0; nisWindowCount = 0;
            lastSync = t;
        end
    end

    for observer = 1:Nw
        dHistory(:,N,observer) = evaluateGlobal( ...
            eta(:,observer),theta,cfg,0,"shared_block");
    end
    result.mode = resultMode;
    result.time = cfg.time;
    result.etaHat = etaHistory;
    result.dHat = dHistory;
    result.thetaChange = thetaChange;
    result.nis = nis;
    result.uploadTimes = uploadTimes;
    result.parameterUploads = sum(cellfun(@numel,uploadTimes));
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue);
    result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2));
    result.accelerationRMSE = sqrt(mean(result.accelerationError.^2));
    result.finalPositionRMSE = result.positionError(end);
    result.parameterCovariance = Pout;
    result.thetaQScale = gammaTheta;
    result.parameterBlockCount = Nb;
    result.packetAssignmentMode = cfg.collaborative.packetAssignmentMode;
    result.observerOwner = cfg.collaborative.observerOwner;
    result.packetEnabled = cfg.collaborative.packetEnabled;
end

function result = simulateHybridOwnerEKFJointGSCase(cfg,data,initial)
%SIMULATEHYBRIDOWNEREKFJOINTGSCASE Proposed two-timescale approximation.
%
% Each watcher runs a fast augmented EKF for [eta_i; theta_owner(i)].
% During the same 60 s window it accumulates an all-block, common-anchor
% Gauss-Newton information factor.  The GS then performs ONE full-joint
% all-layer normal-equation solve, including off-diagonal block Hessian
% terms.  This is deliberately a first real-time implementation: it uses
% local state copies and a single GS linearization, not a full IEKS sweep.

    Nw = cfg.Nw; Nb = cfg.Nblocks; N = cfg.N;
    p = cfg.dnn.arch.nTheta; nGlobal = p*Nb;
    owners = cfg.collaborative.observerOwner;
    thetaReference = initial.shared;
    thetaAnchor = initial.shared;
    thetaGS = thetaAnchor;
    Pglobal = cfg.ekf.Ptheta0*eye(nGlobal);

    eta = data.initialEta;
    thetaLocal = zeros(p,Nw);
    Plocal = zeros(4+p,4+p,Nw);
    G = zeros(4,nGlobal,Nw);
    for observer = 1:Nw
        thetaLocal(:,observer) = thetaGS(:,owners(observer));
        Plocal(:,:,observer) = blkdiag(cfg.ekf.Peta0, ...
            cfg.ekf.Ptheta0*eye(p));
    end

    packetLambda = zeros(nGlobal);
    packetq = zeros(nGlobal,1);
    gammaTheta = ones(1,Nb);
    lastSync = 0;
    nisWindowSum = 0; nisWindowCount = 0;
    uploadTimes = cell(1,Nb);

    etaHistory = zeros(4,N,Nw);
    dHistory = zeros(2,N,Nw);
    thetaChange = zeros(N,Nb);
    nis = zeros(N,Nw);
    gsLinearizedCostDecrease = zeros(1,N);
    etaHistory(:,1,:) = reshape(eta,4,1,Nw);

    for k = 1:N-1
        t = cfg.time(k);
        Pblocks = globalCovarianceBlocks(Pglobal,p,Nb);
        alpha = exp(-cfg.dt/cfg.collaborative.thetaTau);
        Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
        A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];

        for observer = 1:Nw
            owner = owners(observer);
            thetaOwnerPred = thetaAnchor(:,owner) + alpha*( ...
                thetaLocal(:,observer)-thetaAnchor(:,owner));
            thetaShadow = thetaAnchor;
            thetaShadow(:,owner) = thetaOwnerPred;
            [dHat,Jeta,Jall,parameterQd] = evaluateGlobalAllJacobians( ...
                eta(:,observer),thetaShadow,Pblocks,cfg);
            ownerCols = (owner-1)*p+(1:p);
            Jowner = Jall(:,ownerCols);
            etaPred = physicalStep(eta(:,observer),dHat,cfg.dt);
            Feta = A0 + Lmap*Jeta;

            Faug = eye(4+p);
            Faug(1:4,1:4) = Feta;
            Faug(1:4,5:end) = Lmap*Jowner*alpha;
            Faug(5:end,5:end) = alpha*eye(p);
            Qaug = zeros(4+p);
            Qaug(1:4,1:4) = Lmap*(cfg.ekf.qAcceleration*eye(2) + ...
                cfg.ekf.remoteCovarianceScale*parameterQd)*Lmap' + 1e-12*eye(4);
            Qaug(5:end,5:end) = gammaTheta(owner)*cfg.ekf.Ptheta0* ...
                (1-alpha^2)*eye(p);
            Ppred = symmetrizePSD(Faug*Plocal(:,:,observer)*Faug' + Qaug,1e-12);

            [innovation,Heta,R] = bearingInnovation( ...
                data.bearings(observer,k+1),etaPred,cfg.watchers.r(:,observer),cfg);
            Haug = [Heta zeros(1,p)];
            Sstate = Haug*Ppred*Haug' + R;
            Kaug = (Ppred*Haug')/Sstate;
            xNew = [etaPred;thetaOwnerPred] + Kaug*innovation;
            Iaug = eye(4+p);
            Pnew = (Iaug-Kaug*Haug)*Ppred*(Iaug-Kaug*Haug)' + Kaug*R*Kaug';

            % Global total sensitivity for the GS factor.  The local EKF
            % uses only owner parameters, while this factor retains every
            % block and hence its off-diagonal Hessian information.
            Gpred = Feta*G(:,:,observer) + Lmap*Jall;
            Keta = Kaug(1:4);
            Gnew = (eye(4)-Keta*Heta)*Gpred;
            Htheta = Heta*Gpred;
            shadowDelta = zeros(nGlobal,1);
            shadowDelta(ownerCols) = thetaOwnerPred-thetaAnchor(:,owner);
            innovationAnchor = innovation + Htheta*shadowDelta;
            packetLambda = packetLambda + (Htheta'*Htheta)/Sstate;
            packetq = packetq + Htheta'*(innovationAnchor/Sstate);

            eta(:,observer) = xNew(1:4);
            thetaLocal(:,observer) = xNew(5:end);
            Plocal(:,:,observer) = symmetrizePSD(Pnew,1e-12);
            G(:,:,observer) = Gnew;
            etaHistory(:,k+1,observer) = eta(:,observer);
            dHistory(:,k,observer) = dHat;
            nis(k+1,observer) = innovation^2/Sstate;
            nisWindowSum = nisWindowSum + nis(k+1,observer);
            nisWindowCount = nisWindowCount + 1;
        end

        if t-lastSync >= cfg.communication.period-0.5*cfg.dt
            elapsed = max(t-lastSync,cfg.dt);
            alphaWindow = exp(-elapsed/cfg.collaborative.thetaTau);
            thetaPrior = thetaReference + alphaWindow*(thetaGS-thetaReference);
            Qglobal = zeros(nGlobal);
            for block = 1:Nb
                cols = (block-1)*p+(1:p);
                Qglobal(cols,cols) = gammaTheta(block)*cfg.ekf.Ptheta0* ...
                    (1-alphaWindow^2)*eye(p);
            end
            Pprior = symmetrizePSD(alphaWindow^2*Pglobal + Qglobal,1e-12);
            priorInformation = Pprior\eye(nGlobal);
            anchorVector = thetaAnchor(:);
            priorShift = thetaPrior(:)-anchorVector;
            A = symmetrizePSD(priorInformation + packetLambda,1e-12);
            b = packetq + priorInformation*priorShift;
            delta = A\b;
            deltaNorm = norm(delta);
            if deltaNorm > cfg.collaborative.infoStepLimit
                delta = cfg.collaborative.infoStepLimit*delta/deltaNorm;
            end
            thetaProposal = reshape(anchorVector+delta,p,Nb);
            deviation = thetaProposal-thetaReference;
            deviation = max(min(deviation,cfg.dnn.parameterDeviationLimit), ...
                -cfg.dnn.parameterDeviationLimit);
            thetaGS = thetaReference+deviation;
            Pglobal = symmetrizePSD(A\eye(nGlobal), ...
                cfg.collaborative.parameterCovarianceFloor);
            % Actual decrease of J(delta)=0.5*delta'*A*delta-delta'*b
            % from delta=0.  This remains valid when the raw Newton step
            % is norm-clipped above; 0.5*delta'*b would not.
            gsLinearizedCostDecrease(k+1) = delta'*b - 0.5*delta'*A*delta;

            for observer = 1:Nw
                owner = owners(observer);
                thetaShadow = thetaAnchor;
                thetaShadow(:,owner) = thetaLocal(:,observer);
                resetDelta = thetaGS(:)-thetaShadow(:);
                eta(:,observer) = eta(:,observer) + G(:,:,observer)*resetDelta;
                thetaLocal(:,observer) = thetaGS(:,owner);
                Powner = Pglobal((owner-1)*p+(1:p),(owner-1)*p+(1:p));
                Plocal(5:end,5:end,observer) = Powner;
                Plocal(1:4,5:end,observer) = 0;
                Plocal(5:end,1:4,observer) = 0;
                Plocal(:,:,observer) = symmetrizePSD(Plocal(:,:,observer),1e-12);
                G(:,:,observer) = zeros(4,nGlobal);
                etaHistory(:,k+1,observer) = eta(:,observer);
            end
            thetaAnchor = thetaGS;
            for block = 1:Nb
                thetaChange(k+1,block) = norm(thetaGS(:,block)-thetaReference(:,block));
                uploadTimes{block}(end+1) = t;
            end
            if nisWindowCount > 0
                matchedNIS = max(nisWindowSum/(nisWindowCount* ...
                    (1+2*cfg.measurement.positionEnabled)),1e-3);
                gammaTheta = min(max(gammaTheta*matchedNIS^ ...
                    cfg.collaborative.covMatchRate,cfg.collaborative.covMatchBounds(1)), ...
                    cfg.collaborative.covMatchBounds(2));
            end
            packetLambda(:) = 0;
            packetq(:) = 0;
            nisWindowSum = 0; nisWindowCount = 0;
            lastSync = t;
        end
    end

    for observer = 1:Nw
        thetaShadow = thetaAnchor;
        thetaShadow(:,owners(observer)) = thetaLocal(:,observer);
        dHistory(:,N,observer) = evaluateGlobal( ...
            eta(:,observer),thetaShadow,cfg,0,"shared_block");
    end
    result.mode = "hybrid_owner_ekf_full_joint_gs_one_step";
    result.time = cfg.time;
    result.etaHat = etaHistory;
    result.dHat = dHistory;
    result.thetaChange = thetaChange;
    result.nis = nis;
    result.uploadTimes = uploadTimes;
    result.parameterUploads = sum(cellfun(@numel,uploadTimes));
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue);
    result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2));
    result.accelerationRMSE = sqrt(mean(result.accelerationError.^2));
    result.finalPositionRMSE = result.positionError(end);
    result.parameterCovariance = Pglobal;
    result.thetaQScale = gammaTheta;
    result.parameterBlockCount = Nb;
    result.observerOwner = owners;
    result.gsLinearizedCostDecrease = gsLinearizedCostDecrease;
    result.implementationNote = ["One-step joint GS Gauss-Newton approximation; " ...
        "local state-copy cross-covariances are not represented."];
end

function result = simulateCentralizedCommonDNNCase(cfg,data,initial)
%SIMULATECENTRALIZEDCOMMONDNNCASE Ideal synchronized sequential reference.
%
% At every filter step every watcher sends its bearing to the GS.  The GS
% maintains ONE target state and ONE complete additive-DNN parameter vector.
% Thus there are no watcher state copies, no delayed remote blocks, and no
% packet-factor approximation.  This is still an EKF (one linearization per
% step), not an offline/batch DNN fit.  Its full augmented covariance is
% intentionally retained, so it is practical only as a small-DNN reference.

    Nw = cfg.Nw; Nb = cfg.Nblocks; N = cfg.N;
    p = cfg.dnn.arch.nTheta; nTheta = p*Nb; nX = 4+nTheta;
    thetaReference = initial.shared;
    theta = thetaReference(:);
    % Do not average the individual local initial estimates here: that
    % would give the centralized reference a free, unmodelled initial-data
    % fusion advantage.  Start from the same single-watcher prior used by
    % the local filters; all subsequent benefit comes from joint bearings.
    eta = data.initialEta(:,1);
    P = blkdiag(cfg.ekf.Peta0,cfg.ekf.Ptheta0*eye(nTheta));
    gammaTheta = ones(1,Nb); gammaEta = 1;
    alpha = exp(-cfg.dt/cfg.collaborative.thetaTau);
    A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];

    etaHistory = zeros(4,N,Nw);
    dHistory = zeros(2,N,Nw);
    nis = zeros(N,Nw);
    thetaChange = zeros(N,Nb);
    etaHistory(:,1,:) = repmat(eta,1,1,Nw);
    thetaHistoryNorm = zeros(1,N);
    % One raw bearing packet per watcher is sent to GS at each update.
    uploadTimes = cell(1,Nb);

    for k = 1:N-1
        thetaBlocks = reshape(theta,p,Nb);
        Ptheta = P(5:end,5:end);
        Pblocks = globalCovarianceBlocks(Ptheta,p,Nb);
        [dHat,Jeta,Jall] = evaluateGlobalAllJacobians(eta,thetaBlocks,Pblocks,cfg);

        etaPred = physicalStep(eta,dHat,cfg.dt);
        Feta = A0 + Lmap*Jeta;
        Btheta = Lmap*Jall*alpha;
        % Exploit the augmented transition structure instead of forming a
        % dense (4+nTheta)-square FPF' product at every 0.1 s step.
        Pee = P(1:4,1:4); PeT = P(1:4,5:end); PTT = P(5:end,5:end);
        PeePred = Feta*Pee*Feta' + Feta*PeT*Btheta' + ...
            Btheta*PeT'*Feta' + Btheta*PTT*Btheta';
        PeTPred = alpha*(Feta*PeT + Btheta*PTT);
        Qtheta = zeros(nTheta);
        for block = 1:Nb
            cols = (block-1)*p+(1:p);
            Qtheta(cols,cols) = gammaTheta(block)*cfg.ekf.Ptheta0* ...
                (1-alpha^2)*eye(p);
        end
        Qeta = gammaEta*Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap' + ...
            1e-12*eye(4);
        PPred = [PeePred+Qeta, PeTPred; PeTPred', alpha^2*PTT+Qtheta];
        PPred = symmetrizePSD(PPred,1e-12);
        thetaPred = thetaReference(:) + alpha*(theta-thetaReference(:));
        x = [etaPred;thetaPred];

        stepNIS = zeros(1,Nw);
        for observer = 1:Nw
            [innovation,Heta,R] = bearingInnovation( ...
                data.bearings(observer,k+1),x(1:4), ...
                cfg.watchers.r(:,observer),cfg);
            H = [Heta zeros(1,nTheta)];
            S = H*PPred*H' + R;
            K = (PPred*H')/S;
            x = x + K*innovation;
            I = eye(nX);
            PPred = (I-K*H)*PPred*(I-K*H)' + K*R*K';
            PPred = symmetrizePSD(PPred,1e-12);
            stepNIS(observer) = innovation^2/max(S,eps);
        end

        % The existing parameter trust region is retained only as a
        % numerical safeguard; the reference is otherwise a full joint EKF.
        rawTheta = x(5:end);
        step = rawTheta-thetaPred;
        if norm(step) > cfg.dnn.parameterStepLimit
            rawTheta = thetaPred + cfg.dnn.parameterStepLimit*step/norm(step);
        end
        deviation = max(min(rawTheta-thetaReference(:), ...
            cfg.dnn.parameterDeviationLimit),-cfg.dnn.parameterDeviationLimit);
        theta = thetaReference(:)+deviation;
        eta = x(1:4);
        P = PPred;
        nis(k+1,:) = stepNIS;
        matchedNIS = max(mean(stepNIS),1e-3);
        gammaTheta = min(max(gammaTheta*matchedNIS^cfg.collaborative.covMatchRate, ...
            cfg.collaborative.covMatchBounds(1)),cfg.collaborative.covMatchBounds(2));
        gammaEta = min(max(gammaEta*matchedNIS^cfg.collaborative.stateCovMatchRate, ...
            cfg.collaborative.stateCovMatchBounds(1)),cfg.collaborative.stateCovMatchBounds(2));

        thetaBlocks = reshape(theta,p,Nb);
        dNow = evaluateGlobal(eta,thetaBlocks,cfg,1,"shared_block");
        etaHistory(:,k+1,:) = repmat(eta,1,1,Nw);
        dHistory(:,k,:) = repmat(dHat,1,1,Nw);
        for block = 1:Nb
            thetaChange(k+1,block) = norm(thetaBlocks(:,block)-thetaReference(:,block));
            uploadTimes{block}(end+1) = cfg.time(k);
        end
        thetaHistoryNorm(k+1) = norm(theta-thetaReference(:));
        if k == N-1
            dHistory(:,N,:) = repmat(dNow,1,1,Nw);
        end
    end

    result.mode = "centralized_common_state_common_dnn_step_ekf";
    result.time = cfg.time;
    result.etaHat = etaHistory;
    result.dHat = dHistory;
    result.thetaChange = thetaChange;
    result.thetaHistoryNorm = thetaHistoryNorm;
    result.nis = nis;
    result.uploadTimes = uploadTimes;
    result.parameterUploads = sum(cellfun(@numel,uploadTimes));
    result.measurementPacketsToGS = Nw*(N-1);
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue);
    result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2));
    result.accelerationRMSE = sqrt(mean(result.accelerationError.^2));
    result.finalPositionRMSE = result.positionError(end);
    result.parameterCovariance = P(5:end,5:end);
    result.thetaQScale = gammaTheta;
    result.stateQScale = gammaEta;
    result.nisDegreesOfFreedom = 1;
    result.implementationNote = ["Ideal every-step GS reference: all watcher bearings update one " ...
        "common state and one common all-layer DNN; full augmented covariance retained."];
end

function result = simulateCentralizedPeriodicCommonDNNCase(cfg,data,initial)
%SIMULATECENTRALIZEDPERIODICCOMMONDNNCASE Delayed raw-data GS reference.
%
% Watchers still sample bearings every dt, but transmit their buffered raw
% bearings only every cfg.communication.centralizedPeriod.  At receipt the
% GS rewinds to its last synchronized posterior and replays the complete
% window in temporal order.  Thus no sensor information is discarded; this
% isolates the real-time penalty of delayed communication/canonical-model
% updates from a reduction in sensor sample rate.

    Nw = cfg.Nw; Nb = cfg.Nblocks; N = cfg.N;
    p = cfg.dnn.arch.nTheta; nTheta = p*Nb; nX = 4+nTheta;
    thetaReference = initial.shared(:);
    xAnchor = [data.initialEta(:,1);thetaReference];
    PAnchor = blkdiag(cfg.ekf.Peta0,cfg.ekf.Ptheta0*eye(nTheta));
    xOperational = xAnchor;
    gammaTheta = ones(1,Nb); gammaEta = 1;
    alpha = exp(-cfg.dt/cfg.collaborative.thetaTau);
    A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    period = max(cfg.communication.centralizedPeriod,cfg.dt);
    lastSyncTime = 0; windowStartIndex = 1;

    etaHistory = zeros(4,N,Nw); dHistory = zeros(2,N,Nw);
    nis = zeros(N,Nw); thetaChange = zeros(N,Nb);
    etaHistory(:,1,:) = repmat(xOperational(1:4),1,1,Nw);
    uploadTimes = cell(1,Nb); syncTimes = [];

    for k = 1:N-1
        % Real-time trajectory available between communications: propagate
        % the latest canonical GS posterior but do not consume new bearings.
        thetaOp = thetaReference + alpha*(xOperational(5:end)-thetaReference);
        blocksOp = reshape(thetaOp,p,Nb);
        [dOp,~,~,~] = evaluateGlobalAllJacobians(xOperational(1:4), ...
            blocksOp,globalCovarianceBlocks(PAnchor(5:end,5:end),p,Nb),cfg);
        xOperational = [physicalStep(xOperational(1:4),dOp,cfg.dt);thetaOp];

        now = cfg.time(k+1);
        shouldSync = now-lastSyncTime >= period-0.5*cfg.dt || k == N-1;
        if shouldSync
            xReplay = xAnchor; PReplay = PAnchor;
            % The GS now receives all raw bearing packets in this window.
            for replayK = windowStartIndex:k
                theta = thetaReference + alpha*(xReplay(5:end)-thetaReference);
                blocks = reshape(theta,p,Nb);
                Pblocks = globalCovarianceBlocks(PReplay(5:end,5:end),p,Nb);
                [dHat,Jeta,Jall] = evaluateGlobalAllJacobians( ...
                    xReplay(1:4),blocks,Pblocks,cfg);
                Feta = A0 + Lmap*Jeta; Btheta = Lmap*Jall*alpha;
                Pee = PReplay(1:4,1:4); PeT = PReplay(1:4,5:end); PTT = PReplay(5:end,5:end);
                PeePred = Feta*Pee*Feta' + Feta*PeT*Btheta' + ...
                    Btheta*PeT'*Feta' + Btheta*PTT*Btheta';
                PeTPred = alpha*(Feta*PeT + Btheta*PTT);
                Qtheta = zeros(nTheta);
                for block = 1:Nb
                    cols = (block-1)*p+(1:p);
                    Qtheta(cols,cols) = gammaTheta(block)*cfg.ekf.Ptheta0* ...
                        (1-alpha^2)*eye(p);
                end
                Qeta = gammaEta*Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap' + 1e-12*eye(4);
                PReplay = symmetrizePSD([PeePred+Qeta,PeTPred;PeTPred',alpha^2*PTT+Qtheta],1e-12);
                xReplay = [physicalStep(xReplay(1:4),dHat,cfg.dt);theta];
                stepNIS = zeros(1,Nw);
                for observer = 1:Nw
                    [innovation,Heta,R] = bearingInnovation(data.bearings(observer,replayK+1), ...
                        xReplay(1:4),cfg.watchers.r(:,observer),cfg);
                    H = [Heta zeros(1,nTheta)]; S = H*PReplay*H' + R;
                    K = (PReplay*H')/S; xReplay = xReplay + K*innovation;
                    I = eye(nX);
                    PReplay = symmetrizePSD((I-K*H)*PReplay*(I-K*H)' + K*R*K',1e-12);
                    stepNIS(observer) = innovation^2/max(S,eps);
                end
                if cfg.collaborative.enableParameterClipping
                    thetaPred = xReplay(5:end);
                    step = thetaPred-theta;
                    if norm(step) > cfg.dnn.parameterStepLimit
                        thetaPred = theta + cfg.dnn.parameterStepLimit*step/norm(step);
                    end
                    xReplay(5:end) = thetaReference + max(min(thetaPred-thetaReference, ...
                        cfg.dnn.parameterDeviationLimit),-cfg.dnn.parameterDeviationLimit);
                end
                nis(replayK+1,:) = stepNIS;
                if cfg.collaborative.enableCovarianceMatching
                    matchedNIS = max(mean(stepNIS),1e-3);
                    gammaTheta = min(max(gammaTheta*matchedNIS^cfg.collaborative.covMatchRate, ...
                        cfg.collaborative.covMatchBounds(1)),cfg.collaborative.covMatchBounds(2));
                    gammaEta = min(max(gammaEta*matchedNIS^cfg.collaborative.stateCovMatchRate, ...
                        cfg.collaborative.stateCovMatchBounds(1)),cfg.collaborative.stateCovMatchBounds(2));
                end
            end
            xAnchor = xReplay; PAnchor = PReplay; xOperational = xReplay;
            windowStartIndex = k+1; lastSyncTime = now; syncTimes(end+1) = now;
            for block = 1:Nb, uploadTimes{block}(end+1) = now; end
        end
        blocksOp = reshape(xOperational(5:end),p,Nb);
        dNow = evaluateGlobal(xOperational(1:4),blocksOp,cfg,1,"shared_block");
        etaHistory(:,k+1,:) = repmat(xOperational(1:4),1,1,Nw);
        dHistory(:,k,:) = repmat(dNow,1,1,Nw);
        for block = 1:Nb
            thetaChange(k+1,block) = norm(blocksOp(:,block)-thetaReference((block-1)*p+(1:p)));
        end
    end
    dHistory(:,N,:) = dHistory(:,N-1,:);
    result.mode = "centralized_common_state_common_dnn_periodic_replay_ekf";
    result.time = cfg.time; result.etaHat = etaHistory; result.dHat = dHistory;
    result.thetaChange = thetaChange; result.nis = nis; result.uploadTimes = uploadTimes;
    result.parameterUploads = sum(cellfun(@numel,uploadTimes));
    result.measurementPacketsToGS = Nw*(N-1);
    result.syncTimes = syncTimes;
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue);
    result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2));
    result.accelerationRMSE = sqrt(mean(result.accelerationError.^2));
    result.finalPositionRMSE = result.positionError(end);
    result.parameterCovariance = PAnchor(5:end,5:end);
    result.thetaQScale = gammaTheta; result.stateQScale = gammaEta;
    result.nisDegreesOfFreedom = 1;
    result.implementationNote = ["Raw 10 Hz bearings buffered and replayed at periodic GS sync; " ...
        "displayed state is the real-time hold/prediction between sync instants."];
end

function result = simulateCentralizedFactorPacketCase(cfg,data,initial)
%SIMULATECENTRALIZEDFACTORPACKETCASE Common-state 60 s MAP packet test.
%
% Each watcher replaces its raw bearing sequence by its additive normal
% equation contribution C_i,q_i over a common, GS-broadcast reference
% trajectory.  The GS adds those packets BEFORE eliminating the shared
% window trajectory, then performs one damped Gauss-Newton correction for
% the all-layer parameter vector.  This is the correct common-state order:
% sum local factors first; never Schur-complement each watcher state copy.
%
% The parameter is represented by its window-start FOGM coordinate.  Its
% deterministic mean decay within a 60 s window is retained; theta process
% noise inside the window is intentionally omitted in this first packet
% benchmark and remains a documented approximation.

    Nw = cfg.Nw; Nb = cfg.Nblocks; N = cfg.N;
    p = cfg.dnn.arch.nTheta; nTheta = p*Nb;
    thetaRef = initial.shared(:);
    etaAnchor = data.initialEta(:,1);
    thetaAnchor = thetaRef;
    Panchor = blkdiag(cfg.ekf.Peta0,cfg.ekf.Ptheta0*eye(nTheta));
    alpha = exp(-cfg.dt/cfg.collaborative.thetaTau);
    A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    periodSteps = max(1,round(cfg.communication.period/cfg.dt));
    gammaEta = 1;

    etaOperational = etaAnchor; thetaOperational = thetaAnchor;
    etaHistory = zeros(4,N,Nw); dHistory = zeros(2,N,Nw);
    nis = nan(N,Nw); thetaChange = zeros(N,Nb);
    etaHistory(:,1,:) = repmat(etaOperational,1,1,Nw);
    uploadTimes = cell(1,Nb); syncTimes = []; packetDimensions = [];
    windowStart = 1;

    for k = 1:N-1
        thetaOperational = thetaRef + alpha*(thetaOperational-thetaRef);
        bOp = reshape(thetaOperational,p,Nb);
        [dOp,~,~,~] = evaluateGlobalAllJacobians(etaOperational,bOp, ...
            globalCovarianceBlocks(Panchor(5:end,5:end),p,Nb),cfg);
        etaOperational = physicalStep(etaOperational,dOp,cfg.dt);

        atSync = mod(k,periodSteps) == 0 || k == N-1;
        if atSync
            L = k-windowStart+1; nState = 4*(L+1); nVar = nState+nTheta;
            stateIndex = @(node) (4*(node-1)+(1:4));
            thetaIndex = nState+(1:nTheta);
            % Common reference state/FOGM mean trajectory for this window.
            % The iterated mode relinearizes this same packet problem four
            % times; the one-shot mode retains the original single pass.
            xbar = zeros(4,L+1); xbar(:,1) = etaAnchor;
            thetaStart = thetaAnchor;
            thetaBar = zeros(nTheta,L);
            for ell = 1:L
                thetaBar(:,ell) = thetaRef + alpha^(ell-1)*(thetaStart-thetaRef);
                blocks = reshape(thetaBar(:,ell),p,Nb);
                d = evaluateGlobal(xbar(:,ell),blocks,cfg,1,"shared_block");
                xbar(:,ell+1) = physicalStep(xbar(:,ell),d,cfg.dt);
            end
            nGN = 1 + 3*(contains(cfg.communication.mode,"iterated_factor_packet") || ...
                contains(cfg.communication.mode,"damped_factor_packet"));
            for gn = 1:nGN
            A = sparse(nVar,nVar); b = zeros(nVar,1);
            Jstack = sparse(0,nVar); ystack = zeros(0,1);
            % Window-start joint prior retains eta--theta cross covariance.
            Jprior = [stateIndex(1),thetaIndex];
            Wprior = Panchor\eye(4+nTheta);
            A(Jprior,Jprior) = A(Jprior,Jprior) + Wprior;
            b(Jprior) = b(Jprior) - Wprior*([xbar(:,1);thetaStart]-[etaAnchor;thetaAnchor]);
            if contains(cfg.communication.mode,"damped_factor_packet")
                Lprior = chol(symmetrizePSD(Panchor,1e-12),'lower');
                Jp = sparse(4+nTheta,nVar); Jp(:,Jprior) = Lprior\eye(4+nTheta);
                Jstack = [Jstack;Jp];
                ystack = [ystack;-Lprior\([xbar(:,1);thetaStart]-[etaAnchor;thetaAnchor])];
            end

            % GS-owned dynamics factors couple all DNN blocks to the one
            % shared trajectory.  They are not duplicated per watcher.
            for ell = 1:L
                blocks = reshape(thetaBar(:,ell),p,Nb);
                [~,Jeta,Jall,~] = evaluateGlobalAllJacobians( ...
                    xbar(:,ell),blocks,globalCovarianceBlocks( ...
                    Panchor(5:end,5:end),p,Nb),cfg);
                Feta = A0 + Lmap*Jeta;
                Jtheta = -Lmap*Jall*alpha^(ell-1);
                J = zeros(4,nVar);
                J(:,stateIndex(ell)) = -Feta;
                J(:,stateIndex(ell+1)) = eye(4);
                J(:,thetaIndex) = Jtheta;
                Q = gammaEta*Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap' + 1e-12*eye(4);
                A = A + sparse(J')*(Q\J);
                rDyn = xbar(:,ell+1)-physicalStep(xbar(:,ell), ...
                    evaluateGlobal(xbar(:,ell),blocks,cfg,1,"shared_block"),cfg.dt);
                b = b - sparse(J')*(Q\rDyn);
                if contains(cfg.communication.mode,"damped_factor_packet")
                    Lq = chol(Q,'lower');
                    Jstack = [Jstack;sparse(Lq\J)];
                    ystack = [ystack;-Lq\rDyn];
                end
            end

            % Watcher packets: each packet is only C_i,q_i.  Bearing has no
            % direct theta Jacobian; its theta information enters through
            % the common GS dynamics factors above.
            for observer = 1:Nw
                Cpacket = sparse(nState,nState); qpacket = zeros(nState,1);
                for ell = 1:L
                    sample = windowStart+ell;
                    [innovation,Heta,R] = bearingInnovation(data.bearings(observer,sample), ...
                        xbar(:,ell+1),cfg.watchers.r(:,observer),cfg);
                    ids = stateIndex(ell+1);
                    Cpacket(ids,ids) = Cpacket(ids,ids) + (Heta'*Heta)/R;
                    qpacket(ids) = qpacket(ids) + Heta'*innovation/R;
                    if contains(cfg.communication.mode,"damped_factor_packet")
                        Jm = sparse(1,nVar); Jm(1,ids) = Heta/sqrt(R);
                        Jstack = [Jstack;Jm];
                        ystack = [ystack;innovation/sqrt(R)];
                    end
                end
                A(1:nState,1:nState) = A(1:nState,1:nState) + Cpacket;
                b(1:nState) = b(1:nState) + qpacket;
            end

            % Damped GN solve; in iterated mode watchers re-form C_i,q_i at
            % the newly broadcast common reference on the next inner pass.
            damping = 1e-8*max(1,mean(full(diag(A))));
            A = A + damping*speye(nVar);
            if contains(cfg.communication.mode,"damped_factor_packet")
                % Square-root form: sparse QR/least-squares acts directly
                % on whitened residual factors and avoids forming J'J.
                delta = [Jstack;sqrt(damping)*speye(nVar)]\ ...
                    [ystack;zeros(nVar,1)];
            else
                delta = A\b;
            end
            deltaTheta = delta(thetaIndex);
            stepScale = 1;
            for block = 1:Nb
                cols = (block-1)*p+(1:p);
                stepScale = min(stepScale,cfg.dnn.parameterStepLimit/max(norm(deltaTheta(cols)),eps));
            end
            stepScale = min(stepScale,1);
            if contains(cfg.communication.mode,"damped_factor_packet")
                baseCost = factorPacketWindowCost(xbar,thetaStart,etaAnchor,thetaAnchor, ...
                    Panchor,windowStart,L,data,cfg,thetaRef,alpha);
                accepted = false;
                for ls = [1 0.5 0.25 0.125 0.0625]
                    trialScale = stepScale*ls;
                    xTrial = xbar;
                    for node = 1:L+1, xTrial(:,node) = xTrial(:,node)+trialScale*delta(stateIndex(node)); end
                    thetaTrial = thetaStart+trialScale*deltaTheta;
                    if factorPacketWindowCost(xTrial,thetaTrial,etaAnchor,thetaAnchor, ...
                            Panchor,windowStart,L,data,cfg,thetaRef,alpha) < baseCost
                        xbar = xTrial; thetaStart = thetaTrial; accepted = true; break;
                    end
                end
                if ~accepted, break; end
            else
                for node = 1:L+1, xbar(:,node) = xbar(:,node)+stepScale*delta(stateIndex(node)); end
                thetaStart = thetaStart + stepScale*deltaTheta;
            end
            thetaStart = thetaRef + max(min(thetaStart-thetaRef, ...
                cfg.dnn.parameterDeviationLimit),-cfg.dnn.parameterDeviationLimit);
            thetaBar = zeros(nTheta,L);
            for ell = 1:L
                thetaBar(:,ell) = thetaRef + alpha^(ell-1)*(thetaStart-thetaRef);
            end
            end
            thetaAnchor = thetaStart;
            etaAnchor = xbar(:,end);

            % Retain the marginal covariance needed as the next window
            % prior without materializing the full inverse.
            keep = [stateIndex(L+1),thetaIndex];
            selector = sparse(keep,1:numel(keep),1,nVar,numel(keep));
            Panchor = full(selector'*(A\selector));
            Panchor = symmetrizePSD(Panchor,1e-12);
            etaOperational = etaAnchor; thetaOperational = thetaAnchor;
            windowStart = k+1; syncTimes(end+1) = cfg.time(k+1);
            packetDimensions(end+1,:) = [nState,nTheta];
            for block = 1:Nb, uploadTimes{block}(end+1) = cfg.time(k+1); end
        end

        blocksOp = reshape(thetaOperational,p,Nb);
        dNow = evaluateGlobal(etaOperational,blocksOp,cfg,1,"shared_block");
        etaHistory(:,k+1,:) = repmat(etaOperational,1,1,Nw);
        dHistory(:,k,:) = repmat(dNow,1,1,Nw);
        for block = 1:Nb
            thetaChange(k+1,block) = norm(blocksOp(:,block)-thetaRef((block-1)*p+(1:p)));
        end
    end
    dHistory(:,N,:) = dHistory(:,N-1,:);
    result.mode = "common_state_all_layer_factor_packet_gs_60s";
    result.time = cfg.time; result.etaHat = etaHistory; result.dHat = dHistory;
    result.thetaChange = thetaChange; result.nis = nis; result.uploadTimes = uploadTimes;
    result.parameterUploads = sum(cellfun(@numel,uploadTimes));
    result.syncTimes = syncTimes; result.packetStateThetaDimensions = packetDimensions;
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue);
    result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2));
    result.accelerationRMSE = sqrt(mean(result.accelerationError.^2));
    result.finalPositionRMSE = result.positionError(end);
    result.parameterCovariance = Panchor(5:end,5:end);
    result.nisDegreesOfFreedom = 1;
    result.implementationNote = ["Common-state linearized 60 s factor packet MAP. Watchers send " ...
        "bearing C_i,q_i packets; GS adds them before common-trajectory elimination. " ...
        "One GN iteration and deterministic within-window FOGM are used."];
end

function result = simulateCentralizedIteratedFactorPacketCase(cfg,data,initial)
% The shared implementation detects the iterated communication mode and
% performs four relinearize--packet--solve inner passes per GS window.
    result = simulateCentralizedFactorPacketCase(cfg,data,initial);
    result.mode = "common_state_iterated_all_layer_factor_packet_gs_60s";
    result.implementationNote = ["Four-pass common-state iterated factor packet MAP; " ...
        "watchers relinearize C_i,q_i after each GS reference broadcast. " ...
        "Within-window FOGM process noise is still omitted."];
end

function result = simulateWindowStaticThetaFactorGraphCase(cfg,data,initial)
% Exact IEKS/MAP benchmark for the revised multi-rate model: eta is a
% 10-Hz trajectory while one DNN parameter vector is shared by the whole
% communication window.  This first benchmark intentionally supports one
% window only; it isolates the modelling change from inter-window priors.
    L = cfg.N-1; assert(abs(cfg.communication.period-L*cfg.dt) < 0.5*cfg.dt, ...
        'centralized_window_static_dnn_fg_5s currently requires one complete window.');
    Nw = cfg.Nw; Nb = cfg.Nblocks; p = cfg.dnn.arch.nTheta; nTheta = p*Nb;
    nEta = 4*(L+1); nVar = nEta+nTheta;
    thetaRef = initial.shared(:); theta = thetaRef;
    priorEta = data.initialEta(:,1); priorCovEta = cfg.ekf.Peta0;
    priorCovTheta = cfg.ekf.Ptheta0*eye(nTheta);
    alpha = exp(-cfg.dt/cfg.collaborative.thetaTau);
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    qEta = Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap'+1e-12*eye(4);
    Yseed = forwardReplayTrajectoryInitializer([priorEta;theta], ...
        blkdiag(priorCovEta,priorCovTheta),1,L,data,cfg,thetaRef,alpha);
    eta = Yseed(1:4,:);
    maxGN = cfg.factorGraph.maxIterations; costs = nan(1,maxGN); accepted = false(1,maxGN);
    lmDamping = 1e-8;
    for gn = 1:maxGN
        [J,rhs] = buildWindowStaticThetaSquareRootSystem(eta,theta,priorEta,thetaRef, ...
            priorCovEta,priorCovTheta,1,L,data,cfg,qEta);
        baseCost = windowStaticThetaCost(eta,theta,priorEta,thetaRef, ...
            priorCovEta,priorCovTheta,1,L,data,cfg,qEta);
        diagonalScale = max(1,full(mean(sum(J.^2,1)))); acceptedThis = false;
        for lmAttempt = 1:5
            damping = lmDamping*diagonalScale;
            delta = [J;sqrt(damping)*speye(nVar)]\[rhs;zeros(nVar,1)];
            for scale = [1 0.5 0.25 0.125 0.0625]
                candidateEta = eta+scale*reshape(delta(1:nEta),4,L+1);
                candidateTheta = theta+scale*delta(nEta+(1:nTheta));
                deviation = max(min(candidateTheta-thetaRef,cfg.dnn.parameterDeviationLimit), ...
                    -cfg.dnn.parameterDeviationLimit);
                candidateTheta = thetaRef+deviation;
                candidateCost = windowStaticThetaCost(candidateEta,candidateTheta,priorEta,thetaRef, ...
                    priorCovEta,priorCovTheta,1,L,data,cfg,qEta);
                if isfinite(candidateCost) && candidateCost < baseCost
                    eta = candidateEta; theta = candidateTheta; costs(gn) = candidateCost;
                    accepted(gn) = true; acceptedThis = true; lmDamping = max(lmDamping/3,1e-12); break;
                end
            end
            if acceptedThis, break; end
            lmDamping = min(lmDamping*10,1e8);
        end
        if ~acceptedThis, break; end
        if nnz(accepted) >= 2
            ac = costs(accepted); relativeDecrease = (ac(end-1)-ac(end))/max(abs(ac(end-1)),1);
            if relativeDecrease < cfg.factorGraph.relativeCostTolerance, break; end
        end
    end
    blocks = reshape(theta,p,Nb); dStatic = zeros(2,L+1);
    for ell = 1:L
        dStatic(:,ell) = collectFullTrajectoryDNNPackets(eta(:,ell),blocks,cfg);
    end
    dStatic(:,end) = collectFullTrajectoryDNNPackets(eta(:,end),blocks,cfg);
    etaHistory = repmat(reshape(eta,4,L+1,1),1,1,Nw);
    dHistory = repmat(reshape(dStatic,2,L+1,1),1,1,Nw);
    result.mode = "centralized_window_static_theta_square_root_ieks";
    result.time = cfg.time; result.etaHat = etaHistory; result.dHat = dHistory;
    result.thetaChange = repmat(norm(reshape(theta,p,Nb)-reshape(thetaRef,p,Nb),'fro'),cfg.N, Nb);
    result.nis = nan(cfg.N,Nw); result.uploadTimes = cell(1,Nb); result.parameterUploads = 0;
    result.syncTimes = cfg.time(end); result.acceptedCosts = {costs(accepted)};
    result.windowVariableCount = nVar; result.windowStaticTheta = true;
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue);
    result.positionRMSE = sqrt(mean(result.positionError.^2)); result.velocityRMSE = sqrt(mean(result.velocityError.^2));
    result.accelerationRMSE = sqrt(mean(result.accelerationError.^2)); result.finalPositionRMSE = result.positionError(end);
    result.staticThetaEstimate = theta; result.implementationNote = [ ...
        "One theta vector is shared by the full window; all 10-Hz dynamics and bearings jointly update it."];
end

function [J,rhs] = buildWindowStaticThetaSquareRootSystem(eta,theta,priorEta,thetaRef,priorCovEta,priorCovTheta,windowStart,L,data,cfg,qEta)
    nTheta = numel(theta); nEta = 4*(L+1); nVar = nEta+nTheta;
    etaIds = @(node) (node-1)*4+(1:4); thetaIds = nEta+(1:nTheta);
    LpEta = chol(symmetrizePSD(priorCovEta,1e-12),'lower');
    LpTheta = chol(symmetrizePSD(priorCovTheta,1e-12),'lower');
    J = sparse(4+nTheta,nVar); J(1:4,etaIds(1)) = LpEta\eye(4); J(5:end,thetaIds) = LpTheta\eye(nTheta);
    rhs = [-LpEta\(eta(:,1)-priorEta);-LpTheta\(theta-thetaRef)];
    LqEta = chol(qEta,'lower'); A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)]; blocks = reshape(theta,cfg.dnn.arch.nTheta,cfg.Nblocks);
    for ell = 1:L
        [d,Jeta,Jall] = collectFullTrajectoryDNNPackets(eta(:,ell),blocks,cfg);
        row = sparse(4,nVar); row(:,etaIds(ell)) = LqEta\(-A0-Lmap*Jeta);
        row(:,etaIds(ell+1)) = LqEta\eye(4); row(:,thetaIds) = LqEta\(-Lmap*Jall);
        J = [J;row]; rhs = [rhs;-LqEta\(eta(:,ell+1)-physicalStep(eta(:,ell),d,cfg.dt))];
        for watcher = 1:cfg.Nw
            [nu,H,R] = makeWatcherBearingFactorPacket(data,watcher,windowStart+ell,eta(:,ell+1),cfg);
            row = sparse(1,nVar); row(etaIds(ell+1)) = H/sqrt(R); J = [J;row]; rhs = [rhs;nu/sqrt(R)];
        end
    end
end

function cost = windowStaticThetaCost(eta,theta,priorEta,thetaRef,priorCovEta,priorCovTheta,windowStart,L,data,cfg,qEta)
    cost = 0.5*(eta(:,1)-priorEta)'*(priorCovEta\(eta(:,1)-priorEta));
    cost = cost+0.5*(theta-thetaRef)'*(priorCovTheta\(theta-thetaRef));
    blocks = reshape(theta,cfg.dnn.arch.nTheta,cfg.Nblocks);
    for ell = 1:L
        d = collectFullTrajectoryDNNPackets(eta(:,ell),blocks,cfg);
        residual = eta(:,ell+1)-physicalStep(eta(:,ell),d,cfg.dt);
        cost = cost+0.5*residual'*(qEta\residual);
        for watcher = 1:cfg.Nw
            [nu,~,R] = makeWatcherBearingFactorPacket(data,watcher,windowStart+ell,eta(:,ell+1),cfg);
            cost = cost+0.5*nu^2/R;
        end
    end
end

function result = simulateMultiRateThetaFactorGraphCase(cfg,data,initial)
% Exact MAP for eta at dt and theta knots at cfg.factorGraph.thetaKnotInterval.
    L = cfg.N-1; assert(abs(cfg.communication.period-L*cfg.dt) < 0.5*cfg.dt, ...
        'centralized_multirate_dnn_fg_5s currently requires one complete window.');
    knotSteps = round(cfg.factorGraph.thetaKnotInterval/cfg.dt);
    assert(knotSteps >= 1,'thetaKnotInterval must be at least dt.');
    knotOfStep = ceil((1:L)/knotSteps); K = max(knotOfStep);
    Nw = cfg.Nw; Nb = cfg.Nblocks; p = cfg.dnn.arch.nTheta; nTheta = p*Nb;
    nEta = 4*(L+1); nVar = nEta+K*nTheta; thetaRef = initial.shared(:);
    priorEta = data.initialEta(:,1); priorCovEta = cfg.ekf.Peta0; priorCovTheta = cfg.ekf.Ptheta0*eye(nTheta);
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)]; qEta = Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap'+1e-12*eye(4);
    alphaK = exp(-knotSteps*cfg.dt/cfg.collaborative.thetaTau);
    qThetaK = cfg.ekf.Ptheta0*(1-alphaK^2)*eye(nTheta);
    Yseed = forwardReplayTrajectoryInitializer([priorEta;thetaRef],blkdiag(priorCovEta,priorCovTheta), ...
        1,L,data,cfg,thetaRef,exp(-cfg.dt/cfg.collaborative.thetaTau));
    eta = Yseed(1:4,:); thetaKnots = zeros(nTheta,K);
    for q = 1:K
        step = min((q-1)*knotSteps+1,L+1);
        thetaKnots(:,q) = Yseed(5:end,step);
    end
    maxGN = cfg.factorGraph.maxIterations; costs = nan(1,maxGN); accepted = false(1,maxGN); lmDamping = 1e-8;
    for gn = 1:maxGN
        [J,rhs] = buildMultiRateThetaSquareRootSystem(eta,thetaKnots,knotOfStep,priorEta,thetaRef, ...
            priorCovEta,priorCovTheta,1,L,data,cfg,qEta,qThetaK,alphaK);
        baseCost = multiRateThetaCost(eta,thetaKnots,knotOfStep,priorEta,thetaRef, ...
            priorCovEta,priorCovTheta,1,L,data,cfg,qEta,qThetaK,alphaK);
        diagonalScale = max(1,full(mean(sum(J.^2,1)))); acceptedThis = false;
        for lmAttempt = 1:5
            delta = [J;sqrt(lmDamping*diagonalScale)*speye(nVar)]\[rhs;zeros(nVar,1)];
            for scale = [1 0.5 0.25 0.125 0.0625]
                candidateEta = eta+scale*reshape(delta(1:nEta),4,L+1);
                candidateTheta = thetaKnots+scale*reshape(delta(nEta+(1:K*nTheta)),nTheta,K);
                candidateTheta = thetaRef+max(min(candidateTheta-thetaRef,cfg.dnn.parameterDeviationLimit), ...
                    -cfg.dnn.parameterDeviationLimit);
                candidateCost = multiRateThetaCost(candidateEta,candidateTheta,knotOfStep,priorEta,thetaRef, ...
                    priorCovEta,priorCovTheta,1,L,data,cfg,qEta,qThetaK,alphaK);
                if isfinite(candidateCost) && candidateCost < baseCost
                    eta = candidateEta; thetaKnots = candidateTheta; costs(gn) = candidateCost;
                    accepted(gn) = true; acceptedThis = true; lmDamping = max(lmDamping/3,1e-12); break;
                end
            end
            if acceptedThis, break; end
            lmDamping = min(lmDamping*10,1e8);
        end
        if ~acceptedThis, break; end
        if nnz(accepted) >= 2
            ac = costs(accepted);
            if (ac(end-1)-ac(end))/max(abs(ac(end-1)),1) < cfg.factorGraph.relativeCostTolerance, break; end
        end
    end
    dHistoryOne = zeros(2,L+1);
    for ell = 1:L
        dHistoryOne(:,ell) = collectFullTrajectoryDNNPackets(eta(:,ell),reshape(thetaKnots(:,knotOfStep(ell)),p,Nb),cfg);
    end
    dHistoryOne(:,end) = collectFullTrajectoryDNNPackets(eta(:,end),reshape(thetaKnots(:,end),p,Nb),cfg);
    etaHistory = repmat(reshape(eta,4,L+1,1),1,1,Nw); dHistory = repmat(reshape(dHistoryOne,2,L+1,1),1,1,Nw);
    result.mode = "centralized_multirate_theta_square_root_ieks"; result.time = cfg.time;
    result.etaHat = etaHistory; result.dHat = dHistory; result.thetaChange = zeros(cfg.N,Nb); result.nis = nan(cfg.N,Nw);
    result.uploadTimes = cell(1,Nb); result.parameterUploads = 0; result.syncTimes = cfg.time(end); result.acceptedCosts = {costs(accepted)};
    result.windowVariableCount = nVar; result.windowStaticTheta = false; result.thetaKnotInterval = knotSteps*cfg.dt;
    result.thetaKnotCount = K; result.thetaKnotEstimates = thetaKnots;
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:)); result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue); result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2)); result.accelerationRMSE = sqrt(mean(result.accelerationError.^2)); result.finalPositionRMSE = result.positionError(end);
    result.implementationNote = "1 s theta knots with FOGM only between knots; eta and bearing factors remain at 10 Hz.";
end

function [J,rhs] = buildMultiRateThetaSquareRootSystem(eta,thetaKnots,knotOfStep,priorEta,thetaRef,priorCovEta,priorCovTheta,windowStart,L,data,cfg,qEta,qThetaK,alphaK)
    nTheta = size(thetaKnots,1); K = size(thetaKnots,2); nEta = 4*(L+1); nVar = nEta+K*nTheta;
    etaIds = @(node) (node-1)*4+(1:4); thetaIds = @(q) nEta+(q-1)*nTheta+(1:nTheta);
    LpEta = chol(symmetrizePSD(priorCovEta,1e-12),'lower'); LpTheta = chol(symmetrizePSD(priorCovTheta,1e-12),'lower');
    J = sparse(4+nTheta,nVar); J(1:4,etaIds(1)) = LpEta\eye(4); J(5:end,thetaIds(1)) = LpTheta\eye(nTheta);
    rhs = [-LpEta\(eta(:,1)-priorEta);-LpTheta\(thetaKnots(:,1)-thetaRef)];
    LqEta = chol(qEta,'lower'); LqTheta = chol(qThetaK,'lower'); A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)]; Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    for ell = 1:L
        q = knotOfStep(ell); blocks = reshape(thetaKnots(:,q),cfg.dnn.arch.nTheta,cfg.Nblocks);
        [d,Jeta,Jall] = collectFullTrajectoryDNNPackets(eta(:,ell),blocks,cfg);
        row = sparse(4,nVar); row(:,etaIds(ell)) = LqEta\(-A0-Lmap*Jeta); row(:,etaIds(ell+1)) = LqEta\eye(4); row(:,thetaIds(q)) = LqEta\(-Lmap*Jall);
        J = [J;row]; rhs = [rhs;-LqEta\(eta(:,ell+1)-physicalStep(eta(:,ell),d,cfg.dt))];
        for watcher = 1:cfg.Nw
            [nu,H,R] = makeWatcherBearingFactorPacket(data,watcher,windowStart+ell,eta(:,ell+1),cfg);
            row = sparse(1,nVar); row(etaIds(ell+1)) = H/sqrt(R); J = [J;row]; rhs = [rhs;nu/sqrt(R)];
        end
    end
    for q = 1:K-1
        row = sparse(nTheta,nVar); row(:,thetaIds(q)) = -alphaK*(LqTheta\eye(nTheta)); row(:,thetaIds(q+1)) = LqTheta\eye(nTheta);
        J = [J;row]; rhs = [rhs;-LqTheta\(thetaKnots(:,q+1)-thetaRef-alphaK*(thetaKnots(:,q)-thetaRef))];
    end
end

function cost = multiRateThetaCost(eta,thetaKnots,knotOfStep,priorEta,thetaRef,priorCovEta,priorCovTheta,windowStart,L,data,cfg,qEta,qThetaK,alphaK)
    cost = 0.5*(eta(:,1)-priorEta)'*(priorCovEta\(eta(:,1)-priorEta));
    cost = cost+0.5*(thetaKnots(:,1)-thetaRef)'*(priorCovTheta\(thetaKnots(:,1)-thetaRef));
    for ell = 1:L
        blocks = reshape(thetaKnots(:,knotOfStep(ell)),cfg.dnn.arch.nTheta,cfg.Nblocks);
        d = collectFullTrajectoryDNNPackets(eta(:,ell),blocks,cfg); residual = eta(:,ell+1)-physicalStep(eta(:,ell),d,cfg.dt);
        cost = cost+0.5*residual'*(qEta\residual);
        for watcher = 1:cfg.Nw
            [nu,~,R] = makeWatcherBearingFactorPacket(data,watcher,windowStart+ell,eta(:,ell+1),cfg); cost = cost+0.5*nu^2/R;
        end
    end
    for q = 1:size(thetaKnots,2)-1
        residual = thetaKnots(:,q+1)-thetaRef-alphaK*(thetaKnots(:,q)-thetaRef);
        cost = cost+0.5*residual'*(qThetaK\residual);
    end
end

function result = simulateMultiRateOwnerWatcherPacketCase(cfg,data,initial)
% Owner/watchers implementation of the same one-second-knot MAP problem.
    L = cfg.N-1; assert(abs(cfg.communication.period-L*cfg.dt) < 0.5*cfg.dt, ...
        'block_owned_watcher_multirate_packet_fg_5s requires one complete window.');
    knotSteps = round(cfg.factorGraph.thetaKnotInterval/cfg.dt); knotOfStep = ceil((1:L)/knotSteps); K = max(knotOfStep);
    Nw = cfg.Nw; Nb = cfg.Nblocks; p = cfg.dnn.arch.nTheta; nTheta = p*Nb; nEta = 4*(L+1); nVar = nEta+K*nTheta;
    thetaRef = initial.shared(:); priorEta = data.initialEta(:,1); priorCovEta = cfg.ekf.Peta0; priorCovTheta = cfg.ekf.Ptheta0*eye(nTheta);
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)]; qEta = Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap'+1e-12*eye(4);
    alphaK = exp(-knotSteps*cfg.dt/cfg.collaborative.thetaTau); qThetaK = cfg.ekf.Ptheta0*(1-alphaK^2)*eye(nTheta);
    Yseed = forwardReplayTrajectoryInitializer([priorEta;thetaRef],blkdiag(priorCovEta,priorCovTheta),1,L,data,cfg,thetaRef,exp(-cfg.dt/cfg.collaborative.thetaTau));
    eta = Yseed(1:4,:); thetaKnots = zeros(nTheta,K);
    for q = 1:K, thetaKnots(:,q) = Yseed(5:end,min((q-1)*knotSteps+1,L+1)); end
    maxGN = cfg.factorGraph.maxIterations; costs = nan(1,maxGN); accepted = false(1,maxGN); lmDamping = 1e-8;
    communication = initializeDistributedPacketCommunication();
    for gn = 1:maxGN
        packet = buildMultiRateOwnerWatcherPacket(eta,thetaKnots,knotOfStep,priorEta,thetaRef, ...
            priorCovEta,priorCovTheta,1,L,data,cfg,qEta,qThetaK,alphaK);
        communication = addDistributedPacketCommunication(communication,packet.communication);
        baseCost = multiRateThetaCost(eta,thetaKnots,knotOfStep,priorEta,thetaRef,priorCovEta,priorCovTheta,1,L,data,cfg,qEta,qThetaK,alphaK);
        acceptedThis = false;
        for lmAttempt = 1:5
            delta = solveMultiRateOwnerPacketSchur(packet,lmDamping*packet.diagonalScale,K,p);
            for scale = [1 0.5 0.25 0.125 0.0625]
                candidateEta = eta+scale*reshape(delta(1:nEta),4,L+1);
                candidateTheta = thetaKnots+scale*reshape(delta(nEta+(1:K*nTheta)),nTheta,K);
                candidateTheta = thetaRef+max(min(candidateTheta-thetaRef,cfg.dnn.parameterDeviationLimit),-cfg.dnn.parameterDeviationLimit);
                candidateCost = multiRateThetaCost(candidateEta,candidateTheta,knotOfStep,priorEta,thetaRef,priorCovEta,priorCovTheta,1,L,data,cfg,qEta,qThetaK,alphaK);
                if isfinite(candidateCost) && candidateCost < baseCost
                    eta = candidateEta; thetaKnots = candidateTheta; costs(gn) = candidateCost; accepted(gn) = true;
                    acceptedThis = true; lmDamping = max(lmDamping/3,1e-12); break;
                end
            end
            if acceptedThis, break; end
            lmDamping = min(lmDamping*10,1e8);
        end
        if ~acceptedThis, break; end
        if nnz(accepted) >= 2
            ac = costs(accepted);
            if (ac(end-1)-ac(end))/max(abs(ac(end-1)),1) < cfg.factorGraph.relativeCostTolerance, break; end
        end
    end
    communication.gsToOwnerCorrectionScalars = Nb*p;
    dOne = zeros(2,L+1); for ell = 1:L, dOne(:,ell) = collectFullTrajectoryDNNPackets(eta(:,ell),reshape(thetaKnots(:,knotOfStep(ell)),p,Nb),cfg); end
    dOne(:,end) = collectFullTrajectoryDNNPackets(eta(:,end),reshape(thetaKnots(:,end),p,Nb),cfg);
    result.mode = "owner_watcher_multirate_packet_structured_schur"; result.time = cfg.time;
    result.etaHat = repmat(reshape(eta,4,L+1,1),1,1,Nw); result.dHat = repmat(reshape(dOne,2,L+1,1),1,1,Nw);
    result.thetaChange = zeros(cfg.N,Nb); result.nis = nan(cfg.N,Nw); result.uploadTimes = cell(1,Nb); result.parameterUploads = Nb;
    result.syncTimes = cfg.time(end); result.acceptedCosts = {costs(accepted)}; result.windowVariableCount = nVar;
    result.thetaKnotInterval = knotSteps*cfg.dt; result.thetaKnotCount = K; result.thetaKnotEstimates = thetaKnots;
    result.distributedOwnerWatcherPackets = true; result.packetCommunication = communication;
    result.gsReconstructsUFromOwnerB = true;
    result.positionError = vectorRMSE(result.etaHat(1:2,:,:),data.truth(1:2,:)); result.velocityError = vectorRMSE(result.etaHat(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(result.dHat,data.dTrue); result.positionRMSE = sqrt(mean(result.positionError.^2)); result.velocityRMSE = sqrt(mean(result.velocityError.^2)); result.accelerationRMSE = sqrt(mean(result.accelerationError.^2)); result.finalPositionRMSE = result.positionError(end);
    result.linearSolver = "owner_watcher_multirate_packet_schur";
    result.implementationNote = "Owners form one-second-knot DNN/FOGM packets; watchers form bearing packets; GS only aggregates and solves.";
end

function packet = buildMultiRateOwnerWatcherPacket(eta,thetaKnots,knotOfStep,priorEta,thetaRef,priorCovEta,priorCovTheta,windowStart,L,data,cfg,qEta,qThetaK,alphaK)
    p = cfg.dnn.arch.nTheta; Nb = cfg.Nblocks; Nw = cfg.Nw; K = size(thetaKnots,2); nLocal = p*K; nEta = 4*(L+1);
    etaIds = @(node) (node-1)*4+(1:4); LpEta = chol(symmetrizePSD(priorCovEta,1e-12),'lower');
    Heta = zeros(nEta); geta = zeros(nEta,1); WpEta = LpEta\eye(4); Heta(etaIds(1),etaIds(1)) = WpEta'*WpEta; geta(etaIds(1)) = WpEta'*(-LpEta\(eta(:,1)-priorEta));
    ownerPackets = cell(1,Nb);
    for owner = 1:Nb
        cols = (owner-1)*p+(1:p);
        ownerPackets{owner} = makeOwnerMultiRatePacket(owner,eta,thetaKnots(cols,:),knotOfStep,thetaRef(cols),priorCovTheta(cols,cols),L,cfg,qEta,qThetaK(cols,cols),alphaK);
    end
    A = sparse(4*L,nEta); stateRhs = zeros(4*L,1); LqEta = chol(qEta,'lower'); A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)]; Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    for ell = 1:L
        d = zeros(2,1); Jeta = zeros(2,4); for owner = 1:Nb, d = d+ownerPackets{owner}.d(:,ell); Jeta = Jeta+ownerPackets{owner}.Jeta(:,:,ell); end
        rows = (ell-1)*4+(1:4); A(rows,etaIds(ell)) = -LqEta\(A0+Lmap*Jeta); A(rows,etaIds(ell+1)) = LqEta\eye(4);
        stateRhs(rows) = -LqEta\(eta(:,ell+1)-physicalStep(eta(:,ell),d,cfg.dt));
    end
    watcherPackets = cell(1,Nw);
    for watcher = 1:Nw
        watcherPackets{watcher} = makeWatcherTrajectoryBearingPacket(watcher,eta,windowStart,L,data,cfg);
        for ell = 1:L
            h = zeros(1,nEta); h(etaIds(ell+1)) = watcherPackets{watcher}.H(ell,:)/sqrt(watcherPackets{watcher}.R(ell));
            Heta = Heta+h'*h; geta = geta+h'*(watcherPackets{watcher}.nu(ell)/sqrt(watcherPackets{watcher}.R(ell)));
        end
    end
    Heta = Heta+full(A'*A); geta = geta+A'*stateRhs; packet.U = cell(1,Nb); packet.D = cell(1,Nb); packet.g = cell(1,Nb);
    for owner = 1:Nb
        % GS reconstructs U_j from the transmitted 2-by-p DNN Jacobian B_j.
        % LqEta and Lmap are public model terms, not owner-private DNN data.
        Uowner = sparse(4*L,nLocal);
        for ell = 1:L
            q = knotOfStep(ell); Uowner((ell-1)*4+(1:4),(q-1)*p+(1:p)) = ...
                -LqEta\(Lmap*ownerPackets{owner}.B(:,:,ell));
        end
        packet.U{owner} = Uowner; packet.D{owner} = ownerPackets{owner}.D;
        packet.g{owner} = ownerPackets{owner}.g+Uowner'*stateRhs;
    end
    packet.A = A; packet.Heta = Heta; packet.geta = geta; packet.nEta = nEta; packet.nLocal = nLocal; packet.Nb = Nb;
    packet.communication = summarizeDistributedPacketCommunication(ownerPackets,watcherPackets,nEta,nLocal,Nb,Nw,L,p);
    traceH = trace(Heta); for owner = 1:Nb, traceH = traceH+full(trace(packet.D{owner}))+norm(packet.U{owner},'fro')^2; end; packet.diagonalScale = max(1,traceH/(nEta+Nb*nLocal));
end

function ownerPacket = makeOwnerMultiRatePacket(owner,eta,thetaKnots,knotOfStep,thetaRef,priorThetaCov,L,cfg,qEta,qThetaK,alphaK)
    %#ok<INUSD>
    p = size(thetaKnots,1); K = size(thetaKnots,2); localIds = @(q) (q-1)*p+(1:p); nLocal = p*K;
    Lp = chol(symmetrizePSD(priorThetaCov,1e-12),'lower'); Lq = chol(qThetaK,'lower'); Lqe = chol(qEta,'lower'); Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    Wp = Lp\eye(p); Wq = Lq\eye(p); info = Wq'*Wq; ownerPacket.d = zeros(2,L); ownerPacket.Jeta = zeros(2,4,L); ownerPacket.B = zeros(2,p,L); ownerPacket.D = sparse(nLocal,nLocal); ownerPacket.g = zeros(nLocal,1);
    ownerPacket.D(localIds(1),localIds(1)) = Wp'*Wp; ownerPacket.g(localIds(1)) = Wp'*(-Lp\(thetaKnots(:,1)-thetaRef));
    for ell = 1:L
        q = knotOfStep(ell); [ownerPacket.d(:,ell),ownerPacket.Jeta(:,:,ell),Bj] = makeOwnerDNNFactorPacket(owner,eta(:,ell),thetaKnots(:,q),cfg);
        ownerPacket.B(:,:,ell) = Bj;
    end
    for q = 1:K-1
        current = localIds(q); next = localIds(q+1); ownerPacket.D(current,current) = ownerPacket.D(current,current)+alphaK^2*info; ownerPacket.D(current,next) = ownerPacket.D(current,next)-alphaK*info; ownerPacket.D(next,current) = ownerPacket.D(next,current)-alphaK*info; ownerPacket.D(next,next) = ownerPacket.D(next,next)+info;
        weighted = -Lq\(thetaKnots(:,q+1)-thetaRef-alphaK*(thetaKnots(:,q)-thetaRef)); ownerPacket.g(current) = ownerPacket.g(current)+(-alphaK*Wq)'*weighted; ownerPacket.g(next) = ownerPacket.g(next)+Wq'*weighted;
    end
end

function delta = solveMultiRateOwnerPacketSchur(packet,damping,K,p)
    Nb = packet.Nb; nLocal = packet.nLocal; UDU = zeros(size(packet.A,1)); q = zeros(size(packet.A,1),1); factors = cell(1,Nb); V = cell(1,Nb); dinvG = cell(1,Nb);
    for owner = 1:Nb
        factors{owner} = chol((packet.D{owner}+packet.D{owner}')/2+damping*speye(nLocal),'lower'); solveD = @(b) factors{owner}'\(factors{owner}\b);
        V{owner} = solveD(packet.U{owner}'); dinvG{owner} = solveD(packet.g{owner}); UDU = UDU+full(packet.U{owner}*V{owner}); q = q+packet.U{owner}*dinvG{owner};
    end
    Kmatrix = (eye(size(UDU))+UDU+eye(size(UDU))+UDU')/2; Rk = chol(Kmatrix,'lower'); Ksolve = @(b) Rk'\(Rk\b);
    Mtheta = Ksolve(UDU); Mtheta = (Mtheta+Mtheta')/2; Heta = (packet.Heta+packet.Heta')/2+damping*eye(packet.nEta); S = Heta-packet.A'*Mtheta*packet.A;
    Rs = chol((S+S')/2,'lower'); de = Rs'\(Rs\(packet.geta-packet.A'*Ksolve(q))); aDe = packet.A*de; w = Ksolve(q-UDU*aDe);
    thetaGlobal = zeros(K*p*Nb,1);
    for owner = 1:Nb
        solveD = @(b) factors{owner}'\(factors{owner}\b); local = solveD(packet.g{owner}-packet.U{owner}'*aDe)-V{owner}*w;
        for knot = 1:K, thetaGlobal((knot-1)*p*Nb+(owner-1)*p+(1:p)) = local((knot-1)*p+(1:p)); end
    end
    delta = [de;thetaGlobal];
end

function result = simulateFullTrajectoryFactorGraphCase(cfg,data,initial)
%SIMULATEFULLTRAJECTORYFACTORGRAPHCASE Sparse square-root fixed-lag IEKS.
% Each window variable is [eta_k;theta_k].  Unlike the older packet
% prototype, theta is not deterministically eliminated: every time step has
% its own FOGM factor and every watcher contributes only its bearing rows.

    Nw = cfg.Nw; Nb = cfg.Nblocks; p = cfg.dnn.arch.nTheta;
    nTheta = p*Nb; nNode = 4+nTheta; N = cfg.N;
    thetaRef = initial.shared(:);
    priorMean = [data.initialEta(:,1);thetaRef];
    priorCov = blkdiag(cfg.ekf.Peta0,cfg.ekf.Ptheta0*eye(nTheta));
    ownerStore = initializeOwnerParameterStore(thetaRef,p,Nb,cfg.factorGraph.blockOwnedPackets);
    alpha = exp(-cfg.dt/cfg.collaborative.thetaTau);
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    qEta = Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap' + 1e-12*eye(4);
    qTheta = cfg.ekf.Ptheta0*(1-alpha^2)*eye(nTheta);
    etaOperational = priorMean(1:4); thetaOperational = priorMean(5:end);
    etaHistory = zeros(4,N,Nw); dHistory = zeros(2,N,Nw);
    etaHistory(:,1,:) = repmat(etaOperational,1,1,Nw);
    thetaChange = zeros(N,Nb); nis = nan(N,Nw); uploadTimes = cell(1,Nb);
    syncTimes = []; acceptedCosts = {}; windowStart = 1;
    ownerDispatchLog = struct('time',{},'owner',{},'deltaThetaNorm',{},'thetaNorm',{});
    terminalCovarianceAvailable = true;
    packetCommunication = initializeDistributedPacketCommunication();

    for k = 1:N-1
        blocks = reshape(thetaOperational,p,Nb);
        [dNow,~,~] = collectFullTrajectoryDNNPackets(etaOperational,blocks,cfg);
        etaOperational = physicalStep(etaOperational,dNow,cfg.dt);
        thetaOperational = thetaRef + alpha*(thetaOperational-thetaRef);
        atSync = mod(k,round(cfg.communication.period/cfg.dt)) == 0 || k == N-1;
        if atSync
            L = k-windowStart+1; nVar = nNode*(L+1);
            ids = @(node) (node-1)*nNode+(1:nNode);
            % Use the causal, angle-only forward replay as the IEKS
            % initializer.  This provides one estimate per time node but
            % does not replace the subsequent joint MAP optimization.
            Y = forwardReplayTrajectoryInitializer(priorMean,priorCov, ...
                windowStart,L,data,cfg,thetaRef,alpha);
            maxGN = cfg.factorGraph.maxIterations;
            costs = nan(1,maxGN); accepted = false(1,maxGN);
            lmDamping = 1e-8;
            linearSolveIterations = nan(1,maxGN);
            linearSolveFlags = nan(1,maxGN);
            linearSolveRelativeResiduals = nan(1,maxGN);
            structuredCouplingDimensions = nan(1,maxGN);
            ownerBlockDiagonalResiduals = nan(1,maxGN);
            packetReferenceStepRelativeDifference = nan(1,maxGN);
            packetReferenceEtaRelativeDifference = nan(1,maxGN);
            packetReferenceDynamicsRelativeDifference = nan(1,maxGN);
            packetReferenceThetaNorms = nan(1,maxGN);
            packetReferenceThetaDifferenceNorms = nan(1,maxGN);
            packetReferenceDynamicsNorms = nan(1,maxGN);
            packetReferenceDynamicsDifferenceNorms = nan(1,maxGN);
            couplingReciprocalConditions = nan(1,maxGN);
            ownerCholeskyDiagonalRatios = cell(1,maxGN);
            for gn = 1:maxGN
                if cfg.factorGraph.matrixFreeSolver
                    operatorCfg = cfg;
                    if cfg.factorGraph.cacheOwnerPackets
                        operatorCfg.factorGraph.packetCache = buildFullTrajectoryPacketCache(Y,windowStart,L,data,cfg);
                        operatorCfg.factorGraph.packetCacheValid = true;
                    end
                    [rhs,~] = buildFullTrajectoryFactorRHS(Y,priorMean,priorCov,windowStart,L,data,operatorCfg,thetaRef,qEta,qTheta,alpha);
                elseif cfg.factorGraph.packetStructuredSchurSolver
                    if cfg.factorGraph.distributedOwnerWatcherPackets
                        structuredPacket = buildOwnerWatcherStructuredSchurPacket( ...
                            Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha);
                        packetCommunication = addDistributedPacketCommunication( ...
                            packetCommunication,structuredPacket.communication);
                    else
                        structuredPacket = buildPacketOnlyStructuredSchur( ...
                            Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha);
                    end
                    if cfg.factorGraph.packetParityDiagnostic && gn == 1
                        [diagnosticJ,diagnosticRhs] = buildFullTrajectorySquareRootSystem( ...
                            Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha);
                    end
                elseif cfg.factorGraph.timeOrderedSquareRootSolver
                    % Do not assemble the global sparse Jacobian for the
                    % time-chain solver.  The following packet contains the
                    % identical whitened prior, process, and bearing rows,
                    % already partitioned by adjacent time nodes.
                    timeOrderedPacket = buildTimeOrderedSquareRootPacket( ...
                        Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha);
                else
                    J = sparse(0,nVar); rhs = zeros(0,1);
                Lp = chol(symmetrizePSD(priorCov,1e-12),'lower');
                Jp = sparse(nNode,nVar); Jp(:,ids(1)) = Lp\eye(nNode);
                J = [J;Jp]; rhs = [rhs;-Lp\(Y(:,1)-priorMean)];
                LqEta = chol(qEta,'lower'); LqTheta = chol(qTheta,'lower');
                for ell = 1:L
                    x = Y(:,ell); xNext = Y(:,ell+1);
                    blocks = reshape(x(5:end),p,Nb);
                    [d,Jeta,Jall] = collectFullTrajectoryDNNPackets(x(1:4),blocks,cfg);
                    Jd = zeros(4,nVar); Jd(:,ids(ell)) = [-A0-Lmap*Jeta,-Lmap*Jall];
                    nextIds = ids(ell+1);
                    Jd(:,nextIds(1:4)) = eye(4);
                    rd = xNext(1:4)-physicalStep(x(1:4),d,cfg.dt);
                    J = [J;sparse(LqEta\Jd)]; rhs = [rhs;-LqEta\rd];
                    Jt = sparse(nTheta,nVar); Jt(:,ids(ell)) = [sparse(nTheta,4),-alpha*speye(nTheta)];
                    Jt(:,nextIds(5:end)) = speye(nTheta);
                    rt = xNext(5:end)-thetaRef-alpha*(x(5:end)-thetaRef);
                    J = [J;LqTheta\Jt]; rhs = [rhs;-LqTheta\rt];
                    for observer = 1:Nw
                        [nu,Heta,R] = makeWatcherBearingFactorPacket(data,observer,windowStart+ell,xNext(1:4),cfg);
                        % Linearized innovation is nu-H*delta, hence the
                        % square-root least-squares equation is H*delta=nu.
                        Jm = sparse(1,nVar); Jm(1,ids(ell+1)) = [Heta/sqrt(R),zeros(1,nTheta)];
                        J = [J;Jm]; rhs = [rhs;nu/sqrt(R)];
                    end
                end
                end
                baseCost = fullTrajectoryWindowCost(Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha);
                if cfg.factorGraph.matrixFreeSolver
                    diagonalScale = 1;
                elseif cfg.factorGraph.packetStructuredSchurSolver
                    diagonalScale = structuredPacket.diagonalScale;
                elseif cfg.factorGraph.timeOrderedSquareRootSolver
                    diagonalScale = timeOrderedPacket.diagonalScale;
                else
                    diagonalScale = max(1,full(mean(sum(J.^2,1))));
                end
                acceptedThis = false;
                for lmAttempt = 1:5
                    damping = lmDamping*diagonalScale;
                    linearSolveConverged = true;
                    if cfg.factorGraph.matrixFreeSolver
                        normalDiagonal = buildMatrixFreeNormalDiagonal(Y,priorCov,windowStart,L, ...
                            data,operatorCfg,qEta,qTheta,alpha,damping);
                        normalOperator = @(v) matrixFreeNormalOperator(v,Y,priorCov, ...
                            windowStart,L,data,operatorCfg,thetaRef,qEta,qTheta,alpha,damping);
                        normalRHS = fullTrajectoryMatrixFreeOperator([rhs;zeros(nVar,1)], ...
                            'transp',Y,priorCov,windowStart,L,data,operatorCfg,thetaRef,qEta,qTheta,alpha,damping);
                        M = buildBlockJacobiPreconditioner(Y,priorCov,windowStart,L, ...
                            data,operatorCfg,qEta,qTheta,alpha,damping,normalDiagonal);
                        [delta,flag,relres,iter] = pcg(normalOperator,normalRHS, ...
                            cfg.factorGraph.linearSolveTolerance,cfg.factorGraph.linearSolveMaxIterations,M);
                        linearSolveIterations(gn) = iter;
                        linearSolveFlags(gn) = flag;
                        linearSolveRelativeResiduals(gn) = relres;
                        linearSolveConverged = (flag == 0);
                    elseif cfg.factorGraph.packetStructuredSchurSolver
                        [delta,schurInfo] = solvePacketOnlyStructuredSchur(structuredPacket,damping);
                        if cfg.factorGraph.packetParityDiagnostic && gn == 1
                            diagnosticCfg = cfg; diagnosticCfg.factorGraph.structuredSchurSolver = true;
                            diagnosticCfg.factorGraph.packetStructuredSchurSolver = false;
                            [referenceDelta,~] = solveExactTrajectorySchur( ...
                                diagnosticJ,diagnosticRhs,nNode,L,damping,diagnosticCfg);
                            packetReferenceStepRelativeDifference(gn) = norm(delta-referenceDelta)/max(norm(referenceDelta),1);
                            correctionDiagnostic = comparePacketAndReferenceCorrection( ...
                                delta,referenceDelta,nNode,L,p,Nb,structuredPacket);
                            packetReferenceEtaRelativeDifference(gn) = correctionDiagnostic.etaRelativeDifference;
                            packetReferenceDynamicsRelativeDifference(gn) = correctionDiagnostic.dynamicsRelativeDifference;
                            packetReferenceThetaNorms(gn) = correctionDiagnostic.thetaReferenceNorm;
                            packetReferenceThetaDifferenceNorms(gn) = correctionDiagnostic.thetaDifferenceNorm;
                            packetReferenceDynamicsNorms(gn) = correctionDiagnostic.dynamicsReferenceNorm;
                            packetReferenceDynamicsDifferenceNorms(gn) = correctionDiagnostic.dynamicsDifferenceNorm;
                        end
                        linearSolveIterations(gn) = schurInfo.thetaDimension;
                        linearSolveFlags(gn) = 0;
                        linearSolveRelativeResiduals(gn) = schurInfo.relativeNormalResidual;
                        structuredCouplingDimensions(gn) = schurInfo.couplingDimension;
                        ownerBlockDiagonalResiduals(gn) = schurInfo.ownerBlockDiagonalResidual;
                        couplingReciprocalConditions(gn) = schurInfo.couplingReciprocalCondition;
                        ownerCholeskyDiagonalRatios{gn} = schurInfo.ownerCholeskyDiagonalRatios;
                    elseif cfg.factorGraph.timeOrderedSquareRootSolver
                        delta = solveTimeOrderedSquareRootPacket(timeOrderedPacket,damping);
                        linearSolveIterations(gn) = L;
                        linearSolveFlags(gn) = 0;
                        linearSolveRelativeResiduals(gn) = timeOrderedPacketResidual( ...
                            timeOrderedPacket,delta,damping);
                    elseif cfg.factorGraph.exactSchurSolver || cfg.factorGraph.structuredSchurSolver
                        [delta,schurInfo] = solveExactTrajectorySchur(J,rhs,nNode,L,damping,cfg);
                        linearSolveIterations(gn) = schurInfo.thetaDimension;
                        linearSolveFlags(gn) = 0;
                        linearSolveRelativeResiduals(gn) = schurInfo.relativeNormalResidual;
                        structuredCouplingDimensions(gn) = schurInfo.couplingDimension;
                        ownerBlockDiagonalResiduals(gn) = schurInfo.ownerBlockDiagonalResidual;
                    else
                        delta = [J;sqrt(damping)*speye(nVar)]\[rhs;zeros(nVar,1)];
                    end
                    if ~linearSolveConverged
                        % Do not accept an inaccurate Krylov correction.
                        % A larger LM term improves the inner conditioning.
                        lmDamping = min(lmDamping*10,1e8);
                        continue;
                    end
                    for scale = [1 0.5 0.25 0.125 0.0625]
                        trial = Y + scale*reshape(delta,nNode,L+1);
                        trialCost = fullTrajectoryWindowCost(trial,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha);
                        if isfinite(trialCost) && trialCost < baseCost
                            Y = trial; costs(gn) = trialCost; acceptedThis = true;
                            lmDamping = max(lmDamping/3,1e-12);
                            break;
                        end
                    end
                    if acceptedThis, break; end
                    lmDamping = min(lmDamping*10,1e8);
                end
                accepted(gn) = acceptedThis;
                if ~acceptedThis, break; end
                if gn > 1
                    relativeDecrease = (costs(gn-1)-costs(gn))/max(costs(gn-1),1);
                    if relativeDecrease < cfg.factorGraph.relativeCostTolerance
                        break;
                    end
                end
            end
            % Square-root marginalization is needed only if another window
            % follows.  Materializing the full sparse-QR R at a terminal
            % 30/60-s parity window wastes substantial memory.
            if k < N-1 && ~cfg.factorGraph.matrixFreeSolver && ~cfg.factorGraph.packetStructuredSchurSolver
                if cfg.factorGraph.timeOrderedBoundaryElimination
                    priorCov = terminalTimeBlockCovarianceFromSparseFactors(J,nNode,L);
                else
                    [~,Rqr] = qr(J,0); keep = ids(L+1);
                    E = sparse(keep,1:nNode,1,nVar,nNode);
                    priorCov = full(E'*(Rqr\(Rqr'\E))); priorCov = symmetrizePSD(priorCov,1e-12);
                end
            else
                terminalCovarianceAvailable = false;
            end
            solvedMean = Y(:,end);
            if cfg.factorGraph.blockOwnedPackets
                [ownerStore,dispatches] = dispatchGSBlockCorrections( ...
                    ownerStore,priorMean(5:end),solvedMean(5:end),p,cfg.time(k+1));
                ownerDispatchLog = [ownerDispatchLog,dispatches]; %#ok<AGROW>
                % The GS keeps no authoritative theta outside the solve:
                % the next prior is reconstructed from owner-held blocks.
                solvedMean(5:end) = collectOwnerParameterBlocks(ownerStore,p,Nb);
                if cfg.factorGraph.distributedOwnerWatcherPackets
                    packetCommunication.gsToOwnerCorrectionScalars = ...
                        packetCommunication.gsToOwnerCorrectionScalars+Nb*p;
                end
            end
            priorMean = solvedMean; etaOperational = priorMean(1:4); thetaOperational = priorMean(5:end);
            syncTimes(end+1) = cfg.time(k+1); acceptedCosts{end+1} = costs(accepted);
            for block = 1:Nb, uploadTimes{block}(end+1) = cfg.time(k+1); end
        end
        b = reshape(thetaOperational,p,Nb); [dNow,~,~] = collectFullTrajectoryDNNPackets(etaOperational,b,cfg);
        etaHistory(:,k+1,:) = repmat(etaOperational,1,1,Nw); dHistory(:,k,:) = repmat(dNow,1,1,Nw);
        for block = 1:Nb, cols = (block-1)*p+(1:p); thetaChange(k+1,block) = norm(thetaOperational(cols)-thetaRef(cols)); end
    end
    dHistory(:,N,:) = dHistory(:,N-1,:);
    result.mode = "common_state_full_trajectory_sparse_sqrt_ieks";
    result.time = cfg.time; result.etaHat = etaHistory; result.dHat = dHistory; result.thetaChange = thetaChange;
    result.nis = nis; result.uploadTimes = uploadTimes; result.parameterUploads = sum(cellfun(@numel,uploadTimes));
    result.syncTimes = syncTimes; result.acceptedCosts = acceptedCosts; result.windowVariableCount = nNode*(round(cfg.communication.period/cfg.dt)+1);
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:)); result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue); result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2)); result.accelerationRMSE = sqrt(mean(result.accelerationError.^2)); result.finalPositionRMSE = result.positionError(end);
    if terminalCovarianceAvailable
        result.parameterCovariance = priorCov(5:end,5:end);
    else
        result.parameterCovariance = [];
    end
    result.terminalCovarianceAvailable = terminalCovarianceAvailable;
    result.blockOwnedDNNPackets = cfg.factorGraph.blockOwnedPackets;
    result.ownerBlockDispatches = Nb*numel(syncTimes);
    result.ownerParameterStore = ownerStore;
    result.ownerDispatchLog = ownerDispatchLog;
    result.matrixFreeSolver = cfg.factorGraph.matrixFreeSolver;
    result.distributedOwnerWatcherPackets = cfg.factorGraph.distributedOwnerWatcherPackets;
    result.packetCommunication = packetCommunication;
    if cfg.factorGraph.distributedOwnerWatcherPackets
        result.implementationNote = ["Owners evaluate only their own DNN/FOGM trajectory packets; " ...
            "watchers generate bearing packets; GS only aggregates packets and solves the common correction."];
    end
    if cfg.factorGraph.matrixFreeSolver
        result.linearSolver = "matrix_free_pcg";
    elseif cfg.factorGraph.timeOrderedSquareRootSolver
        result.linearSolver = "time_ordered_square_root_chain";
    elseif cfg.factorGraph.packetStructuredSchurSolver
        result.linearSolver = "packet_only_owner_structured_schur";
    elseif cfg.factorGraph.structuredSchurSolver
        result.linearSolver = "owner_local_structured_schur";
    elseif cfg.factorGraph.exactSchurSolver
        result.linearSolver = "exact_normal_equation_schur";
    else
        result.linearSolver = "sparse_square_root_qr";
    end
    result.blockJacobiPreconditioner = cfg.factorGraph.matrixFreeSolver;
    result.ownerPacketCacheEnabled = cfg.factorGraph.matrixFreeSolver && cfg.factorGraph.cacheOwnerPackets;
    result.exactSchurSolver = cfg.factorGraph.exactSchurSolver;
    result.structuredSchurSolver = cfg.factorGraph.structuredSchurSolver;
    result.packetStructuredSchurSolver = cfg.factorGraph.packetStructuredSchurSolver;
    result.ownerLocalElimination = cfg.factorGraph.structuredSchurSolver || cfg.factorGraph.packetStructuredSchurSolver;
    result.lsqrIterations = linearSolveIterations(accepted); % legacy field name
    result.pcgIterations = linearSolveIterations(accepted);
    result.pcgFlags = linearSolveFlags(accepted);
    result.pcgRelativeResiduals = linearSolveRelativeResiduals(accepted);
    result.schurThetaDimensions = linearSolveIterations(accepted);
    result.schurRelativeNormalResiduals = linearSolveRelativeResiduals(accepted);
    result.structuredCouplingDimensions = structuredCouplingDimensions(accepted);
    result.ownerBlockDiagonalResiduals = ownerBlockDiagonalResiduals(accepted);
    result.packetReferenceStepRelativeDifference = packetReferenceStepRelativeDifference(accepted);
    result.packetReferenceEtaRelativeDifference = packetReferenceEtaRelativeDifference(accepted);
    result.packetReferenceDynamicsRelativeDifference = packetReferenceDynamicsRelativeDifference(accepted);
    result.packetReferenceThetaNorms = packetReferenceThetaNorms(accepted);
    result.packetReferenceThetaDifferenceNorms = packetReferenceThetaDifferenceNorms(accepted);
    result.packetReferenceDynamicsNorms = packetReferenceDynamicsNorms(accepted);
    result.packetReferenceDynamicsDifferenceNorms = packetReferenceDynamicsDifferenceNorms(accepted);
    result.couplingReciprocalConditions = couplingReciprocalConditions(accepted);
    result.ownerCholeskyDiagonalRatios = ownerCholeskyDiagonalRatios(accepted);
    if cfg.factorGraph.distributedOwnerWatcherPackets
        result.implementationNote = ["Owners evaluate only their own DNN/FOGM trajectory packets; " ...
            "watchers generate bearing packets; GS only aggregates packets and solves the common correction."];
    else
        result.implementationNote = sprintf("%.3g s full [eta_k;theta_k] trajectory, every-step FOGM, sparse square-root IEKS.",cfg.communication.period);
    end
end

function result = simulateLinearizedOneStepParityCase(cfg,data,initial)
%SIMULATELINEARIZEDONESTEPPARITYCASE Exact KF/batch consistency test.
% All factors are frozen at one prediction reference.  Therefore the
% sequential filter and square-root batch solution must agree numerically.
    assert(cfg.N == 2,"This diagnostic requires T=dt (one transition).");
    p = cfg.dnn.arch.nTheta; Nb = cfg.Nblocks; nTheta = p*Nb; nX = 4+nTheta;
    x0 = [data.initialEta(:,1);initial.shared(:)]; P0 = blkdiag(cfg.ekf.Peta0,cfg.ekf.Ptheta0*eye(nTheta));
    alpha = exp(-cfg.dt/cfg.collaborative.thetaTau);
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)]; A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    thetaPred = initial.shared(:) + alpha*(x0(5:end)-initial.shared(:));
    blocks = reshape(thetaPred,p,Nb);
    [d,Jeta,Jall] = evaluateGlobalAllJacobians(x0(1:4),blocks, ...
        globalCovarianceBlocks(P0(5:end,5:end),p,Nb),cfg);
    F = [A0+Lmap*Jeta,Lmap*Jall*alpha;zeros(nTheta,4),alpha*eye(nTheta)];
    xPred = [physicalStep(x0(1:4),d,cfg.dt);thetaPred];
    Q = blkdiag(Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap' + 1e-12*eye(4), ...
        cfg.ekf.Ptheta0*(1-alpha^2)*eye(nTheta));
    PPred = symmetrizePSD(F*P0*F' + Q,1e-12);

    Hrows = zeros(cfg.Nw,nX); nu = zeros(cfg.Nw,1); Rdiag = zeros(cfg.Nw,1);
    for observer = 1:cfg.Nw
        [nu(observer),Heta,Rdiag(observer)] = bearingInnovation( ...
            data.bearings(observer,2),xPred(1:4),cfg.watchers.r(:,observer),cfg);
        Hrows(observer,1:4) = Heta;
    end

    % Sequential Kalman updates for the single fixed linear measurement model.
    xKF = xPred; PKF = PPred;
    for observer = 1:cfg.Nw
        H = Hrows(observer,:); R = Rdiag(observer);
        innovation = nu(observer)-H*(xKF-xPred);
        S = H*PKF*H'+R; K = (PKF*H')/S; xKF = xKF+K*innovation;
        I = eye(nX); PKF = symmetrizePSD((I-K*H)*PKF*(I-K*H)' + K*R*K',1e-12);
    end

    % The identical linear Gaussian model, expressed as whitened batch rows
    % for delta=[delta_x0; delta_x1].
    Lp = chol(P0,'lower'); Lq = chol(Q,'lower');
    J = [Lp\[eye(nX),zeros(nX)]; Lq\[-F,eye(nX)]];
    rhs = zeros(2*nX,1);
    for observer = 1:cfg.Nw
        row = [zeros(1,nX),Hrows(observer,:)/sqrt(Rdiag(observer))];
        J = [J;sparse(row)]; rhs = [rhs;nu(observer)/sqrt(Rdiag(observer))];
    end
    delta = sparse(J)\rhs; xBatch = xPred+delta(nX+1:end);
    [~,Rqr] = qr(sparse(J),0); E = sparse(nX+(1:nX),1:nX,1,2*nX,nX);
    PBatch = full(E'*(Rqr\(Rqr'\E))); PBatch = symmetrizePSD(PBatch,1e-12);

    stateDifference = norm(xKF-xBatch); covarianceDifference = norm(PKF-PBatch,'fro');
    result.mode = "linearized_one_step_kf_square_root_parity";
    result.linearizedEndpointStateDifference = stateDifference;
    result.linearizedEndpointCovarianceDifference = covarianceDifference;
    result.linearizedParityPassed = stateDifference < 1e-8 && covarianceDifference < 1e-8;
    result.time = cfg.time; result.etaHat = repmat(reshape([x0(1:4),xBatch(1:4)],4,2,1),1,1,cfg.Nw);
    result.dHat = zeros(2,2,cfg.Nw); result.thetaChange = zeros(2,Nb); result.nis = nan(2,cfg.Nw);
    result.uploadTimes = cell(1,Nb); result.parameterUploads = 0;
    result.positionError = vectorRMSE(result.etaHat(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(result.etaHat(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(result.dHat,data.dTrue); result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2)); result.accelerationRMSE = sqrt(mean(result.accelerationError.^2)); result.finalPositionRMSE = result.positionError(end);
    fprintf("Linearized one-step parity: state %.3e, covariance %.3e.\n",stateDifference,covarianceDifference);
end

function [d,Jeta,Jall] = collectFullTrajectoryDNNPackets(eta,thetaBlocks,cfg)
%COLLECTFULLTRAJECTORYDNNPACKETS GS aggregation boundary for stage 1.
% In block-owned mode, each owner computes only its own d_j, A_j, and B_j.
% The GS receives those packets and performs only their additive assembly.
    p = cfg.dnn.arch.nTheta;
    if ~cfg.factorGraph.blockOwnedPackets
        zeroBlocks = zeros(p,p,cfg.Nblocks);
        [d,Jeta,Jall] = evaluateGlobalAllJacobians(eta,thetaBlocks,zeroBlocks,cfg);
        return;
    end
    d = zeros(2,1); Jeta = zeros(2,4); Jall = zeros(2,p*cfg.Nblocks);
    for owner = 1:cfg.Nblocks
        [dj,Aj,Bj] = makeOwnerDNNFactorPacket(owner,eta,thetaBlocks(:,owner),cfg);
        cols = (owner-1)*p+(1:p);
        d = d + dj; Jeta = Jeta + Aj; Jall(:,cols) = Bj;
    end
end

function [dj,Aj,Bj] = makeOwnerDNNFactorPacket(owner,eta,thetaOwner,cfg)
% Owner-local DNN computation.  owner is retained explicitly as packet metadata.
    %#ok<INUSD>
    [dj,Aj,Bj] = blockOutput(eta,thetaOwner,cfg);
end

function [nu,Heta,R] = makeWatcherBearingFactorPacket(data,watcher,sample,eta,cfg)
% Watcher-local whitenable bearing factor packet at the GS reference state.
    [nu,Heta,R] = bearingInnovation(data.bearings(watcher,sample), ...
        eta,cfg.watchers.r(:,watcher),cfg);
end

function cache = buildFullTrajectoryPacketCache(Y,windowStart,L,data,cfg)
%BUILD...CACHE Freeze owner and watcher linearization packets for one GN step.
% The cache is valid only at this fixed Y.  It is rebuilt at every outer
% IEKS iteration, so Krylov Jv/J'v calls reuse identical linear factors.
    p = cfg.dnn.arch.nTheta;
    cache = repmat(struct('d',[],'Jeta',[],'Jall',[], ...
        'nu',[],'Heta',[],'R',[]),1,L);
    for ell = 1:L
        x = Y(:,ell); xn = Y(:,ell+1);
        blocks = reshape(x(5:end),p,cfg.Nblocks);
        [cache(ell).d,cache(ell).Jeta,cache(ell).Jall] = ...
            collectFullTrajectoryDNNPackets(x(1:4),blocks,cfg);
        cache(ell).nu = zeros(cfg.Nw,1);
        cache(ell).Heta = zeros(cfg.Nw,4);
        cache(ell).R = zeros(cfg.Nw,1);
        for watcher = 1:cfg.Nw
            [cache(ell).nu(watcher),cache(ell).Heta(watcher,:), ...
                cache(ell).R(watcher)] = makeWatcherBearingFactorPacket( ...
                data,watcher,windowStart+ell,xn(1:4),cfg);
        end
    end
end

function packet = fullTrajectoryLinearizationPacket(Y,ell,windowStart,data,cfg)
% Return frozen packet when supplied; otherwise form the exact packet.
    if isfield(cfg.factorGraph,'packetCacheValid') && ...
            cfg.factorGraph.packetCacheValid
        packet = cfg.factorGraph.packetCache(ell);
        return;
    end
    x = Y(:,ell); xn = Y(:,ell+1);
    blocks = reshape(x(5:end),cfg.dnn.arch.nTheta,cfg.Nblocks);
    [packet.d,packet.Jeta,packet.Jall] = ...
        collectFullTrajectoryDNNPackets(x(1:4),blocks,cfg);
    packet.nu = zeros(cfg.Nw,1); packet.Heta = zeros(cfg.Nw,4);
    packet.R = zeros(cfg.Nw,1);
    for watcher = 1:cfg.Nw
        [packet.nu(watcher),packet.Heta(watcher,:),packet.R(watcher)] = ...
            makeWatcherBearingFactorPacket(data,watcher,windowStart+ell,xn(1:4),cfg);
    end
end

function store = initializeOwnerParameterStore(theta,p,nBlocks,enabled)
% Persistent canonical parameter state lives with the owners in stage 1.
    store = repmat(struct('owner',0,'theta',[],'lastDispatchDelta',[]),1,nBlocks);
    for owner = 1:nBlocks
        cols = (owner-1)*p+(1:p);
        store(owner).owner = owner;
        store(owner).theta = theta(cols);
        store(owner).lastDispatchDelta = zeros(p,1);
    end
    if ~enabled
        % Kept for a uniform result schema; central modes do not use it.
        for owner = 1:nBlocks, store(owner).theta = []; end
    end
end

function [store,dispatches] = dispatchGSBlockCorrections(store,thetaPrior,thetaSolved,p,time)
% GS sends only delta theta_j; owner j applies it to its local canonical state.
    nBlocks = numel(store);
    dispatches = repmat(struct('time',time,'owner',0,'deltaThetaNorm',0,'thetaNorm',0),1,nBlocks);
    for owner = 1:nBlocks
        cols = (owner-1)*p+(1:p);
        delta = thetaSolved(cols)-thetaPrior(cols);
        store(owner).theta = store(owner).theta + delta;
        store(owner).lastDispatchDelta = delta;
        dispatches(owner).owner = owner;
        dispatches(owner).deltaThetaNorm = norm(delta);
        dispatches(owner).thetaNorm = norm(store(owner).theta);
    end
end

function theta = collectOwnerParameterBlocks(store,p,nBlocks)
% Assemble transient solver input from owner-held blocks after dispatch.
    theta = zeros(p*nBlocks,1);
    for owner = 1:nBlocks
        cols = (owner-1)*p+(1:p);
        theta(cols) = store(owner).theta;
    end
end

function [rhs,nRows] = buildFullTrajectoryFactorRHS(Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha)
    nNode = size(Y,1); nTheta = nNode-4; nRows = nNode + L*(4+nTheta+cfg.Nw);
    rhs = zeros(nRows,1); row = 0;
    Lp = chol(symmetrizePSD(priorCov,1e-12),'lower');
    rhs(1:nNode) = -Lp\(Y(:,1)-priorMean); row = nNode;
    LqEta = chol(qEta,'lower');
    for ell = 1:L
        x = Y(:,ell); xn = Y(:,ell+1);
        packet = fullTrajectoryLinearizationPacket(Y,ell,windowStart,data,cfg);
        rd = xn(1:4)-physicalStep(x(1:4),packet.d,cfg.dt);
        rhs(row+(1:4)) = -LqEta\rd; row = row+4;
        rt = xn(5:end)-thetaRef-alpha*(x(5:end)-thetaRef);
        rhs(row+(1:nTheta)) = -LqTheta\rt; row = row+nTheta;
        for observer = 1:cfg.Nw
            rhs(row+1) = packet.nu(observer)/sqrt(packet.R(observer)); row = row+1;
        end
    end
end

function [J,rhs] = buildFullTrajectorySquareRootSystem(Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha)
% Diagnostic-only explicit square-root system for packet parity checks.
    nNode = size(Y,1); nTheta = nNode-4; nVar = nNode*(L+1); p = cfg.dnn.arch.nTheta;
    ids = @(node) (node-1)*nNode+(1:nNode); Lp = chol(symmetrizePSD(priorCov,1e-12),'lower');
    J = sparse(nNode,nVar); J(:,ids(1)) = Lp\eye(nNode); rhs = -Lp\(Y(:,1)-priorMean);
    LqEta = chol(qEta,'lower'); LqTheta = chol(qTheta,'lower'); A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)]; Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    for ell = 1:L
        x = Y(:,ell); xn = Y(:,ell+1); blocks = reshape(x(5:end),p,cfg.Nblocks);
        [d,Jeta,Jall] = collectFullTrajectoryDNNPackets(x(1:4),blocks,cfg);
        Jd = sparse(4,nVar); Jd(:,ids(ell)) = [-A0-Lmap*Jeta,-Lmap*Jall]; Jd(:,ids(ell+1)) = [eye(4),zeros(4,nTheta)];
        J = [J;sparse(LqEta\Jd)]; rhs = [rhs;-LqEta\(xn(1:4)-physicalStep(x(1:4),d,cfg.dt))];
        Jt = sparse(nTheta,nVar); Jt(:,ids(ell)) = [sparse(nTheta,4),-alpha*speye(nTheta)]; Jt(:,ids(ell+1)) = [sparse(nTheta,4),speye(nTheta)];
        J = [J;LqTheta\Jt]; rhs = [rhs;-LqTheta\(xn(5:end)-thetaRef-alpha*(x(5:end)-thetaRef))];
        for watcher = 1:cfg.Nw
            [nu,H,R] = makeWatcherBearingFactorPacket(data,watcher,windowStart+ell,xn(1:4),cfg);
            row = sparse(1,nVar); row(1,ids(ell+1)) = [H/sqrt(R),zeros(1,nTheta)]; J = [J;row]; rhs = [rhs;nu/sqrt(R)];
        end
    end
end

function packet = buildTimeOrderedSquareRootPacket(Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha)
%BUILD... Direct time-node representation of the same whitened GN system.
% No global J is formed.  At transition ell, processLeft{ell} multiplies
% delta x_ell and processRight{ell} multiplies delta x_(ell+1).
    nNode = size(Y,1); nTheta = nNode-4; p = cfg.dnn.arch.nTheta;
    Nb = cfg.Nblocks; Nw = cfg.Nw;
    Lp = chol(symmetrizePSD(priorCov,1e-12),'lower');
    LqEta = chol(qEta,'lower'); LqTheta = chol(qTheta,'lower');
    A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];

    packet.priorR = Lp\eye(nNode);
    packet.priorRhs = -Lp\(Y(:,1)-priorMean);
    packet.processLeft = cell(1,L);
    packet.processRight = cell(1,L);
    packet.processRhs = cell(1,L);
    packet.measurement = cell(1,L);
    packet.measurementRhs = cell(1,L);
    columnSquares = zeros(nNode,L+1);
    columnSquares(:,1) = sum(packet.priorR.^2,1)';

    for ell = 1:L
        x = Y(:,ell); xn = Y(:,ell+1);
        blocks = reshape(x(5:end),p,Nb);
        [d,Jeta,Jall] = collectFullTrajectoryDNNPackets(x(1:4),blocks,cfg);
        processLeft = [LqEta\[-A0-Lmap*Jeta,-Lmap*Jall]; ...
                       [zeros(nTheta,4),-alpha*(LqTheta\eye(nTheta))]];
        processRight = [LqEta\[eye(4),zeros(4,nTheta)]; ...
                        [zeros(nTheta,4),LqTheta\eye(nTheta)]];
        rd = xn(1:4)-physicalStep(x(1:4),d,cfg.dt);
        rt = xn(5:end)-thetaRef-alpha*(x(5:end)-thetaRef);
        measurement = zeros(Nw,nNode); measurementRhs = zeros(Nw,1);
        for observer = 1:Nw
            [nu,Heta,R] = makeWatcherBearingFactorPacket( ...
                data,observer,windowStart+ell,xn(1:4),cfg);
            measurement(observer,1:4) = Heta/sqrt(R);
            measurementRhs(observer) = nu/sqrt(R);
        end
        packet.processLeft{ell} = processLeft;
        packet.processRight{ell} = processRight;
        packet.processRhs{ell} = [-LqEta\rd;-LqTheta\rt];
        packet.measurement{ell} = measurement;
        packet.measurementRhs{ell} = measurementRhs;
        columnSquares(:,ell) = columnSquares(:,ell)+sum(processLeft.^2,1)';
        columnSquares(:,ell+1) = columnSquares(:,ell+1) + ...
            sum(processRight.^2,1)' + sum(measurement.^2,1)';
    end
    packet.nNode = nNode;
    packet.L = L;
    packet.diagonalScale = max(1,mean(columnSquares(:)));
    packet.rhsNorm = sqrt(norm(packet.priorRhs)^2 + sum(cellfun(@norm,packet.processRhs).^2) + ...
        sum(cellfun(@norm,packet.measurementRhs).^2));
end

function delta = solveTimeOrderedSquareRootPacket(packet,damping)
% Exact square-root elimination of the packet's block-bidiagonal chain.
    nNode = packet.nNode; L = packet.L;
    Rmessage = packet.priorR; dmessage = packet.priorRhs;
    Rdiag = cell(1,L); Rupper = cell(1,L); dconditional = cell(1,L);
    for ell = 1:L
        system = [Rmessage,zeros(nNode); ...
                  packet.processLeft{ell},packet.processRight{ell}; ...
                  zeros(size(packet.measurement{ell},1),nNode),packet.measurement{ell}; ...
                  sqrt(damping)*[eye(nNode),zeros(nNode)]];
        systemRhs = [dmessage;packet.processRhs{ell};packet.measurementRhs{ell};zeros(nNode,1)];
        [Q,R] = qr(system,0); transformedRhs = Q'*systemRhs;
        Rdiag{ell} = R(1:nNode,1:nNode);
        Rupper{ell} = R(1:nNode,nNode+(1:nNode));
        dconditional{ell} = transformedRhs(1:nNode);
        Rmessage = R(nNode+(1:nNode),nNode+(1:nNode));
        dmessage = transformedRhs(nNode+(1:nNode));
    end
    [Q,R] = qr([Rmessage;sqrt(damping)*eye(nNode)],0);
    dfinal = Q'*[dmessage;zeros(nNode,1)];
    x = zeros(nNode,L+1); x(:,L+1) = R\dfinal;
    for ell = L:-1:1
        x(:,ell) = Rdiag{ell}\(dconditional{ell}-Rupper{ell}*x(:,ell+1));
    end
    delta = x(:);
end

function relativeResidual = timeOrderedPacketResidual(packet,delta,damping)
% Diagnostic only; evaluates the same damped LS residual without J.
    nNode = packet.nNode; x = reshape(delta,nNode,packet.L+1);
    residualSquared = norm(packet.priorR*x(:,1)-packet.priorRhs)^2;
    for ell = 1:packet.L
        residualSquared = residualSquared + norm(packet.processLeft{ell}*x(:,ell) + ...
            packet.processRight{ell}*x(:,ell+1)-packet.processRhs{ell})^2;
        residualSquared = residualSquared + norm(packet.measurement{ell}*x(:,ell+1) - ...
            packet.measurementRhs{ell})^2;
    end
    residualSquared = residualSquared + damping*norm(delta)^2;
    relativeResidual = sqrt(residualSquared)/max(packet.rhsNorm,1);
end

function delta = solveTimeOrderedSquareRootLeastSquares(J,rhs,nNode,L,nTheta,Nw,damping)
% Exact QR elimination along the time chain of the linearized factor graph.
% It retains a square-root message only on x_{k+1} while storing the
% conditional factors needed for the final backward substitution.
    node = @(k) (k-1)*nNode+(1:nNode);
    Rmessage = full(J(1:nNode,node(1))); dmessage = rhs(1:nNode);
    Rdiag = cell(1,L); Rupper = cell(1,L); dconditional = cell(1,L);
    for ell = 1:L
        factorStart = nNode+(ell-1)*(nNode+Nw);
        processRows = factorStart+(1:nNode);
        measurementRows = factorStart+nNode+(1:Nw);
        current = node(ell); next = node(ell+1);
        process = [full(J(processRows,current)),full(J(processRows,next))];
        measurement = full(J(measurementRows,next));
        % Eliminate x_k after stacking every factor available at this
        % transition.  This is algebraically identical to the previous
        % two-QR sequence but cuts the QR calls per time step in half.
        system = [Rmessage,zeros(nNode); ...
                  process; ...
                  zeros(Nw,nNode),measurement; ...
                  sqrt(damping)*[eye(nNode),zeros(nNode)]];
        systemRhs = [dmessage;rhs(processRows);rhs(measurementRows);zeros(nNode,1)];
        [Q,R] = qr(system,0); transformedRhs = Q'*systemRhs;
        Rdiag{ell} = R(1:nNode,1:nNode);
        Rupper{ell} = R(1:nNode,nNode+(1:nNode));
        dconditional{ell} = transformedRhs(1:nNode);
        Rmessage = R(nNode+(1:nNode),nNode+(1:nNode));
        dmessage = transformedRhs(nNode+(1:nNode));
    end
    [Q,Rmessage] = qr([Rmessage;sqrt(damping)*eye(nNode)],0);
    dmessage = Q'*[dmessage;zeros(nNode,1)];
    x = cell(1,L+1); x{L+1} = Rmessage\dmessage;
    for ell = L:-1:1
        x{ell} = Rdiag{ell}\(dconditional{ell}-Rupper{ell}*x{ell+1});
    end
    delta = vertcat(x{:});
end

function out = fullTrajectoryMatrixFreeOperator(v,flag,Y,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha,damping)
% Matrix-free whitened factor operator, including sqrt(damping)*I rows.
    nNode = size(Y,1); nTheta = nNode-4; nVar = nNode*(L+1);
    nRows = nNode + L*(4+nTheta+cfg.Nw); ids = @(node) (node-1)*nNode+(1:nNode);
    Lp = chol(symmetrizePSD(priorCov,1e-12),'lower'); LqEta = chol(qEta,'lower'); LqTheta = chol(qTheta,'lower');
    isTranspose = ischar(flag) && strcmp(flag,'transp');
    if ~isTranspose
        out = zeros(nRows+nVar,1); row = 0;
        out(1:nNode) = Lp\v(ids(1)); row = nNode;
        for ell = 1:L
            packet = fullTrajectoryLinearizationPacket(Y,ell,windowStart,data,cfg);
            Jeta = packet.Jeta; Jall = packet.Jall;
            A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)]; Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
            Feta = A0+Lmap*Jeta; node = ids(ell); next = ids(ell+1);
            out(row+(1:4)) = LqEta\(-Feta*v(node(1:4))-Lmap*Jall*v(node(5:end))+v(next(1:4))); row = row+4;
            out(row+(1:nTheta)) = LqTheta\(-alpha*v(node(5:end))+v(next(5:end))); row = row+nTheta;
            for observer = 1:cfg.Nw
                out(row+1) = packet.Heta(observer,:)*v(next(1:4))/sqrt(packet.R(observer)); row = row+1;
            end
        end
        out(nRows+(1:nVar)) = sqrt(damping)*v;
    else
        out = zeros(nVar,1); row = 0;
        out(ids(1)) = Lp'\v(1:nNode); row = nNode;
        for ell = 1:L
            packet = fullTrajectoryLinearizationPacket(Y,ell,windowStart,data,cfg);
            Jeta = packet.Jeta; Jall = packet.Jall;
            A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)]; Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
            Feta = A0+Lmap*Jeta; node = ids(ell); next = ids(ell+1); w = LqEta'\v(row+(1:4));
            out(node(1:4)) = out(node(1:4))-Feta'*w; out(node(5:end)) = out(node(5:end))-(Lmap*Jall)'*w; out(next(1:4)) = out(next(1:4))+w; row = row+4;
            w = LqTheta'\v(row+(1:nTheta)); out(node(5:end)) = out(node(5:end))-alpha*w; out(next(5:end)) = out(next(5:end))+w; row = row+nTheta;
            for observer = 1:cfg.Nw
                out(next(1:4)) = out(next(1:4)) + packet.Heta(observer,:)'*(v(row+1)/sqrt(packet.R(observer))); row = row+1;
            end
        end
        out = out + sqrt(damping)*v(nRows+(1:nVar));
    end
end

function out = rightPreconditionedMatrixFreeOperator(v,flag,inverseScale,Y,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha,damping)
% Right-preconditioned form A*D^{-1}; theta blocks retain owner locality.
    if ischar(flag) && strcmp(flag,'transp')
        out = inverseScale.*fullTrajectoryMatrixFreeOperator(v,flag,Y,priorCov, ...
            windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha,damping);
    else
        out = fullTrajectoryMatrixFreeOperator(inverseScale.*v,flag,Y,priorCov, ...
            windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha,damping);
    end
end

function out = matrixFreeNormalOperator(v,Y,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha,damping)
% (A'*A + damping*I)v via owner-local JVP and VJP calls; no J is stored.
    forward = fullTrajectoryMatrixFreeOperator(v,'notransp',Y,priorCov, ...
        windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha,damping);
    out = fullTrajectoryMatrixFreeOperator(forward,'transp',Y,priorCov, ...
        windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha,damping);
end

function diagonal = buildMatrixFreeNormalDiagonal(Y,priorCov,windowStart,L,data,cfg,qEta,qTheta,alpha,damping)
% Exact diagonal of the current whitened normal operator, assembled blockwise.
    nNode = size(Y,1); nTheta = nNode-4; nVar = nNode*(L+1); ids = @(node) (node-1)*nNode+(1:nNode);
    diagonal = zeros(nVar,1);
    % A diagonal covariance approximation is sufficient for the PCG
    % preconditioner and avoids materializing P^{-1}, even for the prior.
    diagonal(ids(1)) = diagonal(ids(1)) + 1./max(diag(priorCov),1e-16);
    LqEta = chol(qEta,'lower'); LqTheta = chol(qTheta,'lower');
    stateNextDiagonal = sum((LqEta\eye(4)).^2,1)';
    thetaDiagonal = sum((LqTheta\eye(nTheta)).^2,1)';
    for ell = 1:L
        packet = fullTrajectoryLinearizationPacket(Y,ell,windowStart,data,cfg);
        Jeta = packet.Jeta; Jall = packet.Jall;
        A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)]; Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
        Feta = A0+Lmap*Jeta; current = ids(ell); next = ids(ell+1);
        Wcurrent = LqEta\[-Feta,-Lmap*Jall];
        diagonal(current) = diagonal(current) + sum(Wcurrent.^2,1)';
        diagonal(next(1:4)) = diagonal(next(1:4)) + stateNextDiagonal;
        diagonal(current(5:end)) = diagonal(current(5:end)) + alpha^2*thetaDiagonal;
        diagonal(next(5:end)) = diagonal(next(5:end)) + thetaDiagonal;
        for observer = 1:cfg.Nw
            diagonal(next(1:4)) = diagonal(next(1:4)) + ...
                (packet.Heta(observer,:)'.^2)/packet.R(observer);
        end
    end
    diagonal = diagonal + damping;
end

function M = buildBlockJacobiPreconditioner(Y,priorCov,windowStart,L,data,cfg,qEta,qTheta,alpha,damping,fallbackDiagonal)
%BLOCKJACOBIPRECONDITIONER SPD local blocks; cross-block terms remain in A.
    nNode = size(Y,1); p = cfg.dnn.arch.nTheta; Nb = cfg.Nblocks;
    nVar = nNode*(L+1); ids = @(node) (node-1)*nNode+(1:nNode);
    M = sparse(nVar,nVar); LqEta = chol(qEta,'lower');
    QetaInfo = (LqEta\eye(4))'*(LqEta\eye(4));
    qThetaInfo = 1./max(diag(qTheta),1e-16);
    % Prior: diagonal covariance approximation supplies independent SPD
    % state/owner blocks without forming the full prior information matrix.
    first = ids(1);
    M(first(1:4),first(1:4)) = diag(1./max(diag(priorCov(1:4,1:4)),1e-16));
    for owner = 1:Nb
        local = (owner-1)*p+(1:p); idx = first(4+local);
        M(idx,idx) = diag(1./max(diag(priorCov(4+local,4+local)),1e-16));
    end
    A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    for ell = 1:L
        packet = fullTrajectoryLinearizationPacket(Y,ell,windowStart,data,cfg);
        Jeta = packet.Jeta; Jall = packet.Jall;
        Feta = A0+Lmap*Jeta; current = ids(ell); next = ids(ell+1);
        M(current(1:4),current(1:4)) = M(current(1:4),current(1:4)) + (LqEta\Feta)'*(LqEta\Feta);
        M(next(1:4),next(1:4)) = M(next(1:4),next(1:4)) + QetaInfo;
        for owner = 1:Nb
            local = (owner-1)*p+(1:p); currentTheta = current(4+local); nextTheta = next(4+local);
            Wowner = LqEta\(Lmap*Jall(:,local));
            M(currentTheta,currentTheta) = M(currentTheta,currentTheta) + Wowner'*Wowner + diag(alpha^2*qThetaInfo(local));
            M(nextTheta,nextTheta) = M(nextTheta,nextTheta) + diag(qThetaInfo(local));
        end
        for observer = 1:cfg.Nw
            Heta = packet.Heta(observer,:); R = packet.R(observer);
            M(next(1:4),next(1:4)) = M(next(1:4),next(1:4)) + (Heta'*Heta)/R;
        end
    end
    % Keep every local block strictly SPD.  fallbackDiagonal contributes a
    % scale-aware jitter but does not introduce cross-block storage.
    M = M + spdiags(damping + 1e-10*max(fallbackDiagonal,1),0,nVar,nVar);
    M = (M+M')/2;
end

function [delta,info] = solveExactTrajectorySchur(J,rhs,nNode,L,damping,cfg)
%SOLVEEXACTTRAJECTORYSCHUR Exact block elimination of theta from one GN step.
% The partition is deltaY=[deltaEta_0:L; deltaTheta_0:L].  This solves the
% same damped normal equations as [J;sqrt(lambda)I]\\[rhs;0], but exposes
% the state-only Schur system that a later distributed implementation will
% receive.  Cross-owner theta couplings are retained in Htt; none are
% discarded in this benchmark.
    nVar = nNode*(L+1);
    etaIndex = zeros(4*(L+1),1);
    cursor = 0;
    for node = 1:L+1
        cols = (node-1)*nNode+(1:4);
        etaIndex(cursor+(1:4)) = cols;
        cursor = cursor+4;
    end
    thetaMask = true(nVar,1); thetaMask(etaIndex) = false;
    thetaIndex = find(thetaMask);

    H = J'*J + damping*speye(nVar);
    g = J'*rhs;
    HetaEta = full(H(etaIndex,etaIndex));
    HetaTheta = H(etaIndex,thetaIndex);
    HthetaTheta = H(thetaIndex,thetaIndex);
    geta = g(etaIndex); gtheta = g(thetaIndex);

    % Damping makes HthetaTheta SPD in the intended GN solve.  A tiny
    % numerical jitter only protects the Cholesky factorization itself.
    HthetaTheta = (HthetaTheta+HthetaTheta')/2;
    jitter = 1e-12*max(1,full(mean(abs(diag(HthetaTheta)))));
    if cfg.factorGraph.structuredSchurSolver
        [thetaSolve,structuredInfo] = buildOwnerStructuredThetaSolver( ...
            J,nNode,L,damping,cfg);
    else
        Rtheta = chol(HthetaTheta+jitter*speye(size(HthetaTheta,1)),'lower');
        thetaSolve = @(b) Rtheta'\(Rtheta\b);
        structuredInfo = struct('couplingDimension',0, ...
            'ownerBlockDiagonalResidual',nan);
    end
    HthetaEta = HetaTheta';
    thetaHthetaEta = thetaSolve(HthetaEta);
    thetaG = thetaSolve(gtheta);
    S = HetaEta - full(HetaTheta*thetaHthetaEta);
    b = geta - HetaTheta*thetaG;
    S = (S+S')/2;
    Reta = chol(S + jitter*eye(size(S,1)),'lower');
    deltaEta = Reta'\(Reta\b);
    deltaTheta = thetaSolve(gtheta-HthetaEta*deltaEta);
    delta = zeros(nVar,1); delta(etaIndex) = deltaEta; delta(thetaIndex) = deltaTheta;

    normalResidual = H*delta-g;
    info.thetaDimension = numel(thetaIndex);
    info.stateDimension = numel(etaIndex);
    info.relativeNormalResidual = norm(normalResidual)/max(norm(g),1);
    info.couplingDimension = structuredInfo.couplingDimension;
    info.ownerBlockDiagonalResidual = structuredInfo.ownerBlockDiagonalResidual;
end

function [thetaSolve,info] = buildOwnerStructuredThetaSolver(J,nNode,L,damping,cfg)
%BUILDOWNERSTRUCTUREDTHETASOLVER Exact Woodbury inverse action for Htt.
% A dynamics row has theta Jacobian [U_1 ... U_Nb].  Hence its complete
% cross-owner contribution is U'*U.  After subtracting it, the remaining
% prior/FOGM information is block diagonal by owner.  Owners factorize
% those D_j blocks locally; GS solves only I + U*D^{-1}*U'.
    p = cfg.dnn.arch.nTheta; Nb = cfg.Nblocks;
    nThetaNode = nNode-4; nThetaTrajectory = nThetaNode*(L+1);
    stateRows = zeros(4*L,1);
    for ell = 1:L
        factorStart = nNode + (ell-1)*(4+nThetaNode+cfg.Nw);
        stateRows((ell-1)*4+(1:4)) = factorStart+(1:4);
    end
    thetaIndexInFull = zeros(nThetaTrajectory,1);
    cursor = 0;
    for node = 1:L+1
        thetaIndexInFull(cursor+(1:nThetaNode)) = ...
            (node-1)*nNode+(5:nNode);
        cursor = cursor+nThetaNode;
    end
    U = J(stateRows,thetaIndexInFull);
    % Form D from its own factor rows rather than Htt-U'*U.  The latter
    % subtracts two large nearly equal information matrices and loses
    % accuracy in poorly scaled theta directions.
    localRowMask = true(size(J,1),1); localRowMask(stateRows) = false;
    Jlocal = J(localRowMask,thetaIndexInFull);
    D = Jlocal'*Jlocal + damping*speye(nThetaTrajectory);
    localIndices = cell(1,Nb); factors = cell(1,Nb);
    Dlocal = sparse(nThetaTrajectory,nThetaTrajectory);
    for owner = 1:Nb
        index = zeros(p*(L+1),1);
        cursor = 0;
        for node = 1:L+1
            index(cursor+(1:p)) = (node-1)*nThetaNode+(owner-1)*p+(1:p);
            cursor = cursor+p;
        end
        localIndices{owner} = index;
        Dj = (D(index,index)+D(index,index)')/2;
        % D_j already includes the LM term and should be SPD.  Use the
        % identical unperturbed local system as the packet-only solver;
        % only apply a minimal fallback if roundoff defeats Cholesky.
        [R,pchol] = chol(Dj,'lower');
        if pchol ~= 0
            localJitter = 1e-14*max(1,full(mean(abs(diag(Dj)))));
            R = chol(Dj+localJitter*speye(size(Dj,1)),'lower');
        end
        factors{owner} = R;
        Dlocal(index,index) = Dj;
    end
    ownerBlockDiagonalResidual = norm(D-Dlocal,'fro')/max(norm(D,'fro'),1);
    if ownerBlockDiagonalResidual > 1e-10
        error(['Structured Schur requires owner-block-diagonal local information. ' ...
            'The supplied prior has cross-owner information (relative residual %.3e). ' ...
            'Do not silently discard that coupling.'],ownerBlockDiagonalResidual);
    end
    applyDInverse = @(b) applyOwnerLocalInverse(b,localIndices,factors,nThetaTrajectory);
    DinvUt = applyDInverse(U');
    K = speye(size(U,1)) + U*DinvUt;
    K = (K+K')/2;
    Rcoupling = chol(K,'lower');
    thetaSolve = @(b) applyDInverse(b) - DinvUt*(Rcoupling'\(Rcoupling\(U*applyDInverse(b))));
    info.couplingDimension = size(U,1);
    info.ownerBlockDiagonalResidual = ownerBlockDiagonalResidual;
end

function x = applyOwnerLocalInverse(b,localIndices,factors,n)
% Owner-local Cholesky solves; b can contain multiple RHS columns.
    x = zeros(n,size(b,2));
    for owner = 1:numel(localIndices)
        index = localIndices{owner}; R = factors{owner};
        x(index,:) = R'\(R\b(index,:));
    end
end

function packet = buildPacketOnlyStructuredSchur(Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha)
% Build only owner-local and common-state packets; no full J or H exists.
    p = cfg.dnn.arch.nTheta; Nb = cfg.Nblocks; nLocal = p*(L+1); nEta = 4*(L+1);
    etaIds = @(node) (node-1)*4+(1:4); localIds = @(node) (node-1)*p+(1:p);
    if norm(priorCov(1:4,5:end),'fro') > 1e-10 || ...
            norm(priorCov(5:end,5:end)-blkdiagOwnerCovariance(priorCov(5:end,5:end),p,Nb),'fro') > 1e-10
        error('Packet-only structured Schur requires a block-independent incoming prior.');
    end
    LpEta = chol(symmetrizePSD(priorCov(1:4,1:4),1e-12),'lower');
    Heta = zeros(nEta); geta = zeros(nEta,1);
    WpEta = LpEta\eye(4); rpEta = -LpEta\(Y(1:4,1)-priorMean(1:4));
    Heta(etaIds(1),etaIds(1)) = WpEta'*WpEta; geta(etaIds(1)) = WpEta'*rpEta;
    A = sparse(4*L,nEta); stateRhs = zeros(4*L,1);
    packet.U = cell(1,Nb); packet.D = cell(1,Nb); packet.g = cell(1,Nb);
    for owner = 1:Nb
        packet.U{owner} = sparse(4*L,nLocal); packet.D{owner} = sparse(nLocal,nLocal); packet.g{owner} = zeros(nLocal,1);
        cols = (owner-1)*p+(1:p); LpTheta = chol(symmetrizePSD(priorCov(4+cols,4+cols),1e-12),'lower');
        WpTheta = LpTheta\eye(p); rpTheta = -LpTheta\(Y(4+cols,1)-priorMean(4+cols));
        packet.D{owner}(localIds(1),localIds(1)) = WpTheta'*WpTheta;
        packet.g{owner}(localIds(1)) = WpTheta'*rpTheta;
    end
    LqEta = chol(qEta,'lower'); LqTheta = chol(qTheta,'lower');
    for ell = 1:L
        x = Y(:,ell); xn = Y(:,ell+1); blocks = reshape(x(5:end),p,Nb);
        [d,Jeta,Jall] = collectFullTrajectoryDNNPackets(x(1:4),blocks,cfg);
        rows = (ell-1)*4+(1:4); Feta = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)] + [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)]*Jeta;
        A(rows,etaIds(ell)) = -LqEta\Feta; A(rows,etaIds(ell+1)) = LqEta\eye(4);
        stateRhs(rows) = -LqEta\(xn(1:4)-physicalStep(x(1:4),d,cfg.dt));
        for owner = 1:Nb
            cols = (owner-1)*p+(1:p); Bj = Jall(:,cols);
            LqThetaOwner = chol(qTheta(cols,cols),'lower');
            packet.U{owner}(rows,localIds(ell)) = -LqEta\([0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)]*Bj);
            current = localIds(ell); next = localIds(ell+1);
            Wtheta = LqThetaOwner\eye(p); thetaInfo = Wtheta'*Wtheta;
            rt = xn(4+cols)-thetaRef(cols)-alpha*(x(4+cols)-thetaRef(cols));
            % FOGM contributes only adjacent time blocks: D_j is therefore
            % sparse block-tridiagonal in time, not a dense trajectory matrix.
            packet.D{owner}(current,current) = packet.D{owner}(current,current) + alpha^2*thetaInfo;
            packet.D{owner}(current,next) = packet.D{owner}(current,next) - alpha*thetaInfo;
            packet.D{owner}(next,current) = packet.D{owner}(next,current) - alpha*thetaInfo;
            packet.D{owner}(next,next) = packet.D{owner}(next,next) + thetaInfo;
            weightedResidual = -LqThetaOwner\rt;
            packet.g{owner}(current) = packet.g{owner}(current) + (-alpha*Wtheta)'*weightedResidual;
            packet.g{owner}(next) = packet.g{owner}(next) + Wtheta'*weightedResidual;
        end
        for watcher = 1:cfg.Nw
            [nu,H,R] = makeWatcherBearingFactorPacket(data,watcher,windowStart+ell,xn(1:4),cfg);
            h = zeros(1,nEta); h(etaIds(ell+1)) = H/sqrt(R); Heta = Heta+h'*h; geta = geta+h'*(nu/sqrt(R));
        end
    end
    Heta = Heta+full(A'*A); geta = geta+A'*stateRhs;
    for owner = 1:Nb, packet.g{owner} = packet.g{owner}+packet.U{owner}'*stateRhs; end
    packet.A = A; packet.Heta = Heta; packet.geta = geta; packet.nEta = nEta; packet.nLocal = nLocal; packet.Nb = Nb;
    traceH = trace(Heta); for owner = 1:Nb, traceH = traceH+full(trace(packet.D{owner}))+norm(packet.U{owner},'fro')^2; end
    packet.diagonalScale = max(1,traceH/(nEta+Nb*nLocal));
end

function packet = buildOwnerWatcherStructuredSchurPacket(Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha)
%BUILD... Explicit computational ownership boundary for the final design.
% GS broadcasts only eta_0:L.  Owner j evaluates its own DNN block and
% FOGM trajectory, returning {d_j,A_j,B_j,D_j,U_j,g_j}.  Watcher i returns
% its bearing factor.  GS only adds received packets and solves the common
% correction; it never evaluates a DNN block in this path.
    p = cfg.dnn.arch.nTheta; Nb = cfg.Nblocks; nLocal = p*(L+1); nEta = 4*(L+1);
    etaIds = @(node) (node-1)*4+(1:4);
    if norm(priorCov(1:4,5:end),'fro') > 1e-10 || ...
            norm(priorCov(5:end,5:end)-blkdiagOwnerCovariance(priorCov(5:end,5:end),p,Nb),'fro') > 1e-10
        error('Distributed owner packets require a block-independent incoming prior.');
    end

    % GS-owned common-state prior packet.
    LpEta = chol(symmetrizePSD(priorCov(1:4,1:4),1e-12),'lower');
    Heta = zeros(nEta); geta = zeros(nEta,1);
    WpEta = LpEta\eye(4); rpEta = -LpEta\(Y(1:4,1)-priorMean(1:4));
    Heta(etaIds(1),etaIds(1)) = WpEta'*WpEta;
    geta(etaIds(1)) = WpEta'*rpEta;

    % Each owner receives eta reference and retains only theta_j(0:L).
    ownerPackets = cell(1,Nb);
    for owner = 1:Nb
        cols = (owner-1)*p+(1:p);
        ownerPackets{owner} = makeOwnerTrajectorySchurPacket(owner,Y(1:4,:), ...
            Y(4+cols,:),priorMean(4+cols),priorCov(4+cols,4+cols), ...
            thetaRef(cols),L,cfg,qEta,qTheta(cols,cols),alpha);
    end

    % GS aggregates DNN dynamics packets; it does no blockOutput call.
    A = sparse(4*L,nEta); stateRhs = zeros(4*L,1);
    LqEta = chol(qEta,'lower'); A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    for ell = 1:L
        rows = (ell-1)*4+(1:4); d = zeros(2,1); Jeta = zeros(2,4);
        for owner = 1:Nb
            d = d+ownerPackets{owner}.d(:,ell);
            Jeta = Jeta+ownerPackets{owner}.Jeta(:,:,ell);
        end
        Feta = A0+Lmap*Jeta;
        A(rows,etaIds(ell)) = -LqEta\Feta;
        A(rows,etaIds(ell+1)) = LqEta\eye(4);
        stateRhs(rows) = -LqEta\(Y(1:4,ell+1)-physicalStep(Y(1:4,ell),d,cfg.dt));
    end

    % Watchers independently create bearing packets at GS's eta reference.
    watcherPackets = cell(1,cfg.Nw);
    for watcher = 1:cfg.Nw
        watcherPackets{watcher} = makeWatcherTrajectoryBearingPacket( ...
            watcher,Y(1:4,:),windowStart,L,data,cfg);
        for ell = 1:L
            h = zeros(1,nEta); h(etaIds(ell+1)) = ...
                watcherPackets{watcher}.H(ell,:)/sqrt(watcherPackets{watcher}.R(ell));
            Heta = Heta+h'*h;
            geta = geta+h'*(watcherPackets{watcher}.nu(ell)/sqrt(watcherPackets{watcher}.R(ell)));
        end
    end

    Heta = Heta+full(A'*A); geta = geta+A'*stateRhs;
    packet.U = cell(1,Nb); packet.D = cell(1,Nb); packet.g = cell(1,Nb);
    for owner = 1:Nb
        packet.U{owner} = ownerPackets{owner}.U;
        packet.D{owner} = ownerPackets{owner}.D;
        packet.g{owner} = ownerPackets{owner}.g+packet.U{owner}'*stateRhs;
    end
    packet.A = A; packet.Heta = Heta; packet.geta = geta;
    packet.nEta = nEta; packet.nLocal = nLocal; packet.Nb = Nb;
    packet.ownerPackets = ownerPackets; packet.watcherPackets = watcherPackets;
    packet.ownerLocalDNNFactorEvaluations = Nb*L;
    packet.watcherBearingFactorPackets = cfg.Nw*L;
    packet.communication = summarizeDistributedPacketCommunication( ...
        ownerPackets,watcherPackets,nEta,nLocal,Nb,cfg.Nw,L,p);
    traceH = trace(Heta);
    for owner = 1:Nb
        traceH = traceH+full(trace(packet.D{owner}))+norm(packet.U{owner},'fro')^2;
    end
    packet.diagonalScale = max(1,traceH/(nEta+Nb*nLocal));
end

function communication = summarizeDistributedPacketCommunication(ownerPackets,watcherPackets,nEta,nLocal,Nb,Nw,L,p)
% Count transmitted numerical scalars for one IEKS relinearization packet.
% D_j is a deterministic FOGM/prior information block, so GS can construct
% it from the known model.  It is reported separately but not charged to the
% required owner-to-GS payload.  Sparse patterns are assumed known.
    ownerDNNScalars = 0; ownerFOGMGradientScalars = 0; ownerKnownModelScalars = 0;
    for owner = 1:Nb
        local = ownerPackets{owner};
        if isfield(local,'B')
            % B is the transmitted owner Jacobian; GS reconstructs U.
            ownerDNNScalars = ownerDNNScalars+numel(local.d)+numel(local.Jeta)+numel(local.B);
        else
            ownerDNNScalars = ownerDNNScalars+numel(local.d)+numel(local.Jeta)+nnz(local.U);
        end
        ownerFOGMGradientScalars = ownerFOGMGradientScalars+numel(local.g);
        ownerKnownModelScalars = ownerKnownModelScalars+nnz(local.D);
    end
    watcherScalars = 0;
    for watcher = 1:Nw
        local = watcherPackets{watcher};
        watcherScalars = watcherScalars+numel(local.nu)+numel(local.H)+numel(local.R);
    end
    communication = struct;
    communication.ownerToGSScalars = ownerDNNScalars+ownerFOGMGradientScalars;
    communication.ownerToGSFullPacketScalars = communication.ownerToGSScalars+ownerKnownModelScalars;
    communication.ownerDNNScalars = ownerDNNScalars;
    communication.ownerFOGMGradientScalars = ownerFOGMGradientScalars;
    communication.ownerKnownModelScalars = ownerKnownModelScalars;
    communication.watcherToGSScalars = watcherScalars;
    communication.gsReferenceBroadcastScalars = nEta*(Nb+Nw);
    communication.ownerPacketScalarsPerOwner = communication.ownerToGSScalars/Nb;
    communication.ownerFullPacketScalarsPerOwner = communication.ownerToGSFullPacketScalars/Nb;
    communication.watcherPacketScalarsPerWatcher = watcherScalars/Nw;
    communication.ownerLocalThetaTrajectoryScalars = nLocal*Nb; % retained locally, not transmitted
    communication.ownerCount = Nb; communication.watcherCount = Nw;
    communication.windowSteps = L; communication.blockParameterCount = p;
end

function total = initializeDistributedPacketCommunication()
    total = struct('ownerToGSScalars',0,'ownerToGSFullPacketScalars',0, ...
        'ownerDNNScalars',0,'ownerFOGMGradientScalars',0,'ownerKnownModelScalars',0, ...
        'watcherToGSScalars',0, ...
        'gsReferenceBroadcastScalars',0,'gsToOwnerCorrectionScalars',0, ...
        'linearizationPacketCount',0,'ownerPacketScalarsPerOwner',nan, ...
        'ownerFullPacketScalarsPerOwner',nan,'watcherPacketScalarsPerWatcher',nan,'ownerLocalThetaTrajectoryScalars',nan, ...
        'ownerCount',nan,'watcherCount',nan,'windowSteps',nan,'blockParameterCount',nan);
end

function total = addDistributedPacketCommunication(total,communication)
    total.ownerToGSScalars = total.ownerToGSScalars+communication.ownerToGSScalars;
    total.ownerToGSFullPacketScalars = total.ownerToGSFullPacketScalars+communication.ownerToGSFullPacketScalars;
    total.ownerDNNScalars = total.ownerDNNScalars+communication.ownerDNNScalars;
    total.ownerFOGMGradientScalars = total.ownerFOGMGradientScalars+communication.ownerFOGMGradientScalars;
    total.ownerKnownModelScalars = total.ownerKnownModelScalars+communication.ownerKnownModelScalars;
    total.watcherToGSScalars = total.watcherToGSScalars+communication.watcherToGSScalars;
    total.gsReferenceBroadcastScalars = total.gsReferenceBroadcastScalars+communication.gsReferenceBroadcastScalars;
    total.linearizationPacketCount = total.linearizationPacketCount+1;
    total.ownerPacketScalarsPerOwner = communication.ownerPacketScalarsPerOwner;
    total.ownerFullPacketScalarsPerOwner = communication.ownerFullPacketScalarsPerOwner;
    total.watcherPacketScalarsPerWatcher = communication.watcherPacketScalarsPerWatcher;
    total.ownerLocalThetaTrajectoryScalars = communication.ownerLocalThetaTrajectoryScalars;
    total.ownerCount = communication.ownerCount; total.watcherCount = communication.watcherCount;
    total.windowSteps = communication.windowSteps; total.blockParameterCount = communication.blockParameterCount;
end

function ownerPacket = makeOwnerTrajectorySchurPacket(owner,etaTrajectory,thetaTrajectory,priorTheta,priorThetaCov,thetaRef,L,cfg,qEta,qThetaOwner,alpha)
% Runs exclusively on owner j: one DNN block and its local FOGM factors.
    %#ok<INUSD>
    p = size(thetaTrajectory,1); nLocal = p*(L+1);
    localIds = @(node) (node-1)*p+(1:p);
    LpTheta = chol(symmetrizePSD(priorThetaCov,1e-12),'lower');
    LqTheta = chol(qThetaOwner,'lower'); LqEta = chol(qEta,'lower');
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    WpTheta = LpTheta\eye(p); Wtheta = LqTheta\eye(p);
    ownerPacket.d = zeros(2,L); ownerPacket.Jeta = zeros(2,4,L);
    ownerPacket.U = sparse(4*L,nLocal); ownerPacket.D = sparse(nLocal,nLocal);
    ownerPacket.g = zeros(nLocal,1);
    ownerPacket.D(localIds(1),localIds(1)) = WpTheta'*WpTheta;
    ownerPacket.g(localIds(1)) = WpTheta'*(-LpTheta\(thetaTrajectory(:,1)-priorTheta));
    thetaInfo = Wtheta'*Wtheta;
    for ell = 1:L
        [ownerPacket.d(:,ell),ownerPacket.Jeta(:,:,ell),Bj] = ...
            makeOwnerDNNFactorPacket(owner,etaTrajectory(:,ell),thetaTrajectory(:,ell),cfg);
        rows = (ell-1)*4+(1:4); current = localIds(ell); next = localIds(ell+1);
        ownerPacket.U(rows,current) = -LqEta\(Lmap*Bj);
        ownerPacket.D(current,current) = ownerPacket.D(current,current)+alpha^2*thetaInfo;
        ownerPacket.D(current,next) = ownerPacket.D(current,next)-alpha*thetaInfo;
        ownerPacket.D(next,current) = ownerPacket.D(next,current)-alpha*thetaInfo;
        ownerPacket.D(next,next) = ownerPacket.D(next,next)+thetaInfo;
        rt = thetaTrajectory(:,ell+1)-thetaRef-alpha*(thetaTrajectory(:,ell)-thetaRef);
        weightedResidual = -LqTheta\rt;
        ownerPacket.g(current) = ownerPacket.g(current)+(-alpha*Wtheta)'*weightedResidual;
        ownerPacket.g(next) = ownerPacket.g(next)+Wtheta'*weightedResidual;
    end
end

function watcherPacket = makeWatcherTrajectoryBearingPacket(watcher,etaTrajectory,windowStart,L,data,cfg)
% Runs exclusively on watcher i and contains no DNN parameter data.
    watcherPacket.nu = zeros(L,1); watcherPacket.H = zeros(L,4); watcherPacket.R = zeros(L,1);
    for ell = 1:L
        [watcherPacket.nu(ell),watcherPacket.H(ell,:),watcherPacket.R(ell)] = ...
            makeWatcherBearingFactorPacket(data,watcher,windowStart+ell,etaTrajectory(:,ell+1),cfg);
    end
end

function [delta,info] = solvePacketOnlyStructuredSchur(packet,damping)
    Nb = packet.Nb; nLocal = packet.nLocal; nTheta = Nb*nLocal; UDU = zeros(size(packet.A,1)); q = zeros(size(packet.A,1),1);
    factors = cell(1,Nb); V = cell(1,Nb); dinvG = cell(1,Nb); ownerRatios = zeros(1,Nb);
    for owner = 1:Nb
        factors{owner} = chol((packet.D{owner}+packet.D{owner}')/2+damping*speye(nLocal),'lower');
        diagonal = abs(diag(factors{owner}));
        ownerRatios(owner) = max(diagonal)/max(min(diagonal),realmin);
        solveD = @(b) factors{owner}'\(factors{owner}\b);
        V{owner} = solveD(packet.U{owner}'); dinvG{owner} = solveD(packet.g{owner});
        UDU = UDU+full(packet.U{owner}*V{owner}); q = q+packet.U{owner}*dinvG{owner};
    end
    Kmatrix = (eye(size(UDU))+UDU+eye(size(UDU))+UDU')/2;
    K = chol(Kmatrix,'lower');
    Ksolve = @(b) K'\(K\b);
    % UDU - UDU*(I+UDU)^(-1)*UDU equals (I+UDU)^(-1)*UDU.
    % Use the latter to avoid subtracting nearly equal matrices.
    Mtheta = Ksolve(UDU); Mtheta = (Mtheta+Mtheta')/2;
    % LM damping applies to every component of deltaY, including eta.
    HetaDamped = (packet.Heta+packet.Heta')/2+damping*eye(packet.nEta);
    S = HetaDamped-packet.A'*Mtheta*packet.A;
    b = packet.geta-packet.A'*Ksolve(q); R = chol((S+S')/2,'lower'); de = R'\(R\b); aDe = packet.A*de;
    qB = q-UDU*aDe; w = Ksolve(qB); dt = zeros(nTheta,1); sumUdt = zeros(size(q));
    for owner = 1:Nb
        solveD = @(b) factors{owner}'\(factors{owner}\b); idx = (owner-1)*nLocal+(1:nLocal);
        dt(idx) = solveD(packet.g{owner}-packet.U{owner}'*aDe)-V{owner}*w; sumUdt = sumUdt+packet.U{owner}*dt(idx);
    end
    rEta = HetaDamped*de+packet.A'*sumUdt-packet.geta; rTheta = 0;
    for owner = 1:Nb, idx = (owner-1)*nLocal+(1:nLocal); rTheta = rTheta+norm((packet.D{owner}+damping*speye(nLocal))*dt(idx)+packet.U{owner}'*(aDe+sumUdt)-packet.g{owner})^2; end
    delta = zeros(packet.nEta+nTheta,1);
    for node = 1:(packet.nEta/4), delta((node-1)*(4+Nb*(nLocal/(packet.nEta/4)))+(1:4)) = de((node-1)*4+(1:4)); end
    p = nLocal/(packet.nEta/4); for node = 1:(packet.nEta/4), for owner = 1:Nb, fullIdx = (node-1)*(4+Nb*p)+4+(owner-1)*p+(1:p); localIdx = (node-1)*p+(1:p); thetaIdx = (owner-1)*nLocal+localIdx; delta(fullIdx) = dt(thetaIdx); end, end
    info.thetaDimension = nTheta; info.couplingDimension = size(packet.A,1); info.ownerBlockDiagonalResidual = 0;
    info.relativeNormalResidual = sqrt(norm(rEta)^2+rTheta)/max(norm([packet.geta;cell2mat(packet.g(:))]),1);
    info.couplingReciprocalCondition = rcond(Kmatrix);
    info.ownerCholeskyDiagonalRatios = ownerRatios;
end

function Pterminal = terminalTimeBlockCovarianceFromSparseFactors(J,nNode,L)
% Eliminate time blocks in order without forming a global sparse-QR R.
% J contains only prior, adjacent-time dynamics/FOGM, and node-local
% bearing factors, hence H=J'*J is block-tridiagonal in time ordering.
    H = J'*J; nNodes = L+1;
    block = @(node) (node-1)*nNode+(1:nNode);
    S = full(H(block(1),block(1)));
    for node = 1:L
        next = block(node+1); current = block(node);
        B = full(H(current,next));
        C = full(H(next,next));
        S = symmetrizePSD(S,1e-12);
        R = chol(S,'lower');
        S = C - B'*(R'\(R\B));
    end
    S = symmetrizePSD(S,1e-12);
    R = chol(S,'lower');
    Pterminal = R'\(R\eye(nNode));
    Pterminal = symmetrizePSD(Pterminal,1e-12);
end

function B = blkdiagOwnerCovariance(P,p,Nb)
    B = zeros(size(P)); for owner = 1:Nb, idx = (owner-1)*p+(1:p); B(idx,idx) = P(idx,idx); end
end

function info = comparePacketAndReferenceCorrection(packetDelta,referenceDelta,nNode,L,p,Nb,packet)
% Compare physically consequential components, not weak theta null modes.
    nEta = 4*(L+1); etaIndex = zeros(nEta,1); cursor = 0;
    for node = 1:L+1
        etaIndex(cursor+(1:4)) = (node-1)*nNode+(1:4); cursor = cursor+4;
    end
    etaPacket = packetDelta(etaIndex); etaReference = referenceDelta(etaIndex);
    info.etaRelativeDifference = norm(etaPacket-etaReference)/max(norm(etaReference),1);
    dynamicsPacket = zeros(4*L,1); dynamicsReference = zeros(4*L,1);
    nLocal = p*(L+1);
    thetaPacket = zeros(Nb*nLocal,1); thetaReference = zeros(Nb*nLocal,1);
    for owner = 1:Nb
        thetaIndex = zeros(nLocal,1); cursor = 0;
        for node = 1:L+1
            thetaIndex(cursor+(1:p)) = (node-1)*nNode+4+(owner-1)*p+(1:p);
            cursor = cursor+p;
        end
        ownerRange = (owner-1)*nLocal+(1:nLocal);
        thetaPacket(ownerRange) = packetDelta(thetaIndex);
        thetaReference(ownerRange) = referenceDelta(thetaIndex);
        dynamicsPacket = dynamicsPacket+packet.U{owner}*thetaPacket(ownerRange);
        dynamicsReference = dynamicsReference+packet.U{owner}*thetaReference(ownerRange);
    end
    info.thetaReferenceNorm = norm(thetaReference);
    info.thetaDifferenceNorm = norm(thetaPacket-thetaReference);
    info.dynamicsReferenceNorm = norm(dynamicsReference);
    info.dynamicsDifferenceNorm = norm(dynamicsPacket-dynamicsReference);
    info.dynamicsRelativeDifference = norm(dynamicsPacket-dynamicsReference)/max(norm(dynamicsReference),1);
end

function cost = fullTrajectoryWindowCost(Y,priorMean,priorCov,windowStart,L,data,cfg,thetaRef,qEta,qTheta,alpha)
    e = Y(:,1)-priorMean; cost = 0.5*e'*(priorCov\e);
    for ell = 1:L
        x = Y(:,ell); xn = Y(:,ell+1); blocks = reshape(x(5:end),cfg.dnn.arch.nTheta,cfg.Nblocks);
        [d,~,~] = collectFullTrajectoryDNNPackets(x(1:4),blocks,cfg); ed = xn(1:4)-physicalStep(x(1:4),d,cfg.dt);
        et = xn(5:end)-thetaRef-alpha*(x(5:end)-thetaRef); cost = cost+0.5*ed'*(qEta\ed)+0.5*et'*(qTheta\et);
        for observer = 1:cfg.Nw
            [nu,~,R] = makeWatcherBearingFactorPacket(data,observer,windowStart+ell,xn(1:4),cfg);
            cost = cost+0.5*nu^2/R;
        end
    end
end

function Y = forwardReplayTrajectoryInitializer(priorMean,priorCov,windowStart,L,data,cfg,thetaRef,alpha)
% Produce a causal EKF trajectory using only the buffered window bearings.
    p = cfg.dnn.arch.nTheta; Nb = cfg.Nblocks; nTheta = p*Nb;
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    x = priorMean; P = priorCov; Y = zeros(4+nTheta,L+1); Y(:,1) = x;
    for ell = 1:L
        theta = thetaRef + alpha*(x(5:end)-thetaRef);
        blocks = reshape(theta,p,Nb);
        pBlocks = globalCovarianceBlocks(P(5:end,5:end),p,Nb);
        [d,Jeta,Jall] = collectFullTrajectoryDNNPackets(x(1:4),blocks,cfg);
        Feta = A0 + Lmap*Jeta; Btheta = Lmap*Jall*alpha;
        Pee = P(1:4,1:4); PeT = P(1:4,5:end); PTT = P(5:end,5:end);
        PeePred = Feta*Pee*Feta' + Feta*PeT*Btheta' + Btheta*PeT'*Feta' + Btheta*PTT*Btheta';
        PeTPred = alpha*(Feta*PeT + Btheta*PTT);
        Qtheta = cfg.ekf.Ptheta0*(1-alpha^2)*eye(nTheta);
        Qeta = Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap' + 1e-12*eye(4);
        P = symmetrizePSD([PeePred+Qeta,PeTPred;PeTPred',alpha^2*PTT+Qtheta],1e-12);
        x = [physicalStep(x(1:4),d,cfg.dt);theta];
        for observer = 1:cfg.Nw
            [innovation,Heta,R] = makeWatcherBearingFactorPacket(data,observer,windowStart+ell,x(1:4),cfg);
            H = [Heta zeros(1,nTheta)]; S = H*P*H' + R; K = (P*H')/S;
            x = x + K*innovation; I = eye(4+nTheta);
            P = symmetrizePSD((I-K*H)*P*(I-K*H)' + K*R*K',1e-12);
        end
        if cfg.collaborative.enableParameterClipping
            thetaPred = x(5:end); step = thetaPred-theta;
            if norm(step) > cfg.dnn.parameterStepLimit
                thetaPred = theta + cfg.dnn.parameterStepLimit*step/norm(step);
            end
            x(5:end) = thetaRef + max(min(thetaPred-thetaRef,cfg.dnn.parameterDeviationLimit), ...
                -cfg.dnn.parameterDeviationLimit);
        end
        Y(:,ell+1) = x;
    end
end

function cost = factorPacketWindowCost(xbar,thetaStart,etaPrior,thetaPrior, ...
    Pprior,windowStart,L,data,cfg,thetaRef,alpha)
    e0 = [xbar(:,1)-etaPrior;thetaStart-thetaPrior];
    cost = 0.5*e0'*(Pprior\e0);
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    Q = Lmap*(cfg.ekf.qAcceleration*eye(2))*Lmap' + 1e-12*eye(4);
    for ell = 1:L
        theta = thetaRef+alpha^(ell-1)*(thetaStart-thetaRef);
        d = evaluateGlobal(xbar(:,ell),reshape(theta,cfg.dnn.arch.nTheta,cfg.Nblocks),cfg,1,"shared_block");
        e = xbar(:,ell+1)-physicalStep(xbar(:,ell),d,cfg.dt);
        cost = cost+0.5*e'*(Q\e);
        for observer = 1:cfg.Nw
            [nu,~,R] = bearingInnovation(data.bearings(observer,windowStart+ell), ...
                xbar(:,ell+1),cfg.watchers.r(:,observer),cfg);
            cost = cost+0.5*nu^2/R;
        end
    end
end

function result = simulateCanonicalFactorHybridCase(cfg,data,initial)
%SIMULATECANONICALFACTORHYBRIDCASE Two-path hybrid with accepted GS steps.
%
% The fast path updates each watcher's owner block sequentially.  A second,
% canonical-factor path keeps the GS anchor network fixed throughout the
% window and is the ONLY source of joint GS information factors.  This
% separation prevents owner-shadow updates from contaminating cross-block
% Hessian terms.  Candidate GS steps are accepted only when a replayed,
% local-data window likelihood plus the FOGM prior decreases.

    Nw = cfg.Nw; Nb = cfg.Nblocks; N = cfg.N;
    p = cfg.dnn.arch.nTheta; nGlobal = p*Nb;
    owners = cfg.collaborative.observerOwner;
    thetaReference = initial.shared;
    thetaAnchor = initial.shared;
    thetaGS = thetaAnchor;
    Pglobal = cfg.ekf.Ptheta0*eye(nGlobal);
    gammaTheta = ones(1,Nb); gammaEta = 1;

    etaFast = data.initialEta;
    thetaLocal = zeros(p,Nw);
    Pfast = zeros(4+p,4+p,Nw);
    Gfast = zeros(4,nGlobal,Nw);
    etaFac = data.initialEta;
    Pfac = repmat(cfg.ekf.Peta0,1,1,Nw);
    Gfac = zeros(4,nGlobal,Nw);
    for observer = 1:Nw
        thetaLocal(:,observer) = thetaGS(:,owners(observer));
        Pfast(:,:,observer) = blkdiag(cfg.ekf.Peta0,cfg.ekf.Ptheta0*eye(p));
    end
    facStartEta = etaFac;
    facStartP = Pfac;
    windowStartIndex = 1;

    packetLambda = zeros(nGlobal);
    packetq = zeros(nGlobal,1);
    nisWindowSum = 0; nisWindowCount = 0;
    lastSync = 0;
    uploadTimes = cell(1,Nb);
    etaHistory = zeros(4,N,Nw);
    dHistory = zeros(2,N,Nw);
    nis = zeros(N,Nw);
    thetaChange = zeros(N,Nb);
    gsAccepted = false(1,N);
    gsCostBefore = nan(1,N); gsCostAfter = nan(1,N);
    gsResetNorm = zeros(1,N);
    informationScore = nan(1,N);
    syncReason = strings(1,N);
    etaHistory(:,1,:) = reshape(etaFast,4,1,Nw);

    for k = 1:N-1
        t = cfg.time(k);
        Pblocks = globalCovarianceBlocks(Pglobal,p,Nb);
        alpha = exp(-cfg.dt/cfg.collaborative.thetaTau);
        Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
        A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];

        for observer = 1:Nw
            owner = owners(observer);
            ownerCols = (owner-1)*p+(1:p);

            % Fast operational owner-block EKF path.
            thetaOwnerPred = thetaAnchor(:,owner) + alpha*( ...
                thetaLocal(:,observer)-thetaAnchor(:,owner));
            thetaShadow = thetaAnchor;
            thetaShadow(:,owner) = thetaOwnerPred;
            [dFast,JetaFast,JallFast,parameterQd] = evaluateGlobalAllJacobians( ...
                etaFast(:,observer),thetaShadow,Pblocks,cfg);
            Jowner = JallFast(:,ownerCols);
            etaPredFast = physicalStep(etaFast(:,observer),dFast,cfg.dt);
            FetaFast = A0 + Lmap*JetaFast;
            Faug = eye(4+p);
            Faug(1:4,1:4) = FetaFast;
            Faug(1:4,5:end) = Lmap*Jowner*alpha;
            Faug(5:end,5:end) = alpha*eye(p);
            Qaug = zeros(4+p);
            Qaug(1:4,1:4) = gammaEta*(Lmap*(cfg.ekf.qAcceleration*eye(2) + ...
                cfg.ekf.remoteCovarianceScale*parameterQd)*Lmap') + 1e-12*eye(4);
            Qaug(5:end,5:end) = gammaTheta(owner)*cfg.ekf.Ptheta0* ...
                (1-alpha^2)*eye(p);
            PpredFast = symmetrizePSD(Faug*Pfast(:,:,observer)*Faug' + Qaug,1e-12);
            GpredFast = FetaFast*Gfast(:,:,observer) + Lmap*JallFast;
            [etaNewFast,thetaNewFast,PnewFast,GnewFast] = localFastMeasurementUpdates( ...
                etaPredFast,thetaOwnerPred,PpredFast,GpredFast, ...
                data,observer,k+1,cfg);
            etaFast(:,observer) = etaNewFast;
            thetaLocal(:,observer) = thetaNewFast;
            Pfast(:,:,observer) = PnewFast;
            Gfast(:,:,observer) = GnewFast;

            % Canonical-factor path.  Its DNN remains thetaAnchor for the
            % entire window, so all packets share one linearization anchor.
            [dFac,JetaFac,JallFac,parameterQdFac] = evaluateGlobalAllJacobians( ...
                etaFac(:,observer),thetaAnchor,Pblocks,cfg);
            etaPredFac = physicalStep(etaFac(:,observer),dFac,cfg.dt);
            FetaFac = A0 + Lmap*JetaFac;
            Qfac = gammaEta*(Lmap*(cfg.ekf.qAcceleration*eye(2) + ...
                cfg.ekf.remoteCovarianceScale*parameterQdFac)*Lmap') + 1e-12*eye(4);
            PpredFac = symmetrizePSD(FetaFac*Pfac(:,:,observer)*FetaFac' + Qfac,1e-12);
            GpredFac = FetaFac*Gfac(:,:,observer) + Lmap*JallFac;
            [etaNewFac,PnewFac,GnewFac,lambdaAdd,qAdd,nisValue] = ...
                canonicalFactorMeasurementUpdates(etaPredFac,PpredFac,GpredFac, ...
                data,observer,k+1,cfg);
            etaFac(:,observer) = etaNewFac;
            Pfac(:,:,observer) = PnewFac;
            Gfac(:,:,observer) = GnewFac;
            packetLambda = packetLambda + lambdaAdd;
            packetq = packetq + qAdd;
            nis(k+1,observer) = nisValue;
            nisWindowSum = nisWindowSum + nisValue;
            nisWindowCount = nisWindowCount + 1;
            dHistory(:,k,observer) = dFast;
            etaHistory(:,k+1,observer) = etaFast(:,observer);
        end

        [shouldSync,score,reason] = observabilityAwareSyncDecision( ...
            packetLambda,Pglobal,t,lastSync,p,Nb,cfg);
        informationScore(k+1) = score;
        if shouldSync
            elapsed = max(t-lastSync,cfg.dt);
            alphaWindow = exp(-elapsed/cfg.collaborative.thetaTau);
            thetaPrior = thetaReference + alphaWindow*(thetaGS-thetaReference);
            Qglobal = zeros(nGlobal);
            for block = 1:Nb
                cols = (block-1)*p+(1:p);
                Qglobal(cols,cols) = gammaTheta(block)*cfg.ekf.Ptheta0* ...
                    (1-alphaWindow^2)*eye(p);
            end
            Pprior = symmetrizePSD(alphaWindow^2*Pglobal + Qglobal,1e-12);
            priorInformation = Pprior\eye(nGlobal);
            anchorVector = thetaAnchor(:);
            A = symmetrizePSD(priorInformation + packetLambda,1e-12);
            b = packetq + priorInformation*(thetaPrior(:)-anchorVector);
            baseCost = replayWindowMAPCost(data,cfg,facStartEta,facStartP, ...
                thetaAnchor,windowStartIndex,k+1,thetaPrior,Pprior);
            [thetaCandidate,Pcandidate,candidateCost,accepted] = ...
                acceptedJointGSStep(A,b,anchorVector,thetaReference, ...
                baseCost,data,cfg,facStartEta,facStartP,windowStartIndex,k+1, ...
                thetaPrior,Pprior);
            gsAccepted(k+1) = accepted;
            syncReason(k+1) = reason;
            gsCostBefore(k+1) = baseCost;
            gsCostAfter(k+1) = candidateCost;
            if accepted
                thetaGS = thetaCandidate;
                Pglobal = Pcandidate;
            else
                thetaGS = thetaPrior;
                Pglobal = Pprior;
            end

            for observer = 1:Nw
                owner = owners(observer);
                shadow = thetaAnchor;
                shadow(:,owner) = thetaLocal(:,observer);
                resetFast = thetaGS(:)-shadow(:);
                etaFast(:,observer) = etaFast(:,observer) + ...
                    cfg.collaborative.resetGain*Gfast(:,:,observer)*resetFast;
                gsResetNorm(k+1) = max(gsResetNorm(k+1), ...
                    norm(cfg.collaborative.resetGain*Gfast(:,:,observer)*resetFast));
                thetaLocal(:,observer) = thetaGS(:,owner);
                Pfast(1:4,1:4,observer) = cfg.collaborative.syncStateCovarianceInflation* ...
                    Pfast(1:4,1:4,observer);
                Pfast(5:end,5:end,observer) = Pglobal( ...
                    (owner-1)*p+(1:p),(owner-1)*p+(1:p));
                Pfast(1:4,5:end,observer) = 0;
                Pfast(5:end,1:4,observer) = 0;
                Pfast(:,:,observer) = symmetrizePSD(Pfast(:,:,observer),1e-12);
                Gfast(:,:,observer) = zeros(4,nGlobal);

                resetFactor = thetaGS(:)-thetaAnchor(:);
                etaFac(:,observer) = etaFac(:,observer) + ...
                    cfg.collaborative.resetGain*Gfac(:,:,observer)*resetFactor;
                Pfac(:,:,observer) = symmetrizePSD( ...
                    cfg.collaborative.syncStateCovarianceInflation*Pfac(:,:,observer),1e-12);
                Gfac(:,:,observer) = zeros(4,nGlobal);
                etaHistory(:,k+1,observer) = etaFast(:,observer);
            end
            thetaAnchor = thetaGS;
            facStartEta = etaFac;
            facStartP = Pfac;
            windowStartIndex = k+1;
            for block = 1:Nb
                thetaChange(k+1,block) = norm(thetaGS(:,block)-thetaReference(:,block));
                uploadTimes{block}(end+1) = t;
            end
            if nisWindowCount > 0
                matchedNIS = max(nisWindowSum/(nisWindowCount* ...
                    (1+2*cfg.measurement.positionEnabled)),1e-3);
                gammaTheta = min(max(gammaTheta*matchedNIS^ ...
                    cfg.collaborative.covMatchRate,cfg.collaborative.covMatchBounds(1)), ...
                    cfg.collaborative.covMatchBounds(2));
                gammaEta = min(max(gammaEta*matchedNIS^ ...
                    cfg.collaborative.stateCovMatchRate,cfg.collaborative.stateCovMatchBounds(1)), ...
                    cfg.collaborative.stateCovMatchBounds(2));
            end
            packetLambda(:) = 0; packetq(:) = 0;
            nisWindowSum = 0; nisWindowCount = 0;
            lastSync = t;
        end
    end

    for observer = 1:Nw
        shadow = thetaAnchor;
        shadow(:,owners(observer)) = thetaLocal(:,observer);
        dHistory(:,N,observer) = evaluateGlobal(etaFast(:,observer),shadow,cfg,0,"shared_block");
    end
    result.mode = "hybrid_fast_owner_canonical_factor_joint_gs";
    result.time = cfg.time;
    result.etaHat = etaHistory;
    result.dHat = dHistory;
    result.thetaChange = thetaChange;
    result.nis = nis;
    result.uploadTimes = uploadTimes;
    result.parameterUploads = sum(cellfun(@numel,uploadTimes));
    result.positionError = vectorRMSE(etaHistory(1:2,:,:),data.truth(1:2,:));
    result.velocityError = vectorRMSE(etaHistory(3:4,:,:),data.truth(3:4,:));
    result.accelerationError = vectorRMSE(dHistory,data.dTrue);
    result.positionRMSE = sqrt(mean(result.positionError.^2));
    result.velocityRMSE = sqrt(mean(result.velocityError.^2));
    result.accelerationRMSE = sqrt(mean(result.accelerationError.^2));
    result.finalPositionRMSE = result.positionError(end);
    result.parameterCovariance = Pglobal;
    result.thetaQScale = gammaTheta;
    result.stateQScale = gammaEta;
    result.gsAccepted = gsAccepted;
    result.gsCostBefore = gsCostBefore;
    result.gsCostAfter = gsCostAfter;
    result.gsResetNorm = gsResetNorm;
    result.informationScore = informationScore;
    result.syncReason = syncReason;
    result.positionAided = cfg.measurement.positionEnabled;
    result.nisDegreesOfFreedom = 1 + 2*cfg.measurement.positionEnabled;
    result.implementationNote = ["Canonical factor path, damped accepted GS step, and " ...
        "damped state reset; watcher state-copy cross-covariances remain omitted."];
end

function Pblocks = globalCovarianceBlocks(Pglobal,p,nBlocks)
    Pblocks = zeros(p,p,nBlocks);
    for block = 1:nBlocks
        cols = (block-1)*p+(1:p);
        Pblocks(:,:,block) = Pglobal(cols,cols);
    end
end

function [shouldSync,score,reason] = observabilityAwareSyncDecision( ...
    packetLambda,Pglobal,t,lastSync,p,nBlocks,cfg)
    elapsed = t-lastSync;
    scores = zeros(1,nBlocks);
    for block = 1:nBlocks
        cols = (block-1)*p+(1:p);
        Pblock = Pglobal(cols,cols);
        Lambdablock = packetLambda(cols,cols);
        % trace(P*Lambda)/p is the average prior-normalized information.
        scores(block) = max(real(trace(Pblock*Lambdablock))/p,0);
    end
    score = min(scores);
    if ~cfg.collaborative.observabilityAware
        shouldSync = elapsed >= cfg.communication.period-0.5*cfg.dt;
        if shouldSync, reason = "periodic"; else, reason = ""; end
        return;
    end
    enoughInformation = score >= cfg.collaborative.informationTriggerThreshold;
    minIntervalMet = elapsed >= cfg.communication.minInterval-0.5*cfg.dt;
    maxSilenceReached = elapsed >= cfg.communication.maxSilence-0.5*cfg.dt;
    shouldSync = (minIntervalMet && enoughInformation) || maxSilenceReached;
    if maxSilenceReached && ~(minIntervalMet && enoughInformation)
        reason = "max_silence";
    elseif shouldSync
        reason = "information";
    else
        reason = "";
    end
end

function [etaNew,thetaNew,Pnew,Gnew] = localFastMeasurementUpdates( ...
    etaPred,thetaPred,Ppred,Gpred,data,observer,sampleIndex,cfg)
    p = numel(thetaPred);
    [innovation,Heta,R] = bearingInnovation( ...
        data.bearings(observer,sampleIndex),etaPred,cfg.watchers.r(:,observer),cfg);
    H = [Heta zeros(1,p)];
    S = H*Ppred*H' + R;
    K = (Ppred*H')/S;
    x = [etaPred;thetaPred] + K*innovation;
    I = eye(4+p);
    P = (I-K*H)*Ppred*(I-K*H)' + K*R*K';
    G = (eye(4)-K(1:4)*Heta)*Gpred;
    if cfg.measurement.positionEnabled
        Hpos = [eye(2) zeros(2,2+p)];
        Rpos = cfg.measurement.positionSigma^2*eye(2);
        innovationPos = data.positionMeasurements(:,sampleIndex,observer)-x(1:2);
        Spos = Hpos*P*Hpos' + Rpos;
        Kpos = (P*Hpos')/Spos;
        x = x + Kpos*innovationPos;
        I = eye(4+p);
        P = (I-Kpos*Hpos)*P*(I-Kpos*Hpos)' + Kpos*Rpos*Kpos';
        G = (eye(4)-Kpos(1:4,:)*Hpos(:,1:4))*G;
    end
    etaNew = x(1:4);
    thetaNew = x(5:end);
    Pnew = symmetrizePSD(P,1e-12);
    Gnew = G;
end

function [etaNew,Pnew,Gnew,lambdaAdd,qAdd,nisValue] = ...
    canonicalFactorMeasurementUpdates(etaPred,Ppred,Gpred,data,observer,sampleIndex,cfg)
    nGlobal = size(Gpred,2);
    lambdaAdd = zeros(nGlobal); qAdd = zeros(nGlobal,1);
    [innovation,Heta,R] = bearingInnovation( ...
        data.bearings(observer,sampleIndex),etaPred,cfg.watchers.r(:,observer),cfg);
    S = Heta*Ppred*Heta' + R;
    K = (Ppred*Heta')/S;
    Htheta = Heta*Gpred;
    lambdaAdd = lambdaAdd + (Htheta'*Htheta)/S;
    qAdd = qAdd + Htheta'*(innovation/S);
    eta = etaPred + K*innovation;
    I = eye(4);
    P = (I-K*Heta)*Ppred*(I-K*Heta)' + K*R*K';
    G = (I-K*Heta)*Gpred;
    nisValue = innovation^2/S;
    if cfg.measurement.positionEnabled
        Hpos = [eye(2) zeros(2)];
        Rpos = cfg.measurement.positionSigma^2*eye(2);
        innovationPos = data.positionMeasurements(:,sampleIndex,observer)-eta(1:2);
        Spos = Hpos*P*Hpos' + Rpos;
        Kpos = (P*Hpos')/Spos;
        HthetaPos = Hpos*G;
        lambdaAdd = lambdaAdd + HthetaPos'*(Spos\HthetaPos);
        qAdd = qAdd + HthetaPos'*(Spos\innovationPos);
        eta = eta + Kpos*innovationPos;
        P = (I-Kpos*Hpos)*P*(I-Kpos*Hpos)' + Kpos*Rpos*Kpos';
        G = (I-Kpos*Hpos)*G;
        nisValue = nisValue + innovationPos'*(Spos\innovationPos);
    end
    etaNew = eta;
    Pnew = symmetrizePSD(P,1e-12);
    Gnew = G;
end

function [thetaBest,Pbest,costBest,accepted] = acceptedJointGSStep( ...
    A,b,anchorVector,thetaReference,baseCost,data,cfg,etaStart,Pstart, ...
    startIndex,endIndex,thetaPrior,Pprior)
    p = cfg.dnn.arch.nTheta; Nb = cfg.Nblocks;
    thetaBest = reshape(anchorVector,p,Nb);
    Pbest = Pprior;
    costBest = baseCost;
    accepted = false;
    diagonalScale = diag(max(diag(A),1e-12));
    for damping = cfg.collaborative.lmDampingSchedule
        Adamped = symmetrizePSD(A + damping*diagonalScale,1e-12);
        rawDelta = Adamped\b;
        for scale = cfg.collaborative.lineSearchScales
            delta = scale*rawDelta;
            deltaNorm = norm(delta);
            if deltaNorm > cfg.collaborative.infoStepLimit
                delta = cfg.collaborative.infoStepLimit*delta/deltaNorm;
            end
            thetaCandidate = reshape(anchorVector+delta,p,Nb);
            deviation = thetaCandidate-thetaReference;
            thetaCandidate = thetaReference + max(min(deviation, ...
                cfg.dnn.parameterDeviationLimit),-cfg.dnn.parameterDeviationLimit);
            candidateCost = replayWindowMAPCost(data,cfg,etaStart,Pstart, ...
                thetaCandidate,startIndex,endIndex,thetaPrior,Pprior);
            if candidateCost < baseCost-1e-9
                thetaBest = thetaCandidate;
                Pbest = symmetrizePSD(Adamped\eye(size(A)), ...
                    cfg.collaborative.parameterCovarianceFloor);
                costBest = candidateCost;
                accepted = true;
                return;
            end
        end
    end
end

function cost = replayWindowMAPCost(data,cfg,etaStart,Pstart,theta, ...
    startIndex,endIndex,thetaPrior,Pprior)
% Replays only each watcher's own local data; no raw measurement sharing.
    Nw = cfg.Nw;
    eta = etaStart; P = Pstart;
    cost = 0;
    Lmap = [0.5*cfg.dt^2*eye(2);cfg.dt*eye(2)];
    A0 = [eye(2) cfg.dt*eye(2);zeros(2) eye(2)];
    Pblocks = globalCovarianceBlocks(Pprior,cfg.dnn.arch.nTheta,cfg.Nblocks);
    for sample = startIndex+1:endIndex
        for observer = 1:Nw
            [d,Jeta,~,parameterQd] = evaluateGlobalAllJacobians( ...
                eta(:,observer),theta,Pblocks,cfg);
            etaPred = physicalStep(eta(:,observer),d,cfg.dt);
            Feta = A0 + Lmap*Jeta;
            Qeta = Lmap*(cfg.ekf.qAcceleration*eye(2) + ...
                cfg.ekf.remoteCovarianceScale*parameterQd)*Lmap' + 1e-12*eye(4);
            Ppred = symmetrizePSD(Feta*P(:,:,observer)*Feta' + Qeta,1e-12);
            [innovation,Heta,R] = bearingInnovation( ...
                data.bearings(observer,sample),etaPred,cfg.watchers.r(:,observer),cfg);
            S = Heta*Ppred*Heta' + R;
            cost = cost + 0.5*(innovation^2/S + log(max(S,1e-16)));
            K = (Ppred*Heta')/S;
            I = eye(4);
            eta(:,observer) = etaPred + K*innovation;
            P(:,:,observer) = (I-K*Heta)*Ppred*(I-K*Heta)' + K*R*K';
            if cfg.measurement.positionEnabled
                Hpos = [eye(2) zeros(2)];
                Rpos = cfg.measurement.positionSigma^2*eye(2);
                innovationPos = data.positionMeasurements(:,sample,observer)-eta(1:2,observer);
                Spos = Hpos*P(:,:,observer)*Hpos' + Rpos;
                cost = cost + 0.5*(innovationPos'*(Spos\innovationPos) + ...
                    log(max(det(Spos),1e-16)));
                Kpos = (P(:,:,observer)*Hpos')/Spos;
                eta(:,observer) = eta(:,observer) + Kpos*innovationPos;
                P(:,:,observer) = (I-Kpos*Hpos)*P(:,:,observer)*(I-Kpos*Hpos)' + ...
                    Kpos*Rpos*Kpos';
            end
        end
    end
    deltaPrior = theta(:)-thetaPrior(:);
    cost = cost + 0.5*deltaPrior'*(Pprior\deltaPrior);
end

function [etaPred,Feta,Qeta] = predictCollaborativeState(eta,dHat,Jeta,parameterQd,cfg)
    dt = cfg.dt;
    Lmap = [0.5*dt^2*eye(2);dt*eye(2)];
    etaPred = physicalStep(eta,dHat,dt);
    A0 = [eye(2) dt*eye(2);zeros(2) eye(2)];
    Feta = A0 + Lmap*Jeta;
    Qeta = Lmap*(cfg.ekf.qAcceleration*eye(2) + ...
        cfg.ekf.remoteCovarianceScale*parameterQd)*Lmap' + 1e-12*eye(4);
end

function [d,Jeta,Jall,parameterQd] = evaluateGlobalAllJacobians(eta,thetaAll,Ptheta,cfg)
    p = cfg.dnn.arch.nTheta;
    d = zeros(2,1);
    Jeta = zeros(2,4);
    Jall = zeros(2,p*cfg.Nblocks);
    parameterQd = zeros(2);
    for owner = 1:cfg.Nblocks
        [dj,Aj,Bj] = blockOutput(eta,thetaAll(:,owner),cfg);
        cols = (owner-1)*p+(1:p);
        d = d + dj;
        Jeta = Jeta + Aj;
        Jall(:,cols) = Bj;
        parameterQd = parameterQd + Bj*Ptheta(:,:,owner)*Bj';
    end
end

function [d,Jeta,Jall,parameterQd] = evaluateGlobalSelectedJacobians( ...
    eta,thetaAll,Pselected,selectedIdx,cfg)
    pSelected = numel(selectedIdx);
    d = zeros(2,1);
    Jeta = zeros(2,4);
    Jall = zeros(2,pSelected*cfg.Nblocks);
    parameterQd = zeros(2);
    for owner = 1:cfg.Nblocks
        [dj,Aj,Bj] = blockOutput(eta,thetaAll(:,owner),cfg);
        Bselected = Bj(:,selectedIdx);
        cols = (owner-1)*pSelected+(1:pSelected);
        d = d + dj;
        Jeta = Jeta + Aj;
        Jall(:,cols) = Bselected;
        parameterQd = parameterQd + Bselected*Pselected(:,:,owner)*Bselected';
    end
end

function P = blockDiagonalCovariance(Pblocks)
    nBlocks = size(Pblocks,3);
    p = size(Pblocks,1);
    P = zeros(p*nBlocks);
    for block = 1:nBlocks
        cols = (block-1)*p+(1:p);
        P(cols,cols) = Pblocks(:,:,block);
    end
end

function [xPred,F,Q,dHat] = predictNominal(x,cfg)
    dt = cfg.dt;
    dHat = [0;0];
    xPred = physicalStep(x,dHat,dt);
    F = [eye(2) dt*eye(2);zeros(2) eye(2)];
    L = [0.5*dt^2*eye(2);dt*eye(2)];
    Q = L*(cfg.ekf.qAcceleration*eye(2))*L'+1e-12*eye(4);
end

function [xPred,F,Q,dHat] = predictLearned( ...
    x,owner,thetaCache,Pcache,cfg,mode)
    p = cfg.dnn.arch.nTheta; dt = cfg.dt;
    eta = x(1:4); thetaOwner = x(5:end);
    Lmap = [0.5*dt^2*eye(2);dt*eye(2)];

    if mode == "local_only"
        [dHat,Jeta,Jtheta] = blockOutput(eta,thetaOwner,cfg);
        remoteQd = zeros(2);
    else
        thetaCache(:,owner) = thetaOwner;
        [dHat,Jeta,Jtheta,remoteQd] = evaluateGlobal( ...
            eta,thetaCache,cfg,owner,"shared_block",Pcache);
    end

    xPred = x;
    xPred(1:4) = physicalStep(eta,dHat,dt);
    A0 = [eye(2) dt*eye(2);zeros(2) eye(2)];
    F = eye(4+p);
    F(1:4,1:4) = A0+Lmap*Jeta;
    F(1:4,5:end) = Lmap*Jtheta;

    Q = zeros(4+p);
    Q(1:4,1:4) = Lmap*(cfg.ekf.qAcceleration*eye(2)+ ...
        cfg.ekf.remoteCovarianceScale*remoteQd)*Lmap'+1e-12*eye(4);
    Q(5:end,5:end) = cfg.ekf.qTheta*eye(p);
end

function [d,Jeta,Jowner,remoteQd] = evaluateGlobal( ...
    eta,thetaAll,cfg,owner,mode,Pcache)
    if nargin < 6, Pcache = []; end
    d = zeros(2,1); Jeta = zeros(2,4);
    Jowner = zeros(2,cfg.dnn.arch.nTheta);
    remoteQd = zeros(2);
    if mode ~= "shared_block", error('Unexpected global evaluation mode.'); end
    for j = 1:cfg.Nblocks
        [dj,Aj,Bj] = blockOutput(eta,thetaAll(:,j),cfg);
        d = d+dj;
        Jeta = Jeta+Aj;
        if j == owner
            Jowner = Bj;
        elseif ~isempty(Pcache)
            remoteQd = remoteQd+Bj*Pcache(:,:,j)*Bj';
        end
    end
end

function [d,Jeta,Jtheta] = blockOutput(eta,theta,cfg)
    [h,DhDeta,cache] = hiddenFeature(eta,theta,cfg);
    arch = cfg.dnn.arch;
    Wout = reshape(theta(arch.idxOutputW),2,arch.width);
    d = Wout*h;
    Jeta = Wout*DhDeta;

    Jtheta = zeros(2,arch.nTheta);
    S = cell(arch.nHidden,1);
    S{arch.nHidden} = Wout.*(1-cache.z{end}.^2)';
    for ell = arch.nHidden:-1:2
        W = cache.W{ell};
        S{ell-1} = (S{ell}*W).*(1-cache.z{ell}.^2)';
    end
    for ell = 1:arch.nHidden
        zPrev = cache.z{ell};
        Jtheta(:,arch.hidden(ell).idxW) = kron(zPrev',S{ell});
        Jtheta(:,arch.hidden(ell).idxb) = S{ell};
    end
    Jtheta(:,arch.idxOutputW) = kron(h',eye(2));
end

function [h,DhDeta,cache] = hiddenFeature(eta,theta,cfg)
    arch = cfg.dnn.arch;
    z = cell(arch.nHidden+1,1);
    W = cell(arch.nHidden,1);
    z{1} = eta(:)./cfg.dnn.inputScale;
    D = diag(1./cfg.dnn.inputScale);
    for ell = 1:arch.nHidden
        W{ell} = reshape(theta(arch.hidden(ell).idxW), ...
            arch.width,arch.hidden(ell).nIn);
        b = theta(arch.hidden(ell).idxb);
        z{ell+1} = tanh(W{ell}*z{ell}+b);
        D = diag(1-z{ell+1}.^2)*W{ell}*D;
    end
    h = z{end};
    DhDeta = D;
    cache.z = z;
    cache.W = W;
end

function [innovation,H,R] = bearingInnovation(z,eta,watcherR,cfg)
    dr = eta(1:2)-watcherR;
    rho2 = max(dr'*dr,1e-8);
    predicted = atan2(dr(2),dr(1));
    innovation = wrapAngle(z-predicted);
    H = zeros(1,numel(eta));
    H(1:2) = [-dr(2) dr(1)]/rho2;
    R = cfg.measurement.sigma^2;
end

function xNext = physicalStep(x,d,dt)
    xNext = x;
    xNext(1:2) = x(1:2)+dt*x(3:4)+0.5*dt^2*d;
    xNext(3:4) = x(3:4)+dt*d;
end

function d = trueResidual(eta,cfg)
    d = cfg.spiral.velocityGain*(desiredVelocity(eta(1:2),cfg)-eta(3:4));
end

function vDesired = desiredVelocity(r,cfg)
    rho = max(norm(r),0.25);
    uR = r/rho;
    uT = [-uR(2);uR(1)];
    radial = cfg.spiral.radialRate*(1-rho/cfg.spiral.radiusGoal);
    tangential = cfg.spiral.angularRate*rho;
    vDesired = radial*uR+tangential*uT;
end

function e = vectorRMSE(estimate,truth)
    N = size(estimate,2); Nw = size(estimate,3);
    e = zeros(1,N);
    for k = 1:N
        err = reshape(estimate(:,k,:),size(estimate,1),Nw)-truth(:,k);
        e(k) = sqrt(mean(sum(err.^2,1)));
    end
end

function P = symmetrizePSD(P,floorValue)
    P = (P+P')/2;
    [V,D] = eig(P);
    D = diag(max(real(diag(D)),floorValue));
    P = real(V*D*V');
    P = (P+P')/2;
end

function a = wrapAngle(a)
    a = atan2(sin(a),cos(a));
end

function description = architectureDescription(cfg)
    description = struct;
    description.name = "block-structured global-head DNN";
    description.equation = "d_hat(eta)=sum_i W_i^out h_i(eta;theta_i_hidden)";
    description.ownerBlock = "all hidden-layer weights/biases plus W_i^out";
    description.outputActivation = "linear";
    description.globalBias = "omitted to remove duplicated-bias ambiguity";
    description.hiddenLayerCount = cfg.dnn.hiddenLayers;
    description.hiddenWidthPerBlock = cfg.dnn.width;
    description.parametersPerBlock = cfg.dnn.arch.nTheta;
    description.nParameterBlocks = cfg.Nblocks;
    description.globalFeatureWidth = cfg.Nblocks*cfg.dnn.width;
    description.communication = cfg.communication.mode;
end

function summary = makeSummary(out)
    names = ["Nominal EKF";"Local independent block"; ...
        "Shared block-structured global DNN"];
    cases = {out.nominal,out.localOnly,out.sharedBlock};
    positionRMSE = zeros(3,1); velocityRMSE = zeros(3,1);
    accelerationRMSE = zeros(3,1); finalPositionRMSE = zeros(3,1);
    parameterUploads = zeros(3,1);
    for k = 1:3
        positionRMSE(k) = cases{k}.positionRMSE;
        velocityRMSE(k) = cases{k}.velocityRMSE;
        accelerationRMSE(k) = cases{k}.accelerationRMSE;
        finalPositionRMSE(k) = cases{k}.finalPositionRMSE;
        parameterUploads(k) = cases{k}.parameterUploads;
    end
    summary = table(names,positionRMSE,velocityRMSE,accelerationRMSE, ...
        finalPositionRMSE,parameterUploads,'VariableNames', ...
        {'caseName','positionRMSE','velocityRMSE','accelerationRMSE', ...
        'finalPositionRMSE','parameterUploads'});
end

function figs = plotResults(out)
    t = out.cfg.time;
    cases = {out.nominal,out.localOnly,out.sharedBlock};
    labels = {'nominal','local independent block','shared block-structured DNN'};
    colors = lines(3);
    figs.errors = figure('Name','Block-structured global-head DNN EKF');
    tiledlayout(3,1,'TileSpacing','compact');
    titles = {'position estimation error','velocity estimation error', ...
        'acceleration approximation error'};
    fields = {'positionError','velocityError','accelerationError'};
    units = {'RMSE [m]','RMSE [m/s]','RMSE [m/s^2]'};
    for row = 1:3
        nexttile; hold on; grid on;
        for c = 1:3, plot(t,cases{c}.(fields{row}),'LineWidth',1.2, ...
                'Color',colors(c,:)); end
        title(titles{row}); ylabel(units{row});
        if row == 1, legend(labels,'Location','best'); end
    end
    xlabel('time [s]');

    dShared = mean(out.sharedBlock.dHat,3);
    dLocal = mean(out.localOnly.dHat,3);
    figs.acceleration = figure('Name','Global-head acceleration reconstruction');
    tiledlayout(2,1,'TileSpacing','compact');
    for axisID = 1:2
        nexttile; hold on; grid on;
        plot(t,out.trueAcceleration(axisID,:),'k','LineWidth',1.5);
        plot(t,dLocal(axisID,:),'Color',colors(2,:),'LineWidth',1.0);
        plot(t,dShared(axisID,:),'Color',colors(3,:),'LineWidth',1.2);
        ylabel(sprintf('d_%c [m/s^2]','x'+axisID-1));
        if axisID == 1
            legend({'truth','local','shared global block'}, ...
                'Location','best');
        end
    end
    xlabel('time [s]');

    figs.communication = figure('Name','Block-parameter sharing events');
    hold on; grid on;
    for i = 1:out.cfg.Nw
        ti = out.sharedBlock.uploadTimes{i};
        if ~isempty(ti), scatter(ti,i*ones(size(ti)),24,'filled'); end
    end
    xlabel('time [s]'); ylabel('owner watcher');
    if out.cfg.communication.mode == "periodic_60s"
        title('periodic 60 s full block uploads');
    else
        title(sprintf('%s full block uploads', ...
            strrep(char(out.cfg.communication.mode),'_',' ')));
    end
    ylim([0.5 out.cfg.Nw+0.5]);
end
