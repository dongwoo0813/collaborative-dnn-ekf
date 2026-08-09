function cfg = config_step04_GS_DNN_EKF()
%{
Function:
    config_step04_GS_DNN_EKF.m

Purpose:
    Create the configuration for Step 04 GS-assisted collaborative DNN-EKF.

    Step 04 starts from the validated Step 03 local DNN-EKF configuration
    and activates a simple ground-station repository for DNN branch sharing.

    In Step 03, watcher m predicts using only its own local branch:

        d_hat_m(eta) = W_m phi_m(eta).

    In Step 04, watcher m will eventually predict using a composite
    residual assembled from GS-shared branch records:

        d_hat_comp,m(eta)
            =
            sum_{j=1}^{Nw} W_{j|m} phi_j(eta).

    where

        W_{m|m}

    is watcher m's locally estimated branch, and

        W_{j|m}, j ~= m

    are nonlocal branch copies received from the ground station.

Inputs:
    None.

Outputs:
    cfg - Simulation configuration structure.

Notes:
    - This file only defines the Step 04 configuration.
    - The actual GS repository, upload, broadcast, and composite prediction
      logic will be added in later Step 04 files.
    - For the first GS-sharing experiment, residualAmp is set to 5e-4
      because the oracle residual test showed that this case has meaningful
      residual-compensation benefit.
%}

    % Start from the validated Step 03 local DNN-EKF configuration.
    cfg = config_step03_local_DNN_EKF();

    % ---------------------------------------------------------------------
    % Step label and communication mode
    % ---------------------------------------------------------------------
    cfg.step.name = "step04_GS_DNN_EKF";

    cfg.estimator.type = "GS_DNN_EKF";
    cfg.comm.mode = "GS";

    % ---------------------------------------------------------------------
    % Residual scenario
    % ---------------------------------------------------------------------
    % residualAmp = 1e-4 was too weak. Oracle residual compensation improved
    % position RMSE by only about 1 percent.
    %
    % residualAmp = 5e-4 gave about 18 percent oracle improvement, making it
    % a more meaningful scenario for testing whether DNN residual learning
    % and branch sharing can close part of the physical-oracle gap.
    cfg.truth.useResidual = true;

    
    % -------------------------------------------------------------------------
    % Truth residual model
    % -------------------------------------------------------------------------
    % Step 09-B.2:
    % residualFamily is the preferred interface for selecting the truth
    % residual benchmark. residualModel is kept as a backward-compatible alias
    % for older scripts and checks.
    cfg.truth.useResidual = true;
    
    % cfg.truth.residualFamily = "simple_branchwise";
    % cfg.truth.residualModel  = "branchwise";
    
    cfg.truth.residualFamily   = "coupled_nonlinear";
    cfg.truth.residuaModel = "branchwise";
    
    % Current validated residual amplitude used in Step 03--08 comparisons.
    cfg.truth.residualAmp = 5e-4;


    % ---------------------------------------------------------------------
    % DNN residual prediction settings
    % ---------------------------------------------------------------------
    % Step 04 should use realizable learned DNN residuals, not oracle
    % residuals.
    cfg.dnn.useResidualInPrediction = true;
    cfg.dnn.predictionResidualSource = "GS_composite";

    % Start with full residual injection. The Step 03 sweep at residualAmp
    % = 5e-4 showed that betaDNN = 1 performed best for local DNN-EKF.
    cfg.dnn.residualInjectionGain = 1.0;

    % For fair comparison with Step 02 and Step 03 sweeps, initialize theta
    % exactly at zero.
    %
    % This assumes initLocalDNNEKF.m has been fixed so that theta0_std = 0
    % does not call randn.
    cfg.dnn.theta0_std = 0.0;

    % ---------------------------------------------------------------------
    % Ground station sharing settings
    % ---------------------------------------------------------------------
    cfg.gs.enabled = true;


    cfg.gs.bootstrapUpload = true;

    % ---------------------------------------------------------------------
    % Step 09-J bearing-FIM gate metadata settings
    % ---------------------------------------------------------------------
    % Step 09-J.1 only collects passive cumulative geometry support
    % OmegaBar_j. IMPORTANT: this is a legacy geometry heuristic, not the
    % Fisher information obtained from the measurement likelihood.  In
    % particular it omits H, R, dynamics, and DNN-parameter sensitivity.
    % It must not be interpreted as a rigorous information fusion rule for
    % additive residual components.  Use
    % run_step09J6_rigorous_information_diagnostic for the finite-horizon
    % likelihood-based diagnostic.
    % The actual bearing_fim_gated composite residual remains
    % disabled until Step 09-J.3.
    cfg.gs.fimGate.enabled = false;

    % EMA rate for
    %   OmegaBar_k = (1-lambdaOmega) OmegaBar_{k-1} + lambdaOmega Omega_k.
    cfg.gs.fimGate.lambdaOmega = 0.02;

    % Future regularization for OmegaSigma inversion in bearing_fim_gated mode.
    cfg.gs.fimGate.epsilon = 1e-6;

    % Keep false by default so OmegaBar follows the exact EMA recursion.
    cfg.gs.fimGate.normalizeTrace = false;

    % Frame label for residual acceleration and OmegaBar. The present 2-D
    % benchmark uses the inertial simulation frame.
    cfg.gs.fimGate.outputFrame = "inertial";
    cfg.gs.fimGate.accumulationMode = "ema";

    % Output-information fusion for redundant local DNN experts.  Under
    % the local Gaussian approximation, each branch supplies
    % Sigma_d=Jtheta*P(theta|eta)*Jtheta' and Omega_d=inv(Sigma_d).
    % No residual prior is added in the present full-rank simulation.
    cfg.gs.outputInfoFusion.minOutputVariance = 1e-20;
    cfg.gs.outputInfoFusion.approxErrorVariance = 0;
    cfg.gs.outputInfoFusion.rankTolerance = 1e-10;
    cfg.gs.outputInfoFusion.requireFullRank = true;

    % Geometry-information weighted additive DNN.  Eigenvalues smaller
    % than this fraction of a branch's strongest information direction are
    % treated as numerically unobservable.  Weights are normalized within
    % each branch; they are never normalized across watchers.
    cfg.gs.fimWeightedAdditive.relativeEigenvalueFloor = 1e-10;




    % ---------------------------------------------------------------------
    % GS upload / broadcast policy
    % ---------------------------------------------------------------------
    % Upload local branch estimate from each watcher to GS whenever the
    % watcher has completed a measurement update.
    cfg.gs.uploadMode = "after_measurement_update";
    % cfg.gs.uploadMode = "every_step";

    % Options:
    %   "after_measurement_update"  : upload after every local EKF update
    %   "event_contribution_change" : upload only if Delta_i and dwell pass
    %   "every_step"                : upload every step
    %   "never"                     : no upload after bootstrap

    cfg.gs.broadcastMode = "every_step";

    % ---------------------------------------------------------------------
    % Step 05 event-triggered upload parameters
    % ---------------------------------------------------------------------
    cfg.gs.eventDeltaThreshold = 1e-14;     % threshold for branch output change Delta_i
    cfg.gs.eventDwellSteps = 20;            % minimum steps between uploads
    cfg.gs.eventRequireMeasurement = true;  % require measurement update before upload


    % Broadcast the GS branch library to all watchers every time step.
    %
    % This is intentionally simple for Step 04a/04b. Later this can be
    % changed to event-triggered or periodic broadcast.
    cfg.gs.broadcastMode = "every_step";

    % Deterministic nonlocal branch use:
    %   false means nonlocal branch uncertainty is not yet propagated into
    %   watcher covariance.
    %
    % Later we can add covariance aging/inflation from received
    % P_theta_jtheta_j.
    cfg.gs.useNonlocalBranchCovariance = false;

    % No arbitrary Qnonlocal scale is used.
    %
    % The nonlocal covariance magnitude should come from physically or
    % statistically meaningful uncertainty sources:
    %
    %   1. GS branch parameter covariance Ptheta_j
    %   2. GS covariance aging / stale covariance inflation
    %   3. GS acceptance margin
    %   4. residual approximation covariance SresNonlocal
    %   5. Young-inequality coefficients
    %
    % Therefore, cfg.gs.nonlocalCovarianceScale is intentionally not used.
    cfg.gs.SresNonlocal = 0.0;
    cfg.gs.youngMode = "uniform";


    
    % Initial GS branch record status.
    cfg.gs.initialStatus = "empty";

    % Version counter starts at zero.
    cfg.gs.initialVersion = 0;

    % Optional stale-time setting for later.
    cfg.gs.maxStaleTime = Inf;


    % -------------------------------------------------------------------------
    % Step 09-E.1: GS nonlocal branch weighting
    % -------------------------------------------------------------------------
    % "none"   : original GS behavior, all active nonlocal branches use weight 1.
    % "scalar" : all active nonlocal branches use cfg.gs.nonlocalWeight.
    cfg.gs.nonlocalWeightMode = "none";
    cfg.gs.nonlocalWeight = 1.0;
    cfg.gs.nonlocalWeightMin = 0.0;
    cfg.gs.nonlocalWeightMax = 2.0;



    % Option to use block matrix for covariance propagation
    cfg.ekf.useBlockCovPrediction = true;



    % -------------------------------------------------------------------------
    % Measurement availability mode
    % -------------------------------------------------------------------------
    % Step 05-D:
    % Make the current measurement-availability assumption explicit.
    %
    % Current assumption:
    %   Every watcher has a valid angle-only measurement at every EKF update step.
    %
    % This corresponds to:
    %   delta_i^m(k) = 1
    %
    % Future FOV mode should switch these to:
    %   cfg.meas.availabilityMode = "fov";
    %   cfg.fov.enabled = true;
    %
    % Do not activate FOV mode yet.
    cfg.meas.availabilityMode = "always";
    cfg.fov.enabled = false;

    % -------------------------------------------------------------------------
    % FOV measurement-availability configuration
    % -------------------------------------------------------------------------
    % Step 08-A.1:
    % Add FOV configuration fields without activating FOV mode.
    %
    % Current build:
    %   cfg.meas.availabilityMode = "always";
    %   cfg.fov.enabled = false;
    %
    % Therefore, these fields should not affect the current simulation result.
    % They only prepare the interface for future FOV-based measurement gating.
    %
    % Future FOV condition:
    %   dot(r_ti^I / ||r_ti^I||, b_i^I) >= cos(cfg.fov.halfAngleRad)
    %
    % where:
    %   r_ti^I : relative position from watcher i to the target
    %   b_i^I  : camera boresight direction expressed in inertial coordinates
    %
    % Notes:
    %   - halfAngleDeg is the camera half-cone angle.
    %   - rhoMin/rhoMax are optional range gates.
    %   - Set rhoMin = 0 and rhoMax = Inf to disable range gating.
    %   - FOV mode remains guarded in fovAvailable.m until Step 08-A.2.
    cfg.fov.halfAngleDeg = 20.0;                 % camera half-cone angle [deg]
    cfg.fov.halfAngleRad = deg2rad(cfg.fov.halfAngleDeg);
    
    cfg.fov.rhoMin = 0.0;                         % minimum valid range [m]
    cfg.fov.rhoMax = Inf;                         % maximum valid range [m]
    
    cfg.fov.guardUnimplementedMode = true;        % keep FOV mode guarded for now

    cfg.fov.boresightMode = "target_pointing";    % safe default for debugging
    cfg.fov.referencePoint_I = zeros(cfg.dim,1);  % nominal pointing center [m]




    % -------------------------------------------------------------------------
    % Residual branch model
    % -------------------------------------------------------------------------
    % Default old model:
    %     fixed_feature_lip
    %
    % New trainable MLP model:
    %     mlp_general
    cfg.dnn.branchModel = "fixed_feature_lip";

    % Fixed-feature LIP branch size.
    cfg.dnn.nFeatureBranch = 2*cfg.dim + 5;
    cfg.dnn.nThetaBranch = cfg.dim * cfg.dnn.nFeatureBranch;

    % General MLP branch defaults.
    cfg.dnn.mlp.hiddenSizes = [8 8 6];
    cfg.dnn.mlp.activations = ["softplus", "softplus", "tanh"];
    cfg.dnn.mlp.inputMode = "eta_phase";
    cfg.dnn.mlp.rScale = 1000.0;
    cfg.dnn.mlp.vScale = 0.1;







end 
