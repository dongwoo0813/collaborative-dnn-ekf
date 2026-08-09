function results = simulate_GS_DNN_EKF(cfg)
%{
Function:
    simulateGSDNNEKF.m

Purpose:
    Run Step 04 GS-assisted collaborative block-structured DNN-EKF
    simulation.

    This simulation extends Step 03 local DNN-EKF by adding a simple
    ground-station branch repository:

        1. Each watcher estimates its own local augmented state

               X_i = [eta_i; theta_i].

        2. Each watcher uploads only its local branch estimate theta_i and
           P_theta_i theta_i to the GS.

        3. The GS stores one branch record per watcher.

        4. The GS broadcasts valid branch records to all watchers.

        5. Watcher m predicts using the composite residual

               d_comp,m(eta)
                   =
                   d_m(eta; theta_m^local)
                   + sum_{j ~= m} d_j(eta; theta_{j|m}^{GS}).

    Nonlocal branch parameters are NOT appended to the local EKF state.
    They are used only as cached deterministic branch copies in the mean
    prediction for this first Step 04a implementation.

Inputs:
    cfg - configuration from config_step04_GS_DNN_EKF.m

Outputs:
    results - simulation log structure

Notes:
    - This is Step 04a.
    - GS accepts every upload.
    - Bootstrap upload is performed at t0 so every branch has a valid GS
      record from the start. Since theta0_std = 0, these initial branches
      contribute zero residual, but the communication/cache structure is
      active immediately.
    - Nonlocal branch covariance is stored but not injected into watcher.P.
    - Later Step 04b can add nonlocal branch covariance injection.
%}

    dim     = cfg.dim;
    Nw      = cfg.Nw;
    N       = cfg.N;
    time    = cfg.time;

    % Step 10-A.2 deterministic replay.  A replay supplies the realized
    % truth trajectory and watcher kinematics from a previous controller
    % run.  The EKF still runs normally; only the plant/controller path is
    % frozen so estimator comparisons cannot change the geometry.
    replayEnabled = isfield(cfg,"replay") && isfield(cfg.replay,"enabled") && ...
        logical(cfg.replay.enabled);
    if replayEnabled
        validateReplayTrajectory(cfg.replay, 2*dim, dim, N, Nw);
    end

    % ---------------------------------------------------------------------
    % True target state
    % ---------------------------------------------------------------------
    etaTrue = zeros(2*dim, N);
    etaTrue(:,1) = [cfg.target.r0; cfg.target.v0];
    if replayEnabled
        etaTrue(:,1) = cfg.replay.etaTrue(:,1);
    end

    trueResidualLog = zeros(dim, N);
    trueResidualLog(:,1) = computeTrueResidualForLog(time(1), etaTrue(:,1), cfg);

    % ---------------------------------------------------------------------
    % Watcher truth states
    % ---------------------------------------------------------------------
    watcherTruth = initWatcherTruthArray(cfg);

    % ---------------------------------------------------------------------
    % Initialize local DNN-EKFs
    % ---------------------------------------------------------------------
    watchers = initLocalDNNEKF(1, etaTrue(:,1), cfg);

    for i = 2:Nw
        watchers(i) = initLocalDNNEKF(i, etaTrue(:,1), cfg);
    end

    watchers = watchers(:);

    nEta = watchers(1).nEta;
    nTheta = watchers(1).nTheta;
    nX = watchers(1).nX;

    % ---------------------------------------------------------------------
    % Initialize the repository/cache backend.  The legacy GS path uses one
    % global repository.  The Step 10-B P2P path keeps one repository-shaped
    % cache per watcher; only ring-neighbor branch records are copied into a
    % given cache.
    % ---------------------------------------------------------------------
    gsRepo = initGSRepository(cfg);
    p2pEnabled = isP2PCommunication(cfg);
    peerRepos = cell(max(Nw,1),1);
    if p2pEnabled
        for i = 1:Nw
            peerRepos{i} = initGSRepository(cfg);
            peerRepos{i}.enabled = true;
            peerRepos{i}.architecture = "p2p_ring";
        end
    end

    % ---------------------------------------------------------------------
    % Step 04a bootstrap:
    %
    % Upload each watcher's initial local branch to GS, then broadcast the
    % full library to every watcher.
    %
    % With theta0_std = 0, this does not change the initial residual mean.
    % It only activates the GS branch-cache structure.
    %
    % cfg.gs.bootstrapUpload controls whether this initialization is used.
    % Default: true.
    % ---------------------------------------------------------------------
    t0 = time(1);

    doBootstrapUpload = true;

    if isfield(cfg, "gs") && isfield(cfg.gs, "bootstrapUpload")
        doBootstrapUpload = logical(cfg.gs.bootstrapUpload);
    end

    if doBootstrapUpload && ~p2pEnabled

        for i = 1:Nw
            [gsRepo, uploadPacket] = uploadLocalBranchToGS(gsRepo, watchers(i), t0, cfg);
            watchers(i).lastGSUpload = uploadPacket;
        end

        for i = 1:Nw
            [watchers(i), broadcastPacket] = broadcastGSRepositoryToWatcher( ...
                gsRepo, watchers(i), t0, cfg);
            watchers(i).lastGSBroadcast = broadcastPacket;
        end

    elseif doBootstrapUpload
        % Bootstrap only to ring neighbors; there is no global repository.
        for i = 1:Nw
            neighbors = getP2PNeighbors(i,Nw);
            [peerRepos{i}, uploadPacket] = uploadLocalBranchToGS( ...
                peerRepos{i},watchers(i),t0,cfg);
            for nn = neighbors(:).'
                peerRepos{nn}.branch(i) = peerRepos{i}.branch(i);
            end
            watchers(i).lastGSUpload = uploadPacket;
        end
    end

    if doBootstrapUpload && p2pEnabled
        for i = 1:Nw
            [watchers(i), broadcastPacket] = broadcastGSRepositoryToWatcher( ...
                peerRepos{i},watchers(i),t0,cfg);
            watchers(i).lastGSBroadcast = broadcastPacket;
        end
    end

    % ---------------------------------------------------------------------
    % Log allocation
    % ---------------------------------------------------------------------
    etaHatLog = zeros(nEta, N, Nw);
    xhatAugLog = zeros(nX, N, Nw);
    thetaHatLog = zeros(nTheta, N, Nw);

    PdiagLog = zeros(nX, N, Nw);
    PdiagEtaLog = zeros(nEta, N, Nw);
    PdiagThetaLog = zeros(nTheta, N, Nw);


    dnnResidualLog = zeros(dim, N, Nw);

    % DNN residual evaluated at the true state input.
    % This separates pure function-approximation error from state-estimation
    % input error.
    dnnResidualAtTrueEtaLog = zeros(dim, N, Nw);

    branchUsedLog = false(Nw, N, Nw);
    numNonlocalBranchesUsedLog = zeros(N, Nw);



    % ---------------------------------------------------------------------
    % Step 09-J.5b bearing-FIM gate temporal diagnostics.
    %
    % Dimension convention:
    %   fimGateTraceOmegaBar(sourceBranchID, timeIndex, receiverWatcherID)
    %       = trace(OmegaBar_source) as seen by receiver watcher.
    %
    %   fimGateBnorm(sourceBranchID, timeIndex, receiverWatcherID)
    %       = ||B_{source|receiver}||_F.
    %
    %   fimGateOmegaSigmaMinEig(timeIndex, receiverWatcherID)
    %       = lambda_min(OmegaSigma_receiver).
    %
    % These logs are passive diagnostics only. They do not modify the EKF.
    % ---------------------------------------------------------------------
    fimGateTraceOmegaBarLog = NaN(Nw, N, Nw);
    fimGateBnormLog = NaN(Nw, N, Nw);
    fimGateOmegaSigmaMinEigLog = NaN(N, Nw);
    fimGateOmegaSigmaCondLog = NaN(N, Nw);
    fimGateSumIdentityErrorLog = NaN(N, Nw);
    fimGateNumBranchesLog = zeros(N, Nw);




    % Step 09-C.1 diagnostic logs:
    % Decompose the composite DNN residual into local and nonlocal components.
    % This is for analysis only and does not affect the estimator.
    dnnResidualLocalComponentLog = zeros(dim, N, Nw);
    dnnResidualNonlocalComponentLog = zeros(dim, N, Nw);

    % Branch contribution log.
    % Dimension convention:
    %   dnnResidualBranchContrib(:, j, k, i)
    %       = branch j contribution used by watcher i at time index k.
    dnnResidualBranchContribLog = zeros(dim, Nw, N, Nw);


    traceSdNonlocalLog = NaN(N, Nw);
    traceQnonlocalLog = NaN(N, Nw);
    numActiveNonlocalCovBranchesLog = zeros(N, Nw);

    measAvailLog = false(N, Nw);

    fovRangeLog = NaN(size(measAvailLog));
    fovOffBoresightAngleDegLog = NaN(size(measAvailLog));
    
    fovInsideFlagLog = false(size(measAvailLog));
    fovRangeOKFlagLog = false(size(measAvailLog));
    
    measurementDropoutReasonLog = strings(size(measAvailLog));
    measurementDropoutReasonLog(:) = "not_evaluated";

    % ---------------------------------------------------------------------
    % Innovation / NIS diagnostic logs
    % ---------------------------------------------------------------------
    nz = getMeasurementDimensionForLog(cfg);

    innovationLog = NaN(nz, N, Nw);
    SdiagLog = NaN(nz, N, Nw);
    NISLog = NaN(N, Nw);

    % ---------------------------------------------------------------------
    % Adaptive covariance matching logs
    % ---------------------------------------------------------------------
    gammaThetaLog = NaN(N, Nw);
    gammaEpsilonLog = NaN(N, Nw);
    cmRatioLog = NaN(N, Nw);
    cmTraceEmpLog = NaN(N, Nw);
    cmTraceModelLog = NaN(N, Nw);

    % Parameter-learning authority diagnostics.  thetaUpdateNorm contains
    % only the measurement-update correction (prediction drift excluded).
    % The cross-covariance norm measures the indirect path through which a
    % bearing update can change the DNN parameters.
    thetaUpdateNormLog = NaN(N, Nw);
    tracePthetaLog = NaN(N, Nw);
    PthetaEtaFroLog = NaN(N, Nw);

    % ---------------------------------------------------------------------
    % GS diagnostic logs
    % ---------------------------------------------------------------------
    gsVersionLog = zeros(Nw, N);
    gsValidLog = false(Nw, N);
    gsNumTotalUploadsLog = zeros(1, N);

    gsNumIncludedLog = zeros(N, Nw);
    gsIncludedBranchIDsLog = cell(N, Nw);

    % Step 05 event-triggered communication logs.
    % These are useful for checking why each watcher did/did not upload.
    gsUploadDecisionLog = false(N, Nw);              % true if watcher uploads at this step
    gsUploadDeltaLog = NaN(N, Nw);                   % Delta_i branch output change
    gsUploadDwellSatisfiedLog = false(N, Nw);        % dwell-time condition result
    gsUploadMaxSilenceSatisfiedLog = false(N, Nw);   % heartbeat / maximum-silence condition result
    gsUploadMeasSatisfiedLog = false(N, Nw);         % measurement condition result
    gsUploadReasonLog = strings(N, Nw);              % short reason string



    % ---------------------------------------------------------------------
    % Watcher truth / control logs
    % ---------------------------------------------------------------------
    watcherRLog = zeros(dim, N, Nw);
    watcherVLog = zeros(dim, N, Nw);
    watcherULog = zeros(dim, N, Nw);
    watcherTauLog = zeros(3, N, Nw);
    watcherQLog = zeros(4, N, Nw);
    watcherOmegaLog = zeros(3, N, Nw);

    % Step 10-A.1 observability-aware watcher-motion diagnostics.
    % The planner telemetry is passive: it does not change the estimator.
    nCandidateDirections = 1;
    if isfield(cfg.control,"obs") && ...
            isfield(cfg.control.obs,"numCandidateDirections")
        nCandidateDirections = max(1, ...
            round(cfg.control.obs.numCandidateDirections));
    end
    selectedDirectionLog = NaN(dim, N, Nw);
    selectedCandidateIndexLog = zeros(N, Nw);
    selectedScoreLog = NaN(N, Nw);
    candidateScoresLog = NaN(nCandidateDirections, N, Nw);
    candidateInformationMinEigLog = NaN(nCandidateDirections, N, Nw);
    candidateInformationConditionLog = NaN(nCandidateDirections, N, Nw);
    selectedInformationMinEigLog = NaN(N, Nw);
    selectedInformationConditionLog = NaN(N, Nw);
    replanFlagLog = false(N, Nw);
    controllerActiveLog = false(N, Nw);
    cumulativeImpulseLog = zeros(N, Nw);
    cumulativeDeltaVLog = zeros(N, Nw);
    watcherPathLengthLog = zeros(N, Nw);
    watcherDisplacementLog = zeros(dim, N, Nw);
    referenceRLog = NaN(dim, N, Nw);
    referenceVLog = NaN(dim, N, Nw);
    actualLOSChangeLog = NaN(N, Nw);
    actualLOSChangeSigmaLog = NaN(N, Nw);
    actualLOSChangeOverSigmaLog = NaN(N, Nw);
    losAngleLog = NaN(N, Nw);
    predictedRadialVarianceLog = NaN(N, Nw);

    % ---------------------------------------------------------------------
    % Initial logs after bootstrap broadcast
    % ---------------------------------------------------------------------
    for i = 1:Nw

        idxEta = watchers(i).idxEta;
        idxTheta = watchers(i).idxTheta;

        etaHatLog(:,1,i) = watchers(i).xhat(idxEta);
        xhatAugLog(:,1,i) = watchers(i).xhat;
        thetaHatLog(:,1,i) = watchers(i).xhat(idxTheta);

        PdiagLog(:,1,i) = diag(watchers(i).P);
        PdiagEtaLog(:,1,i) = diag(watchers(i).P(idxEta, idxEta));
        PdiagThetaLog(:,1,i) = diag(watchers(i).P(idxTheta, idxTheta));
        thetaUpdateNormLog(1,i) = 0.0;
        tracePthetaLog(1,i) = trace(watchers(i).P(idxTheta, idxTheta));
        PthetaEtaFroLog(1,i) = norm( ...
            watchers(i).P(idxTheta, idxEta), "fro");

        [dnnResidualLog(:,1,i), branchUsed, branchContrib] = ...
            computeDNNResidualForLog(watchers(i), cfg);



        % etaTrue is the full truth-state log, so use the current column only.
        etaTrueNow = etaTrue(:, 1);

        [dnnResidualAtTrueEtaLog(:, 1, i), ~, ~] = computeDNNResidualForLog( ...
            watchers(i), cfg, etaTrueNow);





        branchUsedLog(:,1,i) = branchUsed;
        numNonlocalBranchesUsedLog(1,i) = nnz(branchUsed) - 1;


        [fimGateTraceOmegaBarLog(:,1,i), ...
            fimGateBnormLog(:,1,i), ...
            fimGateOmegaSigmaMinEigLog(1,i), ...
            fimGateOmegaSigmaCondLog(1,i), ...
            fimGateSumIdentityErrorLog(1,i), ...
            fimGateNumBranchesLog(1,i)] = ...
            getFIMGateDiagnosticsForLog(watchers(i), branchUsed, cfg);



        localBranchID = watchers(i).localBranchID;

        dnnResidualBranchContribLog(:,:,1,i) = branchContrib;
        dnnResidualLocalComponentLog(:,1,i) = branchContrib(:, localBranchID);
        dnnResidualNonlocalComponentLog(:,1,i) = ...
            sum(branchContrib(:, setdiff(1:Nw, localBranchID)), 2);


        localBranchID = watchers(i).localBranchID;

        dnnResidualBranchContribLog(:,:,1,i) = branchContrib;
        dnnResidualLocalComponentLog(:,1,i) = branchContrib(:, localBranchID);
        dnnResidualNonlocalComponentLog(:,1,i) = ...
            sum(branchContrib(:, setdiff(1:Nw, localBranchID)), 2);



        [traceSdNonlocalLog(1,i), traceQnonlocalLog(1,i), ...
            numActiveNonlocalCovBranchesLog(1,i)] = ...
                getNonlocalCovInjectionDiagnosticsForLog(watchers(i));

        [gammaThetaLog(1,i), gammaEpsilonLog(1,i), cmRatioLog(1,i), ...
            cmTraceEmpLog(1,i), cmTraceModelLog(1,i)] = ...
                getCovMatchingDiagnosticsForLog(watchers(i));

        watcherRLog(:,1,i) = watcherTruth(i).r;
        watcherVLog(:,1,i) = watcherTruth(i).v;
        watcherULog(:,1,i) = watcherTruth(i).u;
        watcherTauLog(:,1,i) = watcherTruth(i).tau;
        watcherQLog(:,1,i) = watcherTruth(i).q;
        watcherOmegaLog(:,1,i) = watcherTruth(i).omega;

        controllerState = watcherTruth(i).controllerState;
        [selectedDirectionLog(:,1,i), selectedCandidateIndexLog(1,i), ...
            selectedScoreLog(1,i), candidateScoresLog(:,1,i), ...
            candidateInformationMinEigLog(:,1,i), ...
            candidateInformationConditionLog(:,1,i), ...
            selectedInformationMinEigLog(1,i), ...
            selectedInformationConditionLog(1,i), ...
            replanFlagLog(1,i), controllerActiveLog(1,i), ...
            predictedRadialVarianceLog(1,i)] = ...
            extractControllerTelemetry(controllerState, dim, ...
            nCandidateDirections);
        [referenceRLog(:,1,i), referenceVLog(:,1,i)] = ...
            getWatcherReferenceForLog(i, time(1), cfg, watcherTruth(i));
        watcherDisplacementLog(:,1,i) = zeros(dim,1);
        losAngleLog(1,i) = computeLosAngleForLog( ...
            etaTrue(1:dim,1)-watcherTruth(i).r);
        actualLOSChangeSigmaLog(1,i) = getBearingSigmaForLog(cfg);

        if isfield(watchers(i), "lastGSBroadcast")
            gsNumIncludedLog(1,i) = watchers(i).lastGSBroadcast.numIncluded;
            gsIncludedBranchIDsLog{1,i} = watchers(i).lastGSBroadcast.includedBranchIDs;
        end

    end

    [gsVersionLog(:,1), gsValidLog(:,1)] = getGSRepositoryDiagnostics(gsRepo);
    gsNumTotalUploadsLog(1) = gsRepo.numTotalUploads;

    % ---------------------------------------------------------------------
    % Main simulation loop
    % ---------------------------------------------------------------------
    for k = 1:N-1

        t = time(k);
        tNext = time(k+1);

        % 1. Propagate true target from t_k to t_{k+1}, or use the
        % exported truth trajectory during deterministic replay.
        if replayEnabled
            etaTrue(:,k+1) = cfg.replay.etaTrue(:,k+1);
        else
            etaTrue(:,k+1) = propagateRK4( ...
                @(tt,xx) targetTruthDynamics(tt, xx, cfg), ...
                t, etaTrue(:,k), cfg.dt);
        end

        trueResidualLog(:,k+1) = computeTrueResidualForLog(tNext, etaTrue(:,k+1), cfg);

        % Track which watchers had a measurement update and should upload.
        uploadAfterUpdate = false(Nw, 1);

        for i = 1:Nw

            idxEta = watchers(i).idxEta;

            % 2. Propagate watcher state to t_{k+1}.
            %
            % targetInfo.etaHat should be physical eta only, not the full
            % augmented DNN-EKF state.
            targetInfo.etaHat = watchers(i).xhat(idxEta);
            targetInfo.PEta = watchers(i).P(idxEta,idxEta);
            targetInfo.branchID = watchers(i).localBranchID;
            targetInfo.thetaHat = watchers(i).xhat(watchers(i).idxTheta);
            targetInfo.Ptheta = watchers(i).P( ...
                watchers(i).idxTheta,watchers(i).idxTheta);
            % Full local filter snapshot is read only by the optional
            % covariance-rollout maneuver planner.
            targetInfo.filter = watchers(i);
            targetInfo.etaTrue = etaTrue(:,k);   % debugging/analysis only

            if replayEnabled
                [watcherTruth(i), watcherCmd] = replayWatcherStep( ...
                    watcherTruth(i), cfg.replay, k+1, i, dim);
            else
                [watcherTruth(i), watcherCmd] = propagateWatcherStep( ...
                    i, watcherTruth(i), targetInfo, t, cfg);
            end

            watcherState = watcherTruth(i);

            watcherRLog(:,k+1,i) = watcherState.r;
            watcherVLog(:,k+1,i) = watcherState.v;
            watcherULog(:,k+1,i) = watcherCmd.u;
            watcherTauLog(:,k+1,i) = watcherCmd.tau;
            watcherQLog(:,k+1,i) = watcherState.q;
            watcherOmegaLog(:,k+1,i) = watcherState.omega;

            % Step 10-A.1 controller and realized-geometry telemetry.
            controllerState = watcherState.controllerState;
            [selectedDirectionLog(:,k+1,i), ...
                selectedCandidateIndexLog(k+1,i), ...
                selectedScoreLog(k+1,i), candidateScoresNow, ...
                candidateInformationMinEigNow, ...
                candidateInformationConditionNow, ...
                selectedInformationMinEigLog(k+1,i), ...
                selectedInformationConditionLog(k+1,i), ...
                replanFlagLog(k+1,i), controllerActiveLog(k+1,i), ...
                predictedRadialVarianceLog(k+1,i)] = ...
                extractControllerTelemetry(controllerState, dim, ...
                nCandidateDirections);
            candidateScoresLog(:,k+1,i) = candidateScoresNow;
            candidateInformationMinEigLog(:,k+1,i) = ...
                candidateInformationMinEigNow;
            candidateInformationConditionLog(:,k+1,i) = ...
                candidateInformationConditionNow;

            cumulativeImpulseLog(k+1,i) = cumulativeImpulseLog(k,i) + ...
                norm(watcherCmd.u)*cfg.dt;
            cumulativeDeltaVLog(k+1,i) = cumulativeDeltaVLog(k,i) + ...
                norm(watcherCmd.u)/max(watcherState.mass,eps)*cfg.dt;
            watcherPathLengthLog(k+1,i) = watcherPathLengthLog(k,i) + ...
                norm(watcherState.r-watcherRLog(:,k,i));
            watcherDisplacementLog(:,k+1,i) = ...
                watcherState.r-watcherRLog(:,1,i);
            [referenceRLog(:,k+1,i), referenceVLog(:,k+1,i)] = ...
                getWatcherReferenceForLog(i, tNext, cfg, watcherState);

            relativeNow = etaTrue(1:dim,k+1)-watcherState.r;
            losNow = relativeNow/max(norm(relativeNow),eps);
            losAngleLog(k+1,i) = computeLosAngleForLog(relativeNow);
            previousLos = etaTrue(1:dim,k)-watcherRLog(:,k,i);
            previousLos = previousLos/max(norm(previousLos),eps);
            actualLOSChangeLog(k+1,i) = computeLosChangeForLog( ...
                previousLos, losNow);
            actualLOSChangeSigmaLog(k+1,i) = getBearingSigmaForLog(cfg);
            actualLOSChangeOverSigmaLog(k+1,i) = ...
                actualLOSChangeLog(k+1,i)/max( ...
                actualLOSChangeSigmaLog(k+1,i),eps);

            % 3. Generate measurement at t_{k+1}.
            [z, available, measInfo] = measurementModel( ...
                etaTrue(:, k+1), watcherState, cfg, tNext);

            % Use one shared log row for all measurement-availability diagnostics.
            % In this simulator, the EKF update associated with loop index k is logged
            % on row k+1, because row 1 is the initialization row.
            measLogRow = k + 1;
            
            measAvailLog(measLogRow,i) = logical(available);
            
            fovRangeLog(measLogRow,i) = getStructNumericField( ...
                measInfo, "range", NaN);
            
            fovOffBoresightAngleDegLog(measLogRow,i) = getStructNumericField( ...
                measInfo, "offBoresightAngleDeg", NaN);
            
            fovInsideFlagLog(measLogRow,i) = getStructLogicalField( ...
                measInfo, "insideFOV", false);
            
            fovRangeOKFlagLog(measLogRow,i) = getStructLogicalField( ...
                measInfo, "rangeOK", false);
            
            measurementDropoutReasonLog(measLogRow,i) = getStructStringField( ...
                measInfo, "dropoutReason", "unknown");



            % 4. GS-DNN-EKF prediction from t_k to t_{k+1}.
            %
            % DNN_EKF_Predict_Local must already support:
            %
            %   cfg.dnn.predictionResidualSource = "GS_composite".
            %
            % The GS cache used here is the cache available at time t_k.
            watchers(i) = DNN_EKF_Predict_Local(watchers(i), t, cfg);

            % Save the predicted parameters so the subsequent diagnostic
            % isolates the measurement correction from FOGM prediction.
            thetaBeforeMeasurementUpdate = ...
                watchers(i).xhat(watchers(i).idxTheta);

            % 5. Measurement update at t_{k+1}.
            if available

                watchers(i) = DNN_EKF_Update_Local(watchers(i), z, watcherState, cfg);

                thetaUpdateNormLog(k+1,i) = norm( ...
                    watchers(i).xhat(watchers(i).idxTheta) - ...
                    thetaBeforeMeasurementUpdate);

                % Step 09-J.1: update passive local bearing-geometry support
                % only after the bearing measurement has actually been used by
                % the EKF update. This metadata will be uploaded/broadcast in
                % a later step; it is not used by additive GS yet.
                watchers(i) = updateWatcherOmegaBarFromMeasurement( ...
                    watchers(i), z, tNext, cfg);

                nu = watchers(i).lastInnovation;
                S = watchers(i).lastS;

                innovationLog(:,k+1,i) = nu;
                SdiagLog(:,k+1,i) = diag(S);
                NISLog(k+1,i) = nu' * (S \ nu);

                uploadAfterUpdate(i) = true;

            end

        % -----------------------------------------------------------------
        % 6. Upload local branch posteriors to GS.
        %
        % For Step 04a, upload after measurement update only.
        % This is done after all watchers finish their local EKF step, so the
        % order of watchers inside the loop does not create artificial
        % sequential information advantage.
        % -----------------------------------------------------------------
        if shouldUseGS(cfg) && ~p2pEnabled

            for i = 1:Nw

                [doUpload, uploadDecision] = shouldUploadThisStep( ...
                    uploadAfterUpdate(i), watchers(i), gsRepo, tNext, k+1, cfg);

                % Log trigger diagnostics before the repository changes.
                gsUploadDecisionLog(k+1,i) = doUpload;
                gsUploadDeltaLog(k+1,i) = uploadDecision.deltaContribution;
                gsUploadDwellSatisfiedLog(k+1,i) = uploadDecision.dwellSatisfied;
                gsUploadMaxSilenceSatisfiedLog(k+1,i) = uploadDecision.maxSilenceSatisfied;
                gsUploadMeasSatisfiedLog(k+1,i) = uploadDecision.measurementSatisfied;
                gsUploadReasonLog(k+1,i) = uploadDecision.reason;

                if doUpload

                    [gsRepo, uploadPacket] = uploadLocalBranchToGS( ...
                        gsRepo, watchers(i), tNext, cfg);

                    % Store trigger info inside the upload packet for debugging.
                    uploadPacket.triggerDecision = uploadDecision;

                    watchers(i).lastGSUpload = uploadPacket;
                end
            end

        elseif shouldUseGS(cfg) && p2pEnabled
            % P2P upload: evaluate each trigger against the sender's last
            % transmitted copy, then deliver the accepted record to the two
            % ring neighbors from a common pre-update snapshot.
            peerReposBefore = peerRepos;
            p2pTransmit = false(Nw,1);
            for i = 1:Nw
                [doUpload, uploadDecision] = shouldUploadThisStep( ...
                    uploadAfterUpdate(i),watchers(i),peerReposBefore{i}, ...
                    tNext,k+1,cfg);
                gsUploadDecisionLog(k+1,i) = doUpload;
                gsUploadDeltaLog(k+1,i) = uploadDecision.deltaContribution;
                gsUploadDwellSatisfiedLog(k+1,i) = uploadDecision.dwellSatisfied;
                gsUploadMaxSilenceSatisfiedLog(k+1,i) = uploadDecision.maxSilenceSatisfied;
                gsUploadMeasSatisfiedLog(k+1,i) = uploadDecision.measurementSatisfied;
                gsUploadReasonLog(k+1,i) = uploadDecision.reason;

                if doUpload
                    [peerRepos{i},uploadPacket] = uploadLocalBranchToGS( ...
                        peerRepos{i},watchers(i),tNext,cfg);
                    uploadPacket.triggerDecision = uploadDecision;
                    watchers(i).lastGSUpload = uploadPacket;
                    p2pTransmit(i) = true;
                end
            end

            % One synchronous gossip round.  Each transmitting watcher sends
            % its complete current branch cache to its two ring neighbors.
            % Version-based merging lets non-neighbor branches propagate over
            % multiple communication rounds without a central repository.
            peerReposBeforeGossip = peerRepos;
            for i = 1:Nw
                if ~p2pTransmit(i), continue; end
                neighbors = getP2PNeighbors(i,Nw);
                for nn = neighbors(:).'
                    peerRepos{nn} = mergeP2PRepositories( ...
                        peerRepos{nn},peerReposBeforeGossip{i});
                end
            end
        end

            % -------------------------------------------------------------
            % 7. Broadcast GS repository to all watchers.
            %
            % The broadcast happens after uploads at t_{k+1}. Therefore the
            % received branch copies are available for prediction during the
            % next interval [t_{k+1}, t_{k+2}].
            % -------------------------------------------------------------
            if shouldBroadcastThisStep(cfg) && ~p2pEnabled

                for i = 1:Nw

                    [watchers(i), broadcastPacket] = broadcastGSRepositoryToWatcher( ...
                        gsRepo, watchers(i), tNext, cfg);

                    watchers(i).lastGSBroadcast = broadcastPacket;

                end

            elseif shouldBroadcastThisStep(cfg) && p2pEnabled
                for i = 1:Nw
                    [watchers(i),broadcastPacket] = ...
                        broadcastGSRepositoryToWatcher( ...
                        peerRepos{i},watchers(i),tNext,cfg);
                    watchers(i).lastGSBroadcast = broadcastPacket;
                end
            end

        end

        % -----------------------------------------------------------------
        % 8. Log posterior estimate and GS-composite residual after broadcast.
        %
        % The state/covariance logs are unaffected by the broadcast.
        % The residual log reflects the branch library available at t_{k+1}
        % for the next prediction step.
        % -----------------------------------------------------------------
        for i = 1:Nw

            idxEta = watchers(i).idxEta;
            idxTheta = watchers(i).idxTheta;

            etaHatLog(:,k+1,i) = watchers(i).xhat(idxEta);
            xhatAugLog(:,k+1,i) = watchers(i).xhat;
            thetaHatLog(:,k+1,i) = watchers(i).xhat(idxTheta);

            PdiagLog(:,k+1,i) = diag(watchers(i).P);
            PdiagEtaLog(:,k+1,i) = diag(watchers(i).P(idxEta, idxEta));
            PdiagThetaLog(:,k+1,i) = diag(watchers(i).P(idxTheta, idxTheta));
            tracePthetaLog(k+1,i) = trace(watchers(i).P(idxTheta, idxTheta));
            PthetaEtaFroLog(k+1,i) = norm( ...
                watchers(i).P(idxTheta, idxEta), "fro");

            [dnnResidualLog(:,k+1,i), branchUsed, branchContrib] = ...
                computeDNNResidualForLog(watchers(i), cfg);

            branchUsedLog(:,k+1,i) = branchUsed;
            numNonlocalBranchesUsedLog(k+1,i) = nnz(branchUsed) - 1;


            [fimGateTraceOmegaBarLog(:,k+1,i), ...
                fimGateBnormLog(:,k+1,i), ...
                fimGateOmegaSigmaMinEigLog(k+1,i), ...
                fimGateOmegaSigmaCondLog(k+1,i), ...
                fimGateSumIdentityErrorLog(k+1,i), ...
                fimGateNumBranchesLog(k+1,i)] = ...
                getFIMGateDiagnosticsForLog(watchers(i), branchUsed, cfg);




            % etaTrue is the full truth-state log, so use the current column only.
            etaTrueNow = etaTrue(:, k+1);

            [dnnResidualAtTrueEtaLog(:, k+1, i), ~, ~] = computeDNNResidualForLog( ...
                watchers(i), cfg, etaTrueNow);



            localBranchID = watchers(i).localBranchID;
            nonlocalBranchIDs = setdiff(1:Nw, localBranchID);

            dnnResidualBranchContribLog(:,:,k+1,i) = branchContrib;
            dnnResidualLocalComponentLog(:,k+1,i) = branchContrib(:, localBranchID);
            dnnResidualNonlocalComponentLog(:,k+1,i) = ...
                sum(branchContrib(:, nonlocalBranchIDs), 2);

            [traceSdNonlocalLog(k+1,i), traceQnonlocalLog(k+1,i), ...
                numActiveNonlocalCovBranchesLog(k+1,i)] = ...
                    getNonlocalCovInjectionDiagnosticsForLog(watchers(i));

            [gammaThetaLog(k+1,i), gammaEpsilonLog(k+1,i), cmRatioLog(k+1,i), ...
                cmTraceEmpLog(k+1,i), cmTraceModelLog(k+1,i)] = ...
                    getCovMatchingDiagnosticsForLog(watchers(i));

            if isfield(watchers(i), "lastGSBroadcast")
                gsNumIncludedLog(k+1,i) = watchers(i).lastGSBroadcast.numIncluded;
                gsIncludedBranchIDsLog{k+1,i} = watchers(i).lastGSBroadcast.includedBranchIDs;
            end

        end

        [gsVersionLog(:,k+1), gsValidLog(:,k+1)] = getGSRepositoryDiagnostics(gsRepo);
        gsNumTotalUploadsLog(k+1) = gsRepo.numTotalUploads;

    end

    % ---------------------------------------------------------------------
    % Output structure
    % ---------------------------------------------------------------------
    results.time = time;
    results.etaTrue = etaTrue;

    % Eta-only state log for existing metrics and plots.
    results.xhat = etaHatLog;

    % Augmented DNN-EKF logs.
    results.xhatAug = xhatAugLog;
    results.thetaHat = thetaHatLog;

    results.Pdiag = PdiagLog;
    results.PdiagEta = PdiagEtaLog;
    results.PdiagTheta = PdiagThetaLog;

    results.dnnResidual = dnnResidualLog;
    results.trueResidual = trueResidualLog;

    % Step 09-C.1 GS composite residual component diagnostics.
    results.dnnResidualLocalComponent = dnnResidualLocalComponentLog;
    results.dnnResidualNonlocalComponent = dnnResidualNonlocalComponentLog;
    results.dnnResidualBranchContrib = dnnResidualBranchContribLog;
    results.dnnResidualAtTrueEta = dnnResidualAtTrueEtaLog;
    


    results.branchUsed = branchUsedLog;
    results.numNonlocalBranchesUsed = numNonlocalBranchesUsedLog;



    % Step 09-J.5b bearing-FIM gate temporal diagnostics.
    results.fimGateTraceOmegaBar = fimGateTraceOmegaBarLog;
    results.fimGateBnorm = fimGateBnormLog;
    results.fimGateOmegaSigmaMinEig = fimGateOmegaSigmaMinEigLog;
    results.fimGateOmegaSigmaCond = fimGateOmegaSigmaCondLog;
    results.fimGateSumIdentityError = fimGateSumIdentityErrorLog;
    results.fimGateNumBranches = fimGateNumBranchesLog;



    results.traceSdNonlocal = traceSdNonlocalLog;
    results.traceQnonlocal = traceQnonlocalLog;
    results.numActiveNonlocalCovBranches = numActiveNonlocalCovBranchesLog;

    results.measAvail = measAvailLog;

    results.fovRange = fovRangeLog;
    results.fovOffBoresightAngleDeg = fovOffBoresightAngleDegLog;
    
    results.fovInsideFlag = fovInsideFlagLog;
    results.fovRangeOKFlag = fovRangeOKFlagLog;
    
    results.measurementDropoutReason = measurementDropoutReasonLog;


    results.watchersFinal = watchers;

    results.watcherR = watcherRLog;
    results.watcherV = watcherVLog;
    results.watcherU = watcherULog;
    results.watcherTau = watcherTauLog;
    results.watcherQ = watcherQLog;
    results.watcherOmega = watcherOmegaLog;
    results.watcherTruthFinal = watcherTruth;

    % Step 10-A.1 observability-aware watcher-motion diagnostics.
    results.selectedDirection = selectedDirectionLog;
    results.selectedCandidateIndex = selectedCandidateIndexLog;
    results.selectedScore = selectedScoreLog;
    results.candidateScores = candidateScoresLog;
    results.candidateInformationMinEig = ...
        candidateInformationMinEigLog;
    results.candidateInformationCondition = ...
        candidateInformationConditionLog;
    results.selectedInformationMinEig = selectedInformationMinEigLog;
    results.selectedInformationCondition = ...
        selectedInformationConditionLog;
    results.replanFlag = replanFlagLog;
    results.controllerActive = controllerActiveLog;
    results.cumulativeImpulse = cumulativeImpulseLog;
    results.cumulativeDeltaV = cumulativeDeltaVLog;
    results.watcherPathLength = watcherPathLengthLog;
    results.watcherDisplacement = watcherDisplacementLog;
    results.referenceR = referenceRLog;
    results.referenceV = referenceVLog;
    results.actualLOSChange = actualLOSChangeLog;
    results.actualLOSChangeSigma = actualLOSChangeSigmaLog;
    results.actualLOSChangeOverSigma = actualLOSChangeOverSigmaLog;
    results.losAngle = losAngleLog;
    results.predictedRadialVariance = predictedRadialVarianceLog;

    results.innovation = innovationLog;
    results.Sdiag = SdiagLog;
    results.NIS = NISLog;

    results.gammaTheta = gammaThetaLog;
    results.gammaEpsilon = gammaEpsilonLog;
    results.cmRatio = cmRatioLog;
    results.cmTraceEmp = cmTraceEmpLog;
    results.cmTraceModel = cmTraceModelLog;
    results.thetaUpdateNorm = thetaUpdateNormLog;
    results.tracePtheta = tracePthetaLog;
    results.PthetaEtaFro = PthetaEtaFroLog;

    % GS logs.
    results.gsRepoFinal = gsRepo;
    results.gsVersion = gsVersionLog;
    results.gsValid = gsValidLog;
    results.gsNumTotalUploads = gsNumTotalUploadsLog;
    results.gsNumIncluded = gsNumIncludedLog;
    results.gsIncludedBranchIDs = gsIncludedBranchIDsLog;

    % Step 05 event-triggered communication logs.
    results.gsUploadDecision = gsUploadDecisionLog;
    results.gsUploadDelta = gsUploadDeltaLog;
    results.gsUploadDwellSatisfied = gsUploadDwellSatisfiedLog;
    results.gsUploadMaxSilenceSatisfied = gsUploadMaxSilenceSatisfiedLog;
    results.gsUploadMeasSatisfied = gsUploadMeasSatisfiedLog;
    results.gsUploadReason = gsUploadReasonLog;

end

function validateReplayTrajectory(replay,nEta,dim,N,Nw)
%VALIDATEREPLAYTRAJECTORY Validate the minimum deterministic replay contract.
    required = {"etaTrue","watcherR","watcherV"};
    for i = 1:numel(required)
        if ~isfield(replay,required{i})
            error("simulate_GS_DNN_EKF:MissingReplayField", ...
                "cfg.replay.%s is required.",required{i});
        end
    end
    if ~isequal(size(replay.etaTrue),[nEta N]) || ...
            ~isequal(size(replay.watcherR),[dim N Nw]) || ...
            ~isequal(size(replay.watcherV),[dim N Nw])
        error("simulate_GS_DNN_EKF:BadReplaySize", ...
            "Replay truth or watcher arrays do not match cfg dimensions.");
    end
end

function [watcherNext,cmd] = replayWatcherStep(watcherCurrent,replay,k,i,dim)
%REPLAYWATCHERSTEP Inject an exported realized watcher state.
    watcherNext = watcherCurrent;
    watcherNext.r = replay.watcherR(:,k,i);
    watcherNext.v = replay.watcherV(:,k,i);
    cmd.u = zeros(dim,1);
    cmd.tau = zeros(3,1);
    if isfield(replay,"watcherU") && isequal(size(replay.watcherU), ...
            [dim size(replay.watcherR,2) size(replay.watcherR,3)])
        cmd.u = replay.watcherU(:,k,i);
    end
    watcherNext.u = cmd.u;
    watcherNext.tau = cmd.tau;
    if isfield(watcherNext,"controllerState")
        watcherNext.controllerState.replanFlag = false;
        watcherNext.controllerState.activeFlag = false;
    end
end

function [direction,candidateIndex,score,candidateScores, ...
    candidateInformationMinEig,candidateInformationCondition, ...
    selectedInformationMinEig,selectedInformationCondition, ...
    replanFlag,activeFlag,predictedRadialVariance] = ...
    extractControllerTelemetry(controllerState,dim,nCandidateDirections)
%EXTRACTCONTROLLERTELEMETRY Safely unpack Step 10-A.1 controller state.

direction = zeros(dim,1);
candidateIndex = 0;
score = NaN;
candidateScores = NaN(nCandidateDirections,1);
candidateInformationMinEig = NaN(nCandidateDirections,1);
candidateInformationCondition = NaN(nCandidateDirections,1);
selectedInformationMinEig = NaN;
selectedInformationCondition = NaN;
replanFlag = false;
activeFlag = false;
predictedRadialVariance = NaN;

if ~isstruct(controllerState)
    return;
end

if isfield(controllerState,"direction") && ...
        numel(controllerState.direction) == dim
    direction = controllerState.direction(:);
end
if isfield(controllerState,"candidateIndex")
    candidateIndex = double(controllerState.candidateIndex);
end
if isfield(controllerState,"score")
    score = double(controllerState.score);
end
if isfield(controllerState,"candidateScores") && ...
        ~isempty(controllerState.candidateScores)
    values = double(controllerState.candidateScores(:));
    n = min(numel(values),nCandidateDirections);
    candidateScores(1:n) = values(1:n);
end
if isfield(controllerState,"candidateInformationMinEig") && ...
        ~isempty(controllerState.candidateInformationMinEig)
    values = double(controllerState.candidateInformationMinEig(:));
    n = min(numel(values),nCandidateDirections);
    candidateInformationMinEig(1:n) = values(1:n);
end
if isfield(controllerState,"candidateInformationCondition") && ...
        ~isempty(controllerState.candidateInformationCondition)
    values = double(controllerState.candidateInformationCondition(:));
    n = min(numel(values),nCandidateDirections);
    candidateInformationCondition(1:n) = values(1:n);
end
if isfield(controllerState,"selectedInformationMinEig")
    selectedInformationMinEig = double( ...
        controllerState.selectedInformationMinEig);
end
if isfield(controllerState,"selectedInformationCondition")
    selectedInformationCondition = double( ...
        controllerState.selectedInformationCondition);
end
if isfield(controllerState,"replanFlag")
    replanFlag = logical(controllerState.replanFlag);
end
if isfield(controllerState,"activeFlag")
    activeFlag = logical(controllerState.activeFlag);
end
if isfield(controllerState,"predictedRadialVariance")
    predictedRadialVariance = double( ...
        controllerState.predictedRadialVariance);
end

end

function [rRef,vRef] = getWatcherReferenceForLog(i,t,cfg,currentState)
%GETWATCHERREFERENCEFORLOG Return the nominal analytic watcher reference.

rRef = currentState.r;
vRef = currentState.v;
try
    referenceState = watcherTrajectory(i,t,cfg);
    if isfield(referenceState,"r") && isfield(referenceState,"v")
        rRef = referenceState.r(:);
        vRef = referenceState.v(:);
    end
catch
    % Some future controlled configurations may not define an analytic
    % reference. In that case the realized state is the safe fallback.
end

end

function sigma = getBearingSigmaForLog(cfg)
%GETBEARINGSIGMAFORLOG Return the scalar bearing-noise standard deviation.

sigma = NaN;
if isfield(cfg,"meas") && isfield(cfg.meas,"sigmaBearing")
    candidate = double(cfg.meas.sigmaBearing);
    if isscalar(candidate) && isfinite(candidate) && candidate > 0
        sigma = candidate;
    end
elseif isfield(cfg,"meas") && isfield(cfg.meas,"R")
    candidate = double(cfg.meas.R);
    if isscalar(candidate) && isfinite(candidate) && candidate > 0
        sigma = sqrt(candidate);
    end
end

end

function angle = computeLosAngleForLog(relative)
%COMPUTELOSANGLEFORLOG Return the planar LOS angle for plotting/logging.

relative = relative(:);
if numel(relative) >= 2
    angle = atan2(relative(2),relative(1));
else
    angle = NaN;
end

end

function change = computeLosChangeForLog(previousLOS,currentLOS)
%COMPUTELOSCHANGEFORLOG Return absolute angular change between LOS vectors.

previousLOS = previousLOS(:);
currentLOS = currentLOS(:);
if numel(previousLOS) == 2 && numel(currentLOS) == 2
    crossZ = previousLOS(1)*currentLOS(2) - ...
        previousLOS(2)*currentLOS(1);
    dotValue = previousLOS.'*currentLOS;
    change = abs(atan2(crossZ,dotValue));
else
    dotValue = max(-1,min(1,previousLOS.'*currentLOS));
    change = acos(dotValue);
end

end

function nz = getMeasurementDimensionForLog(cfg)
% Return the measurement dimension used to allocate innovation logs.

    switch string(cfg.meas.type)
        case "bearing"
            nz = cfg.dim - 1;
        case "range_bearing"
            nz = cfg.dim;
        case {"relative_position", "direct_residual"}
            nz = cfg.dim;
        otherwise
            error("simulate_GS_DNN_EKF:UnsupportedMeasurementType", ...
                "Unsupported measurement type: %s", string(cfg.meas.type));
    end
end

function tf = shouldUseGS(cfg)
% Return true if GS communication is enabled.

    tf = false;

    if isfield(cfg, "gs") && isfield(cfg.gs, "enabled")
        tf = logical(cfg.gs.enabled);
    end

end

function [tf, decision] = shouldUploadThisStep( ...
    hadMeasurementUpdate, watcher, gsRepo, t, kIndex, cfg)
%{
Function:
    shouldUploadThisStep

Purpose:
    Decide whether one watcher uploads its local DNN branch to the GS.

Modes:
    "after_measurement_update":
        Upload whenever the watcher had a valid EKF measurement update.

    "event_contribution_change":
        Upload only if:
            1. measurement condition is satisfied,
            2. branch contribution change Delta_i is large enough,
            3. dwell-time condition is satisfied.

    "every_step":
        Upload every time step.

    "never":
        Never upload after bootstrap.

Event metric:
    Delta_i = || d_i(eta_hat; theta_i_local)
                 - d_i(eta_hat; theta_i_GS) ||^2

Notes:
    This function does not modify watcher or gsRepo.
%}

    decision = initUploadDecision();

    if ~shouldUseGS(cfg)
        tf = false;
        decision.reason = "GS_disabled";
        return;
    end

    if isfield(cfg.gs, "uploadMode")
        uploadMode = string(cfg.gs.uploadMode);
    else
        uploadMode = "after_measurement_update";
    end

    switch uploadMode

        case "after_measurement_update"
            tf = logical(hadMeasurementUpdate);
            decision.measurementSatisfied = logical(hadMeasurementUpdate);
            decision.dwellSatisfied = true;
            decision.deltaSatisfied = true;
            decision.reason = ternaryString(tf, "measurement_update", "no_measurement_update");

        case "every_step"
            tf = true;
            decision.measurementSatisfied = logical(hadMeasurementUpdate);
            decision.dwellSatisfied = true;
            decision.deltaSatisfied = true;
            decision.reason = "every_step";

        case "never"
            tf = false;
            decision.measurementSatisfied = logical(hadMeasurementUpdate);
            decision.dwellSatisfied = false;
            decision.deltaSatisfied = false;
            decision.reason = "never";

        case "event_contribution_change"
            [tf, decision] = shouldUploadEventContributionChange( ...
                hadMeasurementUpdate, watcher, gsRepo, t, kIndex, cfg);

        otherwise
            error("Unsupported cfg.gs.uploadMode: %s", uploadMode);

    end

end

function [tf, decision] = shouldUploadEventContributionChange( ...
    hadMeasurementUpdate, watcher, gsRepo, t, kIndex, cfg)
%{
Function:
    shouldUploadEventContributionChange

Purpose:
    Step 05 event-triggered GS upload decision for one watcher.

    The local branch is uploaded when:
        1. the measurement condition is satisfied,
        2. the dwell-time condition is satisfied,
        3. either the branch contribution-change threshold is satisfied
           or the maximum-silence / heartbeat condition is satisfied.

Upload rule:
    doUpload =
        measurementSatisfied
        && dwellSatisfied
        && (deltaSatisfied || maxSilenceSatisfied)

Definitions:
    Delta_i:
        Squared branch-output difference between the local branch and the
        current GS copy, evaluated at the local physical estimate.

    eventDwellSteps:
        Minimum number of time steps required between accepted uploads.

    eventMaxSilenceSteps:
        Maximum allowed number of time steps since the last accepted upload.
        If finite, this creates a heartbeat upload even when Delta_i is below
        threshold. If Inf, the heartbeat path is disabled.

Notes:
    - Measurement availability and communication triggering remain separate.
    - The current default behavior is preserved when
      cfg.gs.eventMaxSilenceSteps is absent or Inf.
%}

    decision = initUploadDecision();

    decision.measurementSatisfied = logical(hadMeasurementUpdate);

    if getLogicalField(cfg.gs, "eventRequireMeasurement", true)
        if ~hadMeasurementUpdate
            tf = false;
            decision.reason = "no_measurement_update";
            return;
        end
    end

    branchID = watcher.localBranchID;
    branchRecord = gsRepo.branch(branchID);

    % If GS has no valid copy yet, upload immediately.
    if string(branchRecord.status) ~= "valid"
        tf = true;
        decision.deltaContribution = Inf;
        decision.deltaSatisfied = true;
        decision.dwellSatisfied = true;
        decision.maxSilenceSatisfied = false;
        decision.reason = "no_valid_GS_record";
        decision.kIndex = kIndex;
        return;
    end

    % ------------------------------------------------------------------
    % Contribution-change metric Delta_i.
    % ------------------------------------------------------------------
    deltaContribution = computeBranchContributionChangeForUpload( ...
        watcher, branchRecord, cfg);

    decision.deltaContribution = deltaContribution;

    deltaThreshold = getNumericFieldFromStruct( ...
        cfg.gs, "eventDeltaThreshold", 0.0);

    decision.deltaThreshold = deltaThreshold;
    decision.deltaSatisfied = deltaContribution >= deltaThreshold;

    % ------------------------------------------------------------------
    % Elapsed step count since the last accepted GS upload of this branch.
    % ------------------------------------------------------------------
    if isfield(branchRecord, "lastUpdateTime") && isfinite(branchRecord.lastUpdateTime)
        elapsedSteps = round((t - branchRecord.lastUpdateTime) / cfg.dt);
    else
        elapsedSteps = Inf;
    end

    decision.elapsedStepsSinceUpload = elapsedSteps;

    % ------------------------------------------------------------------
    % Dwell-time condition.
    % ------------------------------------------------------------------
    dwellSteps = getNumericFieldFromStruct(cfg.gs, "eventDwellSteps", 0);
    decision.eventDwellSteps = dwellSteps;

    if dwellSteps <= 0
        decision.dwellSatisfied = true;
    else
        decision.dwellSatisfied = elapsedSteps >= dwellSteps;
    end

    % ------------------------------------------------------------------
    % Maximum-silence / heartbeat condition.
    %
    % Default Inf disables this path and preserves Step 05-B behavior.
    % ------------------------------------------------------------------
    maxSilenceSteps = getNumericFieldFromStruct( ...
        cfg.gs, "eventMaxSilenceSteps", Inf);

    decision.eventMaxSilenceSteps = maxSilenceSteps;

    if isfinite(maxSilenceSteps)
        decision.maxSilenceSatisfied = elapsedSteps >= maxSilenceSteps;
    else
        decision.maxSilenceSatisfied = false;
    end

    % ------------------------------------------------------------------
    % Final event-triggered upload rule.
    % ------------------------------------------------------------------
    tf = decision.measurementSatisfied && ...
         decision.dwellSatisfied && ...
         (decision.deltaSatisfied || decision.maxSilenceSatisfied);

    if tf
        if decision.maxSilenceSatisfied && ~decision.deltaSatisfied
            decision.reason = "max_silence_passed";
        else
            decision.reason = "event_trigger_passed";
        end
    elseif ~decision.deltaSatisfied && ~decision.maxSilenceSatisfied
        decision.reason = "delta_below_threshold";
    elseif ~decision.dwellSatisfied
        decision.reason = "dwell_not_satisfied";
    else
        decision.reason = "event_trigger_failed";
    end

    decision.branchVersion = branchRecord.version;
    decision.kIndex = kIndex;

end


function deltaContribution = computeBranchContributionChangeForUpload( ...
    watcher, branchRecord, cfg)
% Compute Delta_i using the current local eta estimate as the test point.

    idxEta = watcher.idxEta(:);
    idxTheta = watcher.idxTheta(:);

    branchID = watcher.localBranchID;

    etaEval = watcher.xhat(idxEta);

    thetaLocal = watcher.xhat(idxTheta);
    thetaGS = branchRecord.theta;

    % Branch-model-aware contribution change.
    %
    % This upload trigger compares the current local branch output against
    % the most recently stored GS copy at the same etaEval.
    %
    % fixed_feature_lip:
    %   evaluateBranchResidualModel computes W phi.
    %
    % mlp_general:
    %   evaluateBranchResidualModel computes the MLP output.
    [dLocal, ~, ~, ~] = evaluateBranchResidualModel( ...
        branchID, etaEval, thetaLocal, cfg);

    [dGS, ~, ~, ~] = evaluateBranchResidualModel( ...
        branchID, etaEval, thetaGS, cfg);


    diff = dLocal - dGS;

    % Mean over one representative point = squared norm.
    deltaContribution = diff' * diff;

end

function decision = initUploadDecision()
% Small struct used for logging event-trigger decisions.

    decision = struct();

    decision.measurementSatisfied = false;
    decision.deltaSatisfied = false;
    decision.dwellSatisfied = false;
    decision.maxSilenceSatisfied = false;
    
    decision.deltaContribution = NaN;
    decision.deltaThreshold = NaN;
    decision.elapsedStepsSinceUpload = NaN;
    decision.eventDwellSteps = NaN;
    decision.eventMaxSilenceSteps = NaN;

    decision.branchVersion = NaN;
    decision.kIndex = NaN;

    decision.reason = "";

end

function val = getNumericFieldFromStruct(s, fieldName, defaultVal)
% Safe numeric scalar field reader.

    val = defaultVal;

    if isfield(s, fieldName)
        candidate = s.(fieldName);
        if isnumeric(candidate) && isscalar(candidate)
            val = candidate;
        end
    end

end

function val = getLogicalField(s, fieldName, defaultVal)
% Safe logical scalar field reader.

    val = defaultVal;

    if isfield(s, fieldName)
        val = logical(s.(fieldName));
    end

end

function out = ternaryString(condition, trueString, falseString)
% Tiny helper for compact reason strings.

    if condition
        out = string(trueString);
    else
        out = string(falseString);
    end

end

function tf = shouldBroadcastThisStep(cfg)
% Decide whether GS broadcasts the branch repository at this time step.

    if ~shouldUseGS(cfg)
        tf = false;
        return;
    end

    if isfield(cfg.gs, "broadcastMode")
        broadcastMode = string(cfg.gs.broadcastMode);
    else
        broadcastMode = "every_step";
    end

    switch broadcastMode

        case "every_step"
            tf = true;

        case "after_upload"
            % In this simple implementation, upload decisions are not passed
            % into this helper. Treat after_upload as every_step for now.
            tf = true;

        case "never"
            tf = false;

        otherwise
            error("Unsupported cfg.gs.broadcastMode: %s", broadcastMode);

    end

end

function aUnk = computeTrueResidualForLog(t, eta, cfg)
% Compute the true residual acceleration for logging only.

    dim = cfg.dim;
    aUnk = zeros(dim,1);

    if isfield(cfg, "truth") && isfield(cfg.truth, "useResidual")
        if cfg.truth.useResidual
            aUnk = trueResidual(t, eta, cfg);
        end
    end

end

function [dHat, branchUsed, branchContrib] = computeDNNResidualForLog(watcher, cfg, etaOverride)
%COMPUTEDNNRESIDUALFORLOG Evaluate DNN residual only for diagnostics.
%
% Purpose:
%     Log the DNN residual used by each watcher without changing the EKF.
%
% Outputs:
%     dHat
%         Total residual used by the estimator/log.
%
%     branchUsed
%         Logical vector showing which branches contributed.
%
%     branchContrib
%         Individual branch contributions.
%         branchContrib(:,j) is branch j's residual contribution.
%
% Notes:
%     Step 09-C.1 uses this decomposition to separate
%
%         d_GS = d_local + d_nonlocal.
%
%     This function is diagnostic only.


dim = cfg.dim;
Nw = cfg.Nw;

dHat = zeros(dim, 1);
branchUsed = false(Nw, 1);
branchContrib = zeros(dim, Nw);

% eta = watcher.xhat(watcher.idxEta);


if nargin < 3 || isempty(etaOverride)
    eta = watcher.xhat(1:2*cfg.dim);
else
    eta = etaOverride;
end

theta = watcher.xhat(watcher.idxTheta);

residualSource = string(cfg.dnn.predictionResidualSource);

switch residualSource

    case "none"

        % No DNN residual is used.
        return;

    case "local_DNN"

        branchID = watcher.localBranchID;

        % Branch-model-aware local residual for GS simulation diagnostics.
        %
        % This path is used only for logging, but it must match the branch
        % model used in prediction.
        [dLocal, ~, ~, ~] = evaluateBranchResidualModel( ...
            branchID, eta, theta, cfg);

        dHat = dLocal;
        branchContrib(:, branchID) = dLocal;
        branchUsed(branchID) = true;
    case "GS_composite"

        [dComp, ~, branchContribEval, branchUsedEval] = ...
            evaluateWatcherCompositeResidual(watcher, eta, theta, cfg);

        dHat = dComp;
        branchContrib = branchContribEval;
        branchUsed = branchUsedEval;

    case "oracle"

        % Oracle has no branch decomposition. This branch is only for
        % oracle diagnostic runs. The residual-alignment diagnostic
        % handles oracle separately when needed.
        dHat = trueResidual(0.0, eta, cfg);

    otherwise

        error("simulate_GS_DNN_EKF:UnknownResidualSource", ...
            "Unknown cfg.dnn.predictionResidualSource = %s", residualSource);

end

end

function [gammaTheta, gammaEpsilon, ratio, traceEmp, traceModel] = getCovMatchingDiagnosticsForLog(watcher)
%{
Function:
    getCovMatchingDiagnosticsForLog

Purpose:
    Extract adaptive covariance matching diagnostics from a watcher.

Outputs:
    gammaTheta - current Q_theta multiplier
    ratio      - last trace(S_hat)/trace(S_model)
    traceEmp   - last trace(S_hat)
    traceModel - last trace(S_model

Notes:
    If covariance matching has not been initialized yet, this function
    returns safe default values.
%}

    gammaTheta = 1.0;
    gammaEpsilon = 1.0;
    ratio = NaN;
    traceEmp = NaN;
    traceModel = NaN;

    if ~isfield(watcher, "cm")
        return;
    end

    cm = watcher.cm;

    if isfield(cm, "gammaTheta")
        gammaTheta = cm.gammaTheta;
    end

    if isfield(cm, "gammaEpsilon")
        gammaEpsilon = cm.gammaEpsilon;
    end

    if isfield(cm, "lastRatio")
        ratio = cm.lastRatio;
    end

    if isfield(cm, "lastTraceEmp")
        traceEmp = cm.lastTraceEmp;
    end

    if isfield(cm, "lastTraceModel")
        traceModel = cm.lastTraceModel;
    end

end

function [versionVec, validVec] = getGSRepositoryDiagnostics(gsRepo)
% Return GS branch versions and validity flags.

    Nw = numel(gsRepo.branch);

    versionVec = zeros(Nw, 1);
    validVec = false(Nw, 1);

    for j = 1:Nw
        versionVec(j) = gsRepo.branch(j).version;
        validVec(j) = string(gsRepo.branch(j).status) == "valid";
    end

end

function [traceSdNonlocal, traceQnonlocal, numActiveNonlocal] = ...
    getNonlocalCovInjectionDiagnosticsForLog(watcher)
%{
Function:
    getNonlocalCovInjectionDiagnosticsForLog

Purpose:
    Extract Step 04b nonlocal GS-branch covariance injection diagnostics
    from one watcher after prediction.

Inputs:
    watcher - Local watcher structure after DNN_EKF_Predict_Local.

Outputs:
    traceSdNonlocal   - trace of residual-acceleration covariance surrogate
                        S_{d,-m}.
    traceQnonlocal    - trace of injected augmented covariance contribution
                        Q_{X,-m}.
    numActiveNonlocal - number of active nonlocal GS branches used in the
                        covariance injection.

Notes:
    DNN_EKF_Predict_Local stores these diagnostics in

        watcher.lastNonlocalCovInjection.

    If the field is absent, this helper returns NaN for trace quantities
    and zero for the active branch count. This allows the same simulation
    summary helper to be used for Step 03, Step 04a, and Step 04b cases.
%}

    traceSdNonlocal = NaN;
    traceQnonlocal = NaN;
    numActiveNonlocal = 0;

    if ~isfield(watcher, "lastNonlocalCovInjection")
        return;
    end

    diagInfo = watcher.lastNonlocalCovInjection;

    if isfield(diagInfo, "traceSdNonlocal")
        traceSdNonlocal = diagInfo.traceSdNonlocal;
    end

    if isfield(diagInfo, "traceQnonlocal")
        traceQnonlocal = diagInfo.traceQnonlocal;
    end

    if isfield(diagInfo, "numActiveNonlocal")
        numActiveNonlocal = diagInfo.numActiveNonlocal;
    end

end


function value = getStructNumericField(s, fieldName, defaultValue)
%GETSTRUCTNUMERICFIELD Safely read a numeric diagnostic field from a struct.

    value = defaultValue;

    if ~isstruct(s)
        return;
    end

    if ~isfield(s, fieldName)
        return;
    end

    candidate = s.(fieldName);

    if isempty(candidate)
        return;
    end

    if isnumeric(candidate) || islogical(candidate)
        value = double(candidate);
    end

end

function value = getStructLogicalField(s, fieldName, defaultValue)
%GETSTRUCTLOGICALFIELD Safely read a logical diagnostic field from a struct.

    value = logical(defaultValue);

    if ~isstruct(s)
        return;
    end

    if ~isfield(s, fieldName)
        return;
    end

    candidate = s.(fieldName);

    if isempty(candidate)
        return;
    end

    value = logical(candidate);

end

function value = getStructStringField(s, fieldName, defaultValue)
%GETSTRUCTSTRINGFIELD Safely read a string diagnostic field from a struct.

    value = string(defaultValue);

    if ~isstruct(s)
        return;
    end

    if ~isfield(s, fieldName)
        return;
    end

    candidate = s.(fieldName);

    if isempty(candidate)
        return;
    end

    value = string(candidate);

end


function [traceOmegaBarByBranch, BnormByBranch, ...
    omegaSigmaMinEig, omegaSigmaCond, gateSumIdentityError, numGateBranches] = ...
    getFIMGateDiagnosticsForLog(watcher, branchUsed, cfg)
%GETFIMGATEDIAGNOSTICSFORLOG Log bearing-FIM gate internal quantities.
%
% Purpose:
%   Passive time-history diagnostic for Step 09-J.5b.
%
% Logged quantities:
%   traceOmegaBarByBranch(j)
%       trace(OmegaBar_j) as seen by the current recipient watcher.
%
%   BnormByBranch(j)
%       ||B_{j|m}||_F where m is the recipient watcher.
%
%   omegaSigmaMinEig
%       lambda_min(OmegaSigma_m).
%
%   omegaSigmaCond
%       cond(OmegaSigma_m).
%
%   gateSumIdentityError
%       ||sum_j B_{j|m} - I||_F.
%
% Notes:
%   This helper supports the legacy bearing projector gate and the rigorous
%   output-information fusion weights. Additive GS runs return NaNs.

Nw = cfg.Nw;

traceOmegaBarByBranch = NaN(Nw, 1);
BnormByBranch = NaN(Nw, 1);
omegaSigmaMinEig = NaN;
omegaSigmaCond = NaN;
gateSumIdentityError = NaN;
numGateBranches = 0;

compositeMode = "additive";

if isfield(cfg, "gs") && isfield(cfg.gs, "compositeMode")
    compositeMode = string(cfg.gs.compositeMode);
end

if compositeMode ~= "bearing_fim_gated" && ...
        compositeMode ~= "output_information_fusion" && ...
        compositeMode ~= "fim_weighted_additive"
    return;
end

branchUsed = logical(branchUsed(:));

if numel(branchUsed) ~= Nw || ~any(branchUsed)
    return;
end

if compositeMode == "bearing_fim_gated"
    [B, gateDiag] = computeBearingFIMGates(watcher, branchUsed, cfg);
elseif compositeMode == "fim_weighted_additive"
    [B,gateDiag] = computeFIMWeightedAdditiveWeights( ...
        watcher,branchUsed,cfg);
else
    thetaLocal=watcher.xhat(watcher.idxTheta);
    eta=watcher.xhat(watcher.idxEta);
    [~,~,~,branchUsedEvaluated,gateDiag] = ...
        evaluateWatcherCompositeResidual(watcher,eta,thetaLocal,cfg);
    branchUsed=branchUsedEvaluated;
    B=gateDiag.B;
end

for j = 1:Nw
    if ~branchUsed(j)
        continue;
    end

    OmegaBar_j = gateDiag.OmegaBars(:, :, j);
    OmegaBar_j = 0.5 * (OmegaBar_j + OmegaBar_j.');

    traceOmegaBarByBranch(j) = trace(OmegaBar_j);
    BnormByBranch(j) = norm(B(:, :, j), "fro");
end

omegaSigmaMinEig = gateDiag.minEigOmegaSigma;
omegaSigmaCond = gateDiag.condOmegaSigma;
gateSumIdentityError = gateDiag.sumGateIdentityError;
numGateBranches = nnz(branchUsed);

end
