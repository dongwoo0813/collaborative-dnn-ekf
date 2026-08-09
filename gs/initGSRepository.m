function gsRepo = initGSRepository(cfg)
%{
Function:
    initGSRepository.m

Purpose:
    Initialize the ground-station repository for Step 04 GS-assisted
    collaborative DNN-EKF.

    The ground station stores one branch record per watcher:

        branch j stores theta_j, P_theta_jtheta_j, update time, version,
        and status.

    In Step 04, watcher m will later use the GS repository to build a
    composite DNN residual prediction:

        d_hat_comp,m(eta) = sum_{j=1}^{Nw} W_{j|m} phi_j(eta).

    The local branch j = m comes from watcher m itself, while nonlocal
    branches j ~= m come from GS broadcasts.

Inputs:
    cfg - Simulation configuration structure.
          Required fields:
              cfg.Nw
              cfg.dim
              cfg.dnn.nPhi
              cfg.dnn.nThetaPerBranch

          Optional fields:
              cfg.gs.initialStatus
              cfg.gs.initialVersion

Outputs:
    gsRepo - Ground-station repository structure.

Main fields:
    gsRepo.branch(j).theta
        Latest accepted estimate of theta_j.

    gsRepo.branch(j).Ptheta
        Latest accepted covariance of theta_j.

    gsRepo.branch(j).lastUpdateTime
        Time when branch j was last updated at GS.

    gsRepo.branch(j).version
        Version counter for branch j.

    gsRepo.branch(j).status
        "empty" before the first upload, "valid" after an accepted upload.

Notes:
    - This function only initializes the repository.
    - It does not upload, fuse, age, or broadcast any branch records yet.
    - Empty records should not be used in prediction until their status
      becomes "valid".
%}
    
    % ---------------------------------------------------------------------
    % Basic dimensions
    % ---------------------------------------------------------------------
    Nw = cfg.Nw;
    dim = cfg.dim;
    
    if ~isfield(cfg, "dnn")
        error("cfg.dnn is required in initGSRepository.");
    end
    
    % Step 09-I.2:
    % Use branchThetaNumel(...) instead of fixed-feature-only fields such as
    % cfg.dnn.nPhi or cfg.dnn.nThetaPerBranch.
    %
    % fixed_feature_lip:
    %   nTheta = dim * nPhi
    %
    % mlp_general:
    %   nTheta = number of all MLP weights and biases.
    [nTheta, branchInfo] = branchThetaNumel(cfg);
    
    branchModel = string(branchInfo.branchModel);
    
    if isfield(branchInfo, "nPhi")
        nPhi = branchInfo.nPhi;
    else
        % MLP branches do not have a fixed-feature phi dimension.
        nPhi = NaN;
    end

    % ---------------------------------------------------------------------
    % Default GS settings
    % ---------------------------------------------------------------------
    if isfield(cfg, "gs") && isfield(cfg.gs, "initialStatus")
        initialStatus = string(cfg.gs.initialStatus);
    else
        initialStatus = "empty";
    end

    if isfield(cfg, "gs") && isfield(cfg.gs, "initialVersion")
        initialVersion = cfg.gs.initialVersion;
    else
        initialVersion = 0;
    end

    % ---------------------------------------------------------------------
    % Repository metadata
    % ---------------------------------------------------------------------
    gsRepo = struct();

    gsRepo.enabled = true;
    gsRepo.Nw = Nw;
    gsRepo.dim = dim;
    gsRepo.nPhi = nPhi;
    gsRepo.nThetaPerBranch = nTheta;

    % Branch-model metadata.
    %
    % This is required once GS stores general MLP branches, because theta
    % length alone is not enough to interpret a nonlocal branch safely.
    gsRepo.branchModel = branchModel;
    gsRepo.branchInfo = branchInfo;


    gsRepo.createdTime = 0;
    gsRepo.lastGlobalUpdateTime = NaN;
    gsRepo.numTotalUploads = 0;

    % ---------------------------------------------------------------------
    % Empty branch record template
    % ---------------------------------------------------------------------
    emptyRecord = struct();

    emptyRecord.branchID = NaN;
    emptyRecord.sourceWatcherID = NaN;


    % Branch architecture metadata for compatibility checks and diagnostics.
    emptyRecord.branchModel = branchModel;
    emptyRecord.nTheta = nTheta;
    emptyRecord.branchInfo = branchInfo;


    % Stored parameter estimate and covariance.
    %
    % These are initialized as zeros/NaNs, but they should not be used until
    % status becomes "valid".
    emptyRecord.theta = zeros(nTheta, 1);
    emptyRecord.Ptheta = NaN(nTheta, nTheta);
    % Source-side conditional covariance P(theta|eta), used by the
    % output-information fusion mode to remove kinematic-state ambiguity.
    emptyRecord.PthetaConditional = NaN(nTheta, nTheta);


    % Step 09-J.2 bearing-geometry metadata payload.
    %
    % OmegaBar is the cumulative direction-only bearing-FIM support
    % associated with this source watcher's local branch. It is metadata only
    % at this step; the additive GS composite residual does not use it yet.
    emptyRecord.OmegaBar = zeros(dim, dim);
    emptyRecord.numOmegaUpdates = 0;
    emptyRecord.lastLOSUnit = NaN(dim, 1);
    emptyRecord.lastOmegaUpdateTime = NaN;
    emptyRecord.lastMeasTime = NaN;
    emptyRecord.outputFrame = "inertial";




    % Update metadata.
    emptyRecord.lastUpdateTime = NaN;
    emptyRecord.version = initialVersion;
    emptyRecord.status = initialStatus;

    % Upload counters.
    emptyRecord.numUploads = 0;
    emptyRecord.numAcceptedUploads = 0;
    emptyRecord.numRejectedUploads = 0;

    % Optional fields for later event-triggering / novelty tests.
    emptyRecord.lastInnovationNorm = NaN;
    emptyRecord.lastThetaChangeNorm = NaN;
    emptyRecord.lastNoveltyScore = NaN;

    % Optional stale/covariance-aging fields for later.
    emptyRecord.age = Inf;
    emptyRecord.isStale = true;

    % ---------------------------------------------------------------------
    % Create one branch record per watcher
    % ---------------------------------------------------------------------
    gsRepo.branch = repmat(emptyRecord, Nw, 1);

    for j = 1:Nw
        gsRepo.branch(j).branchID = j;
        gsRepo.branch(j).sourceWatcherID = j;
    end

end
