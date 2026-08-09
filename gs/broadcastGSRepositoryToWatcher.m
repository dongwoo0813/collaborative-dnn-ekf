function [watcher, broadcastPacket] = broadcastGSRepositoryToWatcher(gsRepo, watcher, t, cfg)
%{
Function:
    broadcastGSRepositoryToWatcher.m

Purpose:
    Broadcast valid ground-station branch records to one watcher and store
    them as nonlocal branch copies in watcher.gsBranches.

    This is the Step 04 GS-to-watcher broadcast function. It implements the
    one-way communication step

        GS -> watcher m,

    where watcher m receives branch records for j ~= m, but keeps its own
    local branch estimate theta_m inside the local DNN-EKF state

        X_m = [eta_m; theta_m].

    The function intentionally does not overwrite

        watcher.xhat(watcher.idxTheta)
        watcher.P(watcher.idxTheta, watcher.idxTheta)

    because those are watcher m's local posterior estimate and covariance.

Inputs:
    gsRepo  - Ground-station repository structure from initGSRepository.m.
              Required fields:
                  gsRepo.branch(j).theta
                  gsRepo.branch(j).Ptheta
                  gsRepo.branch(j).lastUpdateTime
                  gsRepo.branch(j).version
                  gsRepo.branch(j).status

    watcher - Local DNN-EKF watcher structure.
              Required fields:
                  watcher.id
                  watcher.localBranchID
                  watcher.nTheta

              Optional field:
                  watcher.gsBranches

    t       - Current simulation time [s].

    cfg     - Simulation configuration structure.
              Optional fields:
                  cfg.Nw
                  cfg.gs.maxStaleTime

Outputs:
    watcher - Updated watcher structure with nonlocal branch copies stored in
              watcher.gsBranches(j), j ~= watcher.localBranchID.

    broadcastPacket - Diagnostic structure describing what was broadcast to
                      this watcher at time t.

Main stored fields:
    watcher.gsBranches(j).theta
        Latest GS-provided nonlocal branch parameter copy.

    watcher.gsBranches(j).Ptheta
        Latest GS-provided marginal branch covariance copy.

    watcher.gsBranches(j).version
        GS branch version number.

    watcher.gsBranches(j).active
        True only when branch j is nonlocal, valid, and not too stale.

    watcher.gsBranches(j).usedInPrediction
        Same as active for this first implementation.

Notes:
    - This file only broadcasts/stores branch copies.
    - It does not modify the EKF prediction model yet.
    - It does not propagate nonlocal branch covariance into watcher.P yet.
    - Step 04a treats the GS branch covariance as metadata. Later, when
      cfg.gs.useNonlocalBranchCovariance = true, Ptheta can be mapped into
      physical process covariance through branchJacobianTheta.m.
    - If this function is used inside a MATLAB struct array assignment like

          watchers(i) = broadcastGSRepositoryToWatcher(...),

      then all elements of watchers should already have a gsBranches field.
      The clean way is to add watcher.gsBranches = [] in initLocalDNNEKF.m.
%}

    % ---------------------------------------------------------------------
    % Basic checks
    % ---------------------------------------------------------------------
    if ~isfield(gsRepo, "branch")
        error("gsRepo.branch does not exist. Initialize GS with initGSRepository.m.");
    end

    if ~isfield(watcher, "id")
        error("watcher.id is required in broadcastGSRepositoryToWatcher.");
    end

    if ~isfield(watcher, "localBranchID")
        error("watcher.localBranchID is required in broadcastGSRepositoryToWatcher.");
    end

    if ~isfield(watcher, "nTheta")
        error("watcher.nTheta is required in broadcastGSRepositoryToWatcher.");
    end

    Nw = getNumberOfWatchers(gsRepo, cfg);
    nTheta = watcher.nTheta;
    dim = cfg.dim;


    if numel(gsRepo.branch) < Nw
        error("gsRepo.branch has fewer records than expected Nw = %d.", Nw);
    end

    watcherID = watcher.id;
    localBranchID = watcher.localBranchID;

    if localBranchID < 1 || localBranchID > Nw
        error("Invalid watcher.localBranchID = %d.", localBranchID);
    end

    maxStaleTime = getMaxStaleTime(cfg);

    % ---------------------------------------------------------------------
    % Initialize or repair watcher-side GS branch cache
    % ---------------------------------------------------------------------
    if ~isfield(watcher, "gsBranches") || isempty(watcher.gsBranches)
        watcher.gsBranches = initializeGSBranchCache(Nw, nTheta, dim);
    else
        watcher.gsBranches = ensureGSBranchCacheShape(watcher.gsBranches, Nw, nTheta, dim);
    end

    % ---------------------------------------------------------------------
    % Initialize broadcast diagnostic packet
    % ---------------------------------------------------------------------
    broadcastPacket = struct();

    broadcastPacket.targetWatcherID = watcherID;
    broadcastPacket.localBranchID = localBranchID;
    broadcastPacket.time = t;

    broadcastPacket.includedBranchIDs = [];
    broadcastPacket.skippedLocalBranchID = localBranchID;

    broadcastPacket.numIncluded = 0;
    broadcastPacket.numSkippedLocal = 0;
    broadcastPacket.numSkippedInvalid = 0;
    broadcastPacket.numSkippedStale = 0;
    broadcastPacket.numSkippedOlderThanCache = 0;

    broadcastPacket.branch = initializeGSBranchCache(Nw, nTheta, dim);

    % ---------------------------------------------------------------------
    % Broadcast valid nonlocal branch records
    % ---------------------------------------------------------------------
    for j = 1:Nw

        % -------------------------------------------------------------
        % Do not overwrite the watcher's own branch estimate.
        %
        % watcher m's local branch remains inside watcher.xhat(idxTheta).
        % The local placeholder is kept inactive so composite prediction can
        % later insert the local branch explicitly.
        % -------------------------------------------------------------
        if j == localBranchID
            watcher.gsBranches(j).branchID = j;
            watcher.gsBranches(j).sourceWatcherID = j;
            watcher.gsBranches(j).isLocalBranch = true;
            watcher.gsBranches(j).active = false;
            watcher.gsBranches(j).usedInPrediction = false;
            watcher.gsBranches(j).status = "local_branch_not_overwritten";
            watcher.gsBranches(j).receivedTime = t;

            broadcastPacket.branch(j) = watcher.gsBranches(j);
            broadcastPacket.numSkippedLocal = broadcastPacket.numSkippedLocal + 1;
            continue;
        end

        gsRecord = gsRepo.branch(j);
        status = getRecordStatus(gsRecord);

        % -------------------------------------------------------------
        % Only valid GS records are used.
        % Empty/rejected/quarantined records are retained as inactive cache
        % slots and will not contribute to GS_composite prediction.
        % -------------------------------------------------------------
        if status ~= "valid"
            watcher.gsBranches(j).branchID = j;
            watcher.gsBranches(j).sourceWatcherID = j;
            watcher.gsBranches(j).isLocalBranch = false;
            watcher.gsBranches(j).active = false;
            watcher.gsBranches(j).usedInPrediction = false;
            watcher.gsBranches(j).status = status;
            watcher.gsBranches(j).receivedTime = t;
            watcher.gsBranches(j).age = Inf;
            watcher.gsBranches(j).isStale = true;

            broadcastPacket.branch(j) = watcher.gsBranches(j);
            broadcastPacket.numSkippedInvalid = broadcastPacket.numSkippedInvalid + 1;
            continue;
        end

        age = computeRecordAge(gsRecord, t);
        isStale = age > maxStaleTime;

        cachedVersion = watcher.gsBranches(j).version;
        incomingVersion = gsRecord.version;

        if ~isfinite(cachedVersion)
            cachedVersion = -Inf;
        end

        if incomingVersion < cachedVersion
            % This should not happen with a single GS repository, but this
            % guard prevents an accidental rollback if logs or experiments
            % reuse an older repository object.
            watcher.gsBranches(j).receivedTime = t;
            watcher.gsBranches(j).age = age;
            watcher.gsBranches(j).isStale = isStale;
            watcher.gsBranches(j).active = watcher.gsBranches(j).active && ~isStale;
            watcher.gsBranches(j).usedInPrediction = watcher.gsBranches(j).active;

            broadcastPacket.branch(j) = watcher.gsBranches(j);
            broadcastPacket.numSkippedOlderThanCache = broadcastPacket.numSkippedOlderThanCache + 1;
            continue;
        end

        theta_j = gsRecord.theta(:);
        Ptheta_j = gsRecord.Ptheta;

        validateThetaAndCovariance(theta_j, Ptheta_j, nTheta, j);

        Ptheta_j = 0.5 * (Ptheta_j + Ptheta_j.');

        branchCopy = makeEmptyGSBranchRecord(j, nTheta, dim);

        branchCopy.branchID = j;
        branchCopy.sourceWatcherID = getOptionalField(gsRecord, "sourceWatcherID", j);


        % Preserve branch-model metadata in the watcher-side GS cache.
        branchCopy.branchModel = getOptionalField(gsRecord, "branchModel", "unknown");
        branchCopy.nTheta = getOptionalField(gsRecord, "nTheta", nTheta);
        branchCopy.branchInfo = getOptionalField(gsRecord, "branchInfo", struct());

        branchCopy.theta = theta_j;
        branchCopy.Ptheta = Ptheta_j;
        branchCopy.PthetaConditional = getOptionalField( ...
            gsRecord,"PthetaConditional",Ptheta_j);

        % Step 09-J.2: preserve source-branch bearing-geometry metadata.
        % These fields are passive until compositeMode = "bearing_fim_gated"
        % is added. The recipient will later compute B_{j|m}; GS never
        % broadcasts a precomputed B_j.
        branchCopy.OmegaBar = getOmegaBarField(gsRecord, cfg.dim, j);
        branchCopy.numOmegaUpdates = getOptionalField(gsRecord, "numOmegaUpdates", 0);
        branchCopy.lastLOSUnit = getLOSField(gsRecord, cfg.dim);
        branchCopy.lastOmegaUpdateTime = getOptionalField(gsRecord, "lastOmegaUpdateTime", NaN);
        branchCopy.lastMeasTime = getOptionalField(gsRecord, "lastMeasTime", ...
            branchCopy.lastOmegaUpdateTime);
        branchCopy.outputFrame = getOptionalField(gsRecord, "outputFrame", "inertial");

        branchCopy.lastUpdateTime = gsRecord.lastUpdateTime;
        branchCopy.receivedTime = t;
        branchCopy.age = age;
        branchCopy.isStale = isStale;

        branchCopy.version = incomingVersion;
        branchCopy.status = status;

        branchCopy.isLocalBranch = false;
        branchCopy.active = ~isStale;
        branchCopy.usedInPrediction = branchCopy.active;

        branchCopy.numUploads = getOptionalField(gsRecord, "numUploads", NaN);
        branchCopy.numAcceptedUploads = getOptionalField(gsRecord, "numAcceptedUploads", NaN);
        branchCopy.numRejectedUploads = getOptionalField(gsRecord, "numRejectedUploads", NaN);

        branchCopy.lastInnovationNorm = getOptionalField(gsRecord, "lastInnovationNorm", NaN);
        branchCopy.lastThetaChangeNorm = getOptionalField(gsRecord, "lastThetaChangeNorm", NaN);
        branchCopy.lastNoveltyScore = getOptionalField(gsRecord, "lastNoveltyScore", NaN);

        watcher.gsBranches(j) = branchCopy;
        broadcastPacket.branch(j) = branchCopy;

        if isStale
            broadcastPacket.numSkippedStale = broadcastPacket.numSkippedStale + 1;
        else
            broadcastPacket.includedBranchIDs(end+1) = j; %#ok<AGROW>
            broadcastPacket.numIncluded = broadcastPacket.numIncluded + 1;
        end

    end

end

function Nw = getNumberOfWatchers(gsRepo, cfg)
% Return the number of watcher/branch records expected in the repository.

    if isfield(cfg, "Nw")
        Nw = cfg.Nw;
    elseif isfield(gsRepo, "Nw")
        Nw = gsRepo.Nw;
    else
        Nw = numel(gsRepo.branch);
    end

    if ~isscalar(Nw) || Nw < 1 || floor(Nw) ~= Nw
        error("Nw must be a positive integer.");
    end

end

function maxStaleTime = getMaxStaleTime(cfg)
% Return the maximum allowed GS branch age before deactivation.

    if isfield(cfg, "gs") && isfield(cfg.gs, "maxStaleTime")
        maxStaleTime = cfg.gs.maxStaleTime;
    else
        maxStaleTime = Inf;
    end

    if isempty(maxStaleTime)
        maxStaleTime = Inf;
    end

    if ~isscalar(maxStaleTime) || maxStaleTime < 0
        error("cfg.gs.maxStaleTime must be a nonnegative scalar or Inf.");
    end

end
function gsBranches = initializeGSBranchCache(Nw, nTheta, dim)
% Create an inactive watcher-side GS branch cache.

    emptyRecord = makeEmptyGSBranchRecord(NaN, nTheta, dim);
    gsBranches = repmat(emptyRecord, Nw, 1);

    for j = 1:Nw
        gsBranches(j) = makeEmptyGSBranchRecord(j, nTheta, dim);
    end

end

function gsBranchesOut = ensureGSBranchCacheShape(gsBranchesIn, Nw, nTheta, dim)
% Ensure watcher.gsBranches is a column struct array with Nw records.
%
% The output is rebuilt from the current template so that assignment into a
% MATLAB struct array remains safe even if an older cache had missing fields.

    if ~isstruct(gsBranchesIn)
        error("watcher.gsBranches must be a struct array if it already exists.");
    end

    gsBranchesIn = gsBranchesIn(:);
    gsBranchesOut = initializeGSBranchCache(Nw, nTheta, dim);

    nCopy = min(numel(gsBranchesIn), Nw);

    for j = 1:nCopy
        gsBranchesOut(j) = mergeBranchRecordWithTemplate(gsBranchesIn(j), j, nTheta, dim);
    end

end

function recordOut = mergeBranchRecordWithTemplate(recordIn, branchID, nTheta, dim)
% Copy recognized fields from an existing cache record into a fresh template.

    recordOut = makeEmptyGSBranchRecord(branchID, nTheta, dim);
    fields = fieldnames(recordOut);

    for k = 1:numel(fields)
        f = fields{k};
        if isfield(recordIn, f) && ~isempty(recordIn.(f))
            recordOut.(f) = recordIn.(f);
        end
    end

    if isempty(recordOut.branchID) || ~isfinite(recordOut.branchID)
        recordOut.branchID = branchID;
    end

end

function record = makeEmptyGSBranchRecord(branchID, nTheta, dim)
% Create one inactive GS branch cache record.

    record = struct();

    record.branchID = branchID;
    record.sourceWatcherID = branchID;

    % Branch-model metadata.
    %
    % These fields are filled from GS records for valid nonlocal branches.
    % The inactive template keeps them so MATLAB struct-array assignment is
    % safe even before a valid broadcast.
    record.branchModel = "unknown";
    record.nTheta = nTheta;
    record.branchInfo = struct();

    record.theta = zeros(nTheta, 1);
    record.Ptheta = NaN(nTheta, nTheta);
    record.PthetaConditional = NaN(nTheta, nTheta);

    % Step 09-J.2 geometry metadata cache.
    % The cache template includes these fields so MATLAB struct-array
    % assignment remains safe before any valid GS broadcast occurs.
    record.OmegaBar = zeros(dim, dim);
    record.numOmegaUpdates = 0;
    record.lastLOSUnit = NaN(dim, 1);
    record.lastOmegaUpdateTime = NaN;
    record.lastMeasTime = NaN;
    record.outputFrame = "inertial";

    record.lastUpdateTime = NaN;
    record.receivedTime = NaN;
    record.age = Inf;
    record.isStale = true;

    record.version = 0;
    record.status = "empty";

    record.isLocalBranch = false;
    record.active = false;
    record.usedInPrediction = false;

    record.numUploads = NaN;
    record.numAcceptedUploads = NaN;
    record.numRejectedUploads = NaN;

    record.lastInnovationNorm = NaN;
    record.lastThetaChangeNorm = NaN;
    record.lastNoveltyScore = NaN;

end


function status = getRecordStatus(record)
% Safely read record.status as a string scalar.

    if isfield(record, "status") && ~isempty(record.status)
        status = string(record.status);
    else
        status = "empty";
    end

end

function age = computeRecordAge(record, t)
% Compute current age of a GS branch record.

    if ~isfield(record, "lastUpdateTime") || isempty(record.lastUpdateTime)
        age = Inf;
        return;
    end

    if ~isfinite(record.lastUpdateTime)
        age = Inf;
        return;
    end

    age = t - record.lastUpdateTime;

    if age < 0
        % Avoid negative ages caused by inconsistent test calls.
        age = 0;
    end

end

function validateThetaAndCovariance(theta, Ptheta, nTheta, branchID)
% Check dimensions and numerical validity of a branch parameter record.

    if numel(theta) ~= nTheta
        error("GS branch %d theta has wrong length. Expected %d, got %d.", ...
            branchID, nTheta, numel(theta));
    end

    if any(size(Ptheta) ~= [nTheta, nTheta])
        error("GS branch %d Ptheta has wrong size. Expected %d-by-%d.", ...
            branchID, nTheta, nTheta);
    end

    if any(~isfinite(theta))
        error("GS branch %d theta contains non-finite values.", branchID);
    end

    if any(~isfinite(Ptheta(:)))
        error("GS branch %d Ptheta contains non-finite values.", branchID);
    end

end

function OmegaBar = getOmegaBarField(record, dim, branchID)
%GETOMEGABARFIELD Read and validate record.OmegaBar for GS broadcast.

if isfield(record, "OmegaBar") && ~isempty(record.OmegaBar)
    OmegaBar = double(record.OmegaBar);
else
    OmegaBar = zeros(dim, dim);
end

if any(size(OmegaBar) ~= [dim, dim])
    error("broadcastGSRepositoryToWatcher:BadOmegaBarSize", ...
        "GS branch %d OmegaBar must be %d-by-%d.", branchID, dim, dim);
end

if any(~isfinite(OmegaBar(:)))
    error("broadcastGSRepositoryToWatcher:NonFiniteOmegaBar", ...
        "GS branch %d OmegaBar contains non-finite values.", branchID);
end

% Keep the cache symmetric even if an older upload had small roundoff.
OmegaBar = 0.5 * (OmegaBar + OmegaBar.');

end

function lastLOSUnit = getLOSField(record, dim)
%GETLOSFIELD Read last LOS unit vector from a GS record.

if isfield(record, "lastLOSUnit") && ~isempty(record.lastLOSUnit)
    lastLOSUnit = record.lastLOSUnit(:);
else
    lastLOSUnit = NaN(dim, 1);
end

if numel(lastLOSUnit) ~= dim
    error("broadcastGSRepositoryToWatcher:BadLastLOSUnitSize", ...
        "GS record lastLOSUnit must have length cfg.dim = %d.", dim);
end

end


function value = getOptionalField(s, fieldName, defaultValue)
% Read s.(fieldName) if it exists; otherwise return defaultValue.

    if isfield(s, fieldName) && ~isempty(s.(fieldName))
        value = s.(fieldName);
    else
        value = defaultValue;
    end

end
