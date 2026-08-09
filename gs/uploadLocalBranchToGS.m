function [gsRepo, uploadPacket] = uploadLocalBranchToGS(gsRepo, watcher, t, cfg)
%{
Function:
    uploadLocalBranchToGS.m

Purpose:
    Upload one watcher's local DNN branch estimate to the ground-station
    repository.

    In Step 04, watcher i estimates its own local branch parameter

        theta_i,

    inside the augmented state

        X_i = [eta_i; theta_i].

    This function extracts

        theta_i
        P_theta_i theta_i

    from watcher i and stores them in the GS repository branch record.

Inputs:
    gsRepo  - Ground-station repository structure created by
              initGSRepository.m.

    watcher - Local DNN-EKF watcher structure.
              Required fields:
                  watcher.id
                  watcher.xhat
                  watcher.P
                  watcher.idxTheta
                  watcher.nTheta
                  watcher.localBranchID

    t       - Current simulation time [s].

    cfg     - Simulation configuration structure.

Outputs:
    gsRepo       - Updated ground-station repository.

    uploadPacket - Structure containing the uploaded branch information.

Main equations:
    The uploaded branch parameter is

        theta_i = xhat_i(idxTheta).

    The uploaded branch covariance is

        P_theta_i theta_i = P_i(idxTheta, idxTheta).

Notes:
    - Step 04a accepts every upload.
    - No event-triggering, novelty test, or covariance fusion is used yet.
    - The GS repository simply overwrites branch i with the latest upload.
    - Later, this function can be extended to reject stale or low-novelty
      uploads.
%}

    % ---------------------------------------------------------------------
    % Basic checks
    % ---------------------------------------------------------------------
    if ~isfield(gsRepo, "branch")
        error("gsRepo.branch does not exist. Initialize GS with initGSRepository.m.");
    end

    if ~isfield(watcher, "id")
        error("watcher.id is required in uploadLocalBranchToGS.");
    end

    if ~isfield(watcher, "idxTheta")
        error("watcher.idxTheta is required in uploadLocalBranchToGS.");
    end

    if ~isfield(watcher, "localBranchID")
        error("watcher.localBranchID is required in uploadLocalBranchToGS.");
    end

    branchID = watcher.localBranchID;

    if branchID < 1 || branchID > numel(gsRepo.branch)
        error("Invalid branchID = %d in uploadLocalBranchToGS.", branchID);
    end

    idxTheta = watcher.idxTheta(:);

    theta = watcher.xhat(idxTheta);
    Ptheta = watcher.P(idxTheta, idxTheta);

    Ptheta = 0.5 * (Ptheta + Ptheta.');
    PthetaConditional = conditionalParameterCovariance( ...
        watcher.P,watcher.idxEta,watcher.idxTheta);

    nThetaExpected = gsRepo.nThetaPerBranch;

    if numel(theta) ~= nThetaExpected
        error("Uploaded theta has wrong length. Expected %d, got %d.", ...
            nThetaExpected, numel(theta));
    end

    if any(size(Ptheta) ~= [nThetaExpected, nThetaExpected])
        error("Uploaded Ptheta has wrong size.");
    end


    % Step 09-J.2 geometry metadata carried with the branch parameters.
    % This does not affect the current additive GS prediction; it only makes
    % the future bearing_fim_gated fusion possible.
    geomPayload = getWatcherGeometryPayload(watcher, cfg);


    % ---------------------------------------------------------------------
    % Previous GS record, used only for diagnostics
    % ---------------------------------------------------------------------
    oldRecord = gsRepo.branch(branchID);

    if string(oldRecord.status) == "valid"
        thetaChangeNorm = norm(theta - oldRecord.theta);
    else
        thetaChangeNorm = NaN;
    end

    if isfield(watcher, "lastInnovation") && ~isempty(watcher.lastInnovation)
        innovationNorm = norm(watcher.lastInnovation);
    else
        innovationNorm = NaN;
    end

    % ---------------------------------------------------------------------
    % Build upload packet
    % ---------------------------------------------------------------------
    uploadPacket = struct();

    uploadPacket.sourceWatcherID = watcher.id;
    uploadPacket.branchID = branchID;
    uploadPacket.time = t;

    uploadPacket.theta = theta;
    uploadPacket.Ptheta = Ptheta;
    uploadPacket.PthetaConditional = PthetaConditional;


    % Step 09-J.2 bearing-FIM geometry metadata.
    % OmegaBar describes which residual-output directions have received
    % accumulated bearing support for this source watcher.
    uploadPacket.OmegaBar = geomPayload.OmegaBar;
    uploadPacket.numOmegaUpdates = geomPayload.numOmegaUpdates;
    uploadPacket.lastLOSUnit = geomPayload.lastLOSUnit;
    uploadPacket.lastOmegaUpdateTime = geomPayload.lastOmegaUpdateTime;
    uploadPacket.lastMeasTime = geomPayload.lastMeasTime;
    uploadPacket.outputFrame = geomPayload.outputFrame;



    % Step 09-I.2 branch-model metadata.
    %
    % The GS repository stores this together with theta/Ptheta so that a
    % watcher can later check whether a received nonlocal branch is
    % compatible with its local branch architecture.
    [~, branchInfo] = branchThetaNumel(cfg);

    uploadPacket.branchModel = string(branchInfo.branchModel);
    uploadPacket.nTheta = numel(theta);
    uploadPacket.branchInfo = branchInfo;

    uploadPacket.thetaNorm = norm(theta);
    uploadPacket.thetaChangeNorm = thetaChangeNorm;
    uploadPacket.innovationNorm = innovationNorm;

    uploadPacket.accepted = true;
    uploadPacket.rejectReason = "";

    % ---------------------------------------------------------------------
    % Accept upload and update GS branch record
    % ---------------------------------------------------------------------
    newRecord = oldRecord;

    newRecord.branchID = branchID;
    newRecord.sourceWatcherID = watcher.id;

    newRecord.theta = theta;
    newRecord.Ptheta = Ptheta;
    newRecord.PthetaConditional = PthetaConditional;


    % Store geometry payload in the GS record. Recipients will later compute
    % B_{j|m} from all available OmegaBar matrices; GS does not store B_j.
    newRecord.OmegaBar = uploadPacket.OmegaBar;
    newRecord.numOmegaUpdates = uploadPacket.numOmegaUpdates;
    newRecord.lastLOSUnit = uploadPacket.lastLOSUnit;
    newRecord.lastOmegaUpdateTime = uploadPacket.lastOmegaUpdateTime;
    newRecord.lastMeasTime = uploadPacket.lastMeasTime;
    newRecord.outputFrame = uploadPacket.outputFrame;



    newRecord.branchModel = uploadPacket.branchModel;
    newRecord.nTheta = uploadPacket.nTheta;
    newRecord.branchInfo = uploadPacket.branchInfo;

    newRecord.lastUpdateTime = t;
    newRecord.version = oldRecord.version + 1;
    newRecord.status = "valid";

    newRecord.numUploads = oldRecord.numUploads + 1;
    newRecord.numAcceptedUploads = oldRecord.numAcceptedUploads + 1;

    newRecord.lastInnovationNorm = innovationNorm;
    newRecord.lastThetaChangeNorm = thetaChangeNorm;

    % For now, novelty score is simply the parameter change norm.
    % Later this can be replaced by an event-triggering score.
    newRecord.lastNoveltyScore = thetaChangeNorm;

    newRecord.age = 0;
    newRecord.isStale = false;

    gsRepo.branch(branchID) = newRecord;

    % ---------------------------------------------------------------------
    % Update repository-level metadata
    % ---------------------------------------------------------------------
    if ~isfield(gsRepo, "numTotalUploads")
        gsRepo.numTotalUploads = 0;
    end

    gsRepo.numTotalUploads = gsRepo.numTotalUploads + 1;
    gsRepo.lastGlobalUpdateTime = t;

end


function geomPayload = getWatcherGeometryPayload(watcher, cfg)
%GETWATCHERGEOMETRYPAYLOAD Extract Step 09-J OmegaBar metadata for upload.
%
% The source of truth is the local watcher state. If this function is called
% on an older watcher object without Step 09-J.1 fields, it returns safe
% empty defaults so previous tests do not fail because of missing metadata.

dim = cfg.dim;

if isfield(watcher, "OmegaBar") && ~isempty(watcher.OmegaBar)
    OmegaBar = double(watcher.OmegaBar);
else
    OmegaBar = zeros(dim, dim);
end

if any(size(OmegaBar) ~= [dim, dim])
    error("uploadLocalBranchToGS:BadOmegaBarSize", ...
        "watcher.OmegaBar must be %d-by-%d.", dim, dim);
end

% Symmetrize before upload so downstream gate construction starts from a
% numerically clean support matrix.
OmegaBar = 0.5 * (OmegaBar + OmegaBar.');

if any(~isfinite(OmegaBar(:)))
    error("uploadLocalBranchToGS:NonFiniteOmegaBar", ...
        "watcher.OmegaBar contains non-finite values.");
end

if isfield(watcher, "numOmegaUpdates") && ~isempty(watcher.numOmegaUpdates)
    numOmegaUpdates = watcher.numOmegaUpdates;
else
    numOmegaUpdates = 0;
end

if isfield(watcher, "lastLOSUnit") && ~isempty(watcher.lastLOSUnit)
    lastLOSUnit = watcher.lastLOSUnit(:);
else
    lastLOSUnit = NaN(dim, 1);
end

if numel(lastLOSUnit) ~= dim
    error("uploadLocalBranchToGS:BadLastLOSUnitSize", ...
        "watcher.lastLOSUnit must have length cfg.dim = %d.", dim);
end

if isfield(watcher, "lastOmegaUpdateTime") && ~isempty(watcher.lastOmegaUpdateTime)
    lastOmegaUpdateTime = watcher.lastOmegaUpdateTime;
else
    lastOmegaUpdateTime = NaN;
end

geomPayload = struct();
geomPayload.OmegaBar = OmegaBar;
geomPayload.numOmegaUpdates = numOmegaUpdates;
geomPayload.lastLOSUnit = lastLOSUnit;
geomPayload.lastOmegaUpdateTime = lastOmegaUpdateTime;
geomPayload.lastMeasTime = lastOmegaUpdateTime;
geomPayload.outputFrame = getOutputFrame(cfg);

end

function outputFrame = getOutputFrame(cfg)
%GETOUTPUTFRAME Frame label for residual acceleration and OmegaBar metadata.
%
% The current 2-D simulation outputs residual acceleration in the inertial
% simulation frame. A config field is allowed for future body-frame or 3-D
% extensions without changing the GS payload schema again.

outputFrame = "inertial";

if isfield(cfg, "gs") && isfield(cfg.gs, "fimGate") && ...
        isfield(cfg.gs.fimGate, "outputFrame") && ...
        ~isempty(cfg.gs.fimGate.outputFrame)
    outputFrame = string(cfg.gs.fimGate.outputFrame);
elseif isfield(cfg, "dnn") && isfield(cfg.dnn, "outputFrame") && ...
        ~isempty(cfg.dnn.outputFrame)
    outputFrame = string(cfg.dnn.outputFrame);
end

end
