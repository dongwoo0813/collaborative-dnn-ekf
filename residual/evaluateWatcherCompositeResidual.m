function [dComp, JetaComp, branchContrib, branchUsed, gateDiag] = evaluateWatcherCompositeResidual(watcher, eta, thetaLocal, cfg)
%{
Function:
    evaluateWatcherCompositeResidual.m

Purpose:
    Evaluate the Step 04 GS-assisted composite DNN residual model for one
    watcher.

    Watcher m predicts using

        d_comp,m(eta)
            = d_m(eta; theta_m^local)
              + sum_{j ~= m} w_{j|m} d_j(eta; theta_{j|m}^{GS}),

    where theta_m^local is the local branch parameter inside watcher m's
    augmented EKF state, theta_{j|m}^{GS} are nonlocal branch copies
    received from the ground station and stored in watcher.gsBranches(j),
    and w_{j|m} is a nonlocal branch weight.

Inputs:
    watcher    - Local DNN-EKF watcher structure.
                 Required fields:
                     watcher.localBranchID
                     watcher.nTheta

                 Optional fields:
                     watcher.gsBranches

    eta        - Physical target state estimate.
                 Size: 2*cfg.dim x 1.
                 Definition:
                     eta = [r_t; v_t].

    thetaLocal - Local branch parameter of watcher m.
                 Size: watcher.nTheta x 1.
                 This is normally x(watcher.idxTheta) inside
                 DNN_EKF_Predict_Local.m.

    cfg        - Simulation configuration structure.
                 Required fields:
                     cfg.dim
                     cfg.Nw

                 Optional GS weighting fields:
                     cfg.gs.nonlocalWeightMode
                         "none"   : all usable nonlocal branches use weight 1.
                         "scalar" : all usable nonlocal branches use
                                    cfg.gs.nonlocalWeight.

                     cfg.gs.nonlocalWeight
                     cfg.gs.nonlocalWeightMin
                     cfg.gs.nonlocalWeightMax

Outputs:
    dComp        - Composite residual acceleration.
                   Size: cfg.dim x 1.

    JetaComp     - Jacobian of dComp with respect to eta.
                   Size: cfg.dim x 2*cfg.dim.

    branchContrib - Individual weighted branch residual contributions.
                    Size: cfg.dim x cfg.Nw.
                    Column j is w_j d_j(eta; theta_j).
                    For the local branch, w_j = 1.

    branchUsed    - Logical vector of length cfg.Nw.
                    branchUsed(j) = true if branch j contributed with a
                    nonzero weight.

Main equations:
    Each fixed-feature output-layer branch is

        d_j(eta; theta_j) = W_j phi_j(eta),
        theta_j = vec(W_j).

    Its eta-Jacobian is

        partial d_j / partial eta = W_j partial phi_j / partial eta.

    The weighted composite model is

        d_comp = sum_j w_j d_j,

        partial d_comp / partial eta
            = sum_j w_j partial d_j / partial eta.

Notes:
    - This function only evaluates the mean residual model and its
      eta-Jacobian.
    - It does not modify watcher.xhat or watcher.P.
    - It does not use nonlocal Ptheta yet.
    - Nonlocal branch covariance can later be handled as an additional
      model-uncertainty covariance term.
    - The local branch is always included from thetaLocal with weight 1.
    - A nonlocal GS branch is included only if its cache record is active and
      usedInPrediction is true.
    - Step 09-E.1 adds scalar nonlocal weighting as a diagnostic to test
      whether the GS composite is over-adding nonlocal branch outputs.
%}

    % ---------------------------------------------------------------------
    % Basic dimensions and checks
    % ---------------------------------------------------------------------
    dim = cfg.dim;
    Nw = cfg.Nw;
    nEta = 2 * dim;

    eta = eta(:);
    thetaLocal = thetaLocal(:);

    if numel(eta) ~= nEta
        error("eta has wrong length in evaluateWatcherCompositeResidual. Expected %d, got %d.", ...
            nEta, numel(eta));
    end

    if ~isfield(watcher, "localBranchID")
        error("watcher.localBranchID is required in evaluateWatcherCompositeResidual.");
    end

    if ~isfield(watcher, "nTheta")
        error("watcher.nTheta is required in evaluateWatcherCompositeResidual.");
    end

    if numel(thetaLocal) ~= watcher.nTheta
        error("thetaLocal has wrong length. Expected %d, got %d.", ...
            watcher.nTheta, numel(thetaLocal));
    end

    localBranchID = watcher.localBranchID;

    if localBranchID < 1 || localBranchID > Nw
        error("Invalid watcher.localBranchID = %d.", localBranchID);
    end


    % ---------------------------------------------------------------------
    % Step 09-F.2: GS composite residual mode
    %
    % "additive":
    %     Original behavior.
    %
    % "gated_additive":
    %     Apply branch projection gate B_j(eta) to every branch contribution:
    %
    %         d_j = B_j dRaw_j,
    %         J_j = B_j JRaw_j.
    %
    %     The eta-derivative of B_j is intentionally ignored in this first
    %     implementation.
    % ---------------------------------------------------------------------
    compositeMode = getGSCompositeMode_step09f2(cfg);


    % ---------------------------------------------------------------------
    % Output allocation
    % ---------------------------------------------------------------------
    dComp = zeros(dim, 1);
    JetaComp = zeros(dim, nEta);
    branchContrib = zeros(dim, Nw);
    branchUsed = false(Nw, 1);

    gateDiag = struct();
    gateDiag.enabled = false;
    gateDiag.mode = compositeMode;
    % Cache raw per-branch parameter Jacobians computed during the same
    % forward/backprop pass as d_j and Jeta_j. Prediction covariance code
    % can reuse these instead of evaluating the same MLP again.
    gateDiag.JthetaRawAll = zeros(dim, watcher.nTheta, Nw);


    % ---------------------------------------------------------------------
    % Step 09-J.3: Direction-only bearing-FIM-gated composite residual.
    %
    % This mode needs the final available branch set A_m before it can
    % compute B_{j|m}. Therefore it is evaluated in a separate helper rather
    % than through the old branch-by-branch summation path.
    % ---------------------------------------------------------------------
    if compositeMode == "bearing_fim_gated"
        [dComp, JetaComp, branchContrib, branchUsed, gateDiag] = ...
            evaluateBearingFIMGatedComposite_step09j3( ...
            watcher, eta, thetaLocal, cfg);
        return;
    end

    if compositeMode == "output_information_fusion"
        [dComp, JetaComp, branchContrib, branchUsed, gateDiag] = ...
            evaluateOutputInformationComposite( ...
            watcher, eta, thetaLocal, cfg);
        return;
    end

    if compositeMode == "fim_weighted_additive"
        [dComp,JetaComp,branchContrib,branchUsed,gateDiag] = ...
            evaluateFIMWeightedAdditiveComposite( ...
            watcher,eta,thetaLocal,cfg);
        return;
    end


    % ---------------------------------------------------------------------
    % 1. Always include watcher m's own local branch.
    %
    % The local branch is the one estimated by this watcher's own EKF.
    % Therefore its weight is fixed to 1.
    % ---------------------------------------------------------------------
    [dLocalRaw, JLocalRaw, JthetaLocalRaw] = evaluateOneBranch( ...
        localBranchID, eta, thetaLocal, cfg);
    gateDiag.JthetaRawAll(:, :, localBranchID) = JthetaLocalRaw;

    localWeight = 1.0;

    [dLocal, JLocal] = applyGSCompositeMode_step09f2( ...
        localBranchID, ...
        dLocalRaw, ...
        JLocalRaw, ...
        localWeight, ...
        eta, ...
        cfg, ...
        compositeMode, ...
        true);

    dComp = dComp + dLocal;
    JetaComp = JetaComp + JLocal;

    branchContrib(:, localBranchID) = dLocal;
    branchUsed(localBranchID) = true;

    % ---------------------------------------------------------------------
    % 2. Add active nonlocal GS branch copies.
    %
    % Step 09-E.1:
    %   Nonlocal branches can be attenuated/amplified by a scalar weight.
    %   This is a diagnostic tool for checking whether full-strength
    %   nonlocal summation causes over-addition.
    % ---------------------------------------------------------------------
    if ~isfield(watcher, "gsBranches") || isempty(watcher.gsBranches)
        return;
    end

    nCached = min(numel(watcher.gsBranches), Nw);

    for j = 1:nCached

        if j == localBranchID
            continue;
        end

        branchRecord = watcher.gsBranches(j);

        if ~isUsableGSBranch(branchRecord)
            continue;
        end

        theta_j = branchRecord.theta(:);

        if numel(theta_j) ~= watcher.nTheta
            error("GS branch %d theta has wrong length. Expected %d, got %d.", ...
                j, watcher.nTheta, numel(theta_j));
        end

        branchWeight = getGSBranchWeight_step09e1(watcher, j, cfg);

        % If the weight is zero, this branch is effectively disabled for
        % prediction. Keeping branchUsed false makes numNonlocalBranchesUsed
        % reflect the number of actually contributing nonlocal branches.
        if abs(branchWeight) <= eps
            continue;
        end

        [dRaw_j, JRaw_j, JthetaRaw_j] = evaluateOneBranch( ...
            j, eta, theta_j, cfg);
        gateDiag.JthetaRawAll(:, :, j) = JthetaRaw_j;

        [d_j, J_j] = applyGSCompositeMode_step09f2( ...
            j, ...
            dRaw_j, ...
            JRaw_j, ...
            branchWeight, ...
            eta, ...
            cfg, ...
            compositeMode, ...
            false);

        dComp = dComp + d_j;
        JetaComp = JetaComp + J_j;

        branchContrib(:, j) = d_j;
        branchUsed(j) = true;

    end

end


function [dComp,JetaComp,branchContrib,branchUsed,gateDiag] = ...
    evaluateFIMWeightedAdditiveComposite(watcher,eta,thetaLocal,cfg)
%EVALUATEFIMWEIGHTEDADDITIVECOMPOSITE Geometry-shaped additive DNN sum.
%
% Each branch remains an additive function block. W_j selects the output
% directions supported by that source watcher's accumulated bearing
% geometry; it does not allocate a unit total weight across branches.

    dim=cfg.dim;
    Nw=cfg.Nw;
    nEta=2*dim;
    localBranchID=watcher.localBranchID;
    dRawAll=zeros(dim,Nw);
    JRawAll=zeros(dim,nEta,Nw);
    JthetaRawAll=zeros(dim,watcher.nTheta,Nw);
    branchUsed=false(Nw,1);

    [dRawAll(:,localBranchID),JRawAll(:,:,localBranchID), ...
        JthetaRawAll(:,:,localBranchID)] = evaluateOneBranch( ...
        localBranchID,eta,thetaLocal,cfg);
    branchUsed(localBranchID)=true;

    if isfield(watcher,"gsBranches") && ~isempty(watcher.gsBranches)
        nCached=min(numel(watcher.gsBranches),Nw);
        for j=1:nCached
            if j==localBranchID || ~isUsableGSBranch(watcher.gsBranches(j))
                continue;
            end
            theta_j=watcher.gsBranches(j).theta(:);
            if numel(theta_j)~=watcher.nTheta
                error("evaluateWatcherCompositeResidual:BadGSThetaSize", ...
                    "GS branch %d theta has wrong length. Expected %d, got %d.", ...
                    j,watcher.nTheta,numel(theta_j));
            end
            [dRawAll(:,j),JRawAll(:,:,j),JthetaRawAll(:,:,j)] = ...
                evaluateOneBranch(j,eta,theta_j,cfg);
            branchUsed(j)=true;
        end
    end

    [W,gateDiag]=computeFIMWeightedAdditiveWeights( ...
        watcher,branchUsed,cfg);
    gateDiag.JthetaRawAll=JthetaRawAll;
    dComp=zeros(dim,1);
    JetaComp=zeros(dim,nEta);
    branchContrib=zeros(dim,Nw);
    for j=1:Nw
        if ~branchUsed(j), continue; end
        branchContrib(:,j)=W(:,:,j)*dRawAll(:,j);
        dComp=dComp+branchContrib(:,j);
        JetaComp=JetaComp+W(:,:,j)*JRawAll(:,:,j);
    end
end


function [dComp,JetaComp,branchContrib,branchUsed,gateDiag] = ...
    evaluateOutputInformationComposite(watcher,eta,thetaLocal,cfg)
%EVALUATEOUTPUTINFORMATIONCOMPOSITE Fuse redundant residual experts.
%
% Every available branch estimates the same residual vector.  Its output
% covariance is obtained by propagating the conditional parameter
% covariance through the DNN output Jacobian.  The resulting Gaussian
% estimates are fused in information form.  B_j is held fixed within this
% one EKF linearization; derivatives of the covariance-derived weight are
% intentionally not included in the first-order state Jacobian.

    dim=cfg.dim;
    Nw=cfg.Nw;
    nEta=2*dim;
    localBranchID=watcher.localBranchID;
    dRawAll=zeros(dim,Nw);
    JRawAll=zeros(dim,nEta,Nw);
    JthetaRawAll=zeros(dim,watcher.nTheta,Nw);
    branchUsed=false(Nw,1);

    [dRawAll(:,localBranchID),JRawAll(:,:,localBranchID), ...
        JthetaRawAll(:,:,localBranchID)] = evaluateOneBranch( ...
        localBranchID,eta,thetaLocal,cfg);
    branchUsed(localBranchID)=true;

    if isfield(watcher,"gsBranches") && ~isempty(watcher.gsBranches)
        nCached=min(numel(watcher.gsBranches),Nw);
        for j=1:nCached
            if j==localBranchID || ~isUsableGSBranch(watcher.gsBranches(j))
                continue;
            end
            theta_j=watcher.gsBranches(j).theta(:);
            if numel(theta_j)~=watcher.nTheta
                error("evaluateWatcherCompositeResidual:BadGSThetaSize", ...
                    "GS branch %d theta has wrong length. Expected %d, got %d.", ...
                    j,watcher.nTheta,numel(theta_j));
            end
            [dRawAll(:,j),JRawAll(:,:,j),JthetaRawAll(:,:,j)] = ...
                evaluateOneBranch(j,eta,theta_j,cfg);
            branchUsed(j)=true;
        end
    end

    [B,gateDiag]=computeOutputInformationWeights( ...
        watcher,branchUsed,JthetaRawAll,cfg);
    gateDiag.JthetaRawAll=JthetaRawAll;
    dComp=zeros(dim,1);
    JetaComp=zeros(dim,nEta);
    branchContrib=zeros(dim,Nw);
    for j=1:Nw
        if ~branchUsed(j), continue; end
        branchContrib(:,j)=B(:,:,j)*dRawAll(:,j);
        dComp=dComp+branchContrib(:,j);
        JetaComp=JetaComp+B(:,:,j)*JRawAll(:,:,j);
    end
end


function [dComp, JetaComp, branchContrib, branchUsed, gateDiag] = ...
    evaluateBearingFIMGatedComposite_step09j3(watcher, eta, thetaLocal, cfg)
%{
Subfunction:
    evaluateBearingFIMGatedComposite_step09j3

Purpose:
    Evaluate the Step 09-J.3 direction-only bearing-FIM-gated GS composite
    residual.

Main equations:
    Available branch set:

        A_m = {local branch m} union {valid GS nonlocal branches}.

    Recipient-side geometry gates:

        OmegaSigma_m = sum_{l in A_m} OmegaBar_l + epsilon I,
        B_{j|m}      = OmegaSigma_m \ OmegaBar_j.

    Composite residual and eta-Jacobian:

        d_FIM,m    = sum_{j in A_m} B_{j|m} d_j,
        Jeta_FIM,m = sum_{j in A_m} B_{j|m} Jeta_j.

    In this step B_{j|m} is treated as fixed metadata during one prediction
    call. Thus there is no dB/deta chain-rule term here.
%}

    dim = cfg.dim;
    Nw = cfg.Nw;
    nEta = 2 * dim;

    localBranchID = watcher.localBranchID;

    dRawAll = zeros(dim, Nw);
    JRawAll = zeros(dim, nEta, Nw);
    JthetaRawAll = zeros(dim, watcher.nTheta, Nw);
    branchUsed = false(Nw, 1);

    % Local branch: evaluated from the recipient's EKF-owned thetaLocal.
    [dRawLocal, JRawLocal, JthetaRawLocal] = evaluateOneBranch( ...
        localBranchID, eta, thetaLocal, cfg);

    dRawAll(:, localBranchID) = dRawLocal;
    JRawAll(:, :, localBranchID) = JRawLocal;
    JthetaRawAll(:, :, localBranchID) = JthetaRawLocal;
    branchUsed(localBranchID) = true;

    % Nonlocal branches: evaluated from the recipient's GS cache copies.
    if isfield(watcher, "gsBranches") && ~isempty(watcher.gsBranches)

        nCached = min(numel(watcher.gsBranches), Nw);

        for j = 1:nCached

            if j == localBranchID
                continue;
            end

            branchRecord = watcher.gsBranches(j);

            if ~isUsableGSBranch(branchRecord)
                continue;
            end

            theta_j = branchRecord.theta(:);

            if numel(theta_j) ~= watcher.nTheta
                error("evaluateWatcherCompositeResidual:BadGSThetaSize", ...
                    "GS branch %d theta has wrong length. Expected %d, got %d.", ...
                    j, watcher.nTheta, numel(theta_j));
            end

            [dRaw_j, JRaw_j, JthetaRaw_j] = evaluateOneBranch( ...
                j, eta, theta_j, cfg);

            dRawAll(:, j) = dRaw_j;
            JRawAll(:, :, j) = JRaw_j;
            JthetaRawAll(:, :, j) = JthetaRaw_j;
            branchUsed(j) = true;

        end

    end

    [B, gateDiag] = computeBearingFIMGates(watcher, branchUsed, cfg);
    gateDiag.mode = "bearing_fim_gated";
    gateDiag.JthetaRawAll = JthetaRawAll;

    dComp = zeros(dim, 1);
    JetaComp = zeros(dim, nEta);
    branchContrib = zeros(dim, Nw);

    for j = 1:Nw

        if ~branchUsed(j)
            continue;
        end

        B_j = B(:, :, j);

        % Direction-gated contribution. B_j is metadata-fixed in this step.
        d_j = B_j * dRawAll(:, j);
        J_j = B_j * JRawAll(:, :, j);

        dComp = dComp + d_j;
        JetaComp = JetaComp + J_j;

        branchContrib(:, j) = d_j;

    end

end



function [d_j, Jeta_j, Jtheta_j] = evaluateOneBranch( ...
    branchID, eta, theta_j, cfg)
%{
Subfunction:
    evaluateOneBranch

Purpose:
    Evaluate one residual branch and its eta-Jacobian in a branch-model-aware
    way.

Current Step:
    Step 09-I.1 MLP-compatible GS composite residual.

Why this patch is needed:
    The old implementation assumed the fixed-feature LIP model,

        d_j = W_j phi_j(eta),

    and directly called:
        featureBlock(...)
        featureJacobianEta(...)

    That works only when cfg.dnn.branchModel = "fixed_feature_lip".
    For cfg.dnn.branchModel = "mlp_general", theta_j contains hidden-layer
    weights, hidden biases, output weights, and output bias, so the old
    reshape(theta_j, dim, nPhi) logic is invalid.

Branch-model-aware behavior:
    fixed_feature_lip:
        evaluateBranchResidualModel(...) internally computes W_j phi_j(eta).

    mlp_general:
        evaluateBranchResidualModel(...) internally calls
        branchMLPForwardJacobians(...).

Inputs:
    branchID  - residual branch index
    eta       - physical EKF state [r; v]
    theta_j   - branch parameter vector
    cfg       - simulation configuration

Outputs:
    d_j:
        raw branch residual before additive/gated GS weighting.

    Jeta_j:
        raw branch eta-Jacobian before additive/gated GS weighting.
%}

theta_j = theta_j(:);

[d_j, Jeta_j, Jtheta_j, ~] = evaluateBranchResidualModel( ...
    branchID, eta, theta_j, cfg);

end


function tf = isUsableGSBranch(branchRecord)
% Return true if a GS branch cache record should be used in prediction.

    tf = false;

    if ~isstruct(branchRecord)
        return;
    end

    if ~isfield(branchRecord, "theta") || isempty(branchRecord.theta)
        return;
    end

    if isfield(branchRecord, "active")
        if ~logical(branchRecord.active)
            return;
        end
    else
        return;
    end

    if isfield(branchRecord, "usedInPrediction")
        if ~logical(branchRecord.usedInPrediction)
            return;
        end
    end

    if isfield(branchRecord, "status")
        if string(branchRecord.status) ~= "valid"
            return;
        end
    end

    if isfield(branchRecord, "isStale")
        if logical(branchRecord.isStale)
            return;
        end
    end

    if any(~isfinite(branchRecord.theta(:)))
        return;
    end

    tf = true;

end

function branchWeight = getGSBranchWeight_step09e1(watcher, branchID, cfg)
%{
Function:
    getGSBranchWeight_step09e1

Purpose:
    Return the residual branch weight used in the GS composite model.

Rule:
    local branch:
        weight = 1

    nonlocal branch:
        if cfg.gs.nonlocalWeightMode == "scalar"
            weight = cfg.gs.nonlocalWeight
        otherwise
            weight = 1

Why this helper exists:
    Step 09-E.1 is a diagnostic step. We want to test whether the current
    GS composite residual

        d_GS = d_local + sum d_nonlocal

    is over-adding nonlocal branches. A scalar nonlocal weight gives a clean
    first check before introducing more complicated confidence, covariance,
    age, or innovation-based weighting.
%}

    branchWeight = 1.0;

    if ~isfield(watcher, "localBranchID")
        return;
    end

    % The local branch is always trusted as the EKF-owned branch.
    if branchID == watcher.localBranchID
        branchWeight = 1.0;
        return;
    end

    if ~isfield(cfg, "gs")
        return;
    end

    if ~isfield(cfg.gs, "nonlocalWeightMode")
        return;
    end

    mode = string(cfg.gs.nonlocalWeightMode);

    switch mode

        case "none"
            branchWeight = 1.0;

        case "scalar"
            if isfield(cfg.gs, "nonlocalWeight")
                branchWeight = cfg.gs.nonlocalWeight;
            else
                branchWeight = 1.0;
            end

        otherwise
            branchWeight = 1.0;

    end

    % Clamp the weight to avoid accidental unstable values during sweeps.
    wMin = 0.0;
    wMax = 2.0;

    if isfield(cfg.gs, "nonlocalWeightMin")
        wMin = cfg.gs.nonlocalWeightMin;
    end

    if isfield(cfg.gs, "nonlocalWeightMax")
        wMax = cfg.gs.nonlocalWeightMax;
    end

    branchWeight = min(max(branchWeight, wMin), wMax);

end



function compositeMode = getGSCompositeMode_step09f2(cfg)
%{
Function:
    getGSCompositeMode_step09f2

Purpose:
    Read the GS composite residual mode.

Modes:
    "additive":
        Original GS composite residual model.

    "gated_additive":
        Apply B_j(eta) to each branch output and branch eta-Jacobian.

Default:
    If cfg.gs.compositeMode is missing, use "additive" for backward
    compatibility.
%}

compositeMode = "additive";

if isfield(cfg, "gs") && isfield(cfg.gs, "compositeMode")
    compositeMode = string(cfg.gs.compositeMode);
end

switch compositeMode

    case "additive"
        return;

    case "gated_additive"
        return;

    case "local_full_plus_gated_nonlocal"
        return;

    case "bearing_fim_gated"
        return;

    case "output_information_fusion"
        return;

    case "fim_weighted_additive"
        return;

    otherwise
        error("evaluateWatcherCompositeResidual:UnsupportedCompositeMode", ...
            "Unsupported cfg.gs.compositeMode = %s.", compositeMode);

end


end

function [d_j, Jeta_j] = applyGSCompositeMode_step09f2( ...
    branchID, dRaw_j, JRaw_j, branchWeight, eta, cfg, compositeMode, isLocalBranch)
%{
Function:
    applyGSCompositeMode_step09f2

Purpose:
    Convert a raw branch residual contribution into the contribution used
    by the GS composite model.

Modes:
    additive:
        All branches are used directly.

            d_j = w_j dRaw_j
            J_j = w_j JRaw_j

    gated_additive:
        All branches are projected by B_j(eta).

            d_j = w_j B_j dRaw_j
            J_j = w_j [B_j JRaw_j + dB_j/deta * dRaw_j]

    local_full_plus_gated_nonlocal:
        The local branch keeps full correction authority, while nonlocal
        GS branches are projected.

            local:
                d_i = dRaw_i
                J_i = JRaw_i

            nonlocal:
                d_j = B_j dRaw_j
                J_j = B_j JRaw_j + dB_j/deta * dRaw_j

Why local_full_plus_gated_nonlocal exists:
    Step 09-F.4/F.5 showed that fully gated_additive can improve residual
    approximation but weaken tracking. This hybrid preserves the local
    branch's full residual correction while still reducing nonlocal
    over-counting.
%}

    dim = cfg.dim;

    switch compositeMode

        case "additive"

            d_j = branchWeight * dRaw_j;
            Jeta_j = branchWeight * JRaw_j;

        case "gated_additive"

            [d_j, Jeta_j] = applyBranchGateExact_step09f2( ...
                branchID, dRaw_j, JRaw_j, branchWeight, eta, cfg);

        case "local_full_plus_gated_nonlocal"

            if isLocalBranch
                d_j = branchWeight * dRaw_j;
                Jeta_j = branchWeight * JRaw_j;
            else
                [d_j, Jeta_j] = applyBranchGateExact_step09f2( ...
                    branchID, dRaw_j, JRaw_j, branchWeight, eta, cfg);
            end

        otherwise

            error("evaluateWatcherCompositeResidual:UnsupportedCompositeMode", ...
                "Unsupported compositeMode = %s.", compositeMode);

    end

end




function [d_j, Jeta_j] = applyBranchGateExact_step09f2( ...
    branchID, dRaw_j, JRaw_j, branchWeight, eta, cfg)
% Apply exact eta-chain-rule gate correction:
%
%     d_j = B_j(eta) dRaw_j
%
%     partial d_j / partial eta_k
%       = B_j partial dRaw_j / partial eta_k
%         + partial B_j / partial eta_k dRaw_j.

    dim = cfg.dim;

    [B_j, dB_dEta] = branchGateMatrix(branchID, eta, cfg);

    if ~isequal(size(B_j), [dim, dim])
        error("evaluateWatcherCompositeResidual:InvalidGateSize", ...
            "B_j for branch %d has size %dx%d, expected %dx%d.", ...
            branchID, size(B_j, 1), size(B_j, 2), dim, dim);
    end

    nEta = size(JRaw_j, 2);

    if ~isequal(size(dRaw_j), [dim, 1])
        error("evaluateWatcherCompositeResidual:InvalidRawResidualSize", ...
            "dRaw_j for branch %d has size %dx%d, expected %dx1.", ...
            branchID, size(dRaw_j, 1), size(dRaw_j, 2), dim);
    end

    if size(JRaw_j, 1) ~= dim
        error("evaluateWatcherCompositeResidual:InvalidRawJacobianSize", ...
            "JRaw_j for branch %d has %d rows, expected %d.", ...
            branchID, size(JRaw_j, 1), dim);
    end

    if ~isequal(size(dB_dEta), [dim, dim, nEta])
        error("evaluateWatcherCompositeResidual:InvalidGateDerivativeSize", ...
            "dB_dEta for branch %d has incompatible size.", branchID);
    end

    JGate_j = zeros(dim, nEta);

    for kEta = 1:nEta
        JGate_j(:, kEta) = dB_dEta(:, :, kEta) * dRaw_j;
    end

    d_j = branchWeight * (B_j * dRaw_j);
    Jeta_j = branchWeight * (B_j * JRaw_j + JGate_j);

end
