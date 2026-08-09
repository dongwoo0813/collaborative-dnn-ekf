function [Qnonlocal, SdNonlocal, diagInfo] = ...
    computeNonlocalBranchCovarianceInjection( ...
        watcher, eta, cfg, baseResidualCache)
%{
Function:
    computeNonlocalBranchCovarianceInjection.m

Purpose:
    Compute the nonlocal GS-branch covariance injection term for one
    watcher-local DNN-EKF prediction.

    Watcher m may use nonlocal GS branch copies in the mean prediction, but
    the nonlocal parameters theta_j, j ~= m, are not appended to the local
    EKF state.

    Therefore, their uncertainty is injected as an external model
    uncertainty covariance term

        Q_X,-m = M_X S_d,-m M_X',

    where S_d,-m is a residual-acceleration covariance surrogate.

Composite-mode consistency:
    additive:
        d_nonlocal,j = d_j,
        Jtheta_eff,j = Jtheta_j.

    gated_additive:
        d_nonlocal,j = B_j(eta) d_j,
        Jtheta_eff,j = B_j(eta) Jtheta_j.

    local_full_plus_gated_nonlocal:
        local branch is full, nonlocal branches are gated,
        Jtheta_eff,j = B_j(eta) Jtheta_j for j ~= local branch.

    bearing_fim_gated:
        d_nonlocal,j = B_{j|m} d_j,
        Jtheta_eff,j = B_{j|m} Jtheta_j,

        where

            B_{j|m} = OmegaSigma_m \ OmegaBar_j,

            OmegaSigma_m = sum_{l in A_m} OmegaBar_l + epsilon I.

Inputs:
    watcher - Local watcher structure.
              Required fields:
                  watcher.localBranchID
                  watcher.nEta
                  watcher.nTheta
                  watcher.nX
                  watcher.gsBranches

    eta     - Current local physical estimate eta = [r_t; v_t].
              Size: watcher.nEta x 1.

    cfg     - Simulation configuration.
              Required fields:
                  cfg.dim
                  cfg.Nw
                  cfg.dt

              Optional fields:
                  cfg.gs.useNonlocalBranchCovariance
                  cfg.gs.youngMode
                  cfg.gs.SresNonlocal
                  cfg.gs.compositeMode
                  cfg.gs.fimGate.epsilon
                  cfg.gs.fimGate.normalizeTrace

Outputs:
    Qnonlocal  - Augmented-state covariance contribution.
                 Size: watcher.nX x watcher.nX.

    SdNonlocal - Residual-acceleration covariance surrogate.
                 Size: cfg.dim x cfg.dim.

    diagInfo   - Diagnostic structure.

Notes:
    - This function does not modify watcher.P.
    - This function does not append nonlocal theta_j to the EKF state.
    - It only computes the conservative covariance surrogate.
    - The first implementation uses the uniform Young bound by default:

          a_j = N_nonlocal,

      where N_nonlocal is the number of active nonlocal branches used by
      watcher m.

    - Step 09-J.4 adds consistency with compositeMode = "bearing_fim_gated"
      by applying the same recipient-side B_{j|m} gates to nonlocal
      Jtheta_j before forming Jtheta_eff,j Ptheta_j Jtheta_eff,j'.
%}

    dim = cfg.dim;   % Residual acceleration output dimension.
    Nw  = cfg.Nw;    % Number of watcher branches.

    eta = eta(:);

    if nargin < 4
        baseResidualCache = struct("valid", false);
    end

    if numel(eta) ~= watcher.nEta
        error("computeNonlocalBranchCovarianceInjection:BadEtaSize", ...
            "eta has wrong length. Expected %d, got %d.", ...
            watcher.nEta, numel(eta));
    end

    Qnonlocal = zeros(watcher.nX, watcher.nX);
    SdNonlocal = zeros(dim, dim);

    compositeMode = getCompositeMode_step09J4(cfg);

    diagInfo = struct();
    diagInfo.enabled = false;
    diagInfo.compositeMode = compositeMode;
    diagInfo.branchIDs = [];
    diagInfo.youngCoefficients = zeros(Nw, 1);
    diagInfo.traceSj = zeros(Nw, 1);
    diagInfo.traceRawSj = zeros(Nw, 1);
    diagInfo.traceBgate = NaN(Nw, 1);
    diagInfo.normBgate = NaN(Nw, 1);
    diagInfo.traceSdNonlocal = 0;
    diagInfo.traceQnonlocal = 0;
    diagInfo.numActiveNonlocal = 0;

    % Step 09-J.4 bearing-FIM-gate diagnostics.
    diagInfo.fimGateEnabled = false;
    diagInfo.fimGate = struct();
    diagInfo.gateSumIdentityError = NaN;
    diagInfo.condOmegaSigma = NaN;
    diagInfo.minEigOmegaSigma = NaN;

    if ~isfield(cfg, "gs") || ~isfield(cfg.gs, "useNonlocalBranchCovariance")
        return;
    end

    if ~logical(cfg.gs.useNonlocalBranchCovariance)
        return;
    end

    diagInfo.enabled = true;

    if ~isfield(watcher, "gsBranches") || isempty(watcher.gsBranches)
        return;
    end

    localBranchID = watcher.localBranchID;

    activeBranchIDs = [];

    for j = 1:min(numel(watcher.gsBranches), Nw)

        if j == localBranchID
            continue;
        end

        branchRecord = watcher.gsBranches(j);

        if isUsableNonlocalBranchForCovariance(branchRecord)
            activeBranchIDs(end+1) = j; %#ok<AGROW>
        end

    end

    diagInfo.branchIDs = activeBranchIDs(:);
    diagInfo.numActiveNonlocal = numel(activeBranchIDs);

    if isempty(activeBranchIDs)
        return;
    end

    % ---------------------------------------------------------------------
    % Young coefficients.
    %
    % First implementation:
    %   uniform bound with mu_ij = 1.
    %
    % If there are N_nonlocal active branches, then each coefficient is
    %
    %   a_j = N_nonlocal.
    %
    % This is conservative, but PSD-safe and simple.
    % ---------------------------------------------------------------------
    Nnonlocal = numel(activeBranchIDs);

    youngMode = "uniform";

    if isfield(cfg.gs, "youngMode")
        youngMode = string(cfg.gs.youngMode);
    end

    switch youngMode

        case "uniform"
            aUniform = Nnonlocal;
            youngCoefficients = aUniform * ones(Nw, 1);

        otherwise
            error("computeNonlocalBranchCovarianceInjection:BadYoungMode", ...
                "Unsupported cfg.gs.youngMode: %s", youngMode);

    end

    % ---------------------------------------------------------------------
    % Step 09-J.4:
    % Precompute bearing-FIM gates if covariance injection must be
    % consistent with compositeMode = "bearing_fim_gated".
    %
    % The gate depends on the full available branch set A_m, including the
    % local branch. Only nonlocal branches are summed into SdNonlocal, but
    % local OmegaBar must still be included in OmegaSigma_m.
    % ---------------------------------------------------------------------
    useBearingFIMGate = (compositeMode == "bearing_fim_gated");
    useOutputInformationWeight = ...
        (compositeMode == "output_information_fusion");
    useFIMWeightedAdditive = ...
        (compositeMode == "fim_weighted_additive");
    useMatrixGate = useBearingFIMGate || useOutputInformationWeight || ...
        useFIMWeightedAdditive;

    Bfim = [];

    if useMatrixGate

        branchUsedForGate = false(Nw, 1);
        branchUsedForGate(localBranchID) = true;
        branchUsedForGate(activeBranchIDs) = true;

        useCachedGate = isfield(baseResidualCache, "valid") && ...
            baseResidualCache.valid && ...
            isfield(baseResidualCache, "gateDiag") && ...
            isfield(baseResidualCache.gateDiag, "B") && ...
            isequal(baseResidualCache.branchUsed, branchUsedForGate);

        if useCachedGate
            Bfim = baseResidualCache.gateDiag.B;
            fimGateDiag = baseResidualCache.gateDiag;
        elseif useBearingFIMGate
            [Bfim, fimGateDiag] = computeBearingFIMGates( ...
                watcher, branchUsedForGate, cfg);
        elseif useFIMWeightedAdditive
            [Bfim,fimGateDiag] = computeFIMWeightedAdditiveWeights( ...
                watcher,branchUsedForGate,cfg);
        else
            thetaLocal=watcher.xhat(watcher.idxTheta);
            [~,~,~,branchUsedEvaluated,fimGateDiag] = ...
                evaluateWatcherCompositeResidual( ...
                watcher,eta,thetaLocal,cfg);
            if ~isequal(branchUsedEvaluated,branchUsedForGate)
                error("computeNonlocalBranchCovarianceInjection:BranchSetMismatch", ...
                    "Output-information branch set changed during prediction.");
            end
            Bfim=fimGateDiag.B;
        end

        diagInfo.fimGateEnabled = true;
        diagInfo.fimGate = fimGateDiag;
        diagInfo.gateSumIdentityError = fimGateDiag.sumGateIdentityError;
        diagInfo.condOmegaSigma = fimGateDiag.condOmegaSigma;
        diagInfo.minEigOmegaSigma = fimGateDiag.minEigOmegaSigma;

    end

    % ---------------------------------------------------------------------
    % Build residual-acceleration covariance surrogate:
    %
    %   S_d,-m = sum_j a_j S_d,j + S_res,
    %
    % where
    %
    %   additive:
    %       S_d,j = Jtheta_j Ptheta_j Jtheta_j'
    %
    %   bearing_fim_gated:
    %       S_d,j = (B_{j|m} Jtheta_j) Ptheta_j (B_{j|m} Jtheta_j)'.
    % ---------------------------------------------------------------------
    for idx = 1:Nnonlocal

        j = activeBranchIDs(idx);
        branchRecord = watcher.gsBranches(j);

        if useOutputInformationWeight && ...
                isfield(branchRecord,"PthetaConditional") && ...
                ~isempty(branchRecord.PthetaConditional) && ...
                all(isfinite(branchRecord.PthetaConditional(:)))
            Ptheta_j = branchRecord.PthetaConditional;
        else
            Ptheta_j = branchRecord.Ptheta;
        end
        Ptheta_j = 0.5 * (Ptheta_j + Ptheta_j.');

        theta_j = branchRecord.theta(:);

        if useMatrixGate
            Bgate_j = Bfim(:, :, j);
        else
            Bgate_j = [];
        end

        % Branch-model-aware nonlocal theta sensitivity.
        %
        % fixed_feature_lip:
        %   Jtheta_j = phi_j(eta)^T kron I_dim
        %
        % mlp_general:
        %   Jtheta_j is the MLP output Jacobian with respect to all
        %   hidden/output weights and biases.
        %
        % For gated composite modes, the effective theta sensitivity is
        % gate_j * Jtheta_j because the gate multiplies the branch output
        % but does not depend on theta_j.
        JthetaRawCached = [];
        if isfield(baseResidualCache, "valid") && ...
                baseResidualCache.valid && ...
                isfield(baseResidualCache, "gateDiag") && ...
                isfield(baseResidualCache.gateDiag, "JthetaRawAll") && ...
                size(baseResidualCache.gateDiag.JthetaRawAll, 3) >= j && ...
                baseResidualCache.branchUsed(j)
            JthetaRawCached = ...
                baseResidualCache.gateDiag.JthetaRawAll(:, :, j);
        end

        [JthetaEff_j, JthetaRaw_j, BgateUsed_j] = ...
            effectiveNonlocalThetaJacobian( ...
                j, eta, theta_j, cfg, Bgate_j, JthetaRawCached);

        Sj = JthetaEff_j * Ptheta_j * JthetaEff_j.';
        Sj = 0.5 * (Sj + Sj.');

        rawSj = JthetaRaw_j * Ptheta_j * JthetaRaw_j.';
        rawSj = 0.5 * (rawSj + rawSj.');

        aj = youngCoefficients(j);

        SdNonlocal = SdNonlocal + aj * Sj;

        diagInfo.youngCoefficients(j) = aj;
        diagInfo.traceSj(j) = trace(Sj);
        diagInfo.traceRawSj(j) = trace(rawSj);

        if ~isempty(BgateUsed_j)
            diagInfo.traceBgate(j) = trace(BgateUsed_j);
            diagInfo.normBgate(j) = norm(BgateUsed_j, "fro");
        end

    end

    if isfield(cfg.gs, "SresNonlocal")
        Sres = cfg.gs.SresNonlocal;

        if isscalar(Sres)
            Sres = Sres * eye(dim);
        end

        if any(size(Sres) ~= [dim, dim])
            error("computeNonlocalBranchCovarianceInjection:BadSresNonlocalSize", ...
                "cfg.gs.SresNonlocal must be scalar or %d-by-%d.", dim, dim);
        end

        SdNonlocal = SdNonlocal + Sres;
    end

    SdNonlocal = 0.5 * (SdNonlocal + SdNonlocal.');

    % ---------------------------------------------------------------------
    % Discrete-time acceleration-to-state covariance mapping.
    %
    % eta = [r; v]
    %
    % M_eta =
    %   [0.5*dt^2*I;
    %    dt*I]
    %
    % X = [eta; theta_m]
    %
    % M_X =
    %   [M_eta;
    %    zeros(nTheta, dim)]
    % ---------------------------------------------------------------------
    dt = cfg.dt;

    Mx = zeros(watcher.nX, dim);

    idxR = 1:dim;
    idxV = dim + (1:dim);

    Mx(idxR, :) = 0.5 * dt^2 * eye(dim);
    Mx(idxV, :) = dt * eye(dim);

    Qnonlocal = Mx * SdNonlocal * Mx.';
    Qnonlocal = 0.5 * (Qnonlocal + Qnonlocal.');

    diagInfo.traceSdNonlocal = trace(SdNonlocal);
    diagInfo.traceQnonlocal = trace(Qnonlocal);

end

function [JthetaEff, JthetaRaw, BgateUsed] = ...
    effectiveNonlocalThetaJacobian( ...
        branchID, eta, theta_j, cfg, Bfim_j, JthetaRawCached)
%EFFECTIVENONLOCALTHETAJACOBIAN Branch-model-aware nonlocal theta Jacobian.
%
% Purpose:
%   Compute the effective residual-output sensitivity with respect to a
%   nonlocal GS branch parameter vector theta_j.
%
% Composite-mode consistency:
%   additive:
%       d_nonlocal,j = d_j,
%       JthetaEff    = JthetaRaw.
%
%   gated_additive:
%       d_nonlocal,j = B_j(eta) d_j,
%       JthetaEff    = B_j(eta) JthetaRaw.
%
%   local_full_plus_gated_nonlocal:
%       local branch is full, nonlocal branches are gated,
%       JthetaEff    = B_j(eta) JthetaRaw for j ~= local branch.
%
%   bearing_fim_gated:
%       d_nonlocal,j = B_{j|m} d_j,
%       JthetaEff    = B_{j|m} JthetaRaw.
%
%   In bearing_fim_gated mode, B_{j|m} is supplied by the caller after it
%   computes recipient-side gates from the available OmegaBar set.

    if nargin < 5
        Bfim_j = [];
    end

    if nargin < 6
        JthetaRawCached = [];
    end

    eta = eta(:);
    theta_j = theta_j(:);

    if isempty(JthetaRawCached)
        [~, ~, JthetaRaw, ~] = evaluateBranchResidualModel( ...
            branchID, eta, theta_j, cfg);
    else
        JthetaRaw = JthetaRawCached;
    end

    compositeMode = getCompositeMode_step09J4(cfg);
    BgateUsed = [];

    switch compositeMode

        case "additive"
            JthetaEff = JthetaRaw;

        case {"gated_additive", "local_full_plus_gated_nonlocal"}

            % Old designed tight-frame gate. This is eta-dependent but does
            % not depend on theta_j, so only B_j Jtheta_j is needed here.
            BgateUsed = branchGateMatrix(branchID, eta, cfg);
            JthetaEff = BgateUsed * JthetaRaw;

        case {"bearing_fim_gated", "output_information_fusion", ...
                "fim_weighted_additive"}

            % New Step 09-J gate. This is recipient-side metadata computed
            % from OmegaBar matrices and treated as fixed during this
            % prediction call.
            if isempty(Bfim_j)
                error("computeNonlocalBranchCovarianceInjection:MissingMatrixGate", ...
                    "%s covariance injection requires B_{j|m}.", ...
                    compositeMode);
            end

            BgateUsed = Bfim_j;
            JthetaEff = BgateUsed * JthetaRaw;

        otherwise
            error("computeNonlocalBranchCovarianceInjection:BadCompositeMode", ...
                "Unsupported cfg.gs.compositeMode = %s.", compositeMode);

    end

end

function compositeMode = getCompositeMode_step09J4(cfg)
%GETCOMPOSITEMODE_STEP09J4 Read GS composite mode with additive default.

    compositeMode = "additive";

    if isfield(cfg, "gs") && isfield(cfg.gs, "compositeMode")
        compositeMode = string(cfg.gs.compositeMode);
    end

end

function tf = isUsableNonlocalBranchForCovariance(branchRecord)
%ISUSABLENONLOCALBRANCHFORCOVARIANCE Return true for usable GS covariance data.
%
% A branch must be active, valid, non-stale, and have finite Ptheta.
% The actual theta size is checked later by evaluateBranchResidualModel(...).

    tf = false;

    if ~isstruct(branchRecord)
        return;
    end

    if ~isfield(branchRecord, "active") || ~logical(branchRecord.active)
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

    if ~isfield(branchRecord, "theta") || isempty(branchRecord.theta)
        return;
    end

    if any(~isfinite(branchRecord.theta(:)))
        return;
    end

    if ~isfield(branchRecord, "Ptheta") || isempty(branchRecord.Ptheta)
        return;
    end

    if any(~isfinite(branchRecord.Ptheta(:)))
        return;
    end

    tf = true;

end
