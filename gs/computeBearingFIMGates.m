function [B, gateDiag] = computeBearingFIMGates(watcher, branchUsed, cfg)
%COMPUTEBEARINGFIMGATES Legacy LOS-projector gate.
% OmegaBar is an exponentially smoothed LOS projector.  It is not the FIM
% of the measurement likelihood and the resulting B matrices are not an
% unbiased mean-composition rule when branches represent additive residual
% components.  This function is retained only for reproduction of the
% Step 09-J legacy comparison.
%{
Function:
    computeBearingFIMGates.m

Purpose:
    Compute direction-only bearing-FIM residual fusion gates for one
    recipient watcher.

Main equations:
    For the available branch set A_m,

        OmegaSigma_m = sum_{l in A_m} OmegaBar_l + epsilon I,
        B_{j|m}      = OmegaSigma_m \ OmegaBar_j.

    The composite residual can then use

        d_FIM,m = sum_{j in A_m} B_{j|m} d_j.

Inputs:
    watcher    - Recipient watcher structure. The local branch geometry is
                 read from watcher.OmegaBar. Nonlocal branch geometries are
                 read from watcher.gsBranches(j).OmegaBar.

    branchUsed - Logical vector of length cfg.Nw. Only branches with
                 branchUsed(j)=true are included in OmegaSigma and receive a
                 nonzero gate.

    cfg        - Simulation configuration. Uses:
                    cfg.dim
                    cfg.Nw
                    cfg.gs.fimGate.epsilon
                    cfg.gs.fimGate.normalizeTrace

Outputs:
    B          - Gate stack. B(:,:,j) is B_{j|m}. Size: dim x dim x Nw.

    gateDiag   - Diagnostic structure with OmegaBars, OmegaSigma, gate sum,
                 regularization, and conditioning information.

Notes:
    - This helper does not store B_j in GS. It computes B_{j|m} on the
      recipient side because the gate depends on the currently available
      branch set A_m.
    - B_j is treated as metadata-fixed during one prediction call. Therefore
      no eta-derivative of B_j is used in Step 09-J.3.
    - The current first implementation intentionally ignores range scaling
      and age weighting.
%}

    if ~isfield(cfg, "dim") || ~isfield(cfg, "Nw")
        error("computeBearingFIMGates:MissingConfig", ...
            "cfg.dim and cfg.Nw are required.");
    end

    dim = cfg.dim;
    Nw = cfg.Nw;

    branchUsed = logical(branchUsed(:));

    if numel(branchUsed) ~= Nw
        error("computeBearingFIMGates:BadBranchUsedSize", ...
            "branchUsed must have length cfg.Nw = %d.", Nw);
    end

    epsilon = getFIMGateEpsilon(cfg);
    normalizeTrace = getFIMGateNormalizeTrace(cfg);

    OmegaBars = zeros(dim, dim, Nw);
    OmegaSigma = epsilon * eye(dim);

    for j = 1:Nw

        if ~branchUsed(j)
            continue;
        end

        Omega_j = getBranchOmegaBar(watcher, j, dim);
        Omega_j = sanitizeOmegaBar(Omega_j, dim, j, normalizeTrace);

        OmegaBars(:, :, j) = Omega_j;
        OmegaSigma = OmegaSigma + Omega_j;

    end

    OmegaSigma = 0.5 * (OmegaSigma + OmegaSigma.');

    B = zeros(dim, dim, Nw);

    for j = 1:Nw
        if branchUsed(j)
            % Left division avoids explicitly forming inv(OmegaSigma).
            B(:, :, j) = OmegaSigma \ OmegaBars(:, :, j);
        end
    end

    gateSum = sum(B, 3);

    gateDiag = struct();
    gateDiag.enabled = true;
    gateDiag.branchUsed = branchUsed;
    gateDiag.branchIDs = find(branchUsed).';
    gateDiag.OmegaBars = OmegaBars;
    gateDiag.OmegaSigma = OmegaSigma;
    gateDiag.B = B;
    gateDiag.gateSum = gateSum;
    gateDiag.epsilon = epsilon;
    gateDiag.normalizeTrace = normalizeTrace;
    gateDiag.traceOmegaSigma = trace(OmegaSigma);
    gateDiag.minEigOmegaSigma = min(eig(OmegaSigma));
    gateDiag.condOmegaSigma = cond(OmegaSigma);
    gateDiag.sumGateIdentityError = norm(gateSum - eye(dim), "fro");

end

function OmegaBar = getBranchOmegaBar(watcher, branchID, dim)
%GETBRANCHOMEGABAR Return local or nonlocal OmegaBar for branchID.

    if isfield(watcher, "localBranchID") && branchID == watcher.localBranchID

        if isfield(watcher, "OmegaBar") && ~isempty(watcher.OmegaBar)
            OmegaBar = watcher.OmegaBar;
        else
            OmegaBar = zeros(dim, dim);
        end

        return;

    end

    if isfield(watcher, "gsBranches") && numel(watcher.gsBranches) >= branchID && ...
            isfield(watcher.gsBranches(branchID), "OmegaBar") && ...
            ~isempty(watcher.gsBranches(branchID).OmegaBar)
        OmegaBar = watcher.gsBranches(branchID).OmegaBar;
    else
        OmegaBar = zeros(dim, dim);
    end

end

function OmegaBar = sanitizeOmegaBar(OmegaBar, dim, branchID, normalizeTrace)
%SANITIZEOMEGABAR Validate, symmetrize, and optionally trace-normalize OmegaBar.

    OmegaBar = double(OmegaBar);

    if any(size(OmegaBar) ~= [dim, dim])
        error("computeBearingFIMGates:BadOmegaBarSize", ...
            "OmegaBar for branch %d must be %d-by-%d.", branchID, dim, dim);
    end

    if any(~isfinite(OmegaBar(:)))
        error("computeBearingFIMGates:NonFiniteOmegaBar", ...
            "OmegaBar for branch %d contains non-finite values.", branchID);
    end

    % Remove tiny asymmetry introduced by repeated numerical updates or file
    % transfers through GS metadata.
    OmegaBar = 0.5 * (OmegaBar + OmegaBar.');

    if normalizeTrace
        tr = trace(OmegaBar);
        targetTrace = max(dim - 1, 1);

        % Do not amplify a zero-support branch. A zero OmegaBar means this
        % branch has not accumulated useful bearing geometry yet.
        if tr > eps
            OmegaBar = (targetTrace / tr) * OmegaBar;
        end
    end

end

function epsilon = getFIMGateEpsilon(cfg)
%GETFIMGATEEPSILON Read regularization for OmegaSigma.

    epsilon = 1.0e-6;

    if isfield(cfg, "gs") && isfield(cfg.gs, "fimGate") && ...
            isfield(cfg.gs.fimGate, "epsilon") && ...
            ~isempty(cfg.gs.fimGate.epsilon)
        epsilon = cfg.gs.fimGate.epsilon;
    end

    if ~(isscalar(epsilon) && isfinite(epsilon) && epsilon >= 0)
        error("computeBearingFIMGates:InvalidEpsilon", ...
            "cfg.gs.fimGate.epsilon must be a nonnegative finite scalar.");
    end

end

function normalizeTrace = getFIMGateNormalizeTrace(cfg)
%GETFIMGATENORMALIZETRACE Read optional per-branch OmegaBar trace normalization flag.

    normalizeTrace = false;

    if isfield(cfg, "gs") && isfield(cfg.gs, "fimGate") && ...
            isfield(cfg.gs.fimGate, "normalizeTrace") && ...
            ~isempty(cfg.gs.fimGate.normalizeTrace)
        normalizeTrace = logical(cfg.gs.fimGate.normalizeTrace);
    end

end
