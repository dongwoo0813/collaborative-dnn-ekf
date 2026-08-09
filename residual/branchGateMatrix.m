function [B_j, dB_dEta] = branchGateMatrix(branchID, eta, cfg)
%{
Function:
    branchGateMatrix.m

Purpose:
    Compute the branch gate matrix B_j(eta) for the gated additive shared
    residual model.

Current supported mode:
    cfg.gate.mode = "tight_frame_2d_rt"

For dim = 2:

    eta = [r_x; r_y; v_x; v_y]

    eR = r / ||r||
    eT = [-eR(2); eR(1)]

    psi_j = (j - 1)*pi/Nw
    b_j   = cos(psi_j)*eR + sin(psi_j)*eT

    B_j = (2/Nw) b_j b_j^T.

Optional second output:
    dB_dEta(:,:,k) = partial B_j / partial eta_k.

    Since B_j depends only on r, not v,

        dB_dEta(:,:,1:2) may be nonzero,
        dB_dEta(:,:,3:4) = 0.

Usage in gated residual Jacobian:
    If

        d_j(eta,theta_j) = B_j(eta) dRaw_j(eta,theta_j),

    then

        J_j(:,k)
        =
        B_j * JRaw_j(:,k)
        +
        dB_dEta(:,:,k) * dRaw_j.

Notes:
    - This function still supports one-output calls:
          B_j = branchGateMatrix(...)
      Existing code should not break.
    - If ||r|| is too small, the gate uses a deterministic fallback
      eR = [1;0] and returns zero dB/deta.
%}

    % ---------------------------------------------------------------------
    % Required configuration
    % ---------------------------------------------------------------------
    if ~isfield(cfg, "dim")
        error("branchGateMatrix:MissingDim", ...
            "cfg.dim is required.");
    end

    if ~isfield(cfg, "Nw")
        error("branchGateMatrix:MissingNw", ...
            "cfg.Nw is required.");
    end

    dim = cfg.dim;
    Nw  = cfg.Nw;

    if ~(isscalar(dim) && dim == 2)
        error("branchGateMatrix:UnsupportedDim", ...
            "Only cfg.dim = 2 is supported. Got cfg.dim = %g.", dim);
    end

    if ~(isscalar(Nw) && isfinite(Nw) && Nw >= 2 && abs(Nw - round(Nw)) <= eps)
        error("branchGateMatrix:InvalidNw", ...
            "cfg.Nw must be an integer >= 2.");
    end

    if ~(isscalar(branchID) && isfinite(branchID) && ...
            branchID >= 1 && branchID <= Nw && ...
            abs(branchID - round(branchID)) <= eps)
        error("branchGateMatrix:InvalidBranchID", ...
            "branchID must be an integer satisfying 1 <= branchID <= cfg.Nw.");
    end

    eta = eta(:);
    nEta = 2 * dim;

    if numel(eta) ~= nEta
        error("branchGateMatrix:InvalidEtaLength", ...
            "eta has wrong length. Expected %d, got %d.", nEta, numel(eta));
    end

    if any(~isfinite(eta))
        error("branchGateMatrix:NonFiniteEta", ...
            "eta contains non-finite entries.");
    end

    % ---------------------------------------------------------------------
    % Gate mode
    % ---------------------------------------------------------------------
    gateMode = "tight_frame_2d_rt";

    if isfield(cfg, "gate") && isfield(cfg.gate, "mode")
        gateMode = string(cfg.gate.mode);
    end

    switch gateMode

        case "tight_frame_2d_rt"
            [B_j, dB_dEta] = tightFrame2DRTGate(branchID, eta, cfg);

        otherwise
            error("branchGateMatrix:UnsupportedGateMode", ...
                "Unsupported cfg.gate.mode = %s.", gateMode);

    end

end

function [B_j, dB_dEta] = tightFrame2DRTGate(branchID, eta, cfg)
% Compute the 2-D radial/transverse tight-frame gate and its eta-derivative.

    dim  = cfg.dim;
    Nw   = cfg.Nw;
    nEta = 2 * dim;

    r = eta(1:dim);

    minRange = eps;

    if isfield(cfg, "gate") && isfield(cfg.gate, "minRange")
        minRange = cfg.gate.minRange;
    end

    if ~(isscalar(minRange) && isfinite(minRange) && minRange > 0)
        error("branchGateMatrix:InvalidMinRange", ...
            "cfg.gate.minRange must be a positive finite scalar.");
    end

    dB_dEta = zeros(dim, dim, nEta);

    rNorm = norm(r);

    if rNorm <= minRange
        % Deterministic fallback for singular/near-singular geometry.
        % Since the fallback direction is fixed, the derivative is set to
        % zero.
        eR = [1.0; 0.0];
    else
        eR = r / rNorm;
    end

    c = cos((branchID - 1) * pi / Nw);
    s = sin((branchID - 1) * pi / Nw);

    % A rotates eR into the branch direction b_j.
    A = [ c, -s;
          s,  c];

    b_j = A * eR;

    alpha = 2.0 / Nw;

    B_j = alpha * (b_j * b_j.');
    B_j = 0.5 * (B_j + B_j.');

    % ---------------------------------------------------------------------
    % Analytic derivative dB_j / d eta
    %
    % B_j = alpha * b_j b_j^T,
    % b_j = A eR,
    % eR = r / ||r||.
    %
    % d eR / d r_k = (I - eR eR^T) e_k / ||r||.
    %
    % dB / d r_k = alpha * (db/dr_k * b^T + b * db/dr_k^T).
    % ---------------------------------------------------------------------
    if rNorm > minRange

        Pi_perp = eye(dim) - eR * eR.';

        for k = 1:dim
            deR_drk = Pi_perp(:, k) / rNorm;
            db_drk  = A * deR_drk;

            dB_drk = alpha * (db_drk * b_j.' + b_j * db_drk.');

            % Symmetrize to remove tiny roundoff asymmetry.
            dB_dEta(:, :, k) = 0.5 * (dB_drk + dB_drk.');
        end

    end

end