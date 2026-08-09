function [dHat, branchContrib] = compositeResidual(eta, branchLibrary, cfg)
%{
Function:
    compositeResidual.m

Purpose:
    Compute the composite DNN residual acceleration from multiple branch
    residual models.

    The collaborative DNN-EKF residual model is

        d_hat(eta) = sum_{j=1}^{Nw} d_j(eta; theta_j),

    where each branch has the fixed-feature output-layer form

        d_j(eta; theta_j) = W_j phi_j(eta),

    and

        theta_j = vec(W_j).

    This function is designed to work for later collaborative architectures:

        1. Local-only DNN-EKF:
              branchLibrary contains only the local branch.

        2. GS-assisted DNN-EKF:
              branchLibrary contains the local branch and GS branch copies.

        3. P2P DNN-EKF:
              branchLibrary contains the local branch and peer-cache copies.

Inputs:
    eta            - Target physical state.
                     Size: 2*cfg.dim x 1.
                     Definition:
                         eta = [r_t; v_t].

    branchLibrary  - Branch parameter container.
                     Supported formats:

                     Format 1: struct array
                         branchLibrary(j).theta
                         branchLibrary(j).active     optional logical

                     Format 2: cell array
                         branchLibrary{j} = theta_j

                     Format 3: numeric matrix
                         branchLibrary(:,j) = theta_j

                     Empty or inactive branches are skipped.

    cfg            - Simulation configuration structure.
                     Required fields:
                         cfg.dim
                         cfg.Nw

Outputs:
    dHat           - Composite residual acceleration.
                     Size: cfg.dim x 1.

    branchContrib  - Individual branch contributions.
                     Size: cfg.dim x cfg.Nw.
                     Column j is d_j(eta; theta_j).
                     If branch j is inactive or unavailable, the column is
                     zero.

Main equations:
    For each branch j,

        d_j(eta; theta_j) = W_j phi_j(eta).

    The composite residual is

        d_hat(eta) = sum_j d_j(eta; theta_j).

Notes:
    - This function does not update any EKF states or covariances.
    - It only evaluates the residual model.
    - The struct-array input format is recommended for GS/P2P cases because
      it can later store metadata such as version, timestamp, covariance,
      and branch age.
    - The numeric matrix format is useful for quick tests.
%}

    dim = cfg.dim;
    Nw = cfg.Nw;

    dHat = zeros(dim,1);
    branchContrib = zeros(dim, Nw);

    for j = 1:Nw

        [theta_j, isActive] = getBranchTheta(j, branchLibrary, cfg);

        if ~isActive
            continue;
        end

        d_j = branchOutput(j, eta, theta_j, cfg);

        branchContrib(:,j) = d_j;
        dHat = dHat + d_j;

    end

end

function [theta_j, isActive] = getBranchTheta(j, branchLibrary, cfg)
%{
Function:
    getBranchTheta

Purpose:
    Extract theta_j from a branch library container.

Inputs:
    j             - Branch index.
    branchLibrary - Branch parameter container.
    cfg           - Simulation configuration.

Outputs:
    theta_j       - Branch parameter vector.
    isActive      - Logical flag indicating whether branch j should be used.

Notes:
    - This helper supports struct arrays, cell arrays, and numeric matrices.
    - Empty branches are treated as inactive.
%}

    theta_j = [];
    isActive = false;

    if isempty(branchLibrary)
        return;
    end

    % ---------------------------------------------------------------------
    % Format 1: struct array
    % branchLibrary(j).theta
    % branchLibrary(j).active, optional
    % ---------------------------------------------------------------------
    if isstruct(branchLibrary)

        if numel(branchLibrary) < j
            return;
        end

        if isfield(branchLibrary, "active")
            if ~branchLibrary(j).active
                return;
            end
        end

        if ~isfield(branchLibrary, "theta")
            error("Struct branchLibrary must contain field .theta.");
        end

        theta_j = branchLibrary(j).theta;

        if isempty(theta_j)
            return;
        end

        isActive = true;
        return;
    end

    % ---------------------------------------------------------------------
    % Format 2: cell array
    % branchLibrary{j} = theta_j
    % ---------------------------------------------------------------------
    if iscell(branchLibrary)

        if numel(branchLibrary) < j
            return;
        end

        theta_j = branchLibrary{j};

        if isempty(theta_j)
            return;
        end

        isActive = true;
        return;
    end

    % ---------------------------------------------------------------------
    % Format 3: numeric matrix
    % branchLibrary(:,j) = theta_j
    % ---------------------------------------------------------------------
    if isnumeric(branchLibrary)

        if size(branchLibrary,2) < j
            return;
        end

        theta_j = branchLibrary(:,j);

        if isempty(theta_j)
            return;
        end

        isActive = true;
        return;
    end

    error("Unsupported branchLibrary type.");

end