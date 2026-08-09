function [Omega, uUnit] = bearingDirectionInfoMatrix(u)
%{
File:
    geometry/bearingDirectionInfoMatrix.m

Purpose:
    Build the direction-only bearing information matrix

        Omega = I - u u',

    where u is the line-of-sight unit vector from the watcher to the target,
    expressed in the same frame as the DNN residual acceleration output.

Inputs:
    u
        Line-of-sight direction vector. It does not need to be perfectly
        normalized. Size dim x 1, where dim is usually 2 in the current
        simulation and 3 in the future extension.

Outputs:
    Omega
        Direction-only bearing information/projector matrix. Size dim x dim.
        For unit u, Omega is symmetric positive semidefinite, Omega*u = 0,
        and trace(Omega) = dim - 1.

    uUnit
        Normalized line-of-sight unit vector actually used to form Omega.

Notes:
    - This helper intentionally does not include range scaling 1/rho^2.
      The current bearing-only measurement does not directly provide range.
    - This helper intentionally does not include an age weight. GS shares a
      learned function approximator, not an instantaneous acceleration sample.
%}

    if nargin < 1 || isempty(u)
        error("bearingDirectionInfoMatrix:MissingInput", ...
            "Input LOS vector u is required.");
    end

    u = u(:);
    dim = numel(u);

    if dim < 2
        error("bearingDirectionInfoMatrix:InvalidDimension", ...
            "u must have at least two components.");
    end

    if any(~isfinite(u))
        error("bearingDirectionInfoMatrix:NonFiniteInput", ...
            "u must contain only finite values.");
    end

    uNorm = norm(u);

    if uNorm < eps
        error("bearingDirectionInfoMatrix:ZeroLOS", ...
            "Cannot form a bearing information matrix from a zero LOS vector.");
    end

    uUnit = u / uNorm;

    Omega = eye(dim) - uUnit * uUnit';

    % Symmetrize to remove tiny finite-precision asymmetry.
    Omega = 0.5 * (Omega + Omega');

end
