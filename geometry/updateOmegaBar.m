function [OmegaBarNew, info] = updateOmegaBar( ...
    OmegaBarPrev,u,lambdaOmega,normalizeTrace,accumulationMode)
%{
File:
    geometry/updateOmegaBar.m

Purpose:
    Update a watcher's cumulative direction-only bearing geometry support
    matrix

        OmegaBar_k = (1 - lambdaOmega) OmegaBar_{k-1}
                     + lambdaOmega (I - u_k u_k').

Inputs:
    OmegaBarPrev
        Previous cumulative geometry support matrix. Size dim x dim.

    u
        Current LOS direction vector from watcher to target. Size dim x 1.
        It does not need to be perfectly normalized.

    lambdaOmega
        Exponential moving-average update rate. Typical first value: 0.02.
        Must satisfy 0 < lambdaOmega <= 1.

    normalizeTrace
        Optional logical flag. If true, rescale OmegaBarNew so that
        trace(OmegaBarNew) = dim - 1 whenever its trace is positive.
        Default is false, which preserves the exact EMA recursion above.

Outputs:
    OmegaBarNew
        Updated cumulative geometry support matrix. Size dim x dim.

    info
        Diagnostic structure containing the instantaneous Omega, normalized
        LOS unit vector, trace/eigenvalue diagnostics, and the selected
        lambdaOmega.

Notes:
    - The first implementation uses only direction. No range scaling and no
      age weighting are applied here.
    - Starting from OmegaBarPrev = zeros(dim), the trace approaches dim - 1
      under persistent updates when normalizeTrace = false.
%}

    if nargin < 3 || isempty(lambdaOmega)
        lambdaOmega = 0.02;
    end

    if nargin < 4 || isempty(normalizeTrace)
        normalizeTrace = false;
    end

    if nargin < 5 || isempty(accumulationMode)
        accumulationMode = "ema";
    end
    accumulationMode=string(accumulationMode);

    if ~isscalar(lambdaOmega) || ~isfinite(lambdaOmega) || ...
            lambdaOmega <= 0 || lambdaOmega > 1
        error("updateOmegaBar:InvalidLambda", ...
            "lambdaOmega must be a finite scalar in (0, 1].");
    end

    OmegaBarPrev = double(OmegaBarPrev);

    if size(OmegaBarPrev,1) ~= size(OmegaBarPrev,2)
        error("updateOmegaBar:NonSquareOmegaBar", ...
            "OmegaBarPrev must be square.");
    end

    dim = size(OmegaBarPrev,1);

    if numel(u) ~= dim
        error("updateOmegaBar:DimensionMismatch", ...
            "numel(u) = %d but OmegaBarPrev is %d-by-%d.", ...
            numel(u), dim, dim);
    end

    OmegaBarPrev = 0.5 * (OmegaBarPrev + OmegaBarPrev');

    [Omega, uUnit] = bearingDirectionInfoMatrix(u);

    switch accumulationMode
        case "ema"
            OmegaBarNew = (1-lambdaOmega)*OmegaBarPrev+lambdaOmega*Omega;
        case "cumulative_sum"
            % Measurement-information accumulation without an arbitrary
            % forgetting factor. Per-branch eigenvalue normalization is
            % performed later by computeFIMWeightedAdditiveWeights.
            OmegaBarNew = OmegaBarPrev+Omega;
        otherwise
            error("updateOmegaBar:BadAccumulationMode", ...
                "Unsupported accumulation mode %s.",accumulationMode);
    end
    OmegaBarNew = 0.5 * (OmegaBarNew + OmegaBarNew');

    if logical(normalizeTrace)
        tr = trace(OmegaBarNew);

        % The target trace is dim-1 because each instantaneous bearing
        % projector has trace dim-1.
        targetTrace = dim - 1;

        if tr > eps && targetTrace > 0
            OmegaBarNew = (targetTrace / tr) * OmegaBarNew;
            OmegaBarNew = 0.5 * (OmegaBarNew + OmegaBarNew');
        end
    end

    eigOmegaBar = eig(OmegaBarNew);

    info = struct();
    info.lambdaOmega = lambdaOmega;
    info.accumulationMode = accumulationMode;
    info.normalizeTrace = logical(normalizeTrace);
    info.uUnit = uUnit;
    info.Omega = Omega;
    info.traceOmega = trace(Omega);
    info.traceOmegaBar = trace(OmegaBarNew);
    info.minEigOmega = min(eig(Omega));
    info.minEigOmegaBar = min(eigOmegaBar);
    info.nullResidual = norm(Omega * uUnit);

end
