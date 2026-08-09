function out = run_step09c2_nonlocal_contribution_diagnostics(outCase, halfAngleDeg, seed, residualFamily)
%{
File:
    main/run_step09c2_nonlocal_contribution_diagnostics.m

Purpose:
    Step 09-C.2 nonlocal branch contribution diagnostics.

    After running (out09c1 = run_step08b3_compare_FOV_Local_GS_Oracle(90,
    false, 101, "coupled_nonlinear");)

    Decompose the GS composite residual into

        d_GS = d_local + d_nonlocal,

    and compare each component against the true unknown residual.

Why this step matters:
    Step 09-B.8 showed that GS tracking improvement and GS residual
    approximation improvement are FOV-regime dependent.

    This diagnostic answers a more specific question:

        Does the nonlocal GS contribution point in a useful direction,
        or does it only increase residual magnitude?

Inputs:
    outCase
        Output from:
            run_step08b3_compare_FOV_Local_GS_Oracle(...)

        This function uses outCase.resGS.

    halfAngleDeg
        Optional FOV half-angle used only if outCase is omitted.

    seed
        Optional seed used only if outCase is omitted.

    residualFamily
        Optional residual family used only if outCase is omitted.

Outputs:
    out.summaryTable
        One row each for all / available / dropout samples.

    out.raw
        Raw scalar diagnostic arrays.

Usage:
    out09c2 = run_step09c2_nonlocal_contribution_diagnostics(out09c1);

    out09c2 = run_step09c2_nonlocal_contribution_diagnostics( ...
        [], 90, 101, "coupled_nonlinear");

    disp(out09c2.summaryTable(:, ...
        ["maskName", ...
         "meanNormTrue", ...
         "meanNormLocal", ...
         "meanNormNonlocal", ...
         "meanNormGS", ...
         "meanErrLocal", ...
         "meanErrGS", ...
         "meanGSErrImprovementPct", ...
         "meanCosNonlocalTrue", ...
         "meanProjNonlocalTrue", ...
         "meanNonlocalToTrueNormRatio"]));

%}

    if nargin < 1
        outCase = [];
    end

    if nargin < 2 || isempty(halfAngleDeg)
        halfAngleDeg = 90.0;
    end

    if nargin < 3 || isempty(seed)
        seed = 101;
    end

    if nargin < 4 || isempty(residualFamily)
        residualFamily = "coupled_nonlinear";
    end

    residualFamily = string(residualFamily);

    addpath(genpath(pwd));
    rehash;

    if isempty(outCase)
        outCase = run_step08b3_compare_FOV_Local_GS_Oracle( ...
            halfAngleDeg, false, seed, residualFamily);
    else
        if isfield(outCase, "halfAngleDeg")
            halfAngleDeg = outCase.halfAngleDeg;
        end

        if isfield(outCase, "seed")
            seed = outCase.seed;
        end

        if isfield(outCase, "residualFamily")
            residualFamily = string(outCase.residualFamily);
        end
    end

    if ~isfield(outCase, "resGS")
        error("Step09C2:MissingResGS", ...
            "outCase must contain outCase.resGS.");
    end

    resGS = outCase.resGS;

    requiredFields = [
        "dnnResidual"
        "dnnResidualLocalComponent"
        "dnnResidualNonlocalComponent"
        "trueResidual"
    ];

    for iField = 1:numel(requiredFields)
        if ~isfield(resGS, requiredFields(iField))
            error("Step09C2:MissingField", ...
                "resGS is missing required field: %s", requiredFields(iField));
        end
    end

    dGS = resGS.dnnResidual;
    dLocal = resGS.dnnResidualLocalComponent;
    dNonlocal = resGS.dnnResidualNonlocalComponent;

    [dim, N, Nw] = size(dGS);

    dTrue = expandTrueResidual_step09c2(resGS.trueResidual, dim, N, Nw);

    % ---------------------------------------------------------------------
    % Sanity: d_GS should equal d_local + d_nonlocal.
    % ---------------------------------------------------------------------
    decompError = dGS - dLocal - dNonlocal;
    maxDecompositionError = max(abs(decompError(:)));

    % ---------------------------------------------------------------------
    % Scalar diagnostics
    % ---------------------------------------------------------------------
    normTrue = vecNorm_step09c2(dTrue);
    normLocal = vecNorm_step09c2(dLocal);
    normNonlocal = vecNorm_step09c2(dNonlocal);
    normGS = vecNorm_step09c2(dGS);

    errLocal = vecNorm_step09c2(dLocal - dTrue);
    errNonlocalOnly = vecNorm_step09c2(dNonlocal - dTrue);
    errGS = vecNorm_step09c2(dGS - dTrue);

    dotLocalTrue = vecDot_step09c2(dLocal, dTrue);
    dotNonlocalTrue = vecDot_step09c2(dNonlocal, dTrue);
    dotGSTrue = vecDot_step09c2(dGS, dTrue);

    cosLocalTrue = safeDivide_step09c2(dotLocalTrue, normLocal .* normTrue);
    cosNonlocalTrue = safeDivide_step09c2(dotNonlocalTrue, normNonlocal .* normTrue);
    cosGSTrue = safeDivide_step09c2(dotGSTrue, normGS .* normTrue);

    projLocalTrue = safeDivide_step09c2(dotLocalTrue, normTrue.^2);
    projNonlocalTrue = safeDivide_step09c2(dotNonlocalTrue, normTrue.^2);
    projGSTrue = safeDivide_step09c2(dotGSTrue, normTrue.^2);

    nonlocalToTrueNormRatio = safeDivide_step09c2(normNonlocal, normTrue);
    nonlocalToLocalNormRatio = safeDivide_step09c2(normNonlocal, normLocal);

    gsErrImprovementPct = percentImprovementArray_step09c2(errLocal, errGS);

    validMask = isfinite(normTrue) & normTrue > 1e-14;

    if isfield(resGS, "measAvail")
        measAvail = logical(resGS.measAvail);
        measAvail = alignMaskSize_step09c2(measAvail, N, Nw);
    else
        measAvail = true(N, Nw);
    end

    allMask = validMask;
    availableMask = validMask & measAvail;
    dropoutMask = validMask & ~measAvail;

    maskName = [
        "all"
        "available"
        "dropout"
    ];

    maskCell = {
        allMask
        availableMask
        dropoutMask
    };

    nMask = numel(maskCell);

    sampleCount = NaN(nMask, 1);
    meanNormTrue = NaN(nMask, 1);
    meanNormLocal = NaN(nMask, 1);
    meanNormNonlocal = NaN(nMask, 1);
    meanNormGS = NaN(nMask, 1);

    meanErrLocal = NaN(nMask, 1);
    meanErrNonlocalOnly = NaN(nMask, 1);
    meanErrGS = NaN(nMask, 1);
    meanGSErrImprovementPct = NaN(nMask, 1);

    meanCosLocalTrue = NaN(nMask, 1);
    meanCosNonlocalTrue = NaN(nMask, 1);
    meanCosGSTrue = NaN(nMask, 1);

    meanProjLocalTrue = NaN(nMask, 1);
    meanProjNonlocalTrue = NaN(nMask, 1);
    meanProjGSTrue = NaN(nMask, 1);

    meanNonlocalToTrueNormRatio = NaN(nMask, 1);
    meanNonlocalToLocalNormRatio = NaN(nMask, 1);

    for iMask = 1:nMask

        mask = maskCell{iMask};

        sampleCount(iMask) = nnz(mask);

        meanNormTrue(iMask) = meanMasked_step09c2(normTrue, mask);
        meanNormLocal(iMask) = meanMasked_step09c2(normLocal, mask);
        meanNormNonlocal(iMask) = meanMasked_step09c2(normNonlocal, mask);
        meanNormGS(iMask) = meanMasked_step09c2(normGS, mask);

        meanErrLocal(iMask) = meanMasked_step09c2(errLocal, mask);
        meanErrNonlocalOnly(iMask) = meanMasked_step09c2(errNonlocalOnly, mask);
        meanErrGS(iMask) = meanMasked_step09c2(errGS, mask);
        meanGSErrImprovementPct(iMask) = meanMasked_step09c2(gsErrImprovementPct, mask);

        meanCosLocalTrue(iMask) = meanMasked_step09c2(cosLocalTrue, mask);
        meanCosNonlocalTrue(iMask) = meanMasked_step09c2(cosNonlocalTrue, mask);
        meanCosGSTrue(iMask) = meanMasked_step09c2(cosGSTrue, mask);

        meanProjLocalTrue(iMask) = meanMasked_step09c2(projLocalTrue, mask);
        meanProjNonlocalTrue(iMask) = meanMasked_step09c2(projNonlocalTrue, mask);
        meanProjGSTrue(iMask) = meanMasked_step09c2(projGSTrue, mask);

        meanNonlocalToTrueNormRatio(iMask) = meanMasked_step09c2( ...
            nonlocalToTrueNormRatio, mask);

        meanNonlocalToLocalNormRatio(iMask) = meanMasked_step09c2( ...
            nonlocalToLocalNormRatio, mask);

    end

    summaryTable = table( ...
        repmat(residualFamily, nMask, 1), ...
        repmat(halfAngleDeg, nMask, 1), ...
        repmat(seed, nMask, 1), ...
        maskName, ...
        sampleCount, ...
        meanNormTrue, ...
        meanNormLocal, ...
        meanNormNonlocal, ...
        meanNormGS, ...
        meanErrLocal, ...
        meanErrNonlocalOnly, ...
        meanErrGS, ...
        meanGSErrImprovementPct, ...
        meanCosLocalTrue, ...
        meanCosNonlocalTrue, ...
        meanCosGSTrue, ...
        meanProjLocalTrue, ...
        meanProjNonlocalTrue, ...
        meanProjGSTrue, ...
        meanNonlocalToTrueNormRatio, ...
        meanNonlocalToLocalNormRatio, ...
        repmat(maxDecompositionError, nMask, 1), ...
        'VariableNames', { ...
            'residualFamily', ...
            'halfAngleDeg', ...
            'seed', ...
            'maskName', ...
            'sampleCount', ...
            'meanNormTrue', ...
            'meanNormLocal', ...
            'meanNormNonlocal', ...
            'meanNormGS', ...
            'meanErrLocal', ...
            'meanErrNonlocalOnly', ...
            'meanErrGS', ...
            'meanGSErrImprovementPct', ...
            'meanCosLocalTrue', ...
            'meanCosNonlocalTrue', ...
            'meanCosGSTrue', ...
            'meanProjLocalTrue', ...
            'meanProjNonlocalTrue', ...
            'meanProjGSTrue', ...
            'meanNonlocalToTrueNormRatio', ...
            'meanNonlocalToLocalNormRatio', ...
            'maxDecompositionError'});

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-C.2: Nonlocal contribution diagnostics\n");
    fprintf("============================================================\n");
    fprintf("Residual family = %s\n", residualFamily);
    fprintf("FOV half-angle  = %.3f deg\n", halfAngleDeg);
    fprintf("Seed            = %d\n", seed);
    fprintf("Max dGS - dLocal - dNonlocal error = %.3e\n\n", ...
        maxDecompositionError);

    disp(summaryTable);

    out = struct();
    out.outCase = outCase;
    out.residualFamily = residualFamily;
    out.halfAngleDeg = halfAngleDeg;
    out.seed = seed;
    out.summaryTable = summaryTable;

    out.raw = struct();
    out.raw.normTrue = normTrue;
    out.raw.normLocal = normLocal;
    out.raw.normNonlocal = normNonlocal;
    out.raw.normGS = normGS;
    out.raw.errLocal = errLocal;
    out.raw.errNonlocalOnly = errNonlocalOnly;
    out.raw.errGS = errGS;
    out.raw.cosLocalTrue = cosLocalTrue;
    out.raw.cosNonlocalTrue = cosNonlocalTrue;
    out.raw.cosGSTrue = cosGSTrue;
    out.raw.projLocalTrue = projLocalTrue;
    out.raw.projNonlocalTrue = projNonlocalTrue;
    out.raw.projGSTrue = projGSTrue;
    out.raw.nonlocalToTrueNormRatio = nonlocalToTrueNormRatio;
    out.raw.nonlocalToLocalNormRatio = nonlocalToLocalNormRatio;
    out.raw.measAvail = measAvail;
    out.raw.validMask = validMask;
    out.raw.maxDecompositionError = maxDecompositionError;

end

function dTrueOut = expandTrueResidual_step09c2(dTrueIn, dim, N, Nw)
%EXPANDTRUERESIDUAL_STEP09C2 Convert true residual log to dim x N x Nw.

    sz = size(dTrueIn);

    if numel(sz) == 2
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09C2:BadTrueResidualSize", ...
                "trueResidual must be dim x N or dim x N x Nw.");
        end

        dTrueOut = reshape(dTrueIn, dim, N, 1);
        dTrueOut = repmat(dTrueOut, 1, 1, Nw);
        return;
    end

    if numel(sz) == 3
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09C2:BadTrueResidualSize", ...
                "trueResidual has incompatible first two dimensions.");
        end

        if sz(3) == Nw
            dTrueOut = dTrueIn;
        elseif sz(3) == 1
            dTrueOut = repmat(dTrueIn, 1, 1, Nw);
        else
            error("Step09C2:BadTrueResidualSize", ...
                "trueResidual third dimension must be 1 or Nw.");
        end

        return;
    end

    error("Step09C2:BadTrueResidualSize", ...
        "trueResidual must be dim x N or dim x N x Nw.");

end

function M = vecNorm_step09c2(A)
%VECNORM_STEP09C2 Compute vector norm over the first dimension.

    M = reshape(sqrt(sum(A.^2, 1)), size(A,2), size(A,3));

end

function M = vecDot_step09c2(A, B)
%VECDOT_STEP09C2 Compute vector dot product over the first dimension.

    M = reshape(sum(A .* B, 1), size(A,2), size(A,3));

end

function R = safeDivide_step09c2(A, B)
%SAFEDIVIDE_STEP09C2 Elementwise division with NaN on invalid denominator.

    R = NaN(size(A));

    mask = isfinite(A) & isfinite(B) & abs(B) > 1e-14;

    R(mask) = A(mask) ./ B(mask);

end

function pct = percentImprovementArray_step09c2(baselineValue, testValue)
%PERCENTIMPROVEMENTARRAY_STEP09C2 Positive means testValue is smaller.

    pct = NaN(size(baselineValue));

    mask = isfinite(baselineValue) & baselineValue > 0 & isfinite(testValue);

    pct(mask) = 100 * (baselineValue(mask) - testValue(mask)) ./ baselineValue(mask);

end

function value = meanMasked_step09c2(X, mask)
%MEANMASKED_STEP09C2 Mean of X over mask with NaN omission.

    values = X(mask);

    if isempty(values)
        value = NaN;
    else
        value = mean(values, "omitnan");
    end

end

function maskOut = alignMaskSize_step09c2(maskIn, N, Nw)
%ALIGNMASKSIZE_STEP09C2 Ensure measurement mask is N x Nw.

    if isequal(size(maskIn), [N, Nw])
        maskOut = maskIn;
        return;
    end

    if isequal(size(maskIn), [N, 1])
        maskOut = repmat(maskIn, 1, Nw);
        return;
    end

    if isequal(size(maskIn), [1, N])
        maskOut = repmat(maskIn(:), 1, Nw);
        return;
    end

    error("Step09C2:BadMeasurementMaskSize", ...
        "resGS.measAvail must be N x Nw, N x 1, or 1 x N.");

end