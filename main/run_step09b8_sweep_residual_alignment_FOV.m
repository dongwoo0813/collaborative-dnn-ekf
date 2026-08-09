function out = run_step09b8_sweep_residual_alignment_FOV( ...
    halfAngleListDeg, seed, residualFamily, outFOVSweep)
%{
File:
    main/run_step09b8_sweep_residual_alignment_FOV.m

Purpose:
    Step 09-B.8 residual-alignment FOV sweep.

    This script runs residual alignment diagnostics for each FOV half-angle
    case in a previously computed FOV sweep.

    It is intended mainly for the coupled_nonlinear benchmark, where GS
    tracking improvement should be explained by improved residual
    approximation quality.

Inputs:
    halfAngleListDeg
        FOV half-angle list in degrees.

    seed
        Random seed.

    residualFamily
        Truth residual family, usually "coupled_nonlinear".

    outFOVSweep
        Optional output from:
            run_step08b5_sweep_FOV_angle_multi_metric(...)
        If omitted, this script runs that sweep internally.

Outputs:
    out.alignmentSummaryTable
        Long-format Local / GS / Oracle residual-alignment metrics.

    out.alignmentComparisonTable
        One row per FOV angle comparing GS/Oracle residual alignment
        against Local.

Usage:
    out09b8 = run_step09b8_sweep_residual_alignment_FOV( ...
        [20 40 90], 101, "coupled_nonlinear", out09b7);

    or:

    out09b8 = run_step09b8_sweep_residual_alignment_FOV();
%}

    if nargin < 1 || isempty(halfAngleListDeg)
        halfAngleListDeg = [20 40 90];
    end

    if nargin < 2 || isempty(seed)
        seed = 101;
    end

    if nargin < 3 || isempty(residualFamily)
        residualFamily = "coupled_nonlinear";
    end

    if nargin < 4
        outFOVSweep = [];
    end

    halfAngleListDeg = halfAngleListDeg(:);
    residualFamily = string(residualFamily);

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-B.8: Residual-alignment FOV sweep\n");
    fprintf("============================================================\n");
    fprintf("Residual family = %s\n", residualFamily);
    fprintf("Random seed     = %d\n\n", seed);

    addpath(genpath(pwd));
    rehash;

    if isempty(outFOVSweep)
        outFOVSweep = run_step08b5_sweep_FOV_angle_multi_metric( ...
            halfAngleListDeg, residualFamily, seed);
    end

    nAngle = numel(halfAngleListDeg);

    alignmentOutputCell = cell(nAngle, 1);
    summaryCell = cell(nAngle, 1);
    comparisonCell = cell(nAngle, 1);

    for ia = 1:nAngle

        halfAngleDeg = halfAngleListDeg(ia);

        fprintf("Running residual alignment %d / %d: halfAngle = %.3f deg\n", ...
            ia, nAngle, halfAngleDeg);

        % Reuse existing Step 08-B.5 simulation output.
        outAlign = run_step08b7_residual_alignment_diagnostics( ...
            halfAngleDeg, seed, false, outFOVSweep.caseOutputCell{ia});

        alignmentOutputCell{ia} = outAlign;

        Tsum = outAlign.residualAlignmentSummaryTable;

        Tsum = addvars(Tsum, ...
            repmat(residualFamily, height(Tsum), 1), ...
            repmat(halfAngleDeg, height(Tsum), 1), ...
            repmat(seed, height(Tsum), 1), ...
            'Before', 1, ...
            'NewVariableNames', {'residualFamily', 'halfAngleDeg', 'seed'});

        summaryCell{ia} = Tsum;

        Tcmp = outAlign.comparisonTable;

        Tcmp = addvars(Tcmp, ...
            repmat(residualFamily, height(Tcmp), 1), ...
            'Before', 1, ...
            'NewVariableNames', 'residualFamily');

        comparisonCell{ia} = Tcmp;

    end

    alignmentSummaryTable = vertcat(summaryCell{:});
    alignmentComparisonTable = vertcat(comparisonCell{:});

    fprintf("\n============================================================\n");
    fprintf("Step 09-B.8 residual-alignment comparison summary\n");
    fprintf("============================================================\n");
    disp(alignmentComparisonTable);

    fprintf("\nKey Local / GS / Oracle residual metrics:\n");
    disp(alignmentSummaryTable(:, ...
        ["residualFamily", ...
         "halfAngleDeg", ...
         "caseName", ...
         "meanResErrNorm_mps2", ...
         "meanRelativeResErr", ...
         "meanCosineAlignment", ...
         "meanProjectionRatio", ...
         "meanResErrNorm_dropout_mps2", ...
         "meanCosineAlignment_dropout"]));

    out = struct();
    out.halfAngleListDeg = halfAngleListDeg;
    out.seed = seed;
    out.residualFamily = residualFamily;
    out.outFOVSweep = outFOVSweep;
    out.alignmentOutputCell = alignmentOutputCell;
    out.alignmentSummaryTable = alignmentSummaryTable;
    out.alignmentComparisonTable = alignmentComparisonTable;

    fprintf("\n============================================================\n");
    fprintf("Step 09-B.8 residual-alignment FOV sweep complete.\n");
    fprintf("============================================================\n\n");

end