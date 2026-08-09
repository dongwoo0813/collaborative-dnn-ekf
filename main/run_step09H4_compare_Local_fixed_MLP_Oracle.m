function out = run_step09H4_compare_Local_fixed_MLP_Oracle( ...
    residualFamily, seed, dtOverride, verbose, mlpOverride)
%{
File:
    main/run_step09H4_compare_Local_fixed_MLP_Oracle.m

Purpose:
    Step 09-H.4 Local-only full simulation comparison.

    Compare three always-available cases without touching GS composite logic:

        1. Local fixed_feature_lip DNN-EKF
        2. Local mlp_general DNN-EKF
        3. Oracle residual EKF

Why this runner exists:
    The local init -> predict -> update -> theta-learning path already passed
    structural checks for mlp_general. This runner verifies that the full
    simulation runs to completion and produces finite logs/metrics.

Recommended first run:
    residualFamily = "feedback_sat_disturbance"
    seed           = 101
    dtOverride     = 0.5

Important:
    This is not an MLP tuning script. The first goal is not necessarily that
    MLP beats fixed-feature. The first goal is:

        full simulation completes
        xhat / Pdiag / NIS are finite
        residual logs are generated for mlp_general

Outputs:
    out.summaryTable
        Multi-metric tracking table.

    out.residualSummaryTable
        Operational and same-input residual approximation metrics.

    out.compactComparisonTable
        Key Local fixed / Local MLP / Oracle comparison numbers.
%}

    if nargin < 1 || isempty(residualFamily)
        residualFamily = "feedback_sat_disturbance";
    end

    if nargin < 2 || isempty(seed)
        seed = 101;
    end

    if nargin < 3 || isempty(dtOverride)
        dtOverride = 0.5;
    end

    if nargin < 4 || isempty(verbose)
        verbose = true;
    end

    if nargin < 5 || isempty(mlpOverride)
        mlpOverride = struct();
    end

    residualFamily = string(residualFamily);

    addpath(genpath(pwd));
    rehash;

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 09-H.4: Local fixed-feature / Local MLP / Oracle\n");
        fprintf("============================================================\n");
        fprintf("Residual family = %s\n", residualFamily);
        fprintf("Seed            = %d\n", seed);
        fprintf("dt              = %.6g\n\n", dtOverride);
    end

    % ---------------------------------------------------------------------
    % Base configuration
    % ---------------------------------------------------------------------
    cfgBase = config_step04_GS_DNN_EKF();

    cfgBase.dt = dtOverride;
    cfgBase.time = 0:cfgBase.dt:cfgBase.T;
    cfgBase.N = numel(cfgBase.time);

    cfgBase = applyCommonStep09H4Overrides(cfgBase, residualFamily);

    % ---------------------------------------------------------------------
    % Case 1: Local fixed-feature DNN-EKF
    % ---------------------------------------------------------------------
    cfgFixed = cfgBase;

    cfgFixed.step.name = "step09H4_local_fixed_feature";
    cfgFixed.estimator.type = "local_DNN_EKF";
    cfgFixed.comm.mode = "none";

    cfgFixed.dnn.branchModel = "fixed_feature_lip";
    cfgFixed.dnn.predictionResidualSource = "local_DNN";

    cfgFixed = disableGSForLocalOnlyStep09H4(cfgFixed);

    if verbose
        [nThetaFixed, ~] = branchThetaNumel(cfgFixed);
        fprintf("Running Case 1: Local fixed_feature_lip, nTheta = %d\n", ...
            nThetaFixed);
    end

    rng(seed);
    resFixed = simulateLocalDNNEKF(cfgFixed);
    assertFiniteLocalResultStep09H4(resFixed, "Local fixed_feature_lip");

    % ---------------------------------------------------------------------
    % Case 2: Local general MLP DNN-EKF
    % ---------------------------------------------------------------------
    cfgMLP = cfgBase;

    cfgMLP.step.name = "step09H4_local_mlp_general";
    cfgMLP.estimator.type = "local_DNN_EKF";
    cfgMLP.comm.mode = "none";

    cfgMLP.dnn.branchModel = "mlp_general";
    cfgMLP.dnn.predictionResidualSource = "local_DNN";

    % Explicit MLP architecture for reproducibility.
    %
    % inputMode = "eta_phase":
    %   xi_i = [r/rScale; v/vScale; sin(psi_i); cos(psi_i)]
    cfgMLP.dnn.mlp.hiddenSizes = [12 8 6];
    cfgMLP.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
    cfgMLP.dnn.mlp.inputMode = "eta_phase";

    % Keep these explicit so the full run is independent of future config edits.
    cfgMLP.dnn.mlp.rScale = 1000.0;
    cfgMLP.dnn.mlp.vScale = 0.1;

    % Current intended MLP initialization:
    %   hidden layers nonzero deterministic random
    %   output layer zero
    %
    % This gives initial dHat = 0 but avoids all-zero hidden features.
    cfgMLP.dnn.mlp.thetaInitMode = "random_hidden_zero_output";

    % Optional Step 09-H.5 tuning override.
    %
    % This lets follow-up sweep scripts change only the MLP case while
    % keeping fixed-feature and oracle baselines unchanged.
    cfgMLP = applyNestedOverridesStep09H4(cfgMLP, mlpOverride);

    % cfgMLP = disableGSForLocalOnlyStep09H4(cfgMLP);

    if verbose
        [nThetaMLP, mlpInfo] = branchThetaNumel(cfgMLP);
        fprintf("Running Case 2: Local mlp_general, nTheta = %d\n", nThetaMLP);
        fprintf("    MLP layerSizes = [%s]\n", ...
            sprintf("%d ", mlpInfo.arch.layerSizes));
    end

    rng(seed);
    resMLP = simulateLocalDNNEKF(cfgMLP);
    assertFiniteLocalResultStep09H4(resMLP, "Local mlp_general");

    % ---------------------------------------------------------------------
    % Case 3: Oracle residual EKF
    % ---------------------------------------------------------------------
    cfgOracle = cfgBase;

    cfgOracle.step.name = "step09H4_oracle";
    cfgOracle.estimator.type = "oracle_residual_EKF";
    cfgOracle.comm.mode = "none";

    % The oracle prediction uses trueResidual(...), not the learned branch.
    % Keep fixed_feature_lip here to avoid making the oracle augmented state
    % unnecessarily large.
    cfgOracle.dnn.branchModel = "fixed_feature_lip";
    cfgOracle.dnn.predictionResidualSource = "oracle";

    cfgOracle = disableGSForLocalOnlyStep09H4(cfgOracle);

    if verbose
        [nThetaOracle, ~] = branchThetaNumel(cfgOracle);
        fprintf("Running Case 3: Oracle residual, nTheta = %d\n", ...
            nThetaOracle);
    end

    rng(seed);
    resOracle = simulateLocalDNNEKF(cfgOracle);
    assertFiniteLocalResultStep09H4(resOracle, "Oracle");

    % ---------------------------------------------------------------------
    % Multi-metric estimator evaluation
    % ---------------------------------------------------------------------
    metricsFixed = evaluateEstimatorMetrics( ...
        resFixed, cfgFixed, "Local fixed-feature + always");

    metricsMLP = evaluateEstimatorMetrics( ...
        resMLP, cfgMLP, "Local MLP + always");

    metricsOracle = evaluateEstimatorMetrics( ...
        resOracle, cfgOracle, "Oracle + always");

    summaryTable = [
        metricsFixed.scalarSummary
        metricsMLP.scalarSummary
        metricsOracle.scalarSummary
    ];

    % ---------------------------------------------------------------------
    % Residual approximation metrics
    % ---------------------------------------------------------------------
    residualMetricsFixed = residualErrorSummaryStep09H4( ...
        resFixed, cfgFixed, "Local fixed-feature");

    residualMetricsMLP = residualErrorSummaryStep09H4( ...
        resMLP, cfgMLP, "Local MLP");

    residualMetricsOracle = residualErrorSummaryStep09H4( ...
        resOracle, cfgOracle, "Oracle");

    residualSummaryTable = [
        residualMetricsFixed
        residualMetricsMLP
        residualMetricsOracle
    ];

    % ---------------------------------------------------------------------
    % Compact comparison
    % ---------------------------------------------------------------------
    fixedMean = metricsFixed.meanPosErr;
    mlpMean = metricsMLP.meanPosErr;
    oracleMean = metricsOracle.meanPosErr;

    fixedRms = metricsFixed.rmsPosErr;
    mlpRms = metricsMLP.rmsPosErr;
    oracleRms = metricsOracle.rmsPosErr;

    fixedP95 = metricsFixed.p95PosErr;
    mlpP95 = metricsMLP.p95PosErr;
    oracleP95 = metricsOracle.p95PosErr;

    mlpMeanImprovementPct = percentImprovementStep09H4(fixedMean, mlpMean);
    oracleMeanImprovementPct = percentImprovementStep09H4(fixedMean, oracleMean);

    mlpRmsImprovementPct = percentImprovementStep09H4(fixedRms, mlpRms);
    oracleRmsImprovementPct = percentImprovementStep09H4(fixedRms, oracleRms);

    mlpP95ImprovementPct = percentImprovementStep09H4(fixedP95, mlpP95);
    oracleP95ImprovementPct = percentImprovementStep09H4(fixedP95, oracleP95);

    [nThetaFixed, ~] = branchThetaNumel(cfgFixed);
    [nThetaMLP, ~] = branchThetaNumel(cfgMLP);
    [nThetaOracle, ~] = branchThetaNumel(cfgOracle);

    compactComparisonTable = table( ...
        residualFamily, ...
        seed, ...
        dtOverride, ...
        nThetaFixed, ...
        nThetaMLP, ...
        nThetaOracle, ...
        fixedMean, ...
        mlpMean, ...
        oracleMean, ...
        mlpMeanImprovementPct, ...
        oracleMeanImprovementPct, ...
        fixedRms, ...
        mlpRms, ...
        oracleRms, ...
        mlpRmsImprovementPct, ...
        oracleRmsImprovementPct, ...
        fixedP95, ...
        mlpP95, ...
        oracleP95, ...
        mlpP95ImprovementPct, ...
        oracleP95ImprovementPct, ...
        metricsFixed.meanNIS, ...
        metricsMLP.meanNIS, ...
        metricsOracle.meanNIS, ...
        'VariableNames', { ...
            'residualFamily', ...
            'seed', ...
            'dt', ...
            'nThetaFixed', ...
            'nThetaMLP', ...
            'nThetaOracle', ...
            'fixedMeanPosErr_m', ...
            'mlpMeanPosErr_m', ...
            'oracleMeanPosErr_m', ...
            'mlpMeanImprovementPct_vsFixed', ...
            'oracleMeanImprovementPct_vsFixed', ...
            'fixedRmsPosErr_m', ...
            'mlpRmsPosErr_m', ...
            'oracleRmsPosErr_m', ...
            'mlpRmsImprovementPct_vsFixed', ...
            'oracleRmsImprovementPct_vsFixed', ...
            'fixedP95PosErr_m', ...
            'mlpP95PosErr_m', ...
            'oracleP95PosErr_m', ...
            'mlpP95ImprovementPct_vsFixed', ...
            'oracleP95ImprovementPct_vsFixed', ...
            'fixedMeanNIS', ...
            'mlpMeanNIS', ...
            'oracleMeanNIS'});

    if verbose
        fprintf("\n");
        fprintf("============================================================\n");
        fprintf("Step 09-H.4 multi-metric summary\n");
        fprintf("============================================================\n");
        disp(summaryTable);

        fprintf("\nResidual approximation summary:\n");
        disp(residualSummaryTable);

        fprintf("\nCompact comparison:\n");
        disp(compactComparisonTable);

        fprintf("MLP mean improvement over fixed-feature      = %.3f %%\n", ...
            mlpMeanImprovementPct);
        fprintf("Oracle mean improvement over fixed-feature   = %.3f %%\n", ...
            oracleMeanImprovementPct);
    end

    % ---------------------------------------------------------------------
    % Output
    % ---------------------------------------------------------------------
    out = struct();

    out.residualFamily = residualFamily;
    out.seed = seed;
    out.dt = dtOverride;

    out.cfgBase = cfgBase;
    out.cfgFixed = cfgFixed;
    out.cfgMLP = cfgMLP;
    out.cfgOracle = cfgOracle;

    out.resFixed = resFixed;
    out.resMLP = resMLP;
    out.resOracle = resOracle;

    out.metricsFixed = metricsFixed;
    out.metricsMLP = metricsMLP;
    out.metricsOracle = metricsOracle;

    out.summaryTable = summaryTable;
    out.multiMetricSummaryTable = summaryTable;
    out.residualSummaryTable = residualSummaryTable;
    out.compactComparisonTable = compactComparisonTable;

    out.mlpMeanImprovementPct = mlpMeanImprovementPct;
    out.oracleMeanImprovementPct = oracleMeanImprovementPct;

end

function cfg = applyCommonStep09H4Overrides(cfg, residualFamily)
%APPLYCOMMONSTEP09H4OVERRIDES Shared Local-only benchmark settings.

    residualFamily = string(residualFamily);

    % Use the selected residual family explicitly so this runner is not
    % affected by future config defaults.
    cfg.truth.useResidual = true;
    cfg.truth.residualFamily = residualFamily;

    % Backward-compatible alias for older helper functions.
    if residualFamily == "simple_branchwise"
        cfg.truth.residualModel = "branchwise";
    else
        cfg.truth.residualModel = residualFamily;
    end

    % Keep the same amplitude used in recent Step 09 feedback_sat benchmarks.
    cfg.truth.residualAmp = 5e-4;

    % Fair local comparison:
    %   fixed-feature starts at zero
    %   MLP output layer starts at zero
    cfg.dnn.theta0_std = 0.0;
    cfg.dnn.useResidualInPrediction = true;
    cfg.dnn.residualInjectionGain = 1.0;

    % Keep the current validated covariance prediction path.
    cfg.ekf.useBlockCovPrediction = true;

    % Local-only full simulation: no FOV dropout yet.
    cfg.meas.availabilityMode = "always";

    if isfield(cfg, "fov")
        cfg.fov.enabled = false;
        cfg.fov.guardUnimplementedMode = true;
    end

end

function cfg = disableGSForLocalOnlyStep09H4(cfg)
%DISABLEGSFORLOCALONLYSTEP09H4 Prevent this runner from touching GS logic.

    if isfield(cfg, "gs")
        cfg.gs.enabled = false;
        cfg.gs.bootstrapUpload = false;
        cfg.gs.useNonlocalBranchCovariance = false;
        cfg.gs.broadcastMode = "none";
        cfg.gs.uploadMode = "none";
    end

end

function assertFiniteLocalResultStep09H4(results, caseName)
%ASSERTFINITELOCALRESULTSTEP09H4 Lightweight full-run finite-output check.
%
% This is intentionally not a large sanity test. It only catches the main
% failure modes expected when nTheta changes from 18 to 256:
%   NaN/Inf xhat
%   NaN/Inf covariance diagonal
%   missing finite NIS values

    caseName = string(caseName);

    if ~all(isfinite(results.xhat(:)))
        error("Step09H4:NonFiniteXhat", ...
            "%s produced non-finite entries in results.xhat.", caseName);
    end

    if isfield(results, "thetaHat")
        if ~all(isfinite(results.thetaHat(:)))
            error("Step09H4:NonFiniteThetaHat", ...
                "%s produced non-finite entries in results.thetaHat.", caseName);
        end
    end

    if isfield(results, "Pdiag")
        if ~all(isfinite(results.Pdiag(:)))
            error("Step09H4:NonFinitePdiag", ...
                "%s produced non-finite entries in results.Pdiag.", caseName);
        end
    end

    if isfield(results, "NIS")
        nisFinite = results.NIS(isfinite(results.NIS));

        if isempty(nisFinite)
            error("Step09H4:NoFiniteNIS", ...
                "%s produced no finite NIS entries.", caseName);
        end

        if any(nisFinite < -1e-10)
            error("Step09H4:NegativeNIS", ...
                "%s produced negative finite NIS entries.", caseName);
        end
    end

end

function residualTable = residualErrorSummaryStep09H4(results, cfg, caseName)
%RESIDUALERRORSUMMARYSTEP09H4 Residual approximation summary table.
%
% operational:
%   d_hat evaluated at eta_hat.
%
% same_input:
%   d_hat evaluated at eta_true.
%
% For oracle:
%   d_hat is treated as d_true because the estimator uses trueResidual(...)
%   directly in prediction.

    caseName = string(caseName);

    branchModel = "unknown";
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "branchModel")
        branchModel = string(cfg.dnn.branchModel);
    end

    predictionResidualSource = "unknown";
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "predictionResidualSource")
        predictionResidualSource = string(cfg.dnn.predictionResidualSource);
    end

    [meanOperationalErr, rmsOperationalErr] = residualErrorStatsStep09H4( ...
        results, cfg, "operational");

    [meanSameInputErr, rmsSameInputErr] = residualErrorStatsStep09H4( ...
        results, cfg, "same_input");

    residualTable = table( ...
        caseName, ...
        branchModel, ...
        predictionResidualSource, ...
        meanOperationalErr, ...
        rmsOperationalErr, ...
        meanSameInputErr, ...
        rmsSameInputErr, ...
        'VariableNames', { ...
            'caseName', ...
            'branchModel', ...
            'predictionResidualSource', ...
            'meanOperationalResidualErr', ...
            'rmsOperationalResidualErr', ...
            'meanSameInputResidualErr', ...
            'rmsSameInputResidualErr'});

end

function [meanErr, rmsErr] = residualErrorStatsStep09H4(results, cfg, mode)
%RESIDUALERRORSTATSSTEP09H4 Compute scalar residual approximation errors.

    mode = string(mode);

    switch mode
        case "operational"
            dHat = results.dnnResidual;

        case "same_input"
            dHat = results.dnnResidualAtTrueEta;

        otherwise
            error("Step09H4:BadResidualMode", ...
                "Unsupported residual error mode = %s.", mode);
    end

    [dim, N, Nw] = size(dHat);

    dTrue = expandTrueResidualStep09H4(results.trueResidual, dim, N, Nw);

    residualSource = "unknown";
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "predictionResidualSource")
        residualSource = string(cfg.dnn.predictionResidualSource);
    end

    if residualSource == "oracle"
        % Oracle prediction uses the hidden true residual directly.
        dHat = dTrue;
    end

    errNorm = reshape(sqrt(sum((dHat - dTrue).^2, 1)), N, Nw);

    errAll = errNorm(:);

    meanErr = mean(errAll, "omitnan");
    rmsErr = sqrt(mean(errAll.^2, "omitnan"));

end

function dTrueOut = expandTrueResidualStep09H4(dTrueIn, dim, N, Nw)
%EXPANDTRUERESIDUALSTEP09H4 Convert true residual log to dim x N x Nw.

    sz = size(dTrueIn);

    if numel(sz) == 2
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09H4:BadTrueResidualSize", ...
                "trueResidual must be dim x N or dim x N x Nw.");
        end

        dTrueOut = reshape(dTrueIn, dim, N, 1);
        dTrueOut = repmat(dTrueOut, 1, 1, Nw);
        return;
    end

    if numel(sz) == 3
        if sz(1) ~= dim || sz(2) ~= N
            error("Step09H4:BadTrueResidualSize", ...
                "trueResidual has incompatible first two dimensions.");
        end

        if sz(3) == Nw
            dTrueOut = dTrueIn;
        elseif sz(3) == 1
            dTrueOut = repmat(dTrueIn, 1, 1, Nw);
        else
            error("Step09H4:BadTrueResidualSize", ...
                "trueResidual third dimension must be 1 or Nw.");
        end

        return;
    end

    error("Step09H4:BadTrueResidualSize", ...
        "trueResidual must be dim x N or dim x N x Nw.");

end

function pct = percentImprovementStep09H4(baselineValue, testValue)
%PERCENTIMPROVEMENTSTEP09H4 Positive means testValue is smaller.

    if isfinite(baselineValue) && baselineValue > 0 && isfinite(testValue)
        pct = 100 * (baselineValue - testValue) / baselineValue;
    else
        pct = NaN;
    end

end


function s = applyNestedOverridesStep09H4(s, override)
%APPLYNESTEDOVERRIDESSTEP09H4 Recursively apply nested struct overrides.
%
% Purpose:
%   Small utility for Step 09-H.5 tuning sweeps.
%
% Example:
%   override.dnn.residualInjectionGain = 0.5;
%   override.dnn.thetaSigmaSS = 5e-5;
%
% Then:
%   cfg = applyNestedOverridesStep09H4(cfg, override);
%
% Notes:
%   This helper intentionally only touches fields provided in override.
%   It does not validate the resulting cfg; validation remains inside the
%   estimator/prediction functions.

if isempty(override)
    return;
end

names = fieldnames(override);

for k = 1:numel(names)
    name = names{k};

    if isstruct(override.(name))
        if ~isfield(s, name) || ~isstruct(s.(name))
            s.(name) = struct();
        end

        % Recursively merge nested structs such as override.dnn.mlp.
        s.(name) = applyNestedOverridesStep09H4( ...
            s.(name), override.(name));
    else
        s.(name) = override.(name);
    end
end

end