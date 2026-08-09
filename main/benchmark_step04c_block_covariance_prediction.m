function out = benchmark_step04c_block_covariance_prediction(mode, nRepeats)
%{
File:
    scripts/benchmark_step04c_block_covariance_prediction.m

Purpose:
    Benchmark dense covariance prediction versus Step 04c block covariance
    prediction for the local DNN-EKF / GS-composite DNN-EKF simulation.

Compared options:
    Dense:
        cfg.ekf.useBlockCovPrediction = false

    Block:
        cfg.ekf.useBlockCovPrediction = true

Main metrics:
    - total runtime
    - mean runtime over repeated runs
    - standard deviation
    - speedup = meanDenseTime / meanBlockTime
    - runtime per step
    - runtime per watcher per step

How to run:
    Default GS benchmark:
        out = benchmark_step04c_block_covariance_prediction();

    GS benchmark with 5 repeats:
        out = benchmark_step04c_block_covariance_prediction("gs", 5);

    Local-DNN benchmark:
        out = benchmark_step04c_block_covariance_prediction("local", 3);

    Both local and GS:
        out = benchmark_step04c_block_covariance_prediction("both", 3);

Notes:
    - A warm-up run is performed first and is not timed.
    - Dense and block runs use the same random seed each repeat.
    - The first repeat also checks numerical equivalence.
%}

    if nargin < 1
        mode = "gs";
    end

    if nargin < 2
        nRepeats = 3;
    end

    mode = lower(string(mode));

    clc;
    addpath(genpath(pwd));
    rehash;

    fprintf("\n============================================================\n");
    fprintf("Step 04c benchmark: dense vs block covariance prediction\n");
    fprintf("============================================================\n");

    fprintf("Mode     : %s\n", mode);
    fprintf("Repeats  : %d\n", nRepeats);

    out = struct();
    out.mode = mode;
    out.nRepeats = nRepeats;

    switch mode

        case "local"
            out.local = runOneBenchmarkCase("local", nRepeats);

        case "gs"
            out.gs = runOneBenchmarkCase("gs", nRepeats);

        case "both"
            out.local = runOneBenchmarkCase("local", nRepeats);
            out.gs = runOneBenchmarkCase("gs", nRepeats);

        otherwise
            error("Unknown benchmark mode: %s. Use 'local', 'gs', or 'both'.", mode);

    end

    fprintf("\n============================================================\n");
    fprintf("Step 04c benchmark finished.\n");
    fprintf("============================================================\n");

end

function outCase = runOneBenchmarkCase(caseType, nRepeats)
% Run dense/block benchmark for one simulation case.

    caseType = lower(string(caseType));

    fprintf("\n------------------------------------------------------------\n");
    fprintf("Benchmark case: %s\n", upper(caseType));
    fprintf("------------------------------------------------------------\n");

    seedBase = 100;

    cfgDense = makeBenchmarkConfig(caseType);
    cfgDense.ekf.useBlockCovPrediction = false;

    cfgBlock = cfgDense;
    cfgBlock.ekf.useBlockCovPrediction = true;

    % ------------------------------------------------------------
    % Warm-up runs.
    % ------------------------------------------------------------
    fprintf("Warm-up dense run...\n");
    rng(seedBase);
    runSimulationForCase(caseType, cfgDense);

    fprintf("Warm-up block run...\n");
    rng(seedBase);
    runSimulationForCase(caseType, cfgBlock);

    denseTimes = zeros(nRepeats, 1);
    blockTimes = zeros(nRepeats, 1);

    equivalence = struct();

    for r = 1:nRepeats

        fprintf("\nRepeat %d / %d\n", r, nRepeats);

        % Dense run.
        rng(seedBase + r);
        tDense = tic;
        resDense = runSimulationForCase(caseType, cfgDense);
        denseTimes(r) = toc(tDense);

        fprintf("  Dense runtime = %.4f s\n", denseTimes(r));

        % Block run.
        rng(seedBase + r);
        tBlock = tic;
        resBlock = runSimulationForCase(caseType, cfgBlock);
        blockTimes(r) = toc(tBlock);

        fprintf("  Block runtime = %.4f s\n", blockTimes(r));

        fprintf("  Speedup       = %.4f x\n", denseTimes(r) / blockTimes(r));

        % Numerical equivalence check on the first timed repeat.
        if r == 1
            equivalence = computeEquivalence(resDense, resBlock);

            fprintf("  Equivalence check on repeat 1:\n");
            fprintf("    max |xhat dense - block|   = %.3e\n", equivalence.diff_xhat);
            fprintf("    max |xaug dense - block|   = %.3e\n", equivalence.diff_xaug);
            fprintf("    max |theta dense - block|  = %.3e\n", equivalence.diff_theta);
            fprintf("    max |Pdiag dense - block|  = %.3e\n", equivalence.diff_Pdiag);
            fprintf("    max |NIS dense - block|    = %.3e\n", equivalence.diff_NIS);
        end

    end

    meanDense = mean(denseTimes);
    meanBlock = mean(blockTimes);

    stdDense = std(denseTimes);
    stdBlock = std(blockTimes);

    speedupEach = denseTimes ./ blockTimes;
    meanSpeedup = meanDense / meanBlock;

    Nsteps = numel(resBlock.time);
    Nw = cfgBlock.Nw;

    densePerStep = meanDense / Nsteps;
    blockPerStep = meanBlock / Nsteps;

    densePerWatcherStep = meanDense / (Nsteps * Nw);
    blockPerWatcherStep = meanBlock / (Nsteps * Nw);

    summaryTable = table( ...
        ["Dense"; "Block"], ...
        [meanDense; meanBlock], ...
        [stdDense; stdBlock], ...
        [densePerStep; blockPerStep], ...
        [densePerWatcherStep; blockPerWatcherStep], ...
        'VariableNames', { ...
            'method', ...
            'meanRuntime_sec', ...
            'stdRuntime_sec', ...
            'runtimePerStep_sec', ...
            'runtimePerWatcherStep_sec'});

    fprintf("\n%s benchmark summary\n", upper(caseType));
    disp(summaryTable);

    fprintf("Mean speedup dense/block = %.4f x\n", meanSpeedup);
    fprintf("Per-repeat speedups:\n");
    disp(speedupEach.');

    outCase = struct();

    outCase.caseType = caseType;
    outCase.nRepeats = nRepeats;

    outCase.cfgDense = cfgDense;
    outCase.cfgBlock = cfgBlock;

    outCase.denseTimes = denseTimes;
    outCase.blockTimes = blockTimes;
    outCase.speedupEach = speedupEach;
    outCase.meanSpeedup = meanSpeedup;

    outCase.meanDense = meanDense;
    outCase.meanBlock = meanBlock;
    outCase.stdDense = stdDense;
    outCase.stdBlock = stdBlock;

    outCase.Nsteps = Nsteps;
    outCase.Nw = Nw;

    outCase.summaryTable = summaryTable;
    outCase.equivalence = equivalence;

end

function cfg = makeBenchmarkConfig(caseType)
% Build benchmark configuration.

    caseType = lower(string(caseType));

    cfg = config_step04_GS_DNN_EKF();

    cfg.truth.residualAmp = 5e-4;

    cfg.dnn.theta0_std = 0.0;
    cfg.dnn.residualInjectionGain = 1.0;

    if ~isfield(cfg, "ekf")
        cfg.ekf = struct();
    end

    cfg.ekf.blockPredictionTol = 1e-12;

    % Try to disable plotting flags if such fields exist.
    cfg = disableKnownPlotFlags(cfg);

    switch caseType

        case "local"

            cfg.dnn.predictionResidualSource = "local_DNN";

            if ~isfield(cfg, "gs")
                cfg.gs = struct();
            end

            cfg.gs.enabled = false;

        case "gs"

            cfg.dnn.predictionResidualSource = "GS_composite";

            if ~isfield(cfg, "gs")
                cfg.gs = struct();
            end

            cfg.gs.enabled = true;
            cfg.gs.bootstrapUpload = true;
            cfg.gs.uploadMode = "after_measurement_update";
            cfg.gs.broadcastMode = "every_step";

            % Use the current Step 04b setting.
            cfg.gs.useNonlocalBranchCovariance = true;
            cfg.gs.youngMode = "uniform";
            cfg.gs.SresNonlocal = 0.0;

        otherwise
            error("Unknown benchmark caseType: %s.", caseType);

    end

end

function results = runSimulationForCase(caseType, cfg)
% Run the correct simulation function for the selected case.

    caseType = lower(string(caseType));

    switch caseType

        case "local"
            results = simulateLocalDNNEKF(cfg);

        case "gs"
            results = simulate_GS_DNN_EKF(cfg);

        otherwise
            error("Unknown simulation caseType: %s.", caseType);

    end

end

function eq = computeEquivalence(resDense, resBlock)
% Compute numerical differences between dense and block results.

    eq = struct();

    eq.diff_xhat = maxAbsDiff(resDense.xhat, resBlock.xhat);
    eq.diff_xaug = maxAbsDiff(resDense.xhatAug, resBlock.xhatAug);
    eq.diff_theta = maxAbsDiff(resDense.thetaHat, resBlock.thetaHat);
    eq.diff_Pdiag = maxAbsDiff(resDense.Pdiag, resBlock.Pdiag);
    eq.diff_NIS = maxAbsDiffFinite(resDense.NIS, resBlock.NIS);

end

function cfg = disableKnownPlotFlags(cfg)
% Disable common plotting flags if they exist in the configuration.
% This function is intentionally conservative. It does not assume a fixed
% plotting configuration structure.

    if isfield(cfg, "plot")
        cfg.plot.enabled = false;

        if isfield(cfg.plot, "show")
            cfg.plot.show = false;
        end

        if isfield(cfg.plot, "makePlots")
            cfg.plot.makePlots = false;
        end
    end

    if isfield(cfg, "plots")
        cfg.plots.enabled = false;

        if isfield(cfg.plots, "show")
            cfg.plots.show = false;
        end

        if isfield(cfg.plots, "makePlots")
            cfg.plots.makePlots = false;
        end
    end

    if isfield(cfg, "debug")
        if isfield(cfg.debug, "makePlots")
            cfg.debug.makePlots = false;
        end

        if isfield(cfg.debug, "verbose")
            cfg.debug.verbose = false;
        end
    end

end

function d = maxAbsDiff(a, b)
% Return max absolute difference between two numeric arrays.

    d = max(abs(a(:) - b(:)));

end

function d = maxAbsDiffFinite(a, b)
% Return max absolute difference using only entries finite in both arrays.
% Useful for NIS arrays that may contain NaN when measurements are missing.

    a = a(:);
    b = b(:);

    finiteMask = isfinite(a) & isfinite(b);

    if ~any(finiteMask)
        d = 0;
        return;
    end

    d = max(abs(a(finiteMask) - b(finiteMask)));

    nanMismatch = xor(isnan(a), isnan(b));
    assert(~any(nanMismatch), ...
        "NaN pattern differs between dense and block results.");

end