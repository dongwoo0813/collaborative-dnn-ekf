function diag = run_step09J5b_fim_gate_temporal_diagnostics(out09j5, windowEdges, makePlots)
%{
File:
    main/run_step09J5b_fim_gate_temporal_diagnostics.m

Purpose:
    Analyze time history of direction-only bearing-FIM gates from a
    Step 09-J.5 bearing_fim_gated GS simulation.

Inputs:
    out09j5     - Output from run_step09J5_compare_bearing_fim_gated_MLP(...)
                 after simulate_GS_DNN_EKF has Step 09-J.5b FIM gate logs.

    windowEdges - Optional window edges in seconds.
                  Default: [0 800 900 1200 1400 1500 2000].

    makePlots   - Optional logical. If true, create diagnostic figures.
                  Default: true.

Outputs:
    diag - Structure containing:
              time histories,
              window summary table,
              figure handles.

What this diagnostic checks:
    1. Does trace(OmegaBar_j) warm up over time?
    2. Is OmegaSigma_m well-conditioned?
    3. Are B_{j|m} gate norms too small during 0--800 s?
    4. Does the gate become more selective around 900--1200 s?
    5. Is the gate identity sum_j B_j ≈ I maintained over time?

Interpretation:
    - Small nonlocal B-norm during 0--800 s supports the hypothesis that
      FIM-gated was too conservative early.
    - Large condition number or tiny lambda_min(OmegaSigma) suggests
      numerical or geometry degeneracy.
    - Stable gate identity error near zero suggests the gate algebra is
      internally consistent.
%}

    if nargin < 1 || isempty(out09j5)
        error("run_step09J5b_fim_gate_temporal_diagnostics:MissingInput", ...
            "Input out09j5 is required.");
    end

    if nargin < 2 || isempty(windowEdges)
        windowEdges = [0 800 900 1200 1400 1500 2000];
    end

    if nargin < 3 || isempty(makePlots)
        makePlots = true;
    end

    if ~isfield(out09j5, "resGSFIM")
        error("run_step09J5b_fim_gate_temporal_diagnostics:MissingResGSFIM", ...
            "out09j5.resGSFIM is missing.");
    end

    res = out09j5.resGSFIM;

    requiredFields = [
        "time"
        "fimGateTraceOmegaBar"
        "fimGateBnorm"
        "fimGateOmegaSigmaMinEig"
        "fimGateOmegaSigmaCond"
        "fimGateSumIdentityError"
        "fimGateNumBranches"
    ];

    for k = 1:numel(requiredFields)
        f = requiredFields(k);

        if ~isfield(res, f)
            error("run_step09J5b_fim_gate_temporal_diagnostics:MissingField", ...
                "out09j5.resGSFIM.%s is missing. Patch simulate_GS_DNN_EKF and rerun out09j5.", ...
                f);
        end
    end

    time = res.time(:);
    N = numel(time);

    traceOmegaBar = res.fimGateTraceOmegaBar;
    Bnorm = res.fimGateBnorm;
    lambdaMinOmegaSigma = res.fimGateOmegaSigmaMinEig;
    condOmegaSigma = res.fimGateOmegaSigmaCond;
    gateSumIdentityError = res.fimGateSumIdentityError;
    numGateBranches = res.fimGateNumBranches;

    [Nw, Ntrace, Nreceiver] = size(traceOmegaBar);

    if Ntrace ~= N
        error("run_step09J5b_fim_gate_temporal_diagnostics:BadTraceSize", ...
            "fimGateTraceOmegaBar time dimension does not match res.time.");
    end

    if Nreceiver ~= Nw
        error("run_step09J5b_fim_gate_temporal_diagnostics:BadWatcherSize", ...
            "Expected source and receiver branch dimensions to both equal Nw.");
    end

    % ---------------------------------------------------------------------
    % Aggregate by time.
    %
    % traceOmegaBarBySource(j,k):
    %   mean trace(OmegaBar_j) across recipient watchers that have branch j.
    %
    % meanLocalBnorm(k):
    %   mean ||B_{m|m}||_F across recipient watchers m.
    %
    % meanNonlocalBnorm(k):
    %   mean ||B_{j|m}||_F across j ~= m.
    % ---------------------------------------------------------------------
    traceOmegaBarBySource = NaN(Nw, N);

    for j = 1:Nw
        vals_j = squeeze(traceOmegaBar(j, :, :));  % N x Nreceiver
        traceOmegaBarBySource(j, :) = mean(vals_j, 2, "omitnan").';
    end

    localBnorm = NaN(N, Nw);
    nonlocalBnorm = NaN(N, Nw, Nw);

    for m = 1:Nw
        localBnorm(:, m) = squeeze(Bnorm(m, :, m)).';

        for j = 1:Nw
            if j == m
                continue;
            end
            nonlocalBnorm(:, j, m) = squeeze(Bnorm(j, :, m)).';
        end
    end

    meanLocalBnorm = mean(localBnorm, 2, "omitnan");
    maxLocalBnorm = max(localBnorm, [], 2, "omitnan");

    nonlocalFlat = reshape(nonlocalBnorm, N, []);
    meanNonlocalBnorm = mean(nonlocalFlat, 2, "omitnan");
    maxNonlocalBnorm = max(nonlocalFlat, [], 2, "omitnan");

    meanLambdaMinOmegaSigma = mean(lambdaMinOmegaSigma, 2, "omitnan");
    minLambdaMinOmegaSigma = min(lambdaMinOmegaSigma, [], 2, "omitnan");

    medianCondOmegaSigma = median(condOmegaSigma, 2, "omitnan");
    maxCondOmegaSigma = max(condOmegaSigma, [], 2, "omitnan");

    meanGateIdentityError = mean(gateSumIdentityError, 2, "omitnan");
    maxGateIdentityError = max(gateSumIdentityError, [], 2, "omitnan");

    meanNumGateBranches = mean(numGateBranches, 2, "omitnan");

    % ---------------------------------------------------------------------
    % Window summary.
    % ---------------------------------------------------------------------
    windowSummaryTable = buildFIMGateWindowSummaryStep09J5b( ...
        time, ...
        windowEdges, ...
        traceOmegaBarBySource, ...
        meanLocalBnorm, ...
        meanNonlocalBnorm, ...
        maxNonlocalBnorm, ...
        meanLambdaMinOmegaSigma, ...
        minLambdaMinOmegaSigma, ...
        medianCondOmegaSigma, ...
        maxCondOmegaSigma, ...
        meanGateIdentityError, ...
        maxGateIdentityError, ...
        meanNumGateBranches);

    % ---------------------------------------------------------------------
    % Output structure.
    % ---------------------------------------------------------------------
    diag = struct();
    diag.time = time;
    diag.traceOmegaBarBySource = traceOmegaBarBySource;
    diag.meanLocalBnorm = meanLocalBnorm;
    diag.maxLocalBnorm = maxLocalBnorm;
    diag.meanNonlocalBnorm = meanNonlocalBnorm;
    diag.maxNonlocalBnorm = maxNonlocalBnorm;
    diag.meanLambdaMinOmegaSigma = meanLambdaMinOmegaSigma;
    diag.minLambdaMinOmegaSigma = minLambdaMinOmegaSigma;
    diag.medianCondOmegaSigma = medianCondOmegaSigma;
    diag.maxCondOmegaSigma = maxCondOmegaSigma;
    diag.meanGateIdentityError = meanGateIdentityError;
    diag.maxGateIdentityError = maxGateIdentityError;
    diag.meanNumGateBranches = meanNumGateBranches;
    diag.windowSummaryTable = windowSummaryTable;
    diag.figGate = [];

    if makePlots
        diag.figGate = plotFIMGateTemporalDiagnosticsStep09J5b(diag, out09j5);
    end

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-J.5b: bearing-FIM gate temporal diagnostics\n");
    fprintf("============================================================\n");
    disp(windowSummaryTable);

end

function T = buildFIMGateWindowSummaryStep09J5b( ...
    time, windowEdges, traceOmegaBarBySource, meanLocalBnorm, ...
    meanNonlocalBnorm, maxNonlocalBnorm, meanLambdaMinOmegaSigma, ...
    minLambdaMinOmegaSigma, medianCondOmegaSigma, maxCondOmegaSigma, ...
    meanGateIdentityError, maxGateIdentityError, meanNumGateBranches)
%BUILDFIMGATEWINDOWSUMMARYSTEP09J5B Summarize gate diagnostics by time window.

    nWin = numel(windowEdges) - 1;

    windowName = strings(nWin, 1);
    tStart_s = zeros(nWin, 1);
    tEnd_s = zeros(nWin, 1);

    meanTraceOmegaBar = NaN(nWin, 1);
    minTraceOmegaBar = NaN(nWin, 1);
    maxTraceOmegaBar = NaN(nWin, 1);

    meanLocalBnormWin = NaN(nWin, 1);
    meanNonlocalBnormWin = NaN(nWin, 1);
    maxNonlocalBnormWin = NaN(nWin, 1);

    meanLambdaMinOmegaSigmaWin = NaN(nWin, 1);
    minLambdaMinOmegaSigmaWin = NaN(nWin, 1);

    medianCondOmegaSigmaWin = NaN(nWin, 1);
    maxCondOmegaSigmaWin = NaN(nWin, 1);

    meanGateIdentityErrorWin = NaN(nWin, 1);
    maxGateIdentityErrorWin = NaN(nWin, 1);

    meanNumGateBranchesWin = NaN(nWin, 1);

    traceMeanAcrossSources = mean(traceOmegaBarBySource, 1, "omitnan");
    traceMinAcrossSources = min(traceOmegaBarBySource, [], 1, "omitnan");
    traceMaxAcrossSources = max(traceOmegaBarBySource, [], 1, "omitnan");

    for w = 1:nWin
        t0 = windowEdges(w);
        t1 = windowEdges(w+1);

        idx = time >= t0 & time < t1;

        if w == nWin
            idx = time >= t0 & time <= t1;
        end

        windowName(w) = sprintf("%g-%g s", t0, t1);
        tStart_s(w) = t0;
        tEnd_s(w) = t1;

        meanTraceOmegaBar(w) = mean(traceMeanAcrossSources(idx), "omitnan");
        minTraceOmegaBar(w) = min(traceMinAcrossSources(idx), [], "omitnan");
        maxTraceOmegaBar(w) = max(traceMaxAcrossSources(idx), [], "omitnan");

        meanLocalBnormWin(w) = mean(meanLocalBnorm(idx), "omitnan");
        meanNonlocalBnormWin(w) = mean(meanNonlocalBnorm(idx), "omitnan");
        maxNonlocalBnormWin(w) = max(maxNonlocalBnorm(idx), [], "omitnan");

        meanLambdaMinOmegaSigmaWin(w) = mean(meanLambdaMinOmegaSigma(idx), "omitnan");
        minLambdaMinOmegaSigmaWin(w) = min(minLambdaMinOmegaSigma(idx), [], "omitnan");

        medianCondOmegaSigmaWin(w) = median(medianCondOmegaSigma(idx), "omitnan");
        maxCondOmegaSigmaWin(w) = max(maxCondOmegaSigma(idx), [], "omitnan");

        meanGateIdentityErrorWin(w) = mean(meanGateIdentityError(idx), "omitnan");
        maxGateIdentityErrorWin(w) = max(maxGateIdentityError(idx), [], "omitnan");

        meanNumGateBranchesWin(w) = mean(meanNumGateBranches(idx), "omitnan");
    end

    T = table( ...
        windowName, ...
        tStart_s, ...
        tEnd_s, ...
        meanTraceOmegaBar, ...
        minTraceOmegaBar, ...
        maxTraceOmegaBar, ...
        meanLocalBnormWin, ...
        meanNonlocalBnormWin, ...
        maxNonlocalBnormWin, ...
        meanLambdaMinOmegaSigmaWin, ...
        minLambdaMinOmegaSigmaWin, ...
        medianCondOmegaSigmaWin, ...
        maxCondOmegaSigmaWin, ...
        meanGateIdentityErrorWin, ...
        maxGateIdentityErrorWin, ...
        meanNumGateBranchesWin);

end

function fig = plotFIMGateTemporalDiagnosticsStep09J5b(diag, out09j5)
%PLOTFIMGATETEMPORALDIAGNOSTICSSTEP09J5B Plot gate diagnostic histories.

    time = diag.time;

    fig = figure( ...
        "Name", "Step 09-J.5b Bearing-FIM Gate Temporal Diagnostics", ...
        "Color", "w");

    tiledlayout(fig, 2, 2, ...
        "TileSpacing", "compact", ...
        "Padding", "compact");

    % 1. trace(OmegaBar_j) by source branch.
    nexttile;
    plot(time, diag.traceOmegaBarBySource.', "LineWidth", 1.1);
    addStep09J5bWindowMarkers();
    grid on;
    xlabel("Time [s]");
    ylabel("trace(\OmegaBar_j)");
    title("Accumulated geometry support by source branch");
    legend(compose("branch %d", 1:size(diag.traceOmegaBarBySource, 1)), ...
        "Location", "best");

    % 2. OmegaSigma conditioning.
    nexttile;
    yyaxis left;
    plot(time, diag.meanLambdaMinOmegaSigma, "LineWidth", 1.2);
    hold on;
    plot(time, diag.minLambdaMinOmegaSigma, "--", "LineWidth", 1.0);
    ylabel("\lambda_{min}(\Omega_\Sigma)");

    yyaxis right;
    plot(time, diag.medianCondOmegaSigma, "LineWidth", 1.2);
    plot(time, diag.maxCondOmegaSigma, "--", "LineWidth", 1.0);
    ylabel("cond(\Omega_\Sigma)");

    addStep09J5bWindowMarkers();
    grid on;
    xlabel("Time [s]");
    title("\Omega_\Sigma rank / conditioning");
    legend( ...
        "mean \lambda_{min}", ...
        "min \lambda_{min}", ...
        "median cond", ...
        "max cond", ...
        "Location", "best");

    % 3. Gate norms.
    nexttile;
    plot(time, diag.meanLocalBnorm, "LineWidth", 1.2);
    hold on;
    plot(time, diag.meanNonlocalBnorm, "LineWidth", 1.2);
    plot(time, diag.maxNonlocalBnorm, "--", "LineWidth", 1.0);
    addStep09J5bWindowMarkers();
    grid on;
    xlabel("Time [s]");
    ylabel("Frobenius norm");
    title("Gate norms: local vs nonlocal");
    legend( ...
        "mean ||B_{m|m}||_F", ...
        "mean ||B_{j|m}||_F, j\neqm", ...
        "max ||B_{j|m}||_F, j\neqm", ...
        "Location", "best");

    % 4. Gate algebra consistency.
    nexttile;
    semilogy(time, max(diag.meanGateIdentityError, eps), "LineWidth", 1.2);
    hold on;
    semilogy(time, max(diag.maxGateIdentityError, eps), "--", "LineWidth", 1.0);
    addStep09J5bWindowMarkers();
    grid on;
    xlabel("Time [s]");
    ylabel("||sum_j B_j - I||_F");
    title("Gate identity consistency");
    legend("mean over watchers", "max over watchers", "Location", "best");

    sgtitle(sprintf( ...
        "Step 09-J.5b FIM gate diagnostics, residual=%s, seed=%d, dt=%.3g", ...
        string(out09j5.residualFamily), out09j5.seed, out09j5.dt));

end

function addStep09J5bWindowMarkers()
%ADDSTEP09J5BWINDOWMARKERS Add diagnostic window boundaries.

    windowEdges = [800, 900, 1200, 1400, 1500];

    yl = ylim;

    for k = 1:numel(windowEdges)
        xline(windowEdges(k), ":", "LineWidth", 0.9);
    end

    ylim(yl);

end