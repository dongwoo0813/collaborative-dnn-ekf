function out = debug_step04d_nonlocal_covariance_sources(results, cfg)
%{
Function:
    debug_step04d_nonlocal_covariance_sources

Purpose:
    Diagnose why the Step 04b nonlocal GS-branch covariance injection

        Q_X,-m = M_m * S_d,-m * M_m'

    is numerically small.

    This function decomposes the final nonlocal covariance injection into:

        1. GS branch parameter covariance Ptheta_j
        2. branch-output Jacobian B_j
        3. branch output covariance S_j = B_j Ptheta_j B_j'
        4. Young coefficient a_j
        5. contribution a_j S_j
        6. residual covariance SresNonlocal
        7. discrete-time acceleration-to-state mapping M_m

Inputs:
    results - Output from simulate_GS_DNN_EKF, or out.resGSQ from
              run_step04d_compare_GS_Qnonlocal.

    cfg     - Configuration used for the GS + Qnonlocal run.
              Example:
                  out = run_step04d_compare_GS_Qnonlocal;
                  dbg = debug_step04d_nonlocal_covariance_sources( ...
                            out.resGSQ, out.cfgGSQ);

Outputs:
    out - Struct with:
            out.watcherTable
            out.branchTable
            out.meanTraceSdTotal
            out.meanTraceQFormula
            out.meanTraceQStored

Notes:
    This is a diagnostic helper only. It does not change the filter.

    In the final method, do not use an arbitrary global scale on Qnonlocal.
    The magnitude should come from Ptheta_j, covariance aging/staleness,
    acceptance margins, SresNonlocal, Young coefficients, and M_m.
%}

    if nargin < 2
        error("Usage: out = debug_step04d_nonlocal_covariance_sources(results, cfg)");
    end

    if ~isfield(results, "watchersFinal") || isempty(results.watchersFinal)
        error("results.watchersFinal is missing.");
    end

    if ~isfield(results, "etaTrue") || isempty(results.etaTrue)
        error("results.etaTrue is missing. Need final eta for recomputation.");
    end

    dim = cfg.dim;
    Nw  = cfg.Nw;
    dt  = cfg.dt;

    watchers = results.watchersFinal;
    etaFinalTrue = results.etaTrue(:, end);

    % For double-integrator mapping:
    % M = [0.5*dt^2*I; dt*I; 0],
    % so trace(M*S*M') = (dt^2 + 0.25*dt^4)*trace(S).
    mappingTraceFactor = dt^2 + 0.25 * dt^4;

    % ---------------------------------------------------------------------
    % Watcher-level table data
    % One row = one watcher m.
    % ---------------------------------------------------------------------

    watcherIDVec = [];              % watcher index m
    numActiveVec = [];              % number of nonlocal branches used by m

    traceSdBranchVec = [];          % trace(sum_j a_j*S_j), branch-only Sd
    traceSresVec = [];              % trace(SresNonlocal)
    traceSdTotalVec = [];           % trace(SdFromBranches + Sres)

    traceQFormulaVec = [];          % recomputed trace(Q_X,-m)
    traceQStoredVec = [];           % stored trace(Qnonlocal) from EKF predict
    relTraceQDiffVec = [];          % formula-vs-stored mismatch check

    meanTracePthetaVec = [];        % mean trace(Ptheta_j) over branches j
    meanTraceSjVec = [];            % mean trace(B_j*Ptheta_j*B_j') over j


    % ---------------------------------------------------------------------
    % Branch-level table data
    % One row = one pair (watcher m, nonlocal branch j).
    % ---------------------------------------------------------------------

    branchWatcherIDVec = [];        % watcher m using branch j
    branchIDVec = [];               % nonlocal branch j

    branchAgeVec = [];              % cached branch age, if available
    branchVersionVec = [];          % cached branch version, if available

    tracePthetaVec = [];            % trace(Ptheta_j)
    meanDiagPthetaVec = [];         % average diag(Ptheta_j)
    maxDiagPthetaVec = [];          % max diag(Ptheta_j)

    normBFroVec = [];               % ||B_j||_F
    traceBBtVec = [];               % trace(B_j*B_j')

    traceSjVec = [];                % trace(S_j), S_j = B_j*Ptheta_j*B_j'
    youngCoeffVec = [];             % Young coefficient a_j
    traceAjSjVec = [];              % trace(a_j*S_j)

    contributionFracVec = [];       % trace(a_j*S_j)/trace(sum_l a_l*S_l)
    thetaNormVec = [];              % ||theta_j||

     for m = 1:numel(watchers)

        watcher = watchers(m);

        % Local branch ID. Normally watcher m owns branch m.
        if isfield(watcher, "localBranchID")
            localBranchID = watcher.localBranchID;
        else
            localBranchID = m;
        end

        % Evaluate branch Jacobians at the final estimated physical state.
        % Fallback to true final eta only if xhat/idxEta is missing.
        if isfield(watcher, "xhat") && isfield(watcher, "idxEta")
            etaEval = watcher.xhat(watcher.idxEta);
        else
            etaEval = etaFinalTrue;
        end

        % Nonlocal GS branches actually used by watcher m.
        activeBranchIDs = getActiveNonlocalBranchIDs(watcher, localBranchID, Nw);
        Nnonlocal = numel(activeBranchIDs);

        % Uniform Young bound:
        % if there are Nnonlocal branches, then a_j = Nnonlocal.
        youngCoefficients = zeros(Nw, 1);
        if Nnonlocal > 0
            youngCoefficients(activeBranchIDs) = Nnonlocal;
        end

        % Accumulates sum_j a_j*S_j for watcher m.
        SdFromBranches = zeros(dim, dim);

        % Temporary watcher-local branch summaries.
        localTracePtheta = NaN(Nnonlocal, 1);
        localTraceSj = NaN(Nnonlocal, 1);
        localTraceAjSj = NaN(Nnonlocal, 1);

        for idx = 1:Nnonlocal

            j = activeBranchIDs(idx);
            branchRecord = watcher.gsBranches(j);

            % GS-stored covariance for nonlocal branch j.
            Ptheta_j = branchRecord.Ptheta;
            Ptheta_j = 0.5 * (Ptheta_j + Ptheta_j.');

            % Branch-output Jacobian:
            % B_j = partial d_j(eta;theta_j)/partial theta_j.
            B_j = branchJacobianTheta(j, etaEval, cfg);

            % Output covariance induced by parameter uncertainty:
            % S_j = B_j * Ptheta_j * B_j'.
            Sj = B_j * Ptheta_j * B_j.';
            Sj = 0.5 * (Sj + Sj.');

            % Young-inflated branch contribution.
            aj = youngCoefficients(j);
            AjSj = aj * Sj;

            % Add branch j to watcher m's nonlocal Sd.
            SdFromBranches = SdFromBranches + AjSj;

            % Watcher-level averages over active branches.
            localTracePtheta(idx) = trace(Ptheta_j);
            localTraceSj(idx) = trace(Sj);
            localTraceAjSj(idx) = trace(AjSj);

            % Branch-level diagnostic row.
            branchWatcherIDVec(end+1, 1) = m; %#ok<AGROW>
            branchIDVec(end+1, 1) = j; %#ok<AGROW>
            branchAgeVec(end+1, 1) = getNumericField(branchRecord, "age", NaN); %#ok<AGROW>
            branchVersionVec(end+1, 1) = getNumericField(branchRecord, "version", NaN); %#ok<AGROW>

            % Parameter covariance magnitude.
            tracePthetaVec(end+1, 1) = trace(Ptheta_j); %#ok<AGROW>
            meanDiagPthetaVec(end+1, 1) = mean(diag(Ptheta_j), "omitnan"); %#ok<AGROW>
            maxDiagPthetaVec(end+1, 1) = max(diag(Ptheta_j)); %#ok<AGROW>

            % Branch sensitivity magnitude.
            normBFroVec(end+1, 1) = norm(B_j, "fro"); %#ok<AGROW>
            traceBBtVec(end+1, 1) = trace(B_j * B_j.'); %#ok<AGROW>

            % Output covariance and Young-inflated contribution.
            traceSjVec(end+1, 1) = trace(Sj); %#ok<AGROW>
            youngCoeffVec(end+1, 1) = aj; %#ok<AGROW>
            traceAjSjVec(end+1, 1) = trace(AjSj); %#ok<AGROW>

            % Branch parameter magnitude.
            thetaNormVec(end+1, 1) = norm(branchRecord.theta); %#ok<AGROW>
        end

        % Add optional residual/unmodeled nonlocal covariance.
        Sres = getSresNonlocal(cfg, dim);

        % Total residual-acceleration covariance surrogate:
        % SdTotal = sum_j a_j*S_j + Sres.
        SdTotal = SdFromBranches + Sres;
        SdTotal = 0.5 * (SdTotal + SdTotal.');

        % Recompute trace(Q_X,-m) using double-integrator trace formula.
        QtraceFormula = mappingTraceFactor * trace(SdTotal);

        % Read trace(Qnonlocal) that the actual EKF prediction stored.
        QtraceStored = NaN;
        if isfield(watcher, "lastNonlocalCovInjection")
            info = watcher.lastNonlocalCovInjection;
            if isfield(info, "traceQnonlocal")
                QtraceStored = info.traceQnonlocal;
            end
        end

        % Should be small if this diagnostic matches the actual EKF path.
        relDiff = abs(QtraceFormula - QtraceStored) / max(abs(QtraceStored), eps);

        % Watcher-level diagnostic row.
        watcherIDVec(end+1, 1) = m; %#ok<AGROW>
        numActiveVec(end+1, 1) = Nnonlocal; %#ok<AGROW>
        traceSdBranchVec(end+1, 1) = trace(SdFromBranches); %#ok<AGROW>
        traceSresVec(end+1, 1) = trace(Sres); %#ok<AGROW>
        traceSdTotalVec(end+1, 1) = trace(SdTotal); %#ok<AGROW>
        traceQFormulaVec(end+1, 1) = QtraceFormula; %#ok<AGROW>
        traceQStoredVec(end+1, 1) = QtraceStored; %#ok<AGROW>
        relTraceQDiffVec(end+1, 1) = relDiff; %#ok<AGROW>
        meanTracePthetaVec(end+1, 1) = mean(localTracePtheta, "omitnan"); %#ok<AGROW>
        meanTraceSjVec(end+1, 1) = mean(localTraceSj, "omitnan"); %#ok<AGROW>

        % Branch contribution ratio inside watcher m's SdFromBranches.
        if Nnonlocal > 0
            idxRows = branchWatcherIDVec == m;
            denom = trace(SdFromBranches);

            if abs(denom) > eps
                contributionFracVec(idxRows, 1) = traceAjSjVec(idxRows) / denom;
            else
                contributionFracVec(idxRows, 1) = NaN;
            end
        end

    end

    % Watcher-level summary table.
    % One row = one watcher m.
    watcherTable = table( ...
        watcherIDVec, ...
        numActiveVec, ...
        traceSdBranchVec, ...
        traceSresVec, ...
        traceSdTotalVec, ...
        traceQFormulaVec, ...
        traceQStoredVec, ...
        relTraceQDiffVec, ...
        meanTracePthetaVec, ...
        meanTraceSjVec, ...
        repmat(mappingTraceFactor, numel(watcherIDVec), 1), ...
        'VariableNames', { ...
            'watcherID', ...
            'numActiveNonlocal', ...
            'traceSdFromBranches', ...
            'traceSres', ...
            'traceSdTotal', ...
            'traceQFormula', ...
            'traceQStored', ...
            'relTraceQDiff', ...
            'meanTracePtheta', ...
            'meanTraceSj', ...
            'mappingTraceFactor'});


    % Branch-level summary table.
    % One row = one pair (watcher m, nonlocal branch j).
    branchTable = table( ...
        branchWatcherIDVec, ...
        branchIDVec, ...
        branchAgeVec, ...
        branchVersionVec, ...
        tracePthetaVec, ...
        meanDiagPthetaVec, ...
        maxDiagPthetaVec, ...
        normBFroVec, ...
        traceBBtVec, ...
        traceSjVec, ...
        youngCoeffVec, ...
        traceAjSjVec, ...
        contributionFracVec, ...
        thetaNormVec, ...
        'VariableNames', { ...
            'watcherID', ...
            'branchID', ...
            'age', ...
            'version', ...
            'tracePtheta', ...
            'meanDiagPtheta', ...
            'maxDiagPtheta', ...
            'normBFro', ...
            'traceBBt', ...
            'traceSj', ...
            'youngCoeff', ...
            'traceAjSj', ...
            'contributionFrac', ...
            'thetaNorm'});

    
    fprintf("\n============================================================\n");
    fprintf("Step 04d nonlocal covariance source diagnostics\n");
    fprintf("============================================================\n");

    fprintf("\nWatcher-level decomposition:\n");
    disp(watcherTable);

    fprintf("\nBranch-level decomposition:\n");
    disp(branchTable);

    fprintf("\nKey means:\n");
    fprintf("  mean trace(Ptheta_j)          = %.6e\n", mean(branchTable.tracePtheta, "omitnan"));
    fprintf("  mean trace(S_j)               = %.6e\n", mean(branchTable.traceSj, "omitnan"));
    fprintf("  mean trace(a_j S_j)           = %.6e\n", mean(branchTable.traceAjSj, "omitnan"));
    fprintf("  mean trace(SdTotal)           = %.6e\n", mean(watcherTable.traceSdTotal, "omitnan"));
    fprintf("  mean trace(Qnonlocal formula) = %.6e\n", mean(watcherTable.traceQFormula, "omitnan"));
    fprintf("  mean trace(Qnonlocal stored)  = %.6e\n", mean(watcherTable.traceQStored, "omitnan"));
    fprintf("  mapping trace factor          = %.6e\n", mappingTraceFactor);

    if isfield(cfg, "gs") && isfield(cfg.gs, "nonlocalCovarianceScale")
        fprintf("\nWARNING:\n");
        fprintf("  cfg.gs.nonlocalCovarianceScale exists and equals %.6e.\n", ...
            cfg.gs.nonlocalCovarianceScale);
        fprintf("  For the final method, remove this arbitrary scale or keep it fixed at 1.\n");
    end

    out = struct();
    out.watcherTable = watcherTable;
    out.branchTable = branchTable;
    out.meanTraceSdTotal = mean(watcherTable.traceSdTotal, "omitnan");
    out.meanTraceQFormula = mean(watcherTable.traceQFormula, "omitnan");
    out.meanTraceQStored = mean(watcherTable.traceQStored, "omitnan");
    out.mappingTraceFactor = mappingTraceFactor;

end

function activeBranchIDs = getActiveNonlocalBranchIDs(watcher, localBranchID, Nw)

    activeBranchIDs = [];

    if ~isfield(watcher, "gsBranches") || isempty(watcher.gsBranches)
        return;
    end

    for j = 1:min(numel(watcher.gsBranches), Nw)

        if j == localBranchID
            continue;
        end

        branchRecord = watcher.gsBranches(j);

        if isUsableNonlocalBranch(branchRecord)
            activeBranchIDs(end+1) = j; %#ok<AGROW>
        end

    end

end

function tf = isUsableNonlocalBranch(branchRecord)

    tf = false;

    if ~isstruct(branchRecord)
        return;
    end

    if ~isfield(branchRecord, "active") || ~logical(branchRecord.active)
        return;
    end

    if isfield(branchRecord, "usedInPrediction")
        if ~logical(branchRecord.usedInPrediction)
            return;
        end
    end

    if isfield(branchRecord, "status")
        if string(branchRecord.status) ~= "valid"
            return;
        end
    end

    if isfield(branchRecord, "isStale")
        if logical(branchRecord.isStale)
            return;
        end
    end

    requiredFields = ["theta", "Ptheta"];

    for q = 1:numel(requiredFields)
        if ~isfield(branchRecord, requiredFields(q))
            return;
        end
    end

    if isempty(branchRecord.Ptheta)
        return;
    end

    if any(~isfinite(branchRecord.Ptheta(:)))
        return;
    end

    if any(~isfinite(branchRecord.theta(:)))
        return;
    end

    tf = true;

end

function Sres = getSresNonlocal(cfg, dim)

    Sres = zeros(dim, dim);

    if ~isfield(cfg, "gs") || ~isfield(cfg.gs, "SresNonlocal")
        return;
    end

    val = cfg.gs.SresNonlocal;

    if isscalar(val)
        Sres = val * eye(dim);
    else
        Sres = val;
    end

    if any(size(Sres) ~= [dim, dim])
        error("cfg.gs.SresNonlocal must be scalar or %d-by-%d.", dim, dim);
    end

    Sres = 0.5 * (Sres + Sres.');

end

function val = getNumericField(s, fieldName, defaultVal)

    val = defaultVal;

    if isfield(s, fieldName)
        candidate = s.(fieldName);
        if isnumeric(candidate) && isscalar(candidate)
            val = candidate;
        end
    end

end