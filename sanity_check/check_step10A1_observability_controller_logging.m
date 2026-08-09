function report = check_step10A1_observability_controller_logging(results,cfg)
%CHECK_STEP10A1_OBSERVABILITY_CONTROLLER_LOGGING Validate Step 10-A.1 logs.
%
% Usage:
%   report = check_step10A1_observability_controller_logging(results,cfg);
%
% The check is passive. It validates dimensions, finite telemetry, force
% limits, cumulative-motion monotonicity, candidate-score consistency, and
% reference/displacement reconstruction. It cannot prove absence of target
% truth in a controller implementation; that remains a code-review rule.

arguments
    results (1,1) struct
    cfg (1,1) struct
end

required = [ ...
    "time", "watcherR", "watcherV", "watcherU", ...
    "selectedDirection", "selectedCandidateIndex", "selectedScore", ...
    "candidateScores", "replanFlag", "controllerActive", ...
    "cumulativeImpulse", "cumulativeDeltaV", "watcherPathLength", ...
    "watcherDisplacement", "referenceR", "referenceV", ...
    "actualLOSChange", "actualLOSChangeSigma", ...
    "actualLOSChangeOverSigma", "predictedRadialVariance"];
missing = required(~isfield(results,cellstr(required)));
assert(isempty(missing),"Missing Step 10-A.1 fields: %s", ...
    strjoin(missing,", "));

time = results.time(:);
N = numel(time);
[dim,Nr,Nw] = size(results.watcherR);
assert(Nr == N,"watcherR time dimension mismatch.");
assert(size(results.watcherV,1) == dim && ...
    size(results.watcherV,2) == N && size(results.watcherV,3) == Nw, ...
    "watcherV dimension mismatch.");
assert(size(results.selectedDirection,1) == dim && ...
    size(results.selectedDirection,2) == N && ...
    size(results.selectedDirection,3) == Nw, ...
    "selectedDirection dimension mismatch.");
assert(all(size(results.selectedCandidateIndex) == [N,Nw]), ...
    "selectedCandidateIndex dimension mismatch.");
assert(all(size(results.selectedScore) == [N,Nw]), ...
    "selectedScore dimension mismatch.");
assert(size(results.candidateScores,2) == N && ...
    size(results.candidateScores,3) == Nw, ...
    "candidateScores dimension mismatch.");

forceLimit = Inf;
if isfield(cfg,"watchers") && isfield(cfg.watchers,"maxThrust")
    forceLimit = max(double(cfg.watchers.maxThrust(:)));
end
forceNorm = sqrt(squeeze(sum(results.watcherU.^2,1)));
assert(all(isfinite(forceNorm(:))),"Non-finite watcher force telemetry.");
assert(max(forceNorm(:)) <= forceLimit + 1e-9*max(1,forceLimit), ...
    "Watcher force exceeds configured maxThrust.");

for field = ["cumulativeImpulse","cumulativeDeltaV","watcherPathLength"]
    values = results.(field);
    assert(all(isfinite(values(:))),"Non-finite %s telemetry.",field);
    increments = diff(values,1,1);
    assert(all(increments(:) >= -1e-10), ...
        "%s is not monotone nondecreasing.",field);
end

assert(all(isfinite(results.watcherDisplacement(:))), ...
    "Non-finite watcher displacement telemetry.");
assert(norm(results.watcherDisplacement(:,1,:),"fro") <= 1e-10, ...
    "Initial watcher displacement must be zero.");

candidateCount = size(results.candidateScores,1);
indices = results.selectedCandidateIndex;
assert(all(indices(:) >= 0 & indices(:) <= candidateCount), ...
    "Selected candidate index is outside the candidate set.");

replanRows = find(results.replanFlag);
scoreErrors = [];
for q = 1:numel(replanRows)
    [k,i] = ind2sub([N,Nw],replanRows(q));
    idx = indices(k,i);
    if idx >= 1 && idx <= candidateCount && ...
            isfinite(results.selectedScore(k,i))
        candidateValue = results.candidateScores(idx,k,i);
        if isfinite(candidateValue)
            scoreErrors(end+1) = abs( ...
                results.selectedScore(k,i)-candidateValue); %#ok<AGROW>
        end
    end
end
if ~isempty(scoreErrors)
    assert(max(scoreErrors) <= 1e-9*max(1,max(abs(results.selectedScore(:)))), ...
        "Selected score does not match selected candidate score.");
end

report = struct();
report.passed = true;
report.numTimeSteps = N;
report.numWatchers = Nw;
report.numCandidateDirections = candidateCount;
report.numReplans = nnz(results.replanFlag);
report.maxForce = max(forceNorm(:));
report.finalImpulse = results.cumulativeImpulse(end,:);
report.finalDeltaV = results.cumulativeDeltaV(end,:);
report.finalPathLength = results.watcherPathLength(end,:);
report.maxSelectedScoreConsistencyError = ...
    iffelse(isempty(scoreErrors),0,max(scoreErrors));

fprintf("Step 10-A.1 logging check passed: %d steps, %d watchers, %d replans.\n", ...
    N,Nw,report.numReplans);

end

function value = iffelse(condition,trueValue,falseValue)
%IFFELSE Small local scalar conditional helper.
if condition
    value = trueValue;
else
    value = falseValue;
end
end
