function summary = summarizeObservabilityController(results,cfg)
%SUMMARIZEOBSERVABILITYCONTROLLER Summarize Step 10-A.1 telemetry.
%
% The summary distinguishes requested/configured maneuver quantities from
% realized watcher motion. It is diagnostic only and does not alter filtering
% or controller decisions.

arguments
    results (1,1) struct
    cfg (1,1) struct
end

time = results.time(:);
N = numel(time);
dt = median(diff(time));
[dim,~,Nw] = size(results.watcherR);

forceNorm = sqrt(squeeze(sum(results.watcherU.^2,1)));
if isvector(forceNorm)
    forceNorm = reshape(forceNorm,N,Nw);
end

active = logical(results.controllerActive);
replan = logical(results.replanFlag);

summary = struct();
summary.time = time;
summary.numWatchers = Nw;
summary.totalThrustOnTime = sum(active,1)*dt;
summary.thrustOnTimeFromForce = sum(forceNorm > 1e-12,1)*dt;
summary.numReplans = sum(replan,1);
summary.numDirectionSwitches = zeros(1,Nw);
summary.actualFinalDisplacement = zeros(dim,Nw);
summary.actualFinalBaseline = zeros(1,Nw);
summary.actualMaximumBaseline = zeros(1,Nw);
summary.referenceFinalDisplacement = zeros(dim,Nw);
summary.referenceFinalBaseline = zeros(1,Nw);
summary.finalReferenceDeviation = zeros(dim,Nw);
summary.finalReferenceDeviationNorm = zeros(1,Nw);
summary.maximumReferenceDeviationNorm = zeros(1,Nw);
summary.actualPathLength = results.watcherPathLength(end,:);
summary.finalImpulse = results.cumulativeImpulse(end,:);
summary.finalDeltaV = results.cumulativeDeltaV(end,:);
summary.meanSelectedScore = NaN(1,Nw);
summary.finalSelectedScore = NaN(1,Nw);
summary.meanCandidateScoreMargin = NaN(1,Nw);
summary.meanLOSChange = NaN(1,Nw);
summary.finalLOSChange = NaN(1,Nw);
summary.meanLOSChangeOverSigma = NaN(1,Nw);
summary.finalLOSChangeOverSigma = NaN(1,Nw);
summary.meanSelectedInformationMinEig = NaN(1,Nw);
summary.finalSelectedInformationMinEig = NaN(1,Nw);
summary.meanSelectedInformationCondition = NaN(1,Nw);
summary.finalSelectedInformationCondition = NaN(1,Nw);
summary.meanPredictedRadialVariance = NaN(1,Nw);
summary.finalPredictedRadialVariance = NaN(1,Nw);
summary.informationPerDeltaV = NaN(1,Nw);

for i = 1:Nw
    displacement = results.watcherDisplacement(:,:,i);
    baseline = sqrt(sum(displacement.^2,1));
    summary.actualFinalDisplacement(:,i) = displacement(:,end);
    summary.actualFinalBaseline(i) = baseline(end);
    summary.actualMaximumBaseline(i) = max(baseline);

    referenceDisplacement = results.referenceR(:,:,i) - ...
        results.referenceR(:,1,i);
    referenceBaseline = sqrt(sum(referenceDisplacement.^2,1));
    referenceDeviation = results.watcherR(:,:,i) - ...
        results.referenceR(:,:,i);
    referenceDeviationNorm = sqrt(sum(referenceDeviation.^2,1));
    summary.referenceFinalDisplacement(:,i) = referenceDisplacement(:,end);
    summary.referenceFinalBaseline(i) = referenceBaseline(end);
    summary.finalReferenceDeviation(:,i) = referenceDeviation(:,end);
    summary.finalReferenceDeviationNorm(i) = referenceDeviationNorm(end);
    summary.maximumReferenceDeviationNorm(i) = max(referenceDeviationNorm);

    direction = results.selectedDirection(:,:,i);
    validDirection = sqrt(sum(direction.^2,1)) > 1e-12;
    directionChange = sqrt(sum(diff(direction,1,2).^2,1));
    summary.numDirectionSwitches(i) = nnz( ...
        directionChange > 1e-8 & validDirection(2:end) & validDirection(1:end-1));

    selected = results.selectedScore(:,i);
    selected = selected(isfinite(selected));
    if ~isempty(selected)
        summary.meanSelectedScore(i) = mean(selected);
        summary.finalSelectedScore(i) = selected(end);
    end

    candidate = results.candidateScores(:,:,i);
    margins = NaN(N,1);
    for k = 1:N
        values = candidate(:,k);
        values = values(isfinite(values));
        if numel(values) >= 2
            values = sort(values,"ascend");
            margins(k) = values(2)-values(1);
        end
    end
    margins = margins(isfinite(margins) & replan(:,i));
    if ~isempty(margins)
        summary.meanCandidateScoreMargin(i) = mean(margins);
    end

    summary.meanLOSChange(i) = meanFinite(results.actualLOSChange(:,i));
    summary.finalLOSChange(i) = lastFinite(results.actualLOSChange(:,i));
    summary.meanLOSChangeOverSigma(i) = ...
        meanFinite(results.actualLOSChangeOverSigma(:,i));
    summary.finalLOSChangeOverSigma(i) = ...
        lastFinite(results.actualLOSChangeOverSigma(:,i));
    summary.meanSelectedInformationMinEig(i) = ...
        meanFinite(results.selectedInformationMinEig(:,i));
    summary.finalSelectedInformationMinEig(i) = ...
        lastFinite(results.selectedInformationMinEig(:,i));
    summary.meanSelectedInformationCondition(i) = ...
        meanFinite(results.selectedInformationCondition(:,i));
    summary.finalSelectedInformationCondition(i) = ...
        lastFinite(results.selectedInformationCondition(:,i));
    summary.meanPredictedRadialVariance(i) = ...
        meanFinite(results.predictedRadialVariance(:,i));
    summary.finalPredictedRadialVariance(i) = ...
        lastFinite(results.predictedRadialVariance(:,i));

    if summary.finalDeltaV(i) > eps
        summary.informationPerDeltaV(i) = ...
            summary.finalSelectedInformationMinEig(i)/summary.finalDeltaV(i);
    end
end

summary.maxForce = max(forceNorm,[],1);
summary.configuredMaxThrust = NaN;
if isfield(cfg,"watchers") && isfield(cfg.watchers,"maxThrust")
    summary.configuredMaxThrust = max(double(cfg.watchers.maxThrust(:)));
end

end

function value = meanFinite(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = mean(values);
end
end

function value = lastFinite(values)
idx = find(isfinite(values),1,"last");
if isempty(idx)
    value = NaN;
else
    value = values(idx);
end
end
