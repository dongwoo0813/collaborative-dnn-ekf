function result = run_step10B1_gs_p2p_ablation( ...
    seed,desiredBaseline,simulationTime,maneuverMode,makePlots)
%RUN_STEP10B1_GS_P2P_ABLATION Compare GS and ring-P2P communication.
%
% The target truth, observability-seeking maneuver, initial DNN state, and
% bearing-noise realization are held fixed by the seed.  The four cases are
% continuous upload (after every valid measurement update) and contribution-
% change event-triggered upload, each with either a GS repository or a
% two-neighbor ring peer cache.

if nargin < 1 || isempty(seed), seed = 101; end
if nargin < 2 || isempty(desiredBaseline), desiredBaseline = 100; end
if nargin < 3 || isempty(simulationTime), simulationTime = 600; end
if nargin < 4 || isempty(maneuverMode), maneuverMode = "observability_seeking"; end
if nargin < 5 || isempty(makePlots), makePlots = false; end

architectures = ["gs","gs","p2p_ring","p2p_ring"];
uploadModes = ["after_measurement_update","event_contribution_change", ...
    "after_measurement_update","event_contribution_change"];
caseNames = ["GS-continuous","GS-event","P2P-continuous","P2P-event"];

runs = cell(numel(caseNames),1);
rows = cell(numel(caseNames),1);
for ic = 1:numel(caseNames)
    fprintf("\nStep 10-B1: %s\n",caseNames(ic));
    [out,diagOut,figures] = run_step09J6_transverse_additive_vs_fim( ...
        makePlots,desiredBaseline,simulationTime,maneuverMode,seed, ...
        architectures(ic),uploadModes(ic),"fim");
    runs{ic} = struct("caseName",caseNames(ic),"out",out, ...
        "diag",diagOut,"fig",figures, ...
        "architecture",architectures(ic),"uploadMode",uploadModes(ic));
    s = out.performanceSummary;
    row = s(s.caseName == "FIM-weighted-additive",:);
    row.caseName = caseNames(ic);
    rows{ic} = row;
end

summary = vertcat(rows{:});
result = struct();
result.seed = seed;
result.desiredBaseline = desiredBaseline;
result.simulationTime = simulationTime;
result.maneuverMode = string(maneuverMode);
result.runs = runs;
result.summary = summary;

fprintf("\nStep 10-B1 additive communication comparison\n");
disp(summary(:,{'caseName','positionRMSE','finalPositionRMSE', ...
    'exactRangeRMSE','finalExactRangeRMSE','velocityRMSE', ...
    'finalVelocityRMSE','residualVectorRMSE', ...
    'finalResidualVectorRMSE','meanNIS'}));
end
