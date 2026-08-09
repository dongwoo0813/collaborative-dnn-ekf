function out = run_step10C3_GS_maneuver_duration_ablation( ...
    seed,simulationTime,makePlots,saveArtifacts,timeStep)
%RUN_STEP10C3_GS_MANEUVER_DURATION_ABLATION Coast/short/long comparison.
% Each schedule runs local, GS additive, and GS output-information fusion
% independently with identical initial conditions and bearing-noise seed.

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(simulationTime), simulationTime = 1800; end
    if nargin < 3 || isempty(makePlots), makePlots = false; end
    if nargin < 4 || isempty(saveArtifacts), saveArtifacts = true; end
    if nargin < 5 || isempty(timeStep), timeStep = 0.5; end
    addpath(genpath(pwd));

    common = struct("cooldownDuration",180,"planningHorizon",240, ...
        "rolloutMaxSteps",24);
    short = common; short.impulseDuration = 10;
    long = common; long.impulseDuration = 60;

    fprintf("Step 10-C.3: maneuver-duration ablation (coast / 10 s / 60 s)\n");
    out.coast = run_step10C2_GS_impulse_closed_loop_output_fusion( ...
        seed,simulationTime,false,false,timeStep,common, ...
        struct("translationMode","none"));
    out.short = run_step10C2_GS_impulse_closed_loop_output_fusion( ...
        seed,simulationTime,false,false,timeStep,short);
    out.long = run_step10C2_GS_impulse_closed_loop_output_fusion( ...
        seed,simulationTime,false,false,timeStep,long);

    schedules = ["Coast only";"10 s impulse";"60 s impulse"];
    summaries = {out.coast.summary,out.short.summary,out.long.summary};
    ablation = table();
    for i = 1:numel(summaries)
        s = summaries{i};
        s.schedule = repmat(schedules(i),height(s),1);
        s = movevars(s,"schedule","Before","caseName");
        ablation = [ablation;s]; %#ok<AGROW>
    end
    out.summary = ablation;
    displayNames = ["schedule","caseName","positionRMSE", ...
        "finalPositionRMSE","meanFinalDeltaV","radialNEES"];
    disp(ablation(:,displayNames));

    if saveArtifacts
        if ~isfolder("results"), mkdir("results"); end
        fileName = fullfile("results",sprintf( ...
            "step10C3_GS_maneuver_duration_ablation_seed%d_T%d.mat", ...
            seed,round(simulationTime)));
        save(fileName,"ablation","-v7.3");
        out.artifactFile = string(fileName);
        fprintf("Saved maneuver-duration ablation: %s\n",fileName);
    else
        out.artifactFile = "";
    end

    if makePlots
        out.figures = plot_step10C3_GS_maneuver_duration_ablation(out,true);
    end
end
