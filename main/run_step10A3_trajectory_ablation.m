function ablation = run_step10A3_trajectory_ablation(makePlots,desiredBaseline,simulationTime,seedOverride)
%RUN_STEP10A3_TRAJECTORY_ABLATION Separate maneuver and estimator effects.
%
% Generates coast, transverse, and observability-seeking source
% trajectories.  Each source trajectory is then replayed with both the
% additive and FIM-weighted-additive estimators.

    if nargin < 1, makePlots = false; end
    if nargin < 2, desiredBaseline = 100; end
    if nargin < 3, simulationTime = 600; end
    if nargin < 4, seedOverride = []; end
    modes = ["coast","transverse","observability_seeking"];
    labels = ["coast","transverse","observability-seeking"];
    ablation = struct();
    allRows = table();

    for im = 1:numel(modes)
        fprintf("\nStep 10-A.3 source trajectory: %s\n",labels(im));
        [out,diagOut,figOut] = run_step09J6_transverse_additive_vs_fim( ...
            false,desiredBaseline,simulationTime,modes(im),seedOverride);
        traj = exportStep10A2Trajectory(out,"additive");
        replay = run_step10A2_deterministic_replay(traj,makePlots);

        source = struct();
        source.mode = modes(im);
        source.label = labels(im);
        source.out = out;
        source.diag = diagOut;
        source.figures = figOut;
        source.trajectory = traj;
        source.replay = replay;
        source.controllerSummary = out.controllerSummaryAdd;
        ablation.(matlab.lang.makeValidName(labels(im))) = source;

        rows = replay.summary;
        rows.seed = repmat(out.seed,height(rows),1);
        rows.trajectory = repmat(labels(im),height(rows),1);
        cs = source.controllerSummary;
        rows.meanFinalBaseline = repmat(mean(cs.actualFinalBaseline,"omitnan"),height(rows),1);
        rows.meanPathLength = repmat(mean(cs.actualPathLength,"omitnan"),height(rows),1);
        rows.meanDeltaV = repmat(mean(cs.finalDeltaV,"omitnan"),height(rows),1);
        rows.meanLOSChangeOverSigma = repmat(mean(cs.meanLOSChangeOverSigma,"omitnan"),height(rows),1);
        rows.meanSelectedInformationMinEig = repmat(mean(cs.meanSelectedInformationMinEig,"omitnan"),height(rows),1);
        rows.meanCandidateScoreMargin = repmat(mean(cs.meanCandidateScoreMargin,"omitnan"),height(rows),1);
        rows.meanReplans = repmat(mean(cs.numReplans,"omitnan"),height(rows),1);
        rows.meanDirectionSwitches = repmat(mean(cs.numDirectionSwitches,"omitnan"),height(rows),1);
        rows = movevars(rows,"trajectory","Before",1);
        if isempty(allRows)
            allRows = rows;
        else
            allRows = [allRows; rows]; %#ok<AGROW>
        end
    end

    ablation.summary = allRows;
    fprintf("\nStep 10-A.3 trajectory ablation summary\n");
    disp(allRows);

    if makePlots
        figure("Name","Step 10-A.3 trajectory ablation");
        cats = categorical(allRows.trajectory + " / " + allRows.caseName);
        tiledlayout(1,2);
        nexttile; bar(cats,allRows.finalPositionRMSE); grid on;
        ylabel("final position RMSE [m]"); xtickangle(35);
        nexttile; bar(cats,allRows.finalResidualVectorRMSE); grid on;
        ylabel("final residual RMSE [m/s^2]"); xtickangle(35);
    end
end
