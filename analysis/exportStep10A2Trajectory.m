function trajectory = exportStep10A2Trajectory(out,sourceCase,fileName)
%EXPORTSTEP10A2TRAJECTORY Export a realized plant trajectory for EKF replay.
%
% trajectory = exportStep10A2Trajectory(out,"additive")
% trajectory = exportStep10A2Trajectory(out,"FIM", "results/obsReplay.mat")
%
% The exported object freezes target truth and watcher kinematics.  It does
% not freeze EKF states or DNN parameters: those are intentionally re-run.

    if nargin < 2 || isempty(sourceCase)
        sourceCase = "additive";
    end
    if nargin < 3
        fileName = "";
    end

    sourceCase = lower(string(sourceCase));
    switch sourceCase
        case {"additive","add"}
            res = out.resGSAdd;
            cfg = out.cfgGSAdd;
            sourceLabel = "additive";
        case {"fim","fim_weighted_additive","fim-weighted-additive"}
            res = out.resGSFIM;
            cfg = out.cfgGSFIM;
            sourceLabel = "FIM-weighted-additive";
        otherwise
            error("exportStep10A2Trajectory:UnknownSourceCase", ...
                "sourceCase must be additive or FIM.");
    end

    required = {"time","etaTrue","watcherR","watcherV","watcherU"};
    for i = 1:numel(required)
        assert(isfield(res,required{i}), ...
            "Replay export requires res.%s.",required{i});
    end

    trajectory = struct();
    trajectory.schema = "step10A2_trajectory_v1";
    trajectory.sourceCase = sourceLabel;
    trajectory.seed = out.seed;
    trajectory.time = res.time;
    trajectory.etaTrue = res.etaTrue;
    trajectory.watcherR = res.watcherR;
    trajectory.watcherV = res.watcherV;
    trajectory.watcherU = res.watcherU;
    trajectory.cfg = cfg;

    % Preserve controller diagnostics for trajectory-level analysis.  The
    % replay filter never uses these values to choose a new maneuver.
    controllerFields = {"selectedDirection","selectedCandidateIndex", ...
        "selectedScore","candidateScores","candidateInformationMinEig", ...
        "candidateInformationCondition","selectedInformationMinEig", ...
        "selectedInformationCondition","replanFlag","controllerActive", ...
        "cumulativeImpulse","cumulativeDeltaV","watcherPathLength", ...
        "watcherDisplacement","referenceR","referenceV", ...
        "actualLOSChange","actualLOSChangeSigma", ...
        "actualLOSChangeOverSigma","losAngle","predictedRadialVariance"};
    trajectory.controller = struct();
    for i = 1:numel(controllerFields)
        field = controllerFields{i};
        if isfield(res,field)
            trajectory.controller.(field) = res.(field);
        end
    end

    if strlength(string(fileName)) > 0
        fileName = char(fileName);
        [folder,~,~] = fileparts(fileName);
        if ~isempty(folder) && ~isfolder(folder)
            mkdir(folder);
        end
        save(fileName,"trajectory","-v7.3");
    end
end
