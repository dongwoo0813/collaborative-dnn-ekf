function artifact = save_step10A4_artifacts(result,ansValue,repo,fileName)
%SAVE_STEP10A4_ARTIFACTS Save a completed A4 run and reproducibility data.
%
% Usage from the MATLAB command window:
%   repo = "C:\\...\\Collaborative_DNN_EKF_Simulation_Code";
%   saveInfo = save_step10A4_artifacts(result,ans,repo, ...
%       "results/step10A4_multiseed_101_202_303_T600.mat");
%
% The MAT file contains:
%   result      - complete multi-seed A4 output structure
%   ansSnapshot - the command-window ans table at save time
%   repo        - repository/workspace path
%   metadata    - timestamp and MATLAB release information

    if nargin < 2
        ansValue = [];
    end
    if nargin < 3 || isempty(repo)
        repo = pwd;
    end
    if nargin < 4 || isempty(fileName)
        fileName = fullfile(repo,"results", ...
            "step10A4_multiseed_artifacts.mat");
    end

    if ~(isstruct(result) && isfield(result,"meanTable") && ...
            isfield(result,"stdTable"))
        error("save_step10A4_artifacts:InvalidResult", ...
            "result must be the completed run_step10A4 output.");
    end

    repo = string(repo);
    fileName = char(string(fileName));
    [folder,~,~] = fileparts(fileName);
    if ~isempty(folder) && ~isfolder(folder)
        mkdir(folder);
    end

    metadata = struct();
    metadata.schema = "step10A4_artifacts_v1";
    metadata.savedAt = string(datetime("now","Format", ...
        "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"));
    metadata.matlabVersion = version;
    metadata.computer = computer;
    metadata.repo = repo;
    metadata.seeds = result.seeds;
    metadata.numRows = height(result.allRows);

    ansSnapshot = ansValue;
    save(fileName,"result","ansSnapshot","repo","metadata","-v7.3");

    artifact = struct();
    artifact.fileName = string(fileName);
    artifact.bytes = dir(fileName).bytes;
    artifact.metadata = metadata;
    fprintf("Saved Step 10-A.4 artifacts:\n  %s\n",fileName);
    fprintf("  size = %.2f MB\n",artifact.bytes/1024^2);
end
