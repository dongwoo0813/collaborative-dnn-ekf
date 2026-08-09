%{
check_step09I2_mlp_gs_repository_metadata.m

Purpose:
    Targeted check for Step 09-I.2.

    Verify that the GS repository, upload packet, and watcher-side broadcast
    cache are compatible with cfg.dnn.branchModel = "mlp_general".

What this checks:
    1. initGSRepository uses branchThetaNumel(cfg), not fixed-feature-only
       cfg.dnn.nThetaPerBranch.
    2. GS branch records store nTheta = 256 for the default MLP branch.
    3. uploadLocalBranchToGS stores MLP metadata.
    4. broadcastGSRepositoryToWatcher preserves MLP metadata in
       watcher.gsBranches.
%}

addpath(genpath(pwd));
rehash;

fprintf("\n");
fprintf("============================================================\n");
fprintf("Step 09-I.2 check: MLP GS repository metadata\n");
fprintf("============================================================\n");

cfg = config_step04_GS_DNN_EKF();

cfg.dim = 2;
cfg.Nw = 4;

cfg.dnn.branchModel = "mlp_general";
cfg.dnn.mlp.hiddenSizes = [12 8 6];
cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
cfg.dnn.mlp.inputMode = "eta_phase";
cfg.dnn.mlp.rScale = 1000.0;
cfg.dnn.mlp.vScale = 0.1;

cfg.dnn.Ptheta0 = (7.5e-5)^2;
cfg.dnn.thetaSigmaSS = 7.5e-5;

[nThetaExpected, branchInfoExpected] = branchThetaNumel(cfg);

fprintf("Expected branchModel = %s\n", string(branchInfoExpected.branchModel));
fprintf("Expected nTheta      = %d\n", nThetaExpected);
fprintf("Expected layerSizes  = [%s]\n", ...
    sprintf("%d ", branchInfoExpected.arch.layerSizes));

eta0 = [500.0; -250.0; 0.030; -0.020];

watcher1 = initLocalDNNEKF(1, eta0, cfg);
watcher2 = initLocalDNNEKF(2, eta0, cfg);

gsRepo = initGSRepository(cfg);

if gsRepo.nThetaPerBranch ~= nThetaExpected
    error("Step09I2:BadRepoNTheta", ...
        "GS repo nThetaPerBranch mismatch.");
end

if string(gsRepo.branchModel) ~= "mlp_general"
    error("Step09I2:BadRepoBranchModel", ...
        "GS repo branchModel mismatch.");
end

if ~isfield(gsRepo.branch(1), "branchInfo")
    error("Step09I2:MissingRepoBranchInfo", ...
        "GS repo branch record is missing branchInfo.");
end

% Upload watcher 1 to GS.
[gsRepo, uploadPacket] = uploadLocalBranchToGS(gsRepo, watcher1, 0.0, cfg);

if string(uploadPacket.branchModel) ~= "mlp_general"
    error("Step09I2:BadUploadBranchModel", ...
        "Upload packet branchModel mismatch.");
end

if uploadPacket.nTheta ~= nThetaExpected
    error("Step09I2:BadUploadNTheta", ...
        "Upload packet nTheta mismatch.");
end

if string(gsRepo.branch(1).branchModel) ~= "mlp_general"
    error("Step09I2:BadStoredBranchModel", ...
        "Stored GS branchModel mismatch.");
end

if gsRepo.branch(1).nTheta ~= nThetaExpected
    error("Step09I2:BadStoredNTheta", ...
        "Stored GS nTheta mismatch.");
end

% Broadcast watcher 1's branch to watcher 2.
[watcher2, broadcastPacket] = broadcastGSRepositoryToWatcher( ...
    gsRepo, watcher2, 0.0, cfg);

if ~watcher2.gsBranches(1).active
    error("Step09I2:ExpectedActiveNonlocal", ...
        "Watcher 2 should have active nonlocal branch 1.");
end

if string(watcher2.gsBranches(1).branchModel) ~= "mlp_general"
    error("Step09I2:BadCacheBranchModel", ...
        "Watcher-side cache branchModel mismatch.");
end

if watcher2.gsBranches(1).nTheta ~= nThetaExpected
    error("Step09I2:BadCacheNTheta", ...
        "Watcher-side cache nTheta mismatch.");
end

if ~isfield(watcher2.gsBranches(1).branchInfo, "arch")
    error("Step09I2:MissingCacheArch", ...
        "Watcher-side cache is missing MLP architecture info.");
end

if ~isequal( ...
        watcher2.gsBranches(1).branchInfo.arch.layerSizes, ...
        branchInfoExpected.arch.layerSizes)
    error("Step09I2:BadCacheLayerSizes", ...
        "Watcher-side cache MLP layerSizes mismatch.");
end

fprintf("GS repo nTheta              = %d\n", gsRepo.nThetaPerBranch);
fprintf("Upload packet nTheta        = %d\n", uploadPacket.nTheta);
fprintf("Watcher cache branchModel   = %s\n", ...
    string(watcher2.gsBranches(1).branchModel));
fprintf("Watcher cache layerSizes    = [%s]\n", ...
    sprintf("%d ", watcher2.gsBranches(1).branchInfo.arch.layerSizes));
fprintf("Broadcast included branches = %s\n", ...
    mat2str(broadcastPacket.includedBranchIDs));

fprintf("\nStep 09-I.2 MLP GS repository metadata check passed.\n");