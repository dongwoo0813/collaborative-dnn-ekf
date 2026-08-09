function check_step09J2_gs_omega_bar_metadata()
%{
File:
    sanity_check/check_step09J2_gs_omega_bar_metadata.m

Purpose:
    Targeted Step 09-J.2 check for carrying OmegaBar bearing-geometry
    metadata through the GS repository and watcher-side broadcast cache.

Checks:
    1. A local watcher with an updated OmegaBar uploads OmegaBar to GS.
    2. GS record stores OmegaBar / numOmegaUpdates / lastLOSUnit / outputFrame.
    3. GS broadcast copies those fields into a recipient watcher's
       nonlocal gsBranches cache.
    4. OmegaBar remains symmetric PSD and has the expected dimensions.
    5. The local branch placeholder is still inactive, so this patch only
       adds metadata and does not overwrite the recipient's own branch.

Notes:
    This is not a full simulation regression. It only checks metadata plumbing
    needed before implementing compositeMode = "bearing_fim_gated".
%}

    addpath(genpath(pwd));
    rehash;

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-J.2 check: GS OmegaBar metadata payload\n");
    fprintf("============================================================\n");

    cfg = config_step04_GS_DNN_EKF();
    cfg.dim = 2;
    cfg.Nw = 4;
    cfg.meas.type = "bearing";

    % Use the current main MLP branch configuration so the check also makes
    % sure OmegaBar metadata does not interfere with MLP GS metadata.
    cfg.dnn.branchModel = "mlp_general";
    cfg.dnn.mlp.hiddenSizes = [12 8 6];
    cfg.dnn.mlp.activations = ["softplus", "tanh", "tanh"];
    cfg.dnn.mlp.inputMode = "eta_phase";
    cfg.dnn.mlp.rScale = 1000.0;
    cfg.dnn.mlp.vScale = 0.1;
    cfg.dnn.Ptheta0 = (7.5e-5)^2;
    cfg.dnn.thetaSigmaSS = 7.5e-5;

    eta0 = [500.0; -250.0; 0.030; -0.020];

    watcher1 = initLocalDNNEKF(1, eta0, cfg);
    watcher2 = initLocalDNNEKF(2, eta0, cfg);

    % Give watcher 1 two valid bearing updates so OmegaBar is not zero. These
    % z values are bearing measurements, not true residual labels.
    z1 = 0.35;
    z2 = 1.10;

    [watcher1, info1] = updateWatcherOmegaBarFromMeasurement(watcher1, z1, 0.5, cfg);
    [watcher1, info2] = updateWatcherOmegaBarFromMeasurement(watcher1, z2, 1.0, cfg);

    assert(info1.updated && info2.updated, ...
        "OmegaBar update helper did not report valid updates.");

    OmegaLocal = watcher1.OmegaBar;
    uLocal = watcher1.lastLOSUnit;

    gsRepo = initGSRepository(cfg);

    [gsRepo, uploadPacket] = uploadLocalBranchToGS(gsRepo, watcher1, 1.0, cfg);

    rec = gsRepo.branch(1);

    errUploadOmega = norm(uploadPacket.OmegaBar - OmegaLocal, 'fro');
    errRecordOmega = norm(rec.OmegaBar - OmegaLocal, 'fro');
    errRecordLOS = norm(rec.lastLOSUnit - uLocal);

    [symErrRec, minEigRec, traceRec] = omegaMatrixDiagnostics(rec.OmegaBar);

    fprintf("Upload OmegaBar error        = %.3e\n", errUploadOmega);
    fprintf("GS record OmegaBar error     = %.3e\n", errRecordOmega);
    fprintf("GS record last LOS error     = %.3e\n", errRecordLOS);
    fprintf("GS record symmetry error     = %.3e\n", symErrRec);
    fprintf("GS record min eig            = %.3e\n", minEigRec);
    fprintf("GS record trace              = %.12e\n", traceRec);
    fprintf("GS record numOmegaUpdates    = %d\n", rec.numOmegaUpdates);
    fprintf("GS record outputFrame        = %s\n", string(rec.outputFrame));

    tol = 1e-12;

    assert(errUploadOmega < tol, "Upload packet OmegaBar mismatch.");
    assert(errRecordOmega < tol, "GS record OmegaBar mismatch.");
    assert(errRecordLOS < tol, "GS record lastLOSUnit mismatch.");
    assert(symErrRec < tol, "GS record OmegaBar is not symmetric.");
    assert(minEigRec > -tol, "GS record OmegaBar is not PSD within tolerance.");
    assert(all(size(rec.OmegaBar) == [cfg.dim, cfg.dim]), ...
        "GS record OmegaBar dimension mismatch.");
    assert(rec.numOmegaUpdates == watcher1.numOmegaUpdates, ...
        "GS record numOmegaUpdates mismatch.");
    assert(string(rec.outputFrame) == string(cfg.gs.fimGate.outputFrame), ...
        "GS record outputFrame mismatch.");

    [watcher2, broadcastPacket] = broadcastGSRepositoryToWatcher( ...
        gsRepo, watcher2, 1.0, cfg);

    cache = watcher2.gsBranches(1);

    errCacheOmega = norm(cache.OmegaBar - OmegaLocal, 'fro');
    errCacheLOS = norm(cache.lastLOSUnit - uLocal);

    [symErrCache, minEigCache, traceCache] = omegaMatrixDiagnostics(cache.OmegaBar);

    fprintf("Watcher cache OmegaBar error = %.3e\n", errCacheOmega);
    fprintf("Watcher cache last LOS error = %.3e\n", errCacheLOS);
    fprintf("Watcher cache symmetry error = %.3e\n", symErrCache);
    fprintf("Watcher cache min eig        = %.3e\n", minEigCache);
    fprintf("Watcher cache trace          = %.12e\n", traceCache);
    fprintf("Watcher cache active         = %d\n", cache.active);
    fprintf("Broadcast included branches  = %s\n", ...
        mat2str(broadcastPacket.includedBranchIDs));

    assert(cache.active, "Watcher 2 should have active nonlocal branch 1.");
    assert(errCacheOmega < tol, "Watcher cache OmegaBar mismatch.");
    assert(errCacheLOS < tol, "Watcher cache lastLOSUnit mismatch.");
    assert(symErrCache < tol, "Watcher cache OmegaBar is not symmetric.");
    assert(minEigCache > -tol, "Watcher cache OmegaBar is not PSD within tolerance.");
    assert(all(size(cache.OmegaBar) == [cfg.dim, cfg.dim]), ...
        "Watcher cache OmegaBar dimension mismatch.");
    assert(cache.numOmegaUpdates == watcher1.numOmegaUpdates, ...
        "Watcher cache numOmegaUpdates mismatch.");
    assert(string(cache.outputFrame) == string(cfg.gs.fimGate.outputFrame), ...
        "Watcher cache outputFrame mismatch.");

    % The recipient's own branch must remain local/inactive in the GS cache.
    assert(watcher2.gsBranches(2).isLocalBranch, ...
        "Watcher 2 local branch placeholder should be marked local.");
    assert(~watcher2.gsBranches(2).active, ...
        "Watcher 2 local branch placeholder should not be active as GS nonlocal.");

    fprintf("\nStep 09-J.2 GS OmegaBar metadata check PASSED.\n");

end

function [symErr, minEigVal, traceVal] = omegaMatrixDiagnostics(OmegaBar)
%OMEGAMATRIXDIAGNOSTICS Compact symmetry/PSD diagnostics for OmegaBar.

    OmegaBar = 0.5 * (OmegaBar + OmegaBar');

    symErr = norm(OmegaBar - OmegaBar', 'fro');
    minEigVal = min(eig(OmegaBar));
    traceVal = trace(OmegaBar);

end