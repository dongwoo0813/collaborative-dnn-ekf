function report = check_block_owned_joint_gs_short_sync()
%CHECK_BLOCK_OWNED_JOINT_GS_SHORT_SYNC Exercise one complete GS cycle.
%
% This check verifies the operational path for one global DNN partitioned
% across watcher-owned parameter blocks: all watchers supply information,
% GS forms a joint correction, and one correction is dispatched per block.
% It is a finite-value and dispatch test, not a performance claim.

    seed = 101;
    dt = 0.1;
    syncPeriod = 5.0;
    nWatchers = 4;
    nBlocks = 4;

    out = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,5.1,false,nWatchers,dt,3, ...
        "hybrid_owner_ekf_joint_gs_60s",3,nBlocks, ...
        "all_to_all",[],syncPeriod);

    shared = out.sharedBlock;
    assert(shared.mode == "hybrid_owner_ekf_full_joint_gs_one_step", ...
        "The full-joint GS path was not selected.");
    assert(out.cfg.communication.period == syncPeriod, ...
        "The synchronization-period override was not applied to the GS path.");
    assert(all(isfinite(shared.etaHat(:))) && all(isfinite(shared.dHat(:))), ...
        "The joint GS run produced a non-finite state or DNN output.");
    assert(all(isfinite(shared.thetaChange(:))), ...
        "The joint GS run produced a non-finite parameter correction.");
    assert(shared.parameterUploads == nBlocks, ...
        "Expected exactly one GS dispatch per parameter block.");
    assert(any(shared.thetaChange(:) > 0), ...
        "The GS joint correction did not update any parameter block.");

    report = struct;
    report.passed = true;
    report.seed = seed;
    report.syncPeriod = syncPeriod;
    report.parametersPerBlock = out.cfg.dnn.arch.nTheta;
    report.globalParameterCount = nBlocks*out.cfg.dnn.arch.nTheta;
    report.blockDispatches = shared.parameterUploads;
    report.maxThetaChange = max(shared.thetaChange(:));
    report.positionRMSE = shared.positionRMSE;
    report.accelerationRMSE = shared.accelerationRMSE;
    fprintf(["Block-owned full-joint GS short-sync check passed: " + ...
        "%d parameters, %d block dispatches, max |dtheta|=%.6g.\n"], ...
        report.globalParameterCount,report.blockDispatches,report.maxThetaChange);
end
