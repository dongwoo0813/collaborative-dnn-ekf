function [out09j5, diag09j6] = run_step09J6_seed101_operational()
%RUN_STEP09J6_SEED101_OPERATIONAL Run only the two cases needed by 09-J.6.
% This preserves the Step 09-J.5 benchmark configuration while omitting
% Local, Oracle, and the expensive legacy post-processing diagnostics.

    addpath(genpath(pwd));
    [cfgBase, seed, meta] = config_step09J6_seed101_operational();
    thetaInitStd = meta.thetaInitStd;

    cfgGSAdd = cfgBase;
    cfgGSAdd.step.name = "step09J6_GS_additive_MLP";
    cfgGSAdd.gs.compositeMode = "additive";
    fprintf("Step 09-J.6: running additive GS case...\n");
    rng(seed);
    resGSAdd = simulate_GS_DNN_EKF(cfgGSAdd);

    cfgGSFIM = cfgBase;
    cfgGSFIM.step.name = "step09J6_GS_bearing_FIM_gated_MLP";
    cfgGSFIM.gs.compositeMode = "bearing_fim_gated";
    fprintf("Step 09-J.6: running bearing-FIM-gated GS case...\n");
    rng(seed);
    resGSFIM = simulate_GS_DNN_EKF(cfgGSFIM);

    assert(all(isfinite(resGSAdd.dnnResidual(:))), ...
        "Additive dnnResidual contains non-finite values.");
    assert(all(isfinite(resGSFIM.dnnResidual(:))), ...
        "FIM dnnResidual contains non-finite values.");

    out09j5 = struct("resGSAdd", resGSAdd, "resGSFIM", resGSFIM, ...
        "cfgGSAdd", cfgGSAdd, "cfgGSFIM", cfgGSFIM, ...
        "residualFamily", "feedback_sat_disturbance", "seed", seed, ...
        "thetaInitStd", thetaInitStd);
    diag09j6 = run_step09J6_estimate_norm_cosine_diagnostic( ...
        out09j5, [0 25 50 100 150 200], false);
end
