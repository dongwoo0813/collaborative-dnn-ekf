function check_step09J7_prediction_residual_cache_equivalence()
%{
File:
    sanity_check/check_step09J7_prediction_residual_cache_equivalence.m

Purpose:
    Verify that reusing one base-point residual/Jacobian cache does not
    change DNN-EKF prediction results.

Method:
    Run one prediction from identical watcher states with cache reuse on
    and off, then compare xhat, P, and nonlocal covariance diagnostics for
    additive and bearing-FIM-gated GS composite modes.
%}

    addpath(genpath(pwd));
    cfg = config_step04_GS_DNN_EKF();
    cfg.dnn.branchModel = "mlp_general";
    cfg.dnn.mlp.hiddenSizes = [6 6 6];
    cfg.dnn.mlp.activations = ["softplus", "softplus", "tanh"];
    cfg.dnn.mlp.inputMode = "eta_phase";
    cfg.dnn.mlp.thetaInitMode = "small_random";
    cfg.dnn.mlp.theta0Std = 7.5e-5;
    cfg.dnn.Ptheta0 = (7.5e-5)^2;
    cfg.dnn.predictionResidualSource = "GS_composite";
    cfg.gs.enabled = true;
    cfg.gs.bootstrapUpload = true;
    cfg.gs.useNonlocalBranchCovariance = true;

    eta0 = [cfg.target.r0; cfg.target.v0];
    watchers = repmat(initLocalDNNEKF(1, eta0, cfg), cfg.Nw, 1);
    for i = 1:cfg.Nw
        watchers(i) = initLocalDNNEKF(i, eta0, cfg);
    end

    gsRepo = initGSRepository(cfg);
    for i = 1:cfg.Nw
        [gsRepo, ~] = uploadLocalBranchToGS(gsRepo, watchers(i), 0, cfg);
    end
    for i = 1:cfg.Nw
        [watchers(i), ~] = broadcastGSRepositoryToWatcher( ...
            gsRepo, watchers(i), 0, cfg);
    end

    % Supply a nondegenerate 2-D geometry set for the FIM-gated comparison.
    directions = [1 0; 0 1; 1 1; 1 -1].';
    recipient = watchers(1);
    for j = 1:cfg.Nw
        Omega_j = bearingDirectionInfoMatrix(directions(:,j));
        if j == recipient.localBranchID
            recipient.OmegaBar = Omega_j;
        else
            recipient.gsBranches(j).OmegaBar = Omega_j;
        end
    end

    modes = ["additive", "bearing_fim_gated"];
    tolX = 1e-13;
    tolP = 1e-12;

    for mode = modes
        cfgOn = cfg;
        cfgOn.gs.compositeMode = mode;
        cfgOn.dnn.reusePredictionResidualCache = true;

        cfgOff = cfgOn;
        cfgOff.dnn.reusePredictionResidualCache = false;

        predOn = DNN_EKF_Predict_Local(recipient, 0, cfgOn);
        predOff = DNN_EKF_Predict_Local(recipient, 0, cfgOff);

        xError = norm(predOn.xhat - predOff.xhat, inf);
        PError = norm(predOn.P - predOff.P, "fro");
        qError = abs( ...
            predOn.lastNonlocalCovInjection.traceQnonlocal - ...
            predOff.lastNonlocalCovInjection.traceQnonlocal);

        fprintf("mode=%s: xErr=%.3e, PErr=%.3e, QtraceErr=%.3e\n", ...
            mode, xError, PError, qError);

        assert(xError < tolX, ...
            "Cached and uncached state predictions differ for %s.", mode);
        assert(PError < tolP, ...
            "Cached and uncached covariance predictions differ for %s.", mode);
        assert(qError < tolP, ...
            "Cached and uncached Qnonlocal traces differ for %s.", mode);
    end

    fprintf("Step 09-J.7 prediction residual cache check PASSED.\n");
end
