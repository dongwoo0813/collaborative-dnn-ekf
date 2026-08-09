function F = discretizeAugmentedTransitionBlock(A, dt, watcher, cfg)
%{
File:
    ekf/discretizeAugmentedTransitionBlock.m

Purpose:
    Discretize the block upper-triangular augmented DNN-EKF Jacobian

        A = [A_ee, A_et;
             0,    A_tt]

    without forming A^2 as a full augmented dense product.

Modes:
    cfg.ekf.transitionDiscretization = "euler"
        F = I + dt*A, with exact FOGM F_tt.

    cfg.ekf.transitionDiscretization = "second_order_block"
        F_ee = I + dt*A_ee + 0.5*dt^2*A_ee^2
        F_et = dt*A_et + 0.5*dt^2*(A_ee*A_et + A_et*A_tt)
        F_te = 0
        F_tt = exact FOGM decay or random-walk identity.
%}

    nX = watcher.nX;
    idxEta = watcher.idxEta;
    idxTheta = watcher.idxTheta;

    if ~isscalar(dt) || ~isfinite(dt) || dt <= 0
        error("discretizeAugmentedTransitionBlock:BadDt", ...
            "dt must be a positive finite scalar.");
    end
    if any(size(A) ~= [nX nX])
        error("discretizeAugmentedTransitionBlock:BadASize", ...
            "A must be watcher.nX-by-watcher.nX.");
    end

    Aee = A(idxEta, idxEta);
    Aet = A(idxEta, idxTheta);
    Ate = A(idxTheta, idxEta);
    Att = A(idxTheta, idxTheta);

    tol = 1e-12;
    if max(abs(Ate(:))) > tol
        error("discretizeAugmentedTransitionBlock:NonzeroAte", ...
            "The block method requires A(theta,eta)=0.");
    end

    mode = "euler";
    if isfield(cfg, "ekf") && ...
            isfield(cfg.ekf, "transitionDiscretization")
        mode = string(cfg.ekf.transitionDiscretization);
    end

    switch mode
        case "euler"
            Fee = eye(watcher.nEta) + dt*Aee;
            Fet = dt*Aet;

        case "second_order_block"
            halfDt2 = 0.5*dt^2;
            Fee = eye(watcher.nEta) + dt*Aee + halfDt2*(Aee*Aee);
            Fet = dt*Aet + halfDt2*(Aee*Aet + Aet*Att);

        otherwise
            error("discretizeAugmentedTransitionBlock:BadMode", ...
                "Unsupported transition discretization mode: %s", mode);
    end

    thetaDynamics = "random_walk";
    if isfield(cfg, "dnn") && isfield(cfg.dnn, "thetaDynamics")
        thetaDynamics = string(cfg.dnn.thetaDynamics);
    end

    switch thetaDynamics
        case "random_walk"
            Ftt = eye(watcher.nTheta);

        case "FOGM"
            if ~isfield(cfg.dnn, "thetaTau") || ...
                    ~isscalar(cfg.dnn.thetaTau) || cfg.dnn.thetaTau <= 0
                error("discretizeAugmentedTransitionBlock:BadThetaTau", ...
                    "Positive scalar cfg.dnn.thetaTau is required for FOGM.");
            end
            alphaTheta = exp(-dt/cfg.dnn.thetaTau);
            Ftt = alphaTheta*eye(watcher.nTheta);

        otherwise
            error("discretizeAugmentedTransitionBlock:BadThetaDynamics", ...
                "Unsupported theta dynamics: %s", thetaDynamics);
    end

    F = zeros(nX, nX);
    F(idxEta, idxEta) = Fee;
    F(idxEta, idxTheta) = Fet;
    F(idxTheta, idxTheta) = Ftt;
end
