function check_step09J8_second_order_block_transition()
% Targeted accuracy/structure check for second-order block discretization.

    nEta = 4;
    nTheta = 5;
    nX = nEta + nTheta;
    watcher = struct();
    watcher.nEta = nEta;
    watcher.nTheta = nTheta;
    watcher.nX = nX;
    watcher.idxEta = 1:nEta;
    watcher.idxTheta = nEta + (1:nTheta);

    dt = 0.01;
    tauTheta = 10000;
    Aee = [0 0 1 0; 0 0 0 1; -0.03 0.01 -0.02 0; ...
        0.02 -0.04 0 -0.01];
    Aet = reshape(linspace(-0.02, 0.03, nEta*nTheta), ...
        nEta, nTheta);
    Att = -(1/tauTheta)*eye(nTheta);
    A = [Aee Aet; zeros(nTheta,nEta) Att];

    cfg = struct();
    cfg.dnn.thetaDynamics = "FOGM";
    cfg.dnn.thetaTau = tauTheta;

    cfg.ekf.transitionDiscretization = "euler";
    Feuler = discretizeAugmentedTransitionBlock(A, dt, watcher, cfg);

    cfg.ekf.transitionDiscretization = "second_order_block";
    Fsecond = discretizeAugmentedTransitionBlock(A, dt, watcher, cfg);

    Fref = expm(dt*A);
    errEuler = norm(Feuler-Fref, "fro");
    errSecond = norm(Fsecond-Fref, "fro");
    lowerBlockError = norm(Fsecond(watcher.idxTheta,watcher.idxEta), "fro");
    alpha = exp(-dt/tauTheta);
    thetaBlockError = norm( ...
        Fsecond(watcher.idxTheta,watcher.idxTheta) - ...
        alpha*eye(nTheta), "fro");

    fprintf("Euler transition error       = %.6e\n", errEuler);
    fprintf("Second-order transition error = %.6e\n", errSecond);
    fprintf("Lower block error             = %.6e\n", lowerBlockError);
    fprintf("Exact FOGM block error        = %.6e\n", thetaBlockError);

    assert(errSecond < errEuler, ...
        "Second-order block transition is not more accurate than Euler.");
    assert(lowerBlockError < 1e-14, ...
        "Second-order transition broke the zero lower-left block.");
    assert(thetaBlockError < 1e-14, ...
        "Second-order transition did not retain exact FOGM decay.");

    fprintf("Step 09-J.8 second-order block transition check PASSED.\n");
end
