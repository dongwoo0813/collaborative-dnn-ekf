function check_step09J1_omega_bar_update()
%{
File:
    sanity_check/check_step09J1_omega_bar_update.m

Purpose:
    Targeted Step 09-J.1 check for the new direction-only bearing geometry
    support helpers.

Checks:
    1. Omega = I - u u' is symmetric PSD.
    2. Omega*u ~= 0.
    3. trace(Omega) = dim - 1.
    4. OmegaBar remains symmetric PSD under EMA updates.
    5. initLocalDNNEKF initializes watcher.OmegaBar / numOmegaUpdates /
       lastLOSUnit, and used bearing/range+bearing measurements update them.

Notes:
    This is intentionally not a full simulation regression. It only checks
    the new dimension/projector/cumulative-geometry logic introduced in
    Step 09-J.1.
%}

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 09-J.1 check: OmegaBar bearing-geometry update\n");
    fprintf("============================================================\n");

    addpath(genpath(pwd));
    rehash;

    tol = 1e-12;

    % ------------------------------------------------------------------
    % 1. Instantaneous 2-D direction-only bearing information matrix.
    % ------------------------------------------------------------------
    u2 = [3; 4];
    [Omega2, u2Unit] = bearingDirectionInfoMatrix(u2);

    symErr2 = norm(Omega2 - Omega2', 'fro');
    nullErr2 = norm(Omega2 * u2Unit);
    traceErr2 = abs(trace(Omega2) - (numel(u2Unit) - 1));
    minEig2 = min(eig(Omega2));

    fprintf("2-D Omega symmetry error      = %.3e\n", symErr2);
    fprintf("2-D Omega null error          = %.3e\n", nullErr2);
    fprintf("2-D Omega trace error         = %.3e\n", traceErr2);
    fprintf("2-D Omega min eig             = %.3e\n", minEig2);

    assert(symErr2 < tol, "2-D Omega is not symmetric enough.");
    assert(nullErr2 < tol, "2-D Omega*u is not close to zero.");
    assert(traceErr2 < tol, "2-D trace(Omega) is not dim-1.");
    assert(minEig2 > -tol, "2-D Omega is not PSD within tolerance.");

    % ------------------------------------------------------------------
    % 2. Instantaneous 3-D check for future compatibility.
    % ------------------------------------------------------------------
    u3 = [1; -2; 2];
    [Omega3, u3Unit] = bearingDirectionInfoMatrix(u3);

    symErr3 = norm(Omega3 - Omega3', 'fro');
    nullErr3 = norm(Omega3 * u3Unit);
    traceErr3 = abs(trace(Omega3) - (numel(u3Unit) - 1));
    minEig3 = min(eig(Omega3));

    fprintf("3-D Omega symmetry error      = %.3e\n", symErr3);
    fprintf("3-D Omega null error          = %.3e\n", nullErr3);
    fprintf("3-D Omega trace error         = %.3e\n", traceErr3);
    fprintf("3-D Omega min eig             = %.3e\n", minEig3);

    assert(symErr3 < tol, "3-D Omega is not symmetric enough.");
    assert(nullErr3 < tol, "3-D Omega*u is not close to zero.");
    assert(traceErr3 < tol, "3-D trace(Omega) is not dim-1.");
    assert(minEig3 > -tol, "3-D Omega is not PSD within tolerance.");

    % ------------------------------------------------------------------
    % 3. Cumulative OmegaBar EMA update.
    % ------------------------------------------------------------------
    lambdaOmega = 0.20;
    OmegaBar = zeros(2,2);
    angleList = [0, pi/4, pi/2, 3*pi/4, pi];

    for k = 1:numel(angleList)
        u = [cos(angleList(k)); sin(angleList(k))];
        [OmegaBar, info] = updateOmegaBar(OmegaBar, u, lambdaOmega, false);
    end

    symErrBar = norm(OmegaBar - OmegaBar', 'fro');
    minEigBar = min(eig(OmegaBar));
    traceBar = trace(OmegaBar);

    fprintf("OmegaBar symmetry error       = %.3e\n", symErrBar);
    fprintf("OmegaBar min eig              = %.3e\n", minEigBar);
    fprintf("OmegaBar trace                = %.12e\n", traceBar);
    fprintf("Last instantaneous null error = %.3e\n", info.nullResidual);

    assert(symErrBar < tol, "OmegaBar is not symmetric enough.");
    assert(minEigBar > -tol, "OmegaBar is not PSD within tolerance.");
    assert(traceBar >= -tol && traceBar <= 1 + tol, ...
        "2-D EMA OmegaBar trace should remain in [0, dim-1].");

    % ------------------------------------------------------------------
    % 4. Bearing measurement to LOS vector and watcher metadata update.
    % ------------------------------------------------------------------
    cfg = config_step04_GS_DNN_EKF();
    cfg.dim = 2;
    cfg.meas.type = "bearing";
    cfg.gs.fimGate.lambdaOmega = lambdaOmega;
    cfg.gs.fimGate.normalizeTrace = false;

    rng(101);
    eta0 = [cfg.target.r0; cfg.target.v0];
    watcher = initLocalDNNEKF(1, eta0, cfg);

    assert(isfield(watcher, "OmegaBar"), "watcher.OmegaBar was not initialized.");
    assert(isfield(watcher, "numOmegaUpdates"), "watcher.numOmegaUpdates was not initialized.");
    assert(isfield(watcher, "lastLOSUnit"), "watcher.lastLOSUnit was not initialized.");
    assert(all(size(watcher.OmegaBar) == [cfg.dim, cfg.dim]), ...
        "watcher.OmegaBar has the wrong size.");
    assert(watcher.numOmegaUpdates == 0, ...
        "watcher.numOmegaUpdates should start at zero.");

    z = pi/3;
    [watcher, geomInfo] = updateWatcherOmegaBarFromMeasurement(watcher, z, 0.5, cfg);

    uMeas = [cos(z); sin(z)];
    OmegaExpected = eye(2) - uMeas * uMeas';
    OmegaBarExpected = lambdaOmega * OmegaExpected;

    updateErr = norm(watcher.OmegaBar - OmegaBarExpected, 'fro');
    losErr = norm(watcher.lastLOSUnit - uMeas);

    fprintf("Watcher OmegaBar update error = %.3e\n", updateErr);
    fprintf("Watcher last LOS error        = %.3e\n", losErr);
    fprintf("Watcher numOmegaUpdates       = %d\n", watcher.numOmegaUpdates);

    assert(geomInfo.updated, "updateWatcherOmegaBarFromMeasurement did not report an update.");
    assert(watcher.numOmegaUpdates == 1, "watcher.numOmegaUpdates did not increment.");
    assert(updateErr < tol, "watcher.OmegaBar update does not match the EMA formula.");
    assert(losErr < tol, "watcher.lastLOSUnit does not match the bearing direction.");

    % ------------------------------------------------------------------
    % 5. Range+bearing packets must update geometry from angles only.
    % ------------------------------------------------------------------
    cfg.meas.type = "range_bearing";
    watcherRB = initLocalDNNEKF(1, eta0, cfg);
    [watcherRB, geomInfoRB] = updateWatcherOmegaBarFromMeasurement( ...
        watcherRB, [1234.5; z], 0.5, cfg);
    updateErrRB = norm(watcherRB.OmegaBar - OmegaBarExpected, 'fro');
    losErrRB = norm(watcherRB.lastLOSUnit - uMeas);

    fprintf("2-D range+bearing Omega error = %.3e\n", updateErrRB);
    fprintf("2-D range+bearing LOS error   = %.3e\n", losErrRB);
    assert(geomInfoRB.updated && watcherRB.numOmegaUpdates == 1, ...
        "2-D range_bearing did not update OmegaBar.");
    assert(updateErrRB < tol && losErrRB < tol, ...
        "2-D range_bearing did not use only its bearing component.");

    cfg3 = struct();
    cfg3.dim = 3;
    cfg3.meas.type = "range_bearing";
    cfg3.gs.fimGate.lambdaOmega = lambdaOmega;
    cfg3.gs.fimGate.normalizeTrace = false;
    watcherRB3 = struct("OmegaBar",zeros(3),"numOmegaUpdates",0, ...
        "lastLOSUnit",NaN(3,1),"lastOmegaUpdateTime",NaN, ...
        "lastOmegaUpdate",struct());
    az = pi/4; el = pi/6;
    [watcherRB3, geomInfoRB3] = updateWatcherOmegaBarFromMeasurement( ...
        watcherRB3, [2000; az; el], 0.5, cfg3);
    uRB3 = [cos(el)*cos(az); cos(el)*sin(az); sin(el)];
    expectedRB3 = lambdaOmega*(eye(3)-uRB3*uRB3');
    updateErrRB3 = norm(watcherRB3.OmegaBar-expectedRB3,'fro');
    losErrRB3 = norm(watcherRB3.lastLOSUnit-uRB3);

    fprintf("3-D range+bearing Omega error = %.3e\n", updateErrRB3);
    fprintf("3-D range+bearing LOS error   = %.3e\n", losErrRB3);
    assert(geomInfoRB3.updated && watcherRB3.numOmegaUpdates == 1, ...
        "3-D range_bearing did not update OmegaBar.");
    assert(updateErrRB3 < tol && losErrRB3 < tol, ...
        "3-D range_bearing did not use [azimuth;elevation].");

    fprintf("\nStep 09-J.1 OmegaBar update check PASSED.\n");

end
