cfg = config_step04_GS_DNN_EKF();

dim = cfg.dim;
e1 = zeros(dim,1);
e1(1) = 1;

etaTrue = [10*e1; zeros(dim,1)];

watcherState = struct();
watcherState.r = zeros(dim,1);

cfg.fov.halfAngleDeg = 20.0;
cfg.fov.halfAngleRad = deg2rad(cfg.fov.halfAngleDeg);
cfg.fov.rhoMin = 0.0;
cfg.fov.rhoMax = Inf;

% Case 1: target is directly along boresight.
watcherState.boresight_I = e1;
[available1, info1] = evaluateFOVAvailability(etaTrue, watcherState, cfg);

fprintf("Case 1 available = %d, reason = %s, angleDeg = %.3f\n", ...
    available1, info1.dropoutReason, info1.offBoresightAngleDeg);

% Case 2: boresight points away from target.
watcherState.boresight_I = -e1;
[available2, info2] = evaluateFOVAvailability(etaTrue, watcherState, cfg);

fprintf("Case 2 available = %d, reason = %s, angleDeg = %.3f\n", ...
    available2, info2.dropoutReason, info2.offBoresightAngleDeg);

% Case 3: target is inside cone but below minimum range.
watcherState.boresight_I = e1;
cfg.fov.rhoMin = 20.0;
[available3, info3] = evaluateFOVAvailability(etaTrue, watcherState, cfg);

fprintf("Case 3 available = %d, reason = %s, range = %.3f\n", ...
    available3, info3.dropoutReason, info3.range);5