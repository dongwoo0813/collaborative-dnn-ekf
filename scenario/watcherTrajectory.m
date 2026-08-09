function ws = watcherTrajectory(i, t, cfg)
%{
Function:
    watcherTrajectory.m

Purpose:
    Generate the prescribed trajectory of watcher spacecraft i at time t.

    This function is intentionally dimension-generic. It supports both
    cfg.dim = 2 and cfg.dim = 3. The initial simulation may use 2D motion,
    but the file name and interface are not 2D-specific so that later 3D
    trajectories can be added without changing the simulation architecture.

Inputs:
    i   - Watcher index.
          Type: positive integer.
          Range: 1 <= i <= cfg.Nw.

    t   - Current simulation time.
          Type: scalar.
          Units: seconds.

    cfg - Simulation configuration structure.
          Required fields:
              cfg.dim
              cfg.Nw
              cfg.watchers.radius
              cfg.watchers.omega
              cfg.watchers.inclination
              cfg.scenario.watcherModel

Outputs:
    ws  - Watcher state structure.
          Fields:
              ws.r - watcher position vector, size cfg.dim x 1
              ws.v - watcher velocity vector, size cfg.dim x 1
              ws.a - watcher acceleration vector, size cfg.dim x 1

Main equations:
    For cfg.dim = 2, the prescribed watcher trajectory is

        r_w,i(t) = R [ cos(omega t + psi_i);
                       sin(omega t + psi_i) ],

        v_w,i(t) = R omega [ -sin(omega t + psi_i);
                               cos(omega t + psi_i) ],

    where

        psi_i = 2*pi*(i-1)/cfg.Nw.

    For cfg.dim = 3, an inclined circular trajectory is used:

        r_w,i(t) = R [ cos(theta_i);
                       sin(theta_i) cos(inc);
                       sin(theta_i) sin(inc) ],

    with theta_i = omega t + psi_i.

Notes:
    - This is a prescribed trajectory, not a controlled watcher dynamics model.
    - Later, when attitude/orbit control is added, this function can be replaced
      or extended by a watcher dynamics propagator.
%}
    
    [r, v, a] = unmaneuveredWatcherTrajectory(i, t, cfg);

    % Optional calibrated observability maneuver. The maneuver is an
    % analytically integrated finite acceleration pulse superposed on the
    % existing prescribed circular trajectory. Its direction is frozen at
    % the burn start using the nominal watcher-to-target LOS, so r, v, and
    % a remain mutually consistent and deterministic.
    [deltaR, deltaV, deltaA] = prescribedObservabilityPerturbation( ...
        i, t, cfg);

    r = r + deltaR;
    v = v + deltaV;
    a = a + deltaA;
    
    ws.r = r;
    ws.v = v;
    ws.a = a;

end

function [r, v, a] = unmaneuveredWatcherTrajectory(i, t, cfg)
%UNMANEUVEREDWATCHERTRAJECTORY Reference motion before active excitation.

    dim = cfg.dim;
    R = cfg.watchers.radius;
    omega = cfg.watchers.omega;
    psi = 2*pi*(i-1)/cfg.Nw;

    switch string(cfg.scenario.watcherModel)
        case "prescribed_orbit"
            theta = omega*t + psi;
            if dim == 2
                direction = [cos(theta); sin(theta)];
                tangent = [-sin(theta); cos(theta)];
            elseif dim == 3
                inc = cfg.watchers.inclination;
                direction = [cos(theta); ...
                    sin(theta)*cos(inc); sin(theta)*sin(inc)];
                tangent = [-sin(theta); ...
                    cos(theta)*cos(inc); cos(theta)*sin(inc)];
            else
                error("Unsupported cfg.dim. Use cfg.dim = 2 or cfg.dim = 3.");
            end
            r = R * direction;
            v = R * omega * tangent;
            a = -R * omega^2 * direction;

        case "matched_velocity_coast"
            % Watchers start at the same phased positions as the circular
            % case, but coast with the nominal target velocity. Therefore,
            % under the nominal double-integrator model, each natural
            % watcher-to-target relative position remains constant.
            if dim == 2
                r0 = R * [cos(psi); sin(psi)];
            elseif dim == 3
                inc = cfg.watchers.inclination;
                r0 = R * [cos(psi); ...
                    sin(psi)*cos(inc); sin(psi)*sin(inc)];
            else
                error("Unsupported cfg.dim. Use cfg.dim = 2 or cfg.dim = 3.");
            end

            if isfield(cfg.watchers, "coastVelocity")
                v = cfg.watchers.coastVelocity(:);
            else
                v = cfg.target.v0(:);
            end
            if numel(v) ~= dim
                error("cfg.watchers.coastVelocity has incompatible size.");
            end
            r = r0 + v*t;
            a = zeros(dim,1);

        otherwise
            error("Unknown cfg.scenario.watcherModel: %s", ...
                string(cfg.scenario.watcherModel));
    end
end

function [deltaR, deltaV, deltaA] = ...
    prescribedObservabilityPerturbation(i, t, cfg)
%PRESCRIBEDOBSERVABILITYPERTURBATION Known active-sensing excitation.

    dim = cfg.dim;
    deltaR = zeros(dim,1);
    deltaV = zeros(dim,1);
    deltaA = zeros(dim,1);

    if ~isfield(cfg, "control") || ~isfield(cfg.control, "obs")
        return;
    end

    obs = cfg.control.obs;
    if ~isfield(obs, "enabled") || ~logical(obs.enabled)
        return;
    end

    mode = string(getFieldOrDefault(obs, "mode", "none"));
    if mode == "none"
        return;
    end

    startTime = getFieldOrDefault(obs, "startTime", 0.0);
    burnDuration = getFieldOrDefault(obs, "burnDuration", 0.0);
    acceleration = getFieldOrDefault(obs, "acceleration", 0.0);

    if burnDuration <= 0 || acceleration <= 0 || t <= startTime
        return;
    end

    direction = maneuverDirectionAtBurnStart(i, startTime, cfg, mode);

    if mode == "transverse_alternating"
        direction = (-1)^(i-1) * direction;
    end

    elapsed = t - startTime;
    burnElapsed = min(elapsed, burnDuration);

    % Constant calibrated acceleration during the burn, followed by coast.
    % This is the finite-burn counterpart of the paper's impulsive maneuver.
    deltaA = acceleration * direction * double(elapsed < burnDuration);
    deltaV = acceleration * burnElapsed * direction;

    if elapsed <= burnDuration
        deltaR = 0.5 * acceleration * elapsed^2 * direction;
    else
        deltaRAtBurnEnd = ...
            0.5 * acceleration * burnDuration^2 * direction;
        deltaVAtBurnEnd = acceleration * burnDuration * direction;
        deltaR = deltaRAtBurnEnd + ...
            deltaVAtBurnEnd * (elapsed - burnDuration);
    end

end

function direction = maneuverDirectionAtBurnStart(i, startTime, cfg, mode)
%MANEUVERDIRECTIONATBURNSTART Parallel/transverse initial-LOS direction.

    [watcherBase, ~, ~] = ...
        unmaneuveredWatcherTrajectory(i, startTime, cfg);

    % The nominal target trajectory is used only to select the maneuver
    % direction. In the present double-integrator case this equals r0+v0*t.
    targetNominal = cfg.target.r0 + cfg.target.v0 * startTime;
    los = targetNominal - watcherBase;
    losNorm = norm(los);
    if losNorm <= 1e-12
        error("Cannot define observability maneuver at zero nominal range.");
    end
    los = los / losNorm;

    switch mode
        case "parallel"
            direction = los;

        case {"transverse", "transverse_alternating"}
            if cfg.dim == 2
                direction = [-los(2); los(1)];
            else
                % Select the coordinate axis least aligned with the LOS,
                % then project it into the LOS-normal plane.
                [~, axisID] = min(abs(los));
                axisVector = zeros(3,1);
                axisVector(axisID) = 1;
                direction = axisVector - los * (los' * axisVector);
                direction = direction / norm(direction);
            end

        otherwise
            error("Unknown cfg.control.obs.mode: %s", mode);
    end
end

function value = getFieldOrDefault(s, fieldName, defaultValue)
    if isfield(s, fieldName)
        value = s.(fieldName);
    else
        value = defaultValue;
    end
end
