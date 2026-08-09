function [watcherNext, cmd] = propagateWatcherStep(i, watcherCurrent, targetInfo, t, cfg)
%{
Function:
    propagateWatcherStep.m

Purpose:
    Propagate or update watcher spacecraft i from t_k to t_{k+1}.

    This function is the abstraction layer between the simulation loop and
    the watcher motion model.

    In the current Step 01 baseline:

        cfg.watchers.motionMode = "prescribed",

    so the watcher state is obtained from watcherTrajectory(i,t_{k+1},cfg).

    Later:

        cfg.watchers.motionMode = "controlled",

    can propagate watcher translational and attitude dynamics using
    thruster force and torque commands.

Inputs:
    i              - Watcher index.

    watcherCurrent - Current watcher truth-state structure at time t_k.

    targetInfo     - Target information structure.
                     Suggested fields:
                         targetInfo.etaHat
                         targetInfo.etaTrue

    t              - Current simulation time t_k.

    cfg            - Simulation configuration structure.
                     Required fields:
                         cfg.dt
                         cfg.dim
                         cfg.watchers.motionMode

Outputs:
    watcherNext    - Watcher truth-state structure at time t_{k+1}.

    cmd            - Control command structure.
                     Fields:
                         cmd.u
                         cmd.tau

Main equations:
    Prescribed mode:

        r_w(t_{k+1}), v_w(t_{k+1})
            = watcherTrajectory(i,t_{k+1},cfg).

    Controlled translational mode:

        dot r_w = v_w,

        dot v_w = u_w / m_w.

    Attitude dynamics are not activated in Step 01, but placeholders are
    preserved:

        q_w, omega_w, tau_w.

Notes:
    - This function keeps the simulation loop independent of whether watcher
      motion is prescribed or controlled.
    - The measurement model only needs watcherNext.r for now.
%}

    dt = cfg.dt;
    tNext = t + dt;

    [cmd,controllerState] = watcherController( ...
        i,watcherCurrent,targetInfo,t,cfg);

    switch cfg.watchers.motionMode

        case "prescribed"

            ws = watcherTrajectory(i, tNext, cfg);

            watcherNext = watcherCurrent;
            watcherNext.r = ws.r;
            watcherNext.v = ws.v;
            if isfield(ws, "a")
                cmd.u = watcherCurrent.mass * ws.a;
            else
                cmd.u = zeros(cfg.dim,1);
            end
            watcherNext.u = cmd.u;
            watcherNext.tau = zeros(3,1);
            watcherNext.controllerState = controllerState;

        case "controlled"

            watcherNext = watcherCurrent;

            x0 = [watcherCurrent.r; watcherCurrent.v];

            f = @(tt,xx) watcherTranslationalDynamics( ...
                tt, xx, cmd.u, watcherCurrent.mass, cfg);

            xNext = propagateRK4(f, t, x0, dt);

            watcherNext.r = xNext(1:cfg.dim);
            watcherNext.v = xNext(cfg.dim+1:2*cfg.dim);
            watcherNext.u = cmd.u;
            watcherNext.tau = cmd.tau;
            watcherNext.controllerState = controllerState;

            % Attitude dynamics can be added here later.
            % For now, keep attitude fixed.
            if ~cfg.watchers.useAttitude
                watcherNext.q = watcherCurrent.q;
                watcherNext.omega = watcherCurrent.omega;
            end

        otherwise
            error("Unknown cfg.watchers.motionMode.");
    end

end

function xdot = watcherTranslationalDynamics(t, x, u, mass, cfg)
%{
Function:
    watcherTranslationalDynamics

Purpose:
    Continuous-time translational dynamics for controlled watcher motion.

Inputs:
    t     - Current time.
    x     - Watcher translational state [r_w; v_w].
    u     - Translational force command.
    mass  - Watcher mass.
    cfg   - Simulation configuration.

Outputs:
    xdot  - Time derivative of x.

Main equations:
    dot r_w = v_w,

    dot v_w = u_w / m_w.

Notes:
    - Environmental/orbital terms can be added later.
%}

    dim = cfg.dim;

    r = x(1 : dim);
    v = x(dim+1 : 2*dim);

    rdot = v;
    vdot = u / mass;

    xdot = [rdot; vdot];

end
