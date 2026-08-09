function watcherState = initWatcherTruth(i, cfg)
%{
Function:
    initWatcherTruth.m

Purpose:
    Initialize the truth state of watcher spacecraft i.

    In the current Step 01 simulation, watcher motion is prescribed.
    However, this function creates a watcher truth-state structure that can
    later be propagated by thruster dynamics and attitude dynamics.

Inputs:
    i   - Watcher index.
          Type: positive integer.
          Range: 1 <= i <= cfg.Nw.

    cfg - Simulation configuration structure.
          Required fields:
              cfg.dim
              cfg.time
              cfg.watchers.mass
              cfg.watchers.useAttitude

Outputs:
    watcherState - Watcher truth-state structure.
                   Fields:
                       watcherState.id
                       watcherState.r
                       watcherState.v
                       watcherState.q
                       watcherState.omega
                       watcherState.mass
                       watcherState.u
                       watcherState.tau

Main equations:
    The translational watcher state is

        x_{w,i} = [r_{w,i}; v_{w,i}].

    For now, r_{w,i}(0) and v_{w,i}(0) are obtained from the prescribed
    trajectory

        [r_{w,i}(0), v_{w,i}(0)] = watcherTrajectory(i,t_0,cfg).

    Later, controlled watcher dynamics can use

        dot r_w = v_w,

        dot v_w = u_w / m_w.

    The attitude placeholders use scalar-last quaternion convention

        q = [q_v; q_s].

Notes:
    - The watcher state is assumed known exactly by its own onboard system
      in this baseline.
    - The target EKF does not estimate watcher states.
%}

    t0 = cfg.time(1);

    ws0 = watcherTrajectory(i, t0, cfg);

    watcherState.id = i;
    watcherState.r = ws0.r;
    watcherState.v = ws0.v;

    % Scalar-last identity quaternion q = [qx; qy; qz; qs].
    watcherState.q = [0; 0; 0; 1];
    watcherState.omega = zeros(3,1);

    if isscalar(cfg.watchers.mass)
        watcherState.mass = cfg.watchers.mass;
    else
        watcherState.mass = cfg.watchers.mass(i);
    end

    watcherState.u = zeros(cfg.dim,1);
    watcherState.tau = zeros(3,1);
    watcherState.controllerState = struct( ...
        'direction',zeros(cfg.dim,1), ...
        'nextReplanTime',-Inf, ...
        'candidateIndex',0, ...
        'score',NaN, ...
        'candidateScores',[], ...
        'candidateInformationMinEig',[], ...
        'candidateInformationCondition',[], ...
        'predictedRadialVariance',NaN, ...
        'selectedInformationMinEig',NaN, ...
        'selectedInformationCondition',NaN, ...
        'replanFlag',false, ...
        'activeFlag',false, ...
        'lastPlanTime',NaN);

end
