function cfg = config_step01_physical_EKF()

    cfg.dim = 2;          % later: 3
    cfg.Nw  = 4;          % later: 6, 8, ...
    
    cfg.dt = 0.5;
    cfg.T  = 2000;
    cfg.time = 0:cfg.dt:cfg.T;
    cfg.N = numel(cfg.time);
    
    cfg.scenario.targetModel = "double_integrator";
    cfg.scenario.watcherModel = "prescribed_orbit";
    
    cfg.target.r0 = zeros(cfg.dim,1);
    cfg.target.v0 = zeros(cfg.dim,1);
    
    if cfg.dim == 2
        cfg.target.r0 = [500; 0];
        cfg.target.v0 = [0; 0.2];
    elseif cfg.dim == 3
        cfg.target.r0 = [500; 0; 50];
        cfg.target.v0 = [0; 0.2; 0.01];
    end


    % Watcher motion mode
    %   "prescribed" : watcher motion is generated analytically by watcherTrajectory.m
    %   "controlled" : watcher motion is propagated by thruster/control dynamics
    cfg.watchers.motionMode = "prescribed";
    
    % Basic translational control/dynamics parameters for later use
    cfg.watchers.mass = 20 * ones(cfg.Nw,1);      % kg
    cfg.watchers.maxThrust = 1e-3;                % N
    
    % Attitude placeholders for later spacecraft attitude/FOV control
    cfg.watchers.useAttitude = false;
    cfg.watchers.J = eye(3);                      % kg*m^2, placeholder inertia
    cfg.watchers.maxTorque = 1e-5;                % N*m
    
    % Controller mode
    cfg.control.translationMode = "none";         % later: "PD", "MPC", ...
    cfg.control.attitudeMode = "none";            % later: "PD_attitude", ...

    cfg.watchers.radius = 1000;
    cfg.watchers.omega  = 2*pi/2000;
    cfg.watchers.inclination = deg2rad(30);
    
    cfg.meas.type = "bearing";
    cfg.meas.sigmaBearing = deg2rad(0.2);
    
    if cfg.dim == 2
        cfg.meas.R = cfg.meas.sigmaBearing^2;
    elseif cfg.dim == 3
        cfg.meas.R = cfg.meas.sigmaBearing^2 * eye(2);
    end
    
    cfg.ekf.r0_err_std = 50;
    cfg.ekf.v0_err_std = 0.1;
    
    cfg.ekf.P0_pos = 50^2;
    cfg.ekf.P0_vel = 0.1^2;
    cfg.ekf.q_acc  = 1e-6;

end