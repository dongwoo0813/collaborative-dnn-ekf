function cfg = config_step08_FOV_GS_DNN_EKF()
%{
File:
    config/config_step08_FOV_GS_DNN_EKF.m

Purpose:
    Step 08 FOV-enabled GS DNN-EKF configuration.

    This config starts from the validated Step 04/05 GS DNN-EKF config and
    activates FOV-based measurement availability.

    Use this config when you want to run GS DNN-EKF with intermittent
    angle-only measurements caused by camera FOV dropout.

Base config:
    config_step04_GS_DNN_EKF()

Main changes:
    cfg.meas.availabilityMode = "fov";
    cfg.fov.enabled = true;
    cfg.fov.guardUnimplementedMode = false;

Default FOV geometry:
    cfg.fov.boresightMode = "inertial_fixed";
    cfg.fov.boresightInertial = e1;
    cfg.fov.halfAngleDeg = 1.0;

Notes:
    - This config intentionally creates intermittent measurements.
    - Range gating is disabled by default.
    - Cone-based dropout is the primary dropout mechanism.
    - For always-available measurement, use config_step04_GS_DNN_EKF().
%}

    % Start from the validated GS-assisted DNN-EKF configuration.
    cfg = config_step04_GS_DNN_EKF();

    % ---------------------------------------------------------------------
    % Step label
    % ---------------------------------------------------------------------
    cfg.step.name = "step08_FOV_GS_DNN_EKF";

    % ---------------------------------------------------------------------
    % Activate FOV-based measurement availability
    % ---------------------------------------------------------------------
    % This changes the measurement availability indicator delta_i^m(k).
    %
    % In Step 04 default config:
    %   cfg.meas.availabilityMode = "always";
    %   cfg.fov.enabled = false;
    %
    % In this Step 08 config:
    %   cfg.meas.availabilityMode = "fov";
    %   cfg.fov.enabled = true;
    %
    % Therefore, EKF measurement updates occur only when the target is inside
    % the watcher camera FOV.
    cfg.meas.availabilityMode = "fov";
    cfg.fov.enabled = true;
    cfg.fov.guardUnimplementedMode = false;

    % ---------------------------------------------------------------------
    % FOV cone configuration
    % ---------------------------------------------------------------------
    % Use a fixed inertial boresight to generate real cone-based dropout.
    % This is simple, repeatable, and useful for debugging.
    cfg.fov.boresightMode = "inertial_fixed";

    cfg.fov.boresightInertial = zeros(cfg.dim,1);
    cfg.fov.boresightInertial(1) = 1.0;

    % A narrow half-angle intentionally creates intermittent measurements.
    cfg.fov.halfAngleDeg = 1.0;
    cfg.fov.halfAngleRad = deg2rad(cfg.fov.halfAngleDeg);

    % Disable range gating by default so dropout is caused by outside_fov.
    cfg.fov.rhoMin = 0.0;
    cfg.fov.rhoMax = Inf;

    % ---------------------------------------------------------------------
    % GS communication policy
    % ---------------------------------------------------------------------
    % Keep the validated GS communication path.
    %
    % Do not use "every_step" upload here unless we make a separate safe
    % every-step upload test. With always-available measurements,
    % after_measurement_update is already effectively frequent, but with FOV
    % dropout it correctly blocks uploads when there is no measurement update.
    cfg.gs.uploadMode = "after_measurement_update";
    cfg.gs.broadcastMode = "every_step";

end