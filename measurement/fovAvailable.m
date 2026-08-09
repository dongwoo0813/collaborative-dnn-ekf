function [available, info] = fovAvailable(etaTrue, watcherState, cfg)
%{
File:
    measurement/fovAvailable.m

Purpose:
    Step 08-A.4a measurement-availability wrapper with diagnostics.

    This function determines whether watcher i has a valid measurement at
    the current time step. It separates measurement availability from
    GS/P2P communication triggering.

Supported modes:
    1. cfg.meas.availabilityMode = "always"
       cfg.fov.enabled = false

       Every watcher has a valid angle-only measurement at every EKF update
       step:

           delta_i^m(k) = 1.

    2. cfg.meas.availabilityMode = "fov"
       cfg.fov.enabled = true
       cfg.fov.guardUnimplementedMode = false

       Measurement availability is computed by evaluateFOVAvailability.m.

Inputs:
    etaTrue
        True target physical state eta = [r_t; v_t].

    watcherState
        Watcher state/geometry structure. For FOV mode, it should contain
        watcherState.r and optionally watcherState.boresight_I.

    cfg
        Simulation configuration structure.

Outputs:
    available
        Logical scalar.
        true  -> EKF measurement update may run.
        false -> prediction-only EKF step.

    info
        Diagnostic structure. In FOV mode, this preserves fields from
        evaluateFOVAvailability.m, such as:
            info.dropoutReason
            info.range
            info.insideFOV
            info.insideCone
            info.rangeOK
            info.offBoresightAngleDeg
%}

    % ---------------------------------------------------------------------
    % Read mode flags with backward-compatible defaults.
    % ---------------------------------------------------------------------
    availabilityMode = "always";

    if isfield(cfg, 'meas') && isfield(cfg.meas, 'availabilityMode')
        availabilityMode = string(cfg.meas.availabilityMode);
    end

    fovEnabled = false;

    if isfield(cfg, 'fov') && isfield(cfg.fov, 'enabled')
        fovEnabled = logical(cfg.fov.enabled);
    end

    guardFOVMode = true;

    if isfield(cfg, 'fov') && isfield(cfg.fov, 'guardUnimplementedMode')
        guardFOVMode = logical(cfg.fov.guardUnimplementedMode);
    end

    % ---------------------------------------------------------------------
    % Default diagnostic info.
    % This prevents unavailable measurements from returning empty diagnostics.
    % ---------------------------------------------------------------------
    info = struct();
    info.available = false;
    info.insideFOV = false;
    info.insideCone = false;
    info.rangeOK = false;
    info.range = NaN;
    info.rho = NaN;
    info.offBoresightAngleDeg = NaN;
    info.offBoresightAngleRad = NaN;
    info.dropoutReason = "unavailable";

    % ---------------------------------------------------------------------
    % Measurement-availability mode switch.
    % ---------------------------------------------------------------------
    switch availabilityMode

        case "always"

            if fovEnabled
                error( ...
                    "fovAvailable:InvalidAlwaysModeConfig", ...
                    "Invalid measurement configuration: cfg.meas.availabilityMode = 'always' but cfg.fov.enabled = true. For Step 08-A, use cfg.fov.enabled = false unless running explicit FOV tests.");
            end

            available = true;

            % In always mode, the measurement is available by assumption.
            % Geometry-specific quantities are left as NaN because no FOV
            % geometry is evaluated in this mode.
            info.available = true;
            info.insideFOV = true;
            info.insideCone = true;
            info.rangeOK = true;
            info.dropoutReason = "available";

        case "fov"

            if ~fovEnabled
                error( ...
                    "fovAvailable:InvalidFovModeConfig", ...
                    "Invalid measurement configuration: cfg.meas.availabilityMode = 'fov' but cfg.fov.enabled = false. Set cfg.fov.enabled = true for FOV mode.");
            end

            if guardFOVMode
                error( ...
                    "fovAvailable:FovModeGuarded", ...
                    "FOV mode is still guarded. To run the Step 08-A FOV geometry path, set cfg.fov.guardUnimplementedMode = false explicitly.");
            end

            % Main Step 08-A.4a change:
            % Preserve the diagnostic info instead of discarding it.
            [available, info] = evaluateFOVAvailability(etaTrue, watcherState, cfg);
            available = logical(available);
            info.available = available;

        otherwise

            error( ...
                "fovAvailable:UnsupportedAvailabilityMode", ...
                "Unsupported cfg.meas.availabilityMode: %s", char(availabilityMode));

    end

end