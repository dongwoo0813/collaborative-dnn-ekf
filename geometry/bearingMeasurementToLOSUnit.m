function u = bearingMeasurementToLOSUnit(z, cfg)
%{
File:
    geometry/bearingMeasurementToLOSUnit.m

Purpose:
    Convert the current bearing measurement into an inertial LOS unit vector
    from watcher to target.

Inputs:
    z
        Bearing measurement.
        For cfg.dim = 2: scalar angle atan2(rho_y, rho_x).
        For cfg.dim = 3: [azimuth; elevation].

    cfg
        Simulation configuration with cfg.dim.

Outputs:
    u
        Unit LOS vector in the same inertial frame used by the residual
        acceleration output. Size cfg.dim x 1.

Notes:
    - This uses the actual bearing measurement z rather than the hidden true
      target position, so the geometry support follows the measurement signal
      that drives the EKF update.
    - The measurement can be noisy; that is acceptable for this first
      direction-only support diagnostic.
%}

    if ~isfield(cfg, "dim")
        error("bearingMeasurementToLOSUnit:MissingDim", ...
            "cfg.dim is required.");
    end

    dim = cfg.dim;
    z = z(:);

    if dim == 2

        if numel(z) ~= 1
            error("bearingMeasurementToLOSUnit:Invalid2DBearing", ...
                "2-D bearing measurement must be scalar.");
        end

        az = z(1);
        u = [cos(az); sin(az)];

    elseif dim == 3

        if numel(z) ~= 2
            error("bearingMeasurementToLOSUnit:Invalid3DBearing", ...
                "3-D bearing measurement must be [azimuth; elevation].");
        end

        az = z(1);
        el = z(2);

        u = [cos(el) * cos(az);
             cos(el) * sin(az);
             sin(el)];

    else
        error("bearingMeasurementToLOSUnit:UnsupportedDimension", ...
            "Unsupported cfg.dim = %d. Use 2 or 3.", dim);
    end

    u = u / norm(u);

end
