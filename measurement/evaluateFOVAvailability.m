function [available, info] = evaluateFOVAvailability(etaTrue, watcherState, cfg)
%{
File:
    measurement/evaluateFOVAvailability.m

Purpose:
    Step 08-A.2 FOV-geometry availability helper.

    This function evaluates whether the target is inside watcher i's camera
    field of view and optional valid range interval.

    It is a pure geometry helper. It does not generate measurements and it
    does not update the EKF.

Main FOV condition:
    Let

        r_ti^I = r_t^I - r_i^I

    be the relative position from watcher i to the target, expressed in the
    inertial frame. Let b_i^I be the camera boresight direction, also
    expressed in the inertial frame.

    The target is inside the FOV cone if

        dot(r_ti^I / ||r_ti^I||, b_i^I) >= cos(theta_FOV).

    Optional range gate:

        rhoMin <= ||r_ti^I|| <= rhoMax.

Inputs:
    etaTrue
        True target physical state eta = [r_t; v_t], size 2*cfg.dim x 1.

    watcherState
        Watcher state structure.
        Required:
            watcherState.r
        Optional:
            watcherState.boresight_I
            watcherState.boreSight_I

    cfg
        Simulation configuration.
        Required:
            cfg.dim
        Optional:
            cfg.fov.halfAngleRad
            cfg.fov.halfAngleDeg
            cfg.fov.rhoMin
            cfg.fov.rhoMax
            cfg.fov.boresightMode
            cfg.fov.boresightInertial

Outputs:
    available
        Logical scalar. true if the target is inside the FOV cone and range
        gates.

    info
        Diagnostic struct with FOV geometry and dropout reason.

Notes:
    - This helper is not yet wired into fovAvailable.m by default.
    - Default boresight mode is "target_pointing", which points the camera
      directly at the target and therefore only tests range gating.
    - For unit tests, set watcherState.boresight_I or cfg.fov.boresightMode
      = "inertial_fixed".
%}

    dim = cfg.dim;

    rTarget = etaTrue(1:dim);
    rWatcher = watcherState.r(:);

    rRel = rTarget(:) - rWatcher(:);
    rho = norm(rRel);

    info = initFOVInfo();
    info.range  = rho;
    info.rho    = rho;

    halfAngleRad = getFOVHalfAngleRad(cfg);
    halfAngleDeg = rad2deg(halfAngleRad);

    rhoMin = getNumericFieldNested(cfg, "fov", "rhoMin", 0.0);
    rhoMax = getNumericFieldNested(cfg, "fov", "rhoMax", Inf);

    info.halfAngleRad = halfAngleRad;
    info.halfAngleDeg = halfAngleDeg;
    info.rhoMin = rhoMin;
    info.rhoMax = rhoMax;

    if rho <= eps
        available = false;

        info.available = false;
        info.insideFOV = false;
        info.insideCone = false;
        info.rangeOK = false;
        info.dropoutReason = "zero_range";
        return;
    end

    losUnit_I = rRel / rho;

    boresight_I = getFOVBoresightInertial(losUnit_I, watcherState, cfg);
    boresightNorm = norm(boresight_I);

    if boresightNorm <= eps
        error("evaluateFOVAvailability:InvalidBoresight", ...
            "FOV boresight vector has near-zero norm.");
    end

    boresight_I = boresight_I / boresightNorm;

    cosAngle = dot(losUnit_I, boresight_I);
    cosAngle = max(-1.0, min(1.0, cosAngle));

    offAngleRad = acos(cosAngle);
    offAngleDeg = rad2deg(offAngleRad);

    coneTolerance = getNumericFieldNested(cfg, "fov", "coneTolerance", 1e-12);

    insideCone = cosAngle >= cos(halfAngleRad) - coneTolerance;
    aboveMinRange = rho >= rhoMin;
    belowMaxRange = rho <= rhoMax;
    rangeOK = aboveMinRange && belowMaxRange;

    available = insideCone && rangeOK;

    info.available = available;
    info.insideFOV = available;
    info.insideCone = insideCone;
    info.rangeOK = rangeOK;
    info.aboveMinRange = aboveMinRange;
    info.belowMaxRange = belowMaxRange;
    info.cosAngle = cosAngle;
    info.offBoresightAngleRad = offAngleRad;
    info.offBoresightAngleDeg = offAngleDeg;
    info.losUnit_I = losUnit_I;
    info.boresight_I = boresight_I;

    if available
        info.dropoutReason = "available";
    elseif ~aboveMinRange
        info.dropoutReason = "range_too_small";
    elseif ~belowMaxRange
        info.dropoutReason = "range_too_large";
    elseif ~insideCone
        info.dropoutReason = "outside_fov";
    else
        info.dropoutReason = "unavailable";
    end

end

function info = initFOVInfo()
% Initialize FOV diagnostic struct with stable fields.

    info = struct();

    info.available = false;
    info.insideFOV = false;
    info.insideCone = false;
    info.rangeOK = false;
    info.aboveMinRange = false;
    info.belowMaxRange = false;

    info.range = NaN;
    info.rho = NaN;
    info.rhoMin = NaN;
    info.rhoMax = NaN;

    info.halfAngleRad = NaN;
    info.halfAngleDeg = NaN;

    info.cosAngle = NaN;
    info.offBoresightAngleRad = NaN;
    info.offBoresightAngleDeg = NaN;

    info.losUnit_I = [];
    info.boresight_I = [];

    info.dropoutReason = "unavailable";

end

function halfAngleRad = getFOVHalfAngleRad(cfg)
% Read FOV half-angle in radians with safe fallback.

    if isfield(cfg, "fov") && isfield(cfg.fov, "halfAngleRad")
        halfAngleRad = cfg.fov.halfAngleRad;
        return;
    end

    if isfield(cfg, "fov") && isfield(cfg.fov, "halfAngleDeg")
        halfAngleRad = deg2rad(cfg.fov.halfAngleDeg);
        return;
    end

    halfAngleRad = deg2rad(20.0);

end

function boresight_I = getFOVBoresightInertial(losUnit_I, watcherState, cfg)
%GETFOVBORESIGHTINERTIAL Get camera boresight in inertial coordinates.
%
% Priority:
%   1. watcherState.boresight_I
%   2. watcherState.boreSight_I
%   3. cfg.fov.boresightMode
%
% Supported cfg.fov.boresightMode:
%   "target_pointing"
%       Boresight is aligned with the current true line of sight.
%       This is useful for debugging because it disables cone dropout.
%
%   "reference_pointing"
%       Boresight points from watcher position to cfg.fov.referencePoint_I.
%       This is the first realistic FOV mode because it does not use the
%       true target LOS as the camera direction.
%
%   "inertial_fixed"
%       Boresight is a fixed inertial vector cfg.fov.boresightInertial.
%
% Default:
%   "target_pointing"

    if isfield(watcherState, 'boresight_I')
        boresight_I = watcherState.boresight_I(:);
        return;
    end

    if isfield(watcherState, 'boreSight_I')
        boresight_I = watcherState.boreSight_I(:);
        return;
    end

    boresightMode = "target_pointing";

    if isfield(cfg, 'fov') && isfield(cfg.fov, 'boresightMode')
        boresightMode = string(cfg.fov.boresightMode);
    end

    switch boresightMode

        case "target_pointing"

            % Debug/safe mode:
            % Camera points exactly along the true LOS.
            boresight_I = losUnit_I(:);

        case "reference_pointing"

            if ~isfield(watcherState, 'r')
                error("evaluateFOVAvailability:MissingWatcherPosition", ...
                    "reference_pointing mode requires watcherState.r.");
            end

            rWatcher = watcherState.r(:);
            rRef = getVectorFieldNested(cfg, 'fov', 'referencePoint_I', zeros(size(rWatcher)));

            refRel = rRef(:) - rWatcher(:);
            refRange = norm(refRel);

            if refRange <= eps
                error("evaluateFOVAvailability:InvalidReferencePoint", ...
                    "reference_pointing boresight is undefined because watcherState.r equals cfg.fov.referencePoint_I.");
            end

            % Camera points toward the nominal reference point, not
            % necessarily toward the true target.
            boresight_I = refRel / refRange;

        case "inertial_fixed"

            if isfield(cfg, 'fov') && isfield(cfg.fov, 'boresightInertial')
                boresight_I = cfg.fov.boresightInertial(:);
            else
                boresight_I = zeros(numel(losUnit_I), 1);
                boresight_I(1) = 1.0;
            end

        otherwise

            error("evaluateFOVAvailability:UnsupportedBoresightMode", ...
                "Unsupported cfg.fov.boresightMode: %s", char(boresightMode));

    end

end

function value = getNumericFieldNested(s, parentName, childName, defaultValue)
% Safely read s.(parentName).(childName), otherwise use defaultValue.

    value = defaultValue;

    if isfield(s, parentName)
        parent = s.(parentName);

        if isfield(parent, childName)
            value = parent.(childName);
        end
    end

end


function value = getVectorFieldNested(s, parentName, childName, defaultValue)
% GETVECTORFIELDNESTED Safely read a nested vector field.
%
% Example:
%   value = getVectorFieldNested(cfg, 'fov', 'referencePoint_I', zeros(3,1));

    value = defaultValue;

    if isfield(s, parentName)
        parent = s.(parentName);

        if isfield(parent, childName)
            candidate = parent.(childName);

            if ~isempty(candidate)
                value = candidate(:);
            end
        end
    end

end