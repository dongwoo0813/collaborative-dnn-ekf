function [watcher, geomInfo] = updateWatcherOmegaBarFromMeasurement(watcher, z, tMeas, cfg)
%{
File:
    geometry/updateWatcherOmegaBarFromMeasurement.m

Purpose:
    Update one watcher's local cumulative bearing-geometry support after a
    valid bearing measurement has actually been used by the EKF update.

    This is Step 09-J.1 local metadata only. It does not change the EKF state
    vector xhat, the covariance P, GS upload, GS broadcast, or the composite
    residual mode.

Inputs:
    watcher
        Local DNN-EKF watcher structure. This function reads/writes:
            watcher.OmegaBar
            watcher.numOmegaUpdates
            watcher.lastLOSUnit
            watcher.lastOmegaUpdateTime
            watcher.lastOmegaUpdate

    z
        Available bearing measurement used in the EKF update.

    tMeas
        Time associated with this measurement update.

    cfg
        Simulation configuration.
        Optional fields:
            cfg.gs.fimGate.lambdaOmega
            cfg.gs.fimGate.normalizeTrace

Outputs:
    watcher
        Same watcher with updated OmegaBar diagnostics.

    geomInfo
        Diagnostic structure returned by updateOmegaBar plus measurement time
        and update count.

Notes:
    - cfg.gs.fimGate.enabled is intentionally not required here. Step 09-J.1
      only collects passive geometry metadata. The enabled flag can be used
      later when compositeMode = "bearing_fim_gated" is implemented.
%}

    geomInfo = struct();
    geomInfo.updated = false;
    geomInfo.reason = "not_updated";

    if ~isfield(cfg, "meas") || ~isfield(cfg.meas, "type")
        geomInfo.reason = "not_bearing_measurement";
        return;
    end

    % A range+bearing packet carries the same angular information used by
    % the direction-only FIM gate. Strip the leading range component before
    % converting the measurement to an LOS unit vector.
    measType = string(cfg.meas.type);
    switch measType
        case "bearing"
            zBearing = z;
        case "range_bearing"
            z = z(:);
            expectedLength = 1 + max(cfg.dim - 1, 1);
            if numel(z) ~= expectedLength
                error("updateWatcherOmegaBarFromMeasurement:BadRangeBearingSize", ...
                    "range_bearing measurement must have length %d for cfg.dim=%d.", ...
                    expectedLength, cfg.dim);
            end
            zBearing = z(2:end);
        otherwise
            geomInfo.reason = "not_bearing_measurement";
            return;
    end

    dim = cfg.dim;

    if ~isfield(watcher, "OmegaBar") || isempty(watcher.OmegaBar)
        watcher.OmegaBar = zeros(dim, dim);
    end

    if ~isfield(watcher, "numOmegaUpdates") || isempty(watcher.numOmegaUpdates)
        watcher.numOmegaUpdates = 0;
    end

    if ~isfield(watcher, "lastLOSUnit") || isempty(watcher.lastLOSUnit)
        watcher.lastLOSUnit = NaN(dim, 1);
    end

    lambdaOmega = getFimGateNumericField(cfg, "lambdaOmega", 0.02);
    normalizeTrace = getFimGateLogicalField(cfg, "normalizeTrace", false);
    accumulationMode = getFimGateStringField( ...
        cfg,"accumulationMode","ema");

    u = bearingMeasurementToLOSUnit(zBearing, cfg);

    [OmegaBarNew, updateInfo] = updateOmegaBar( ...
        watcher.OmegaBar,u,lambdaOmega,normalizeTrace,accumulationMode);

    watcher.OmegaBar = OmegaBarNew;
    watcher.numOmegaUpdates = watcher.numOmegaUpdates + 1;
    watcher.lastLOSUnit = u;
    watcher.lastOmegaUpdateTime = tMeas;

    geomInfo = updateInfo;
    geomInfo.updated = true;
    geomInfo.reason = "bearing_measurement_used";
    geomInfo.time = tMeas;
    geomInfo.numOmegaUpdates = watcher.numOmegaUpdates;

    % Keep the latest diagnostics on the watcher for later GS payload checks.
    watcher.lastOmegaUpdate = geomInfo;

end

function val=getFimGateStringField(cfg,fieldName,defaultVal)
    val=string(defaultVal);
    if isfield(cfg,"gs") && isfield(cfg.gs,"fimGate") && ...
            isfield(cfg.gs.fimGate,fieldName) && ...
            ~isempty(cfg.gs.fimGate.(fieldName))
        val=string(cfg.gs.fimGate.(fieldName));
    end
end

function val = getFimGateNumericField(cfg, fieldName, defaultVal)
%GETFIMGATENUMERICFIELD Safe numeric reader for cfg.gs.fimGate.*.

    val = defaultVal;

    if ~isfield(cfg, "gs") || ~isfield(cfg.gs, "fimGate")
        return;
    end

    if isfield(cfg.gs.fimGate, fieldName)
        candidate = cfg.gs.fimGate.(fieldName);
        if isnumeric(candidate) && isscalar(candidate) && isfinite(candidate)
            val = candidate;
        end
    end

end

function val = getFimGateLogicalField(cfg, fieldName, defaultVal)
%GETFIMGATELOGICALFIELD Safe logical reader for cfg.gs.fimGate.*.

    val = logical(defaultVal);

    if ~isfield(cfg, "gs") || ~isfield(cfg.gs, "fimGate")
        return;
    end

    if isfield(cfg.gs.fimGate, fieldName)
        val = logical(cfg.gs.fimGate.(fieldName));
    end

end
