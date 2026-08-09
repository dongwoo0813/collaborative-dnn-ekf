function [z, available, info] = measurementModel(etaTrue, watcherState, cfg, tMeas)
%{
File:
    measurement/measurementModel.m

Purpose:
    Generate a watcher measurement if the measurement is available.

    This function separates two operations:
        1. Check measurement availability.
        2. Generate the actual measurement.

    In always-available mode, availability is always true.

    In FOV mode, availability is determined by fovAvailable.m, which may
    also return diagnostic information such as range, off-boresight angle,
    and dropout reason.

Inputs:
    etaTrue
        True target physical state eta = [r_t; v_t].

    watcherState
        Watcher truth/state structure.

    cfg
        Simulation configuration.

Outputs:
    z
        Measurement vector. Empty if measurement is unavailable.

    available
        Logical scalar indicating whether a valid measurement exists.

    info
        Diagnostic structure. In FOV mode, this includes fields such as:
            info.dropoutReason
            info.range
            info.insideFOV
            info.offBoresightAngleDeg

Notes:
    Measurement availability is not communication triggering. If available
    is false, the EKF should perform prediction only.
%}

    if nargin < 4
        tMeas = 0.0;
    end

    % -------------------------------------------------------------------------
    % Input-shape guard
    % -------------------------------------------------------------------------
    % etaTrue must be one physical state vector:
    %
    %   etaTrue = [r_t; v_t]
    %
    % with size 2*cfg.dim by 1.
    %
    % Passing the full etaTrue history matrix is a serious bug because the
    % measurement would be generated from the wrong state entries.
    if ~isvector(etaTrue) || numel(etaTrue) ~= 2*cfg.dim
        error("measurementModel:InvalidEtaTrueShape", ...
            "etaTrue must be a vector of length 2*cfg.dim. Got size [%s].", ...
            num2str(size(etaTrue)));
    end
    
    etaTrue = etaTrue(:);


    % Check whether a measurement is available and preserve diagnostics.
    [available, info] = fovAvailable(etaTrue, watcherState, cfg);

    if ~available
        z = [];

        % Keep diagnostic info even when no measurement is generated.
        info.available = false;
        return;
    end

    switch cfg.meas.type

        case "bearing"

            z = bearingMeasurement(etaTrue, watcherState, cfg, tMeas);

        case "range_bearing"

            zhatTrue = measurementPrediction(etaTrue, watcherState, cfg);
            if cfg.dim == 2
                noise = [cfg.meas.sigmaRange * randn; ...
                         cfg.meas.sigmaBearing * randn];
            else
                noise = [cfg.meas.sigmaRange * randn; ...
                         cfg.meas.sigmaBearing * randn(2,1)];
            end
            z = zhatTrue + noise;

        case "relative_position"

            zhatTrue = measurementPrediction(etaTrue, watcherState, cfg);
            z = zhatTrue + cfg.meas.sigmaPosition * randn(cfg.dim,1);

        case "direct_residual"

            % Oracle learning control: every local branch is taught an
            % equal 1/Nw share, so their additive GS sum targets d_true.
            targetScale = 1 / max(cfg.Nw, 1);
            if isfield(cfg.meas, "directResidualTargetScale")
                targetScale = cfg.meas.directResidualTargetScale;
            end
            dTrue = trueResidual(tMeas, etaTrue, cfg);
            z = targetScale * dTrue + ...
                cfg.meas.sigmaDirectResidual * randn(cfg.dim,1);

        otherwise

            error("measurementModel:UnknownMeasurementType", ...
                "Unknown measurement type: %s", string(cfg.meas.type));

    end

    info.available = true;
    info.measurementType = string(cfg.meas.type);

end
