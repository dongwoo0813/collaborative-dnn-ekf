function out = run_step08_FOV_GS_DNN_EKF()
%{
File:
    scripts/run_step08_FOV_GS_DNN_EKF.m

Purpose:
    Run GS DNN-EKF with FOV-based intermittent measurements.

    This is the first standard run script for Step 08 FOV mode.
    It starts from the validated Step 04 GS config, then activates FOV
    measurement availability inside this script.

What this script checks:
    1. FOV mode runs over the normal simulation horizon.
    2. Measurement availability is intermittent.
    3. Dropout reasons are logged.
    4. GS event/upload logic respects measurement availability.
    5. Position error can be plotted under FOV dropout.

Notes:
    This script does not create a separate config_step08 file. The base
    config remains config_step04_GS_DNN_EKF.m.
%}

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 08-B.1: FOV-enabled GS DNN-EKF run\n");
    fprintf("============================================================\n\n");

    % ---------------------------------------------------------------------
    % Base validated GS config
    % ---------------------------------------------------------------------
    cfg = config_step04_GS_DNN_EKF();

    % ---------------------------------------------------------------------
    % Activate FOV scenario locally in this run script
    % ---------------------------------------------------------------------
    cfg.step.name = "step08_FOV_GS_DNN_EKF";

    cfg.meas.availabilityMode = "fov";
    cfg.fov.enabled = true;
    cfg.fov.guardUnimplementedMode = false;

    % Cone-based dropout scenario.
    cfg.fov.boresightMode = "inertial_fixed";

    cfg.fov.boresightInertial = zeros(cfg.dim,1);
    cfg.fov.boresightInertial(1) = 1.0;

    cfg.fov.halfAngleDeg = 20.0;
    cfg.fov.halfAngleRad = deg2rad(cfg.fov.halfAngleDeg);

    % Disable range gating so unavailable measurements come from FOV cone.
    cfg.fov.rhoMin = 0.0;
    cfg.fov.rhoMax = Inf;

    % Keep validated GS communication path.
    cfg.gs.uploadMode = "after_measurement_update";
    cfg.gs.broadcastMode = "every_step";

    % ---------------------------------------------------------------------
    % Run simulation
    % ---------------------------------------------------------------------
    rng(100);
    res = simulate_GS_DNN_EKF(cfg);

    % ---------------------------------------------------------------------
    % Position error summary
    % ---------------------------------------------------------------------
    errPos = computePositionErrorNorm_step08(res, cfg);

    meanPosErr = mean(errPos, "omitnan");
    finalPosErr = errPos(end);

    fprintf("Position error summary:\n");
    fprintf("  Mean position error  = %.6f m\n", meanPosErr);
    fprintf("  Final position error = %.6f m\n\n", finalPosErr);

    % ---------------------------------------------------------------------
    % Measurement availability summary
    % ---------------------------------------------------------------------
    activeRows = 2:(cfg.N-1);

    measAvail = logical(res.measAvail(activeRows,:));
    numAvailable = nnz(measAvail);
    numTotal = numel(measAvail);

    fprintf("Measurement availability:\n");
    fprintf("  Available entries = %d / %d\n", numAvailable, numTotal);
    fprintf("  Availability rate = %.3f %%\n\n", 100*numAvailable/numTotal);

    % ---------------------------------------------------------------------
    % Dropout reason summary
    % ---------------------------------------------------------------------
    reasons = string(res.measurementDropoutReason(activeRows,:));
    reasonList = unique(reasons(:));

    fprintf("Dropout reason summary:\n");

    for idx = 1:numel(reasonList)
        reason = reasonList(idx);
        count = nnz(reasons == reason);

        fprintf("  %-18s : %d\n", reason, count);
    end

    fprintf("\n");

    % ---------------------------------------------------------------------
    % GS upload summary
    % ---------------------------------------------------------------------
    fprintf("GS upload summary:\n");

    if isfield(res, "gsUploadCount")
        fprintf("  Final GS upload count = %d\n", res.gsUploadCount(end));
    end

    if isfield(res, "gsUploadDecision")
        fprintf("  Logged GS upload decisions = %d\n", nnz(res.gsUploadDecision));
    end

    fprintf("\n");

    % ---------------------------------------------------------------------
    % Plot
    % ---------------------------------------------------------------------
    figure;
    plot(cfg.time(:), errPos(:), "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Position error norm [m]");
    title("Step 08-B.1 FOV-enabled GS DNN-EKF");

    % Optional measurement availability plot.
    figure;
    plot(cfg.time(:), sum(logical(res.measAvail), 2), "LineWidth", 1.5);
    grid on;
    xlabel("Time [s]");
    ylabel("Number of watchers with measurement");
    title("FOV measurement availability count");

    % ---------------------------------------------------------------------
    % Output
    % ---------------------------------------------------------------------
    out = struct();
    out.cfg = cfg;
    out.res = res;
    out.errPos = errPos;

    out.meanPosErr = meanPosErr;
    out.finalPosErr = finalPosErr;

    out.numMeasurementAvailable = numAvailable;
    out.numMeasurementTotal = numTotal;
    out.measurementAvailabilityRate = numAvailable / numTotal;

    out.uniqueReasons = reasonList;

    fprintf("============================================================\n");
    fprintf("Step 08-B.1 run complete.\n");
    fprintf("============================================================\n\n");

end

function errPos = computePositionErrorNorm_step08(res, cfg)
%COMPUTEPOSITIONERRORNORM_STEP08 Compute mean watcher position error.
%
% This helper is intentionally robust to different result storage formats:
%
%   etaTrue:
%       2*dim x Nt
%       Nt x 2*dim
%
%   estimate:
%       nx x Nt
%       Nt x nx
%       nx x Nt x Nw
%       Nt x nx x Nw
%
% Output:
%   errPos:
%       NtPlot x 1 position-error norm averaged over watchers when a
%       watcher dimension exists.
%
% Notes:
%   Only the first dim rows/components are used as target position.

    dim = cfg.dim;
    NtPlot = numel(cfg.time);

    % ------------------------------------------------------------
    % Read truth state.
    % ------------------------------------------------------------
    if isfield(res, "etaTrue")
        etaTrue = res.etaTrue;
    elseif isfield(res, "xTrue")
        etaTrue = res.xTrue;
    else
        error("run_step08:MissingTruth", ...
            "Could not find etaTrue or xTrue in results.");
    end

    etaTrue = orientStateTimeArray_step08(etaTrue, 2*dim);

    rTrue = etaTrue(1:dim,:);
    NtTruth = size(rTrue, 2);

    % ------------------------------------------------------------
    % Read estimate state.
    % ------------------------------------------------------------
    if isfield(res, "xhat")
        xhat = res.xhat;
    elseif isfield(res, "xhatAug")
        xhat = res.xhatAug;
    elseif isfield(res, "etaHat")
        xhat = res.etaHat;
    else
        error("run_step08:MissingEstimate", ...
            "Could not find xhat, xhatAug, or etaHat in results.");
    end

    % ------------------------------------------------------------
    % Compute position error.
    % ------------------------------------------------------------
    if ndims(xhat) == 2

        xhat = orientStateTimeArray_step08(xhat, dim);
        rHat = xhat(1:dim,:);

        NtCommon = min(size(rHat,2), NtTruth);

        errValid = vecnorm(rHat(:,1:NtCommon) - rTrue(:,1:NtCommon), 2, 1).';

    elseif ndims(xhat) == 3

        xhat = orientStateTimeWatcherArray_step08(xhat, dim);

        % After orientation:
        %   xhat = nx x Nt x Nw
        rHat = xhat(1:dim,:,:);

        NtCommon = min(size(rHat,2), NtTruth);
        Nw = size(rHat,3);

        errByWatcher = NaN(NtCommon, Nw);

        for iw = 1:Nw
            rHat_i = rHat(:,1:NtCommon,iw);
            rTrue_i = rTrue(:,1:NtCommon);

            errByWatcher(:,iw) = vecnorm(rHat_i - rTrue_i, 2, 1).';
        end

        % One plot line: mean position error over watchers.
        errValid = mean(errByWatcher, 2, "omitnan");

    else

        error("run_step08:UnsupportedEstimateShape", ...
            "Unsupported estimate array dimension: ndims(xhat) = %d.", ndims(xhat));

    end

    % ------------------------------------------------------------
    % Return vector compatible with cfg.time for plotting.
    % ------------------------------------------------------------
    errPos = NaN(NtPlot, 1);

    NtCopy = min(NtPlot, numel(errValid));
    errPos(1:NtCopy) = errValid(1:NtCopy);

end

function X = orientStateTimeArray_step08(X, minStateDim)
%ORIENTSTATETIMEARRAY_STEP08 Convert a 2D array to state x time.
%
% If X is time x state, transpose it. If X is already state x time, leave it.

    if ndims(X) ~= 2
        error("run_step08:Expected2DArray", ...
            "Expected a 2D state-time array.");
    end

    nRow = size(X,1);
    nCol = size(X,2);

    % If rows are too small to contain the state, but columns can, transpose.
    if nRow < minStateDim && nCol >= minStateDim
        X = X.';
        return;
    end

    % If both dimensions could contain states, prefer the common convention
    % state x time. Do nothing.
end

function X = orientStateTimeWatcherArray_step08(X, minStateDim)
%ORIENTSTATETIMEWATCHERARRAY_STEP08 Convert 3D estimate to state x time x watcher.
%
% Supported input conventions:
%   state x time x watcher
%   time x state x watcher

    if ndims(X) ~= 3
        error("run_step08:Expected3DArray", ...
            "Expected a 3D state-time-watcher array.");
    end

    n1 = size(X,1);
    n2 = size(X,2);

    if n1 >= minStateDim
        % Already state x time x watcher.
        return;
    end

    if n2 >= minStateDim
        % Convert time x state x watcher -> state x time x watcher.
        X = permute(X, [2 1 3]);
        return;
    end

    error("run_step08:Invalid3DEstimateShape", ...
        "Could not infer state dimension from estimate size [%s].", ...
        num2str(size(X)));
end
