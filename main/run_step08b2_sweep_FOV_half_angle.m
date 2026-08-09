function out = run_step08b2_sweep_FOV_half_angle(halfAngleListDeg)
%{
File:
    scripts/run_step08b2_sweep_FOV_half_angle.m

Purpose:
    Sweep FOV half-angle for the Step 08 FOV-enabled GS DNN-EKF scenario.

    The goal is to choose a useful FOV severity level for later comparison:
        Local DNN vs GS composite vs Oracle under intermittent measurements.

What is swept:
    cfg.fov.halfAngleDeg

Fixed FOV scenario:
    cfg.meas.availabilityMode = "fov";
    cfg.fov.enabled = true;
    cfg.fov.guardUnimplementedMode = false;
    cfg.fov.boresightMode = "inertial_fixed";
    cfg.fov.boresightInertial = e1;
    cfg.fov.rhoMin = 0;
    cfg.fov.rhoMax = Inf;

Metrics:
    - measurement availability rate
    - available count
    - outside_fov count
    - mean position error
    - final position error
    - GS upload decisions

Usage:
    out = run_step08b2_sweep_FOV_half_angle();

    or

    out = run_step08b2_sweep_FOV_half_angle([1 2 5 10 20 40 90]);
%}

    if nargin < 1 || isempty(halfAngleListDeg)
        halfAngleListDeg = [1 2 5 10 20 40 90];
    end

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Step 08-B.2: FOV half-angle severity sweep\n");
    fprintf("============================================================\n\n");

    nCase = numel(halfAngleListDeg);

    meanPosErr = NaN(nCase,1);
    finalPosErr = NaN(nCase,1);

    availableCount = NaN(nCase,1);
    totalCount = NaN(nCase,1);
    availabilityRate = NaN(nCase,1);

    outsideFOVCount = NaN(nCase,1);
    rangeTooSmallCount = NaN(nCase,1);
    rangeTooLargeCount = NaN(nCase,1);

    gsUploadDecisionCount = NaN(nCase,1);

    resultsCell = cell(nCase,1);
    cfgCell = cell(nCase,1);

    for ic = 1:nCase

        halfAngleDeg = halfAngleListDeg(ic);

        cfg = config_step04_GS_DNN_EKF();

        % ------------------------------------------------------------
        % Activate FOV mode locally.
        % ------------------------------------------------------------
        cfg.step.name = "step08b2_FOV_half_angle_sweep";

        cfg.meas.availabilityMode = "fov";
        cfg.fov.enabled = true;
        cfg.fov.guardUnimplementedMode = false;

        % Fixed inertial boresight for repeatable cone-based dropout.
        cfg.fov.boresightMode = "inertial_fixed";
        cfg.fov.boresightInertial = zeros(cfg.dim,1);
        cfg.fov.boresightInertial(1) = 1.0;

        cfg.fov.halfAngleDeg = halfAngleDeg;
        cfg.fov.halfAngleRad = deg2rad(cfg.fov.halfAngleDeg);

        % Disable range gating so dropout reason is mainly outside_fov.
        cfg.fov.rhoMin = 0.0;
        cfg.fov.rhoMax = Inf;

        % Keep validated GS communication policy.
        cfg.gs.uploadMode = "after_measurement_update";
        cfg.gs.broadcastMode = "every_step";

        fprintf("Running FOV half-angle = %.3f deg ...\n", halfAngleDeg);

        rng(100);
        res = simulate_GS_DNN_EKF(cfg);

        activeRows = 2:(cfg.N-1);

        measAvail = logical(res.measAvail(activeRows,:));
        reasons = string(res.measurementDropoutReason(activeRows,:));

        availableCount(ic) = nnz(measAvail);
        totalCount(ic) = numel(measAvail);
        availabilityRate(ic) = availableCount(ic) / totalCount(ic);

        outsideFOVCount(ic) = nnz(reasons == "outside_fov");
        rangeTooSmallCount(ic) = nnz(reasons == "range_too_small");
        rangeTooLargeCount(ic) = nnz(reasons == "range_too_large");

        if isfield(res, "gsUploadDecision")
            gsUploadDecisionCount(ic) = nnz(logical(res.gsUploadDecision(activeRows,:)));
        end

        errPos = computePositionErrorNorm_step08b2(res, cfg);

        meanPosErr(ic) = mean(errPos, "omitnan");
        finalPosErr(ic) = errPos(end);

        resultsCell{ic} = res;
        cfgCell{ic} = cfg;

        fprintf("  availability rate = %.3f %%\n", 100*availabilityRate(ic));
        fprintf("  mean pos error    = %.6f m\n", meanPosErr(ic));
        fprintf("  final pos error   = %.6f m\n\n", finalPosErr(ic));

    end

    summaryTable = table( ...
        halfAngleListDeg(:), ...
        availableCount, ...
        totalCount, ...
        100*availabilityRate, ...
        outsideFOVCount, ...
        rangeTooSmallCount, ...
        rangeTooLargeCount, ...
        gsUploadDecisionCount, ...
        meanPosErr, ...
        finalPosErr, ...
        'VariableNames', { ...
            'halfAngleDeg', ...
            'availableCount', ...
            'totalCount', ...
            'availabilityRatePercent', ...
            'outsideFOVCount', ...
            'rangeTooSmallCount', ...
            'rangeTooLargeCount', ...
            'gsUploadDecisionCount', ...
            'meanPosErr_m', ...
            'finalPosErr_m'});

    fprintf("Summary table:\n");
    disp(summaryTable);

    % ------------------------------------------------------------
    % Plot availability rate.
    % ------------------------------------------------------------
    figure;
    plot(halfAngleListDeg(:), 100*availabilityRate(:), "-o", "LineWidth", 1.5);
    grid on;
    xlabel("FOV half-angle [deg]");
    ylabel("Measurement availability rate [%]");
    title("Step 08-B.2 FOV severity sweep: availability");

    % ------------------------------------------------------------
    % Plot mean position error.
    % ------------------------------------------------------------
    figure;
    plot(halfAngleListDeg(:), meanPosErr(:), "-o", "LineWidth", 1.5);
    grid on;
    xlabel("FOV half-angle [deg]");
    ylabel("Mean position error [m]");
    title("Step 08-B.2 FOV severity sweep: mean position error");

    out = struct();
    out.halfAngleListDeg = halfAngleListDeg(:);
    out.summaryTable = summaryTable;
    out.cfgCell = cfgCell;
    out.resultsCell = resultsCell;

    fprintf("============================================================\n");
    fprintf("Step 08-B.2 FOV half-angle sweep complete.\n");
    fprintf("============================================================\n\n");

end

function errPos = computePositionErrorNorm_step08b2(res, cfg)
%COMPUTEPOSITIONERRORNORM_STEP08B2 Compute mean watcher position error.
%
% Supports:
%   truth:
%       etaTrue or xTrue
%
%   estimate:
%       xhat, xhatAug, or etaHat
%
% Handles 2D or 3D estimate arrays. If estimate has watcher dimension, this
% returns the mean position error over watchers.

    dim = cfg.dim;
    NtPlot = numel(cfg.time);

    if isfield(res, "etaTrue")
        etaTrue = res.etaTrue;
    elseif isfield(res, "xTrue")
        etaTrue = res.xTrue;
    else
        error("step08b2:MissingTruth", ...
            "Could not find etaTrue or xTrue in results.");
    end

    etaTrue = orientStateTimeArray_step08b2(etaTrue, 2*dim);
    rTrue = etaTrue(1:dim,:);
    NtTruth = size(rTrue, 2);

    if isfield(res, "xhat")
        xhat = res.xhat;
    elseif isfield(res, "xhatAug")
        xhat = res.xhatAug;
    elseif isfield(res, "etaHat")
        xhat = res.etaHat;
    else
        error("step08b2:MissingEstimate", ...
            "Could not find xhat, xhatAug, or etaHat in results.");
    end

    if ndims(xhat) == 2

        xhat = orientStateTimeArray_step08b2(xhat, dim);
        rHat = xhat(1:dim,:);

        NtCommon = min(size(rHat,2), NtTruth);
        errValid = vecnorm(rHat(:,1:NtCommon) - rTrue(:,1:NtCommon), 2, 1).';

    elseif ndims(xhat) == 3

        xhat = orientStateTimeWatcherArray_step08b2(xhat, dim);

        rHat = xhat(1:dim,:,:);

        NtCommon = min(size(rHat,2), NtTruth);
        Nw = size(rHat,3);

        errByWatcher = NaN(NtCommon, Nw);

        for iw = 1:Nw
            rHat_i = rHat(:,1:NtCommon,iw);
            rTrue_i = rTrue(:,1:NtCommon);

            errByWatcher(:,iw) = vecnorm(rHat_i - rTrue_i, 2, 1).';
        end

        errValid = mean(errByWatcher, 2, "omitnan");

    else

        error("step08b2:UnsupportedEstimateShape", ...
            "Unsupported estimate array dimension: ndims(xhat) = %d.", ndims(xhat));

    end

    errPos = NaN(NtPlot, 1);

    NtCopy = min(NtPlot, numel(errValid));
    errPos(1:NtCopy) = errValid(1:NtCopy);

end

function X = orientStateTimeArray_step08b2(X, minStateDim)
%ORIENTSTATETIMEARRAY_STEP08B2 Convert a 2D array to state x time.

    if ndims(X) ~= 2
        error("step08b2:Expected2DArray", ...
            "Expected a 2D state-time array.");
    end

    nRow = size(X,1);
    nCol = size(X,2);

    if nRow < minStateDim && nCol >= minStateDim
        X = X.';
    end

end

function X = orientStateTimeWatcherArray_step08b2(X, minStateDim)
%ORIENTSTATETIMEWATCHERARRAY_STEP08B2 Convert 3D estimate to state x time x watcher.

    if ndims(X) ~= 3
        error("step08b2:Expected3DArray", ...
            "Expected a 3D state-time-watcher array.");
    end

    n1 = size(X,1);
    n2 = size(X,2);

    if n1 >= minStateDim
        return;
    end

    if n2 >= minStateDim
        X = permute(X, [2 1 3]);
        return;
    end

    error("step08b2:Invalid3DEstimateShape", ...
        "Could not infer state dimension from estimate size [%s].", ...
        num2str(size(X)));

end