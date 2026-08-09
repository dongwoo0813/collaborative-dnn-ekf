function results = simulateLocalDNNEKF(cfg)
%{
Function:
    simulateLocalDNNEKF.m

Purpose:
    Run Step 03 local DNN-EKF simulation.

    Each watcher estimates its own augmented state

        X_i = [eta_i; theta_i],

    where

        eta_i = [r_t; v_t]

    and theta_i is the local fixed-feature output-layer DNN branch
    parameter vector.

    The measurement still depends only on eta_i, so DNNEKFUpdateLocal.m
    uses

        H_X = [H_eta, zeros(nz,nTheta_i)].

    Therefore theta_i is updated only through the predicted cross covariance
    P_{theta eta}.

Additional Step 09-D.3 diagnostic logging:
    This version logs two residual estimates:

        results.dnnResidual
            d_hat(eta_hat), the residual actually used by the EKF.

        results.dnnResidualAtTrueEta
            d_hat(eta_true), the same learned DNN evaluated at the true
            state input.

    These two logs allow comparison of

        || d_hat(eta_hat)  - d_true(eta_true) ||
        || d_hat(eta_true) - d_true(eta_true) ||

    so we can separate operational EKF correction quality from pure
    residual-function approximation quality.

Inputs:
    cfg - configuration from config_step03_local_DNN_EKF.m

Outputs:
    results - simulation log structure

Notes:
    - No GS sharing.
    - No peer-to-peer sharing.
    - Watcher motion is still prescribed unless cfg changes it.
    - The truth dynamics may include trueResidual, but the estimator only
      uses its local DNN branch approximation.
%}

    dim     = cfg.dim;
    Nw      = cfg.Nw;
    N       = cfg.N;
    time    = cfg.time;
    replayEnabled = isfield(cfg,"replay") && isfield(cfg.replay,"enabled") && logical(cfg.replay.enabled);
    if replayEnabled
        validateReplayTrajectoryLocal(cfg.replay,2*dim,dim,N,Nw);
    end

    % ---------------------------------------------------------------------
    % True target state
    % ---------------------------------------------------------------------
    etaTrue = zeros(2*dim, N);
    etaTrue(:,1) = [cfg.target.r0; cfg.target.v0];

    trueResidualLog = zeros(dim, N);
    trueResidualLog(:,1) = computeTrueResidualForLog(time(1), etaTrue(:,1), cfg);

    % ---------------------------------------------------------------------
    % Watcher truth states
    % ---------------------------------------------------------------------
    watcherTruth = initWatcherTruthArray(cfg);

    % ---------------------------------------------------------------------
    % Initialize local DNN-EKFs
    % ---------------------------------------------------------------------
    watchers = initLocalDNNEKF(1, etaTrue(:,1), cfg);

    for i = 2:Nw
        watchers(i) = initLocalDNNEKF(i, etaTrue(:,1), cfg);
    end

    watchers = watchers(:);

    nEta = watchers(1).nEta;
    nTheta = watchers(1).nTheta;
    nX = watchers(1).nX;

    % ---------------------------------------------------------------------
    % Log allocation
    % ---------------------------------------------------------------------
    etaHatLog = zeros(nEta, N, Nw);
    xhatAugLog = zeros(nX, N, Nw);
    thetaHatLog = zeros(nTheta, N, Nw);

    PdiagLog = zeros(nX, N, Nw);
    PdiagEtaLog = zeros(nEta, N, Nw);
    PdiagThetaLog = zeros(nTheta, N, Nw);

    % Operational DNN residual:
    %   d_hat(eta_hat)
    % This is the residual actually used by the EKF prediction.
    dnnResidualLog = zeros(dim, N, Nw);

    % Same-input DNN residual:
    %   d_hat(eta_true)
    % This separates residual-function learning quality from eta_hat input
    % error.
    dnnResidualAtTrueEtaLog = zeros(dim, N, Nw);

    measAvailLog = false(N, Nw);

    % ---------------------------------------------------------------------
    % Innovation / NIS diagnostic logs
    % ---------------------------------------------------------------------
    if cfg.dim == 2
        nz = 1;
    elseif cfg.dim == 3
        nz = 2;
    else
        error("Unsupported cfg.dim. Use cfg.dim = 2 or cfg.dim = 3.");
    end

    innovationLog = NaN(nz, N, Nw);
    SdiagLog = NaN(nz, N, Nw);
    NISLog = NaN(N, Nw);

    % ---------------------------------------------------------------------
    % Adaptive covariance matching logs
    % ---------------------------------------------------------------------
    gammaThetaLog = NaN(N, Nw);
    cmRatioLog = NaN(N, Nw);
    cmTraceEmpLog = NaN(N, Nw);
    cmTraceModelLog = NaN(N, Nw);

    watcherRLog = zeros(dim, N, Nw);
    watcherVLog = zeros(dim, N, Nw);
    watcherULog = zeros(dim, N, Nw);
    watcherTauLog = zeros(3, N, Nw);
    watcherQLog = zeros(4, N, Nw);
    watcherOmegaLog = zeros(3, N, Nw);

    % Live observability-impulse controller telemetry.  This mirrors the
    % GS simulator so a local branch can participate in a fair closed-loop
    % maneuver comparison.
    controllerActiveLog = false(N,Nw);
    predictedRadialVarianceLog = NaN(N,Nw);
    cumulativeImpulseLog = zeros(N,Nw);
    cumulativeDeltaVLog = zeros(N,Nw);

    % ---------------------------------------------------------------------
    % Initial logs
    % ---------------------------------------------------------------------
    for i = 1:Nw

        idxEta = watchers(i).idxEta;
        idxTheta = watchers(i).idxTheta;

        etaHatLog(:,1,i) = watchers(i).xhat(idxEta);
        xhatAugLog(:,1,i) = watchers(i).xhat;
        thetaHatLog(:,1,i) = watchers(i).xhat(idxTheta);

        PdiagLog(:,1,i) = diag(watchers(i).P);
        PdiagEtaLog(:,1,i) = diag(watchers(i).P(idxEta, idxEta));
        PdiagThetaLog(:,1,i) = diag(watchers(i).P(idxTheta, idxTheta));

        % Operational residual evaluated at eta_hat.
        dnnResidualLog(:,1,i) = computeLocalDNNResidualForLog( ...
            watchers(i), cfg);

        % Same-input residual evaluated at eta_true(:,1).
        dnnResidualAtTrueEtaLog(:,1,i) = computeLocalDNNResidualForLog( ...
            watchers(i), cfg, etaTrue(:,1));

        [gammaThetaLog(1,i), cmRatioLog(1,i), ...
            cmTraceEmpLog(1,i), cmTraceModelLog(1,i)] = ...
                getCovMatchingDiagnosticsForLog(watchers(i));

        watcherRLog(:,1,i) = watcherTruth(i).r;
        watcherVLog(:,1,i) = watcherTruth(i).v;
        watcherULog(:,1,i) = watcherTruth(i).u;
        watcherTauLog(:,1,i) = watcherTruth(i).tau;
        watcherQLog(:,1,i) = watcherTruth(i).q;
        watcherOmegaLog(:,1,i) = watcherTruth(i).omega;
        [controllerActiveLog(1,i),predictedRadialVarianceLog(1,i)] = ...
            getLocalControllerTelemetry(watcherTruth(i));

    end

    % ---------------------------------------------------------------------
    % Main simulation loop
    % ---------------------------------------------------------------------
    for k = 1:N-1

        t = time(k);
        tNext = time(k+1);
        logIdx = k + 1;

        % 1. Propagate true target from t_k to t_{k+1}.
        if replayEnabled
            etaTrue(:,logIdx) = cfg.replay.etaTrue(:,logIdx);
        else
            etaTrue(:,logIdx) = propagateRK4( ...
                @(tt,xx) targetTruthDynamics(tt, xx, cfg), ...
                t, etaTrue(:,k), cfg.dt);
        end

        trueResidualLog(:,logIdx) = computeTrueResidualForLog( ...
            tNext, etaTrue(:,logIdx), cfg);

        for i = 1:Nw

            idxEta = watchers(i).idxEta;
            idxTheta = watchers(i).idxTheta;

            % 2. Propagate watcher state to t_{k+1}.
            %
            % Important:
            %   targetInfo.etaHat should be physical eta only, not the full
            %   augmented DNN-EKF state.
            targetInfo.etaHat = watchers(i).xhat(idxEta);
            targetInfo.PEta = watchers(i).P(idxEta,idxEta);
            targetInfo.filter = watchers(i);
            targetInfo.etaTrue = etaTrue(:,k);   % debugging/analysis only

            if replayEnabled
                [watcherTruth(i), watcherCmd] = replayWatcherStepLocal( ...
                    watcherTruth(i),cfg.replay,logIdx,i,dim);
            else
                [watcherTruth(i), watcherCmd] = propagateWatcherStep( ...
                    i, watcherTruth(i), targetInfo, t, cfg);
            end

            watcherState = watcherTruth(i);

            watcherRLog(:,logIdx,i) = watcherState.r;
            watcherVLog(:,logIdx,i) = watcherState.v;
            watcherULog(:,logIdx,i) = watcherCmd.u;
            watcherTauLog(:,logIdx,i) = watcherCmd.tau;
            watcherQLog(:,logIdx,i) = watcherState.q;
            watcherOmegaLog(:,logIdx,i) = watcherState.omega;
            [controllerActiveLog(logIdx,i), ...
                predictedRadialVarianceLog(logIdx,i)] = ...
                getLocalControllerTelemetry(watcherState);
            cumulativeImpulseLog(logIdx,i) = cumulativeImpulseLog(k,i) + ...
                norm(watcherCmd.u)*cfg.dt;
            cumulativeDeltaVLog(logIdx,i) = cumulativeDeltaVLog(k,i) + ...
                norm(watcherCmd.u)/max(watcherState.mass,eps)*cfg.dt;

            % 3. Generate measurement at t_{k+1}.
            [z, available, ~] = measurementModel(etaTrue(:,logIdx), watcherState, cfg, tNext);

            % 4. DNN-EKF prediction from t_k to t_{k+1}.
            watchers(i) = DNN_EKF_Predict_Local(watchers(i), t, cfg);

            % 5. DNN-EKF measurement update at t_{k+1}.
            %
            % DNNEKFUpdateLocal internally builds
            %
            %   H_X = [H_eta, zeros(nz,nTheta)].
            %
            if available
                watchers(i) = DNN_EKF_Update_Local(watchers(i), z, watcherState, cfg);

                % Step 09-J.1: update passive local bearing-geometry support
                % only after the bearing measurement has actually been used by
                % the EKF update. This does not alter xhat or P.
                watchers(i) = updateWatcherOmegaBarFromMeasurement( ...
                    watchers(i), z, tNext, cfg);

                % -------------------------------------------------------------
                % Innovation / NIS logging
                %
                % NIS = nu' inv(S) nu
                %
                % For a consistent EKF, NIS should roughly behave like a
                % chi-square random variable with nz degrees of freedom.
                % -------------------------------------------------------------
                nu = watchers(i).lastInnovation;
                S = watchers(i).lastS;

                innovationLog(:,logIdx,i) = nu;
                SdiagLog(:,logIdx,i) = diag(S);
                NISLog(logIdx,i) = nu' * (S \ nu);
            end

            % 6. Log posterior estimate at t_{k+1}.
            etaHatLog(:,logIdx,i) = watchers(i).xhat(idxEta);
            xhatAugLog(:,logIdx,i) = watchers(i).xhat;
            thetaHatLog(:,logIdx,i) = watchers(i).xhat(idxTheta);

            PdiagLog(:,logIdx,i) = diag(watchers(i).P);
            PdiagEtaLog(:,logIdx,i) = diag(watchers(i).P(idxEta, idxEta));
            PdiagThetaLog(:,logIdx,i) = diag(watchers(i).P(idxTheta, idxTheta));

            % Operational residual evaluated at the filter estimate eta_hat.
            dnnResidualLog(:,logIdx,i) = computeLocalDNNResidualForLog( ...
                watchers(i), cfg);

            % Same-input residual evaluated at the true state eta_true.
            %
            % This does not affect the EKF. It is only a diagnostic log for
            % checking whether the learned residual function itself improves.
            dnnResidualAtTrueEtaLog(:,logIdx,i) = computeLocalDNNResidualForLog( ...
                watchers(i), cfg, etaTrue(:,logIdx));

            [gammaThetaLog(logIdx,i), cmRatioLog(logIdx,i), ...
                cmTraceEmpLog(logIdx,i), cmTraceModelLog(logIdx,i)] = ...
                getCovMatchingDiagnosticsForLog(watchers(i));

            measAvailLog(logIdx,i) = available;

        end
    end

    % ---------------------------------------------------------------------
    % Output structure
    % ---------------------------------------------------------------------
    results.time = time;
    results.etaTrue = etaTrue;

    % Keep this eta-only so existing physical metrics and plots still work.
    results.xhat = etaHatLog;

    % New augmented DNN-EKF logs.
    results.xhatAug = xhatAugLog;
    results.thetaHat = thetaHatLog;

    results.Pdiag = PdiagLog;
    results.PdiagEta = PdiagEtaLog;
    results.PdiagTheta = PdiagThetaLog;

    % Residual logs.
    results.dnnResidual = dnnResidualLog;
    results.dnnResidualAtTrueEta = dnnResidualAtTrueEtaLog;
    results.trueResidual = trueResidualLog;

    results.measAvail = measAvailLog;

    results.watchersFinal = watchers;

    results.watcherR = watcherRLog;
    results.watcherV = watcherVLog;
    results.watcherU = watcherULog;
    results.watcherTau = watcherTauLog;
    results.watcherQ = watcherQLog;
    results.watcherOmega = watcherOmegaLog;
    results.watcherTruthFinal = watcherTruth;
    results.controllerActive = controllerActiveLog;
    results.predictedRadialVariance = predictedRadialVarianceLog;
    results.cumulativeImpulse = cumulativeImpulseLog;
    results.cumulativeDeltaV = cumulativeDeltaVLog;

    results.innovation = innovationLog;
    results.Sdiag = SdiagLog;
    results.NIS = NISLog;

    results.gammaTheta = gammaThetaLog;
    results.cmRatio = cmRatioLog;
    results.cmTraceEmp = cmTraceEmpLog;
    results.cmTraceModel = cmTraceModelLog;

end

function [active,radialVariance] = getLocalControllerTelemetry(watcherState)
%GETLOCALCONTROLLERTELEMETRY Extract common impulse-controller fields.
    active = false;
    radialVariance = NaN;
    if ~isfield(watcherState,"controllerState") || ...
            ~isstruct(watcherState.controllerState)
        return;
    end
    state = watcherState.controllerState;
    if isfield(state,"activeFlag"), active = logical(state.activeFlag); end
    if isfield(state,"predictedRadialVariance")
        radialVariance = double(state.predictedRadialVariance);
    end
end

function validateReplayTrajectoryLocal(replay,nEta,dim,N,Nw)
    if ~isfield(replay,"etaTrue") || ~isfield(replay,"watcherR") || ~isfield(replay,"watcherV") || ...
            ~isequal(size(replay.etaTrue),[nEta N]) || ...
            ~isequal(size(replay.watcherR),[dim N Nw]) || ...
            ~isequal(size(replay.watcherV),[dim N Nw])
        error("simulateLocalDNNEKF:BadReplayTrajectory", ...
            "Replay trajectory dimensions do not match local DNN-EKF configuration.");
    end
end

function [watcherNext,cmd] = replayWatcherStepLocal(watcherCurrent,replay,k,i,dim)
    watcherNext = watcherCurrent;
    watcherNext.r = replay.watcherR(:,k,i);
    watcherNext.v = replay.watcherV(:,k,i);
    cmd.u = zeros(dim,1); cmd.tau = zeros(3,1);
    if isfield(replay,"watcherU")
        cmd.u = replay.watcherU(:,k,i);
    end
    watcherNext.u = cmd.u; watcherNext.tau = cmd.tau;
end

function aUnk = computeTrueResidualForLog(t, eta, cfg)
% Compute the true residual acceleration for logging only.

    dim = cfg.dim;
    aUnk = zeros(dim,1);

    if isfield(cfg, "truth") && isfield(cfg.truth, "useResidual")
        if cfg.truth.useResidual
            aUnk = trueResidual(t, eta, cfg);
        end
    end

end

function dHat = computeLocalDNNResidualForLog(watcher, cfg, etaOverride)
%{
Function:
    computeLocalDNNResidualForLog

Purpose:
    Compute the current local DNN residual estimate for logging only.

Current Step:
    Step 09-H.4 local MLP full-simulation support.

Why this helper uses evaluateBranchResidualModel:
    The old implementation called branchOutput(...), which only supports the
    fixed-feature LIP branch. The full Local MLP simulation needs this logging
    path to use the same branch-model wrapper as prediction.

Inputs:
    watcher      - local DNN-EKF watcher structure
    cfg          - simulation configuration
    etaOverride  - optional physical input eta

Behavior:
    If etaOverride is omitted:
        evaluate d_hat at watcher.xhat(idxEta), i.e. eta_hat.

    If etaOverride is provided:
        evaluate d_hat at etaOverride, e.g. eta_true(:,k).

Notes:
    This helper does not modify the watcher or the EKF. It is used only for
    diagnostic residual logs:
        results.dnnResidual
        results.dnnResidualAtTrueEta
%}

if nargin < 3 || isempty(etaOverride)
    eta = watcher.xhat(watcher.idxEta);
else
    eta = etaOverride;
end

theta = watcher.xhat(watcher.idxTheta);

branchID = watcher.localBranchID;

% Branch-model-aware residual evaluation.
%
% fixed_feature_lip:
%   d_i = W_i phi_i(eta)
%
% mlp_general:
%   d_i = MLP_i(xi_i(eta); theta_i)
%
% Only dHat is needed for logging here; the Jacobians are discarded.
[dHat, ~, ~, ~] = evaluateBranchResidualModel( ...
    branchID, eta, theta, cfg);

end

function [gammaTheta, ratio, traceEmp, traceModel] = getCovMatchingDiagnosticsForLog(watcher)
%{
Function:
    getCovMatchingDiagnosticsForLog

Purpose:
    Extract adaptive covariance matching diagnostics from a watcher.

Outputs:
    gammaTheta - current Q_theta multiplier
    ratio      - last trace(S_hat)/trace(S_model)
    traceEmp   - last trace(S_hat)
    traceModel - last trace(S_model)

Notes:
    If covariance matching has not been initialized yet, this function
    returns safe default values.
%}

    gammaTheta = 1.0;
    ratio = NaN;
    traceEmp = NaN;
    traceModel = NaN;

    if ~isfield(watcher, "cm")
        return;
    end

    cm = watcher.cm;

    if isfield(cm, "gammaTheta")
        gammaTheta = cm.gammaTheta;
    end

    if isfield(cm, "lastRatio")
        ratio = cm.lastRatio;
    end

    if isfield(cm, "lastTraceEmp")
        traceEmp = cm.lastTraceEmp;
    end

    if isfield(cm, "lastTraceModel")
        traceModel = cm.lastTraceModel;
    end

end
