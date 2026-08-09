function results = simulatePhysicalEKF(cfg)
%{
Function:
    simulatePhysicalEKF.m

Purpose:
    Run the Step 01 physical EKF simulation for multiple watcher spacecraft.

    Each watcher estimates the same target physical state

        eta = [r_t; v_t],

    using its own bearing-only measurements.

    This simulation does not yet include:
        - DNN residual approximation,
        - branch parameters theta_i,
        - ground-station communication,
        - peer-to-peer communication,
        - watcher control.

Inputs:
    cfg      - Simulation configuration structure.
               Required fields:
                   cfg.dim
                   cfg.Nw
                   cfg.N
                   cfg.time
                   cfg.dt
                   cfg.target.r0
                   cfg.target.v0

Outputs:
    results  - Simulation results structure.
               Fields:
                   results.time
                   results.etaTrue
                   results.xhat
                   results.Pdiag
                   results.measAvail
                   results.watchersFinal

Main equations:
    Target truth propagation:

        eta_{k+1} = RK4(f_true, t_k, eta_k, dt).

    Physical EKF prediction:

        eta_hat_{i,k+1}^- = F eta_hat_{i,k}^+,

        P_{i,k+1}^- = F P_{i,k}^+ F^T + Q.

    Measurement update at t_{k+1}:

        nu_{i,k+1} = z_{i,k+1} - h_i(eta_hat_{i,k+1}^-),

        eta_hat_{i,k+1}^+
            = eta_hat_{i,k+1}^- + K_{i,k+1} nu_{i,k+1}.

Notes:
    - Measurement availability is handled through measurementModel.m.
    - In Step 01, fovAvailable.m may simply return true.
    - The simulation is dimension-generic through cfg.dim.
    - This file keeps the EKF naming convention:
          simulatePhysicalEKF.m
          initPhysicalEKF.m
          EKFPredictPhysical.m
          EKFUpdatePhysical.m
%}

    dim = cfg.dim;
    Nw = cfg.Nw;
    N = cfg.N;
    time = cfg.time;
    replayEnabled = isfield(cfg,"replay") && isfield(cfg.replay,"enabled") && logical(cfg.replay.enabled);
    if replayEnabled
        validateReplayTrajectoryPhysical(cfg.replay,2*dim,dim,N,Nw);
    end


    etaTrue = zeros(2*dim, N);
    etaTrue(:,1) = [cfg.target.r0; cfg.target.v0];
    
    trueResidualLog = zeros(dim, N);
    trueResidualLog(:,1) = computeTrueResidualForLog(time(1), etaTrue(:,1), cfg);

    
    % Initialize watcher structures.
    watcherTruth = initWatcherTruthArray(cfg);

    % The first assignment defines the struct fields:
    %   id, xhat, P, lastInnovation, lastS.
    watchers = initPhysicalEKF(1, etaTrue(:,1), cfg);
    
    for i = 2:Nw
        watchers(i) = initPhysicalEKF(i, etaTrue(:,1), cfg);
    end
    
    watchers = watchers(:);


    % Log Allocation
    xhatLog = zeros(2*dim, N, Nw);
    PdiagLog = zeros(2*dim, N, Nw);
    measAvailLog = false(N, Nw);
    NISLog = NaN(N, Nw);


    watcherRLog = zeros(dim, N, Nw);
    watcherVLog = zeros(dim, N, Nw);
    watcherULog = zeros(dim, N, Nw);
    watcherTauLog = zeros(3, N, Nw);
    watcherQLog = zeros(4, N, Nw);
    watcherOmegaLog = zeros(3, N, Nw);


    for i = 1:Nw
        xhatLog(:,1,i) = watchers(i).xhat;
        PdiagLog(:,1,i) = diag(watchers(i).P);
    
        watcherState0 = watcherTrajectory(i, time(1), cfg);
        
        watcherRLog(:,1,i) = watcherTruth(i).r;
        watcherVLog(:,1,i) = watcherTruth(i).v;
        watcherULog(:,1,i) = watcherTruth(i).u;
        watcherTauLog(:,1,i) = watcherTruth(i).tau;
        watcherQLog(:,1,i) = watcherTruth(i).q;
        watcherOmegaLog(:,1,i) = watcherTruth(i).omega;
    end

    for k = 1:N-1

        t = time(k);
        tNext = time(k+1);

        % 1. Truth propagation from t_k to t_{k+1}
        if replayEnabled
            etaTrue(:,k+1) = cfg.replay.etaTrue(:,k+1);
        else
            etaTrue(:,k+1) = propagateRK4( ...
                @(tt,xx) targetTruthDynamics(tt, xx, cfg), ...
                t, etaTrue(:,k), cfg.dt);
        end

        trueResidualLog(:,k+1) = computeTrueResidualForLog(tNext, etaTrue(:,k+1), cfg);

        for i = 1:Nw

            % 2. Watcher state at t_{k+1}
            targetInfo.etaHat = watchers(i).xhat;

            targetInfo.etaTrue = etaTrue(:,k);   % only for debugging/analysis
            
            if replayEnabled
                [watcherTruth(i), watcherCmd] = replayWatcherStepPhysical( ...
                    watcherTruth(i),cfg.replay,k+1,i,dim);
            else
                [watcherTruth(i), watcherCmd] = propagateWatcherStep( ...
                    i, watcherTruth(i), targetInfo, t, cfg);
            end
            
            watcherState = watcherTruth(i);
            
            watcherRLog(:,k+1,i) = watcherState.r;
            watcherVLog(:,k+1,i) = watcherState.v;
            watcherULog(:,k+1,i) = watcherCmd.u;
            watcherTauLog(:,k+1,i) = watcherCmd.tau;
            watcherQLog(:,k+1,i) = watcherState.q;
            watcherOmegaLog(:,k+1,i) = watcherState.omega;

            % 3. Measurement at t_{k+1}
            [z, available, ~] = measurementModel(etaTrue(:,k+1), watcherState, cfg, tNext);

            % 4. EKF prediction from t_k to t_{k+1}
            watchers(i) = EKFPredictPhysical(watchers(i), t, cfg);

            % 5. EKF correction using z_{k+1}
            if available
                watchers(i) = EKFUpdatePhysical(watchers(i), z, watcherState, cfg);
                nu = watchers(i).lastInnovation;
                S = watchers(i).lastS;
                NISLog(k+1,i) = nu' * (S \ nu);
            end

            % 6. Log posterior estimate at t_{k+1}
            xhatLog(:,k+1,i) = watchers(i).xhat;
            PdiagLog(:,k+1,i) = diag(watchers(i).P);
            measAvailLog(k+1,i) = available;

        end
    end

    results.time = time;
    results.etaTrue = etaTrue;
    results.xhat = xhatLog;
    results.Pdiag = PdiagLog;
    results.measAvail = measAvailLog;
    results.NIS = NISLog;
    results.watchersFinal = watchers;

    results.watcherR = watcherRLog;
    results.watcherV = watcherVLog;
    results.watcherU = watcherULog;
    results.watcherTau = watcherTauLog;
    results.watcherQ = watcherQLog;
    results.watcherOmega = watcherOmegaLog;
    results.watcherTruthFinal = watcherTruth;

    results.trueResidual = trueResidualLog;

end


function aUnk = computeTrueResidualForLog(t, eta, cfg)
%{
Function:
    computeTrueResidualForLog

Purpose:
    Compute the true residual acceleration for logging purposes.

    This helper keeps Step 01 safe. If the residual is not explicitly
    enabled by

        cfg.truth.useResidual = true,

    then this function returns zero acceleration.

Inputs:
    t      - Current simulation time.
    eta    - True target state eta = [r_t; v_t].
    cfg    - Simulation configuration structure.

Outputs:
    aUnk   - True residual acceleration used for logging.
             Size: cfg.dim x 1.

Main equations:
    If residual is disabled:

        d_unk = 0.

    If residual is enabled:

        d_unk = trueResidual(t,eta,cfg).

Notes:
    - This function is only for logging.
    - The actual truth dynamics are still computed by targetTruthDynamics.m.
%}

    dim = cfg.dim;

    aUnk = zeros(dim,1);

    if isfield(cfg, "truth") && isfield(cfg.truth, "useResidual")
        if cfg.truth.useResidual
            aUnk = trueResidual(t, eta, cfg);
        end
    end

end

function validateReplayTrajectoryPhysical(replay,nEta,dim,N,Nw)
    if ~isfield(replay,"etaTrue") || ~isfield(replay,"watcherR") || ~isfield(replay,"watcherV") || ...
            ~isequal(size(replay.etaTrue),[nEta N]) || ...
            ~isequal(size(replay.watcherR),[dim N Nw]) || ...
            ~isequal(size(replay.watcherV),[dim N Nw])
        error("simulatePhysicalEKF:BadReplayTrajectory", ...
            "Replay trajectory dimensions do not match physical EKF configuration.");
    end
end

function [watcherNext,cmd] = replayWatcherStepPhysical(watcherCurrent,replay,k,i,dim)
    watcherNext = watcherCurrent;
    watcherNext.r = replay.watcherR(:,k,i);
    watcherNext.v = replay.watcherV(:,k,i);
    cmd.u = zeros(dim,1); cmd.tau = zeros(3,1);
    if isfield(replay,"watcherU")
        cmd.u = replay.watcherU(:,k,i);
    end
    watcherNext.u = cmd.u; watcherNext.tau = cmd.tau;
end
