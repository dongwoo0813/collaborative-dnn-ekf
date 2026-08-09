function z = bearingMeasurement(etaTrue, watcherState, cfg, tMeas)
%{
Function:
    bearingMeasurement.m

Purpose:
    Generate a noisy bearing measurement from watcher to target.

    This function computes

        z = h(eta_true) + v,

    where h(eta_true) is obtained from measurementPrediction.m.

Inputs:
    etaTrue      - True target physical state.
                   Size: 2*cfg.dim x 1.

    watcherState - Watcher state structure.
                   Required fields:
                       watcherState.r - watcher position, cfg.dim x 1.

    cfg          - Simulation configuration structure.
                   Required fields:
                       cfg.dim
                       cfg.meas.sigmaBearing

Outputs:
    z            - Noisy bearing measurement.
                   Scalar for cfg.dim = 2.
                   Vector [azimuth; elevation] for cfg.dim = 3.

Main equations:
    z = h(eta_true) + v,

    where

        v ~ N(0, R).

Notes:
    - This function adds measurement noise.
    - The deterministic function h(eta) is implemented in
      measurementPrediction.m.
    - FOV/dropout logic should not be implemented here.
%}

    if nargin < 4, tMeas = 0; end
    zhatTrue = measurementPrediction(etaTrue, watcherState, cfg);
    replayNoise = [];
    if isfield(cfg,"replay") && isfield(cfg.replay,"bearingNoise")
        idx = min(size(cfg.replay.bearingNoise,1), ...
            max(1,round(tMeas/cfg.dt)+1));
        watcherID = 1;
        if isfield(watcherState,"id"), watcherID = watcherState.id; end
        replayNoise = cfg.replay.bearingNoise(idx,watcherID);
    end
    
    if cfg.dim == 2
        if isempty(replayNoise)
            noise = cfg.meas.sigmaBearing * randn;
        else
            noise = cfg.meas.sigmaBearing * replayNoise;
        end
    elseif cfg.dim == 3
        if isempty(replayNoise)
            noise = cfg.meas.sigmaBearing * randn(2,1);
        else
            noise = cfg.meas.sigmaBearing * replayNoise(:);
        end
    else
        error("Unsupported cfg.dim. Use cfg.dim = 2 or cfg.dim = 3.");
    end
    
    z = zhatTrue + noise;

end
