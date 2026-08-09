function watcherTruth = initWatcherTruthArray(cfg)
%{
Function:
    initWatcherTruthArray.m

Purpose:
    Initialize the truth-state structure array for all watcher spacecraft.

Inputs:
    cfg - Simulation configuration structure.
          Required fields:
              cfg.Nw
              cfg.dim
              cfg.time

Outputs:
    watcherTruth - Struct array of watcher truth states.
                   Size: cfg.Nw x 1.

Main equations:
    For each watcher i,

        watcherTruth(i) = initWatcherTruth(i,cfg).

Notes:
    - This avoids MATLAB "subscripted assignment between dissimilar
      structures" errors by initializing the first watcher explicitly.
%}

    Nw = cfg.Nw;

    watcherTruth = initWatcherTruth(1, cfg);

    for i = 2:Nw
        watcherTruth(i) = initWatcherTruth(i, cfg);
    end

    watcherTruth = watcherTruth(:);

end