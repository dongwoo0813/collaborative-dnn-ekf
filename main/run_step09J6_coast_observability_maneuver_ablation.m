function sweep = run_step09J6_coast_observability_maneuver_ablation(makePlots)
%RUN_STEP09J6_COAST_OBSERVABILITY_MANEUVER_ABLATION Run N0-N3 coast study.
%
% N0 removes the prescribed circular motion. Each watcher begins at its
% former phased circular position and then coasts with the nominal target
% velocity. Therefore its nominal watcher-to-target relative position is
% constant. N1-N3 add the same calibrated burns used in the M study.
%
% Usage:
%   sweep09j6Coast = ...
%       run_step09J6_coast_observability_maneuver_ablation(true);

    if nargin < 1
        makePlots = true;
    end

    sweep = run_step09J6_observability_maneuver_ablation( ...
        makePlots, "coast");
end
