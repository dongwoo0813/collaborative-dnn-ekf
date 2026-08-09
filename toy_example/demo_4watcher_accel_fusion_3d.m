function result = demo_4watcher_accel_fusion_3d(makePlots,simulationTime,dt,seed)
%DEMO_4WATCHER_ACCEL_FUSION_3D Azimuth-elevation local EKF fusion demo.
%
% Each watcher independently produces the filtered point estimate
%
%   aHat_i(k|k) = E[a_k | z_i(1:k)]
%
% of the current 3-D target acceleration. Although it is one point estimate
% at time k, it contains the complete local azimuth-elevation history.
%
% The fusion projector uses the prior predicted LOS employed by the EKF
% Jacobian. Each watcher shares its LOS-transverse acceleration constraint,
% and the full acceleration point estimate is reconstructed as
%
%   aHat_f = [sum_i Pperp_i]^dagger sum_i Pperp_i aHat_i(k|k).
%
% The baseline assumes alpha_i = 1 for all four watchers.

if nargin < 1 || isempty(makePlots),      makePlots = true; end
if nargin < 2 || isempty(simulationTime), simulationTime = 40; end
if nargin < 3 || isempty(dt),             dt = 0.02; end
if nargin < 4 || isempty(seed),           seed = 10; end

result = run_4watcher_accel_fusion_EKF( ...
    3, makePlots, simulationTime, dt, seed);
end
