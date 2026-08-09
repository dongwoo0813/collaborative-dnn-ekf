function result = demo_4watcher_accel_fusion_2d(makePlots,simulationTime,dt,seed)
%DEMO_4WATCHER_ACCEL_FUSION_2D Bearing-only local EKF fusion demo.
%
% Each watcher independently produces the filtered point estimate
%
%   aHat_i(k|k) = E[a_k | z_i(1:k)]
%
% of the current 2-D target acceleration. Although it is one point estimate
% at time k, it contains the complete local bearing-measurement history.
%
% The fusion projector uses the prior predicted LOS employed by the EKF
% Jacobian. Each watcher shares only its current transverse directional
% constraint, and the full acceleration point estimate is reconstructed as
%
%   aHat_f = [sum_i Pperp_i]^dagger sum_i Pperp_i aHat_i(k|k).
%
% The baseline assumes alpha_i = 1 for all four watchers.
clc
close all

if nargin < 1 || isempty(makePlots),      makePlots = true; end
if nargin < 2 || isempty(simulationTime), simulationTime = 40; end
if nargin < 3 || isempty(dt),             dt = 0.05; end
if nargin < 4 || isempty(seed),           seed = 9; end

result = run_4watcher_accel_fusion_EKF( ...
    2, makePlots, simulationTime, dt, seed);
end
