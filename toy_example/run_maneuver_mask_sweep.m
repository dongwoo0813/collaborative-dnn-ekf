function sweep = run_maneuver_mask_sweep( ...
    dim,simulationTime,dt,seed,sweepMode)
%RUN_MANEUVER_MASK_SWEEP Compare selective watcher maneuver sets.
%
% PURPOSE
% -------
% Run the same distributed angle-only [r;v;a] experiment repeatedly with:
%
%   - the same truth,
%   - the same initial range scales,
%   - the same measurement-noise seed,
%   - the same filter settings,
%
% while changing only options.maneuverWatcherMask.
%
% This isolates the tradeoff between:
%
%   post-burn acceleration accuracy,
%   weighted directional geometry,
%   and total maneuver effort.
%
% INPUTS
% ------
% dim             2 or 3.
% simulationTime  Duration [s].
% dt              Sampling interval [s].
% seed            Paired random seed used for every mask.
% sweepMode       'core' or 'all_combinations'.
%
% OUTPUT
% ------
% sweep.summary   One row per maneuver mask.
% sweep.results   Full result structure for each mask.
% sweep.figure    RMSE/cost and geometry diagnostic figures.

if nargin < 1 || isempty(dim),             dim = 2; end
if nargin < 2 || isempty(simulationTime),  simulationTime = 120; end
if nargin < 3 || isempty(dt),              dt = 0.05; end
if nargin < 4 || isempty(seed),            seed = 73; end
if nargin < 5 || isempty(sweepMode),       sweepMode = 'core'; end

validateattributes(dim,{'numeric'},{'scalar','integer'});
if ~ismember(dim,[2,3])
    error('dim must be 2 or 3.');
end
sweepMode = validatestring( ...
    sweepMode,{'core','all_combinations'});

Nw = 4;
maskSet = makeMaskSet(sweepMode,Nw);
numCases = size(maskSet,1);

baseOptions = struct();
baseOptions.truthAccelerationMode = 'constant';

if dim == 2
    baseOptions.constantAcceleration = [0.15;0.20];
else
    baseOptions.constantAcceleration = [0.15;0.20;-0.10];
end

baseOptions.activeWatcherMode = 'all';
baseOptions.activeWatcherIndex = 1;
baseOptions.initialRangeScale = [0.55,0.80,1.25,1.55];

baseOptions.burnAcceleration = 2.0;
baseOptions.burnStartTime = 10.0;
baseOptions.burnDuration = 5.0;

baseOptions.sigmaBearingDeg = 0.2;
baseOptions.sigmaAzimuthDeg = 0.2;
baseOptions.sigmaElevationDeg = 0.2;

% Constant-acceleration observability validation.
baseOptions.sigmaJerk = 0;

% Only maneuvering watchers contribute to weighted fusion after burn start.
baseOptions.nonManeuverFusionWeight = 0;
baseOptions.postBurnSettlingTime = 20.0;

results = cell(numCases,1);

maskLabel = strings(numCases,1);
maskCode = strings(numCases,1);
numManeuverWatchers = zeros(numCases,1);
totalDeltaV = zeros(numCases,1);

postLocalMeanAccelerationRMSE = nan(numCases,1);
postEqualGeometryAccelerationRMSE = nan(numCases,1);
postWeightedGeometryAccelerationRMSE = nan(numCases,1);

postWeightedFullRankRate = nan(numCases,1);
postWeightedMedianCondition = nan(numCases,1);
postWeightedMaxCondition = nan(numCases,1);
maxAngularSignatureSigma = nan(numCases,1);

for c = 1:numCases
    mask = logical(maskSet(c,:));

    options = baseOptions;
    options.maneuverWatcherMask = mask;

    fprintf('\nMask %2d/%2d: [%s]\n', ...
        c,numCases,sprintf('%d ',mask));

    % Same seed in every call gives paired truth/noise/initialization.
    results{c} = run_local_accel_observability_experiment( ...
        dim,false,simulationTime,dt,seed,options);

    pulse = results{c}.pulsePair;

    maskLabel(c) = makeMaskLabel(mask);
    maskCode(c) = sprintf('%d%d%d%d',mask);
    numManeuverWatchers(c) = nnz(mask);
    totalDeltaV(c) = pulse.totalDeltaV;

    postLocalMeanAccelerationRMSE(c) = ...
        pulse.postActiveMeanAccelerationRMSE;
    postEqualGeometryAccelerationRMSE(c) = ...
        pulse.postEqualGeometryAccelerationRMSE;
    postWeightedGeometryAccelerationRMSE(c) = ...
        pulse.postManeuverWeightedGeometryAccelerationRMSE;

    postWeightedFullRankRate(c) = ...
        pulse.postWeightedGeometryFullRankRate;
    postWeightedMedianCondition(c) = ...
        pulse.postWeightedGeometryMedianCondition;
    postWeightedMaxCondition(c) = ...
        pulse.postWeightedGeometryMaxCondition;

    signature = pulse.angleSignatureSigma(:,mask);
    maxAngularSignatureSigma(c) = maxFiniteLocal(signature(:));

    % Targeted maneuver-cost sanity check.
    expectedDeltaV = ...
        nnz(mask)*2*options.burnAcceleration*options.burnDuration;

    if abs(totalDeltaV(c)-expectedDeltaV) > ...
            1e-8*max(1,expectedDeltaV)
        warning(['Unexpected total Delta-V for mask %s: ' ...
            'computed %.12g, expected %.12g.'], ...
            maskCode(c),totalDeltaV(c),expectedDeltaV);
    end
end

summary = table( ...
    maskLabel,maskCode,numManeuverWatchers,totalDeltaV, ...
    postLocalMeanAccelerationRMSE, ...
    postEqualGeometryAccelerationRMSE, ...
    postWeightedGeometryAccelerationRMSE, ...
    postWeightedFullRankRate, ...
    postWeightedMedianCondition, ...
    postWeightedMaxCondition, ...
    maxAngularSignatureSigma);

% Rank valid solutions first, then acceleration RMSE, then maneuver cost.
rankPenalty = postWeightedFullRankRate < 1;
summary = addvars(summary,rankPenalty,'After','totalDeltaV');
summary = sortrows(summary, ...
    {'rankPenalty', ...
     'postWeightedGeometryAccelerationRMSE', ...
     'totalDeltaV'}, ...
    {'ascend','ascend','ascend'});

disp(summary);

sweep = struct();
sweep.dimension = dim;
sweep.simulationTime = simulationTime;
sweep.dt = dt;
sweep.seed = seed;
sweep.sweepMode = sweepMode;
sweep.maskSet = maskSet;
sweep.baseOptions = baseOptions;
sweep.results = results;
sweep.summary = summary;
sweep.figure = plotSweep(summary);
end

function maskSet = makeMaskSet(mode,Nw)
%MAKEMASKSET Return representative or exhaustive maneuver masks.

switch mode
    case 'core'
        % No maneuver; representative singles; same-axis and cross-axis
        % pairs; one triple; and all watchers.
        maskSet = logical([ ...
            0 0 0 0; ...
            1 0 0 0; ...
            0 0 1 0; ...
            1 1 0 0; ...
            0 0 1 1; ...
            1 0 1 0; ...
            0 1 0 1; ...
            1 1 1 0; ...
            1 1 1 1]);

    case 'all_combinations'
        values = 0:(2^Nw-1);
        maskSet = false(numel(values),Nw);

        for c = 1:numel(values)
            bits = dec2bin(values(c),Nw)-'0';
            maskSet(c,:) = logical(bits);
        end
end
end

function label = makeMaskLabel(mask)
%MAKEMASKLABEL Human-readable watcher list.

index = find(mask);

if isempty(index)
    label = "none";
else
    label = "W"+join(string(index),"+W");
end
end

function value = maxFiniteLocal(x)
%MAXFINITELOCAL Maximum finite value, or NaN.

x = x(isfinite(x));
if isempty(x)
    value = nan;
else
    value = max(x);
end
end

function fig = plotSweep(summary)
%PLOTSWEEP Compare post-burn error, maneuver cost, and geometry.

fig = figure( ...
    'Name','Selective maneuver-mask sweep', ...
    'Color','w');

tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(summary.totalDeltaV, ...
    summary.postWeightedGeometryAccelerationRMSE,'o', ...
    'LineWidth',1.1);
grid on;
xlabel('total maneuver effort');
ylabel('post weighted accel. RMSE');
title('accuracy versus maneuver effort');

for i = 1:height(summary)
    text(summary.totalDeltaV(i), ...
        summary.postWeightedGeometryAccelerationRMSE(i), ...
        "  "+summary.maskLabel(i));
end

nexttile;
bar(categorical(summary.maskLabel), ...
    summary.postWeightedFullRankRate);
grid on;
ylabel('post full-rank rate');
ylim([0,1.05]);
title('weighted directional reconstruction rank');

nexttile;
semilogy(categorical(summary.maskLabel), ...
    max(summary.postWeightedMedianCondition,1),'o-', ...
    'LineWidth',1.1);
grid on;
xlabel('maneuver mask');
ylabel('median weighted condition');
title('post-burn weighted geometry conditioning');
end
