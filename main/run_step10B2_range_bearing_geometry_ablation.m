function result = run_step10B2_range_bearing_geometry_ablation( ...
    seed,simulationTime,makePlots)
%RUN_STEP10B2_RANGE_BEARING_GEOMETRY_ABLATION GS/P2P geometry comparison.
% Watchers coast, range+bearing is continuously available, and communication
% occurs after each measurement update.  The only residual-composition
% comparison is unweighted additive versus globally normalized geometry
% fusion B_j=(sum_l Omega_l+epsilon I)\Omega_j.

if nargin < 1 || isempty(seed), seed = 101; end
if nargin < 2 || isempty(simulationTime), simulationTime = 200; end
if nargin < 3 || isempty(makePlots), makePlots = false; end

architectures = ["gs","p2p_ring"];
architectureNames = ["GS","P2P-full-library-gossip"];
runs = cell(2,1);
rows = cell(4,1);

for ia = 1:2
    fprintf("\nStep 10-B2 architecture: %s\n",architectureNames(ia));
    [out,diagInfo,plotInfo] = ...
        run_step09J6_range_bearing_additive_vs_fim( ...
        false,architectures(ia),simulationTime,seed);
    runs{ia} = struct("architecture",architectures(ia), ...
        "architectureName",architectureNames(ia),"out",out, ...
        "diag",diagInfo,"plots",plotInfo);
    s = out.performanceSummary;
    s.architecture = repmat(architectureNames(ia),height(s),1);
    s.composition = s.caseName;
    s.caseName = architectureNames(ia) + "-" + s.caseName;
    rows{2*ia-1} = s(1,:);
    rows{2*ia} = s(2,:);
end

summary = vertcat(rows{:});
summary = movevars(summary,"architecture","After","caseName");
summary = movevars(summary,"composition","After","architecture");

result = struct("seed",seed,"simulationTime",simulationTime, ...
    "measurementType","range_bearing", ...
    "communicationMode","after_measurement_update", ...
    "runs",{runs},"summary",summary);

fprintf("\nStep 10-B2 range-bearing geometry comparison\n");
disp(summary(:,{'caseName','positionRMSE','finalPositionRMSE', ...
    'exactRangeRMSE','finalExactRangeRMSE','velocityRMSE', ...
    'residualVectorRMSE','finalResidualVectorRMSE', ...
    'meanCosine','finalCosine','meanNormRatio','meanNIS'}));

result.figures = gobjects(0);
if makePlots
    result.figures = makeStep10B2TimeHistoryPlots(result);
end
end

function figures = makeStep10B2TimeHistoryPlots(result)
%MAKESTEP10B2TIMEHISTORYPLOTS Combined GS/P2P time-history comparison.

caseNames = ["GS-additive","GS-global-geometry", ...
    "P2P-additive","P2P-global-geometry"];
residuals = { ...
    result.runs{1}.out.resGSAdd, ...
    result.runs{1}.out.resGSFIM, ...
    result.runs{2}.out.resGSAdd, ...
    result.runs{2}.out.resGSFIM};
colors = lines(4);
lineStyles = {'-','--','-','--'};

fig = figure('Name','Step 10-B2 range-bearing geometry time histories');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
for ic = 1:4
    h = computeTimeHistories(residuals{ic});
    nexttile(1);
    plot(h.time,h.positionRMSE,'Color',colors(ic,:), ...
        'LineStyle',lineStyles{ic},'LineWidth',1.25); hold on;
    nexttile(2);
    plot(h.time,h.rangeRMSE,'Color',colors(ic,:), ...
        'LineStyle',lineStyles{ic},'LineWidth',1.25); hold on;
    nexttile(3);
    plot(h.time,h.velocityRMSE,'Color',colors(ic,:), ...
        'LineStyle',lineStyles{ic},'LineWidth',1.25); hold on;
    nexttile(4);
    plot(h.time,h.residualRMSE,'Color',colors(ic,:), ...
        'LineStyle',lineStyles{ic},'LineWidth',1.25); hold on;
end

nexttile(1); title('Position error'); ylabel('RMSE [m]'); grid on;
legend(caseNames,'Location','best');
nexttile(2); title('Exact-range error'); ylabel('RMSE [m]'); grid on;
nexttile(3); title('Velocity error'); ylabel('RMSE [m/s]');
xlabel('time [s]'); grid on;
nexttile(4); title('Residual-vector error'); ylabel('RMSE [m/s^2]');
xlabel('time [s]'); grid on;

figures = fig;
if ~exist('results','dir'), mkdir('results'); end
fileName = sprintf('step10B2_time_history_seed%d_T%d.png', ...
    result.seed,round(result.simulationTime));
exportgraphics(fig,fullfile('results',fileName),'Resolution',200);
end

function h = computeTimeHistories(res)
%COMPUTETIMEHISTORIES RMS over watchers at every simulation time.

[nEta,N,Nw] = size(res.xhat);
dim = nEta/2;
truth = repmat(reshape(res.etaTrue,nEta,N,1),1,1,Nw);
ep = res.xhat(1:dim,:,:) - truth(1:dim,:,:);
ev = res.xhat(dim+(1:dim),:,:) - truth(dim+(1:dim),:,:);
positionNorm2 = reshape(sum(ep.^2,1),N,Nw);
velocityNorm2 = reshape(sum(ev.^2,1),N,Nw);

truthR = repmat(reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
rhoTrue = truthR-res.watcherR;
rhoHat = res.xhat(1:dim,:,:)-res.watcherR;
rangeTrue = reshape(sqrt(sum(rhoTrue.^2,1)),N,Nw);
rangeHat = reshape(sqrt(sum(rhoHat.^2,1)),N,Nw);
rangeError2 = (rangeHat-rangeTrue).^2;

trueResidual = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
residualError2 = reshape(sum((res.dnnResidual-trueResidual).^2,1),N,Nw);

h = struct();
h.time = res.time(:);
h.positionRMSE = sqrt(mean(positionNorm2,2,'omitnan'));
h.rangeRMSE = sqrt(mean(rangeError2,2,'omitnan'));
h.velocityRMSE = sqrt(mean(velocityNorm2,2,'omitnan'));
h.residualRMSE = sqrt(mean(residualError2,2,'omitnan'));
end
