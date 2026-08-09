function out = run_step09J6_oracle_dynamics_ablation(makePlots)
%RUN_STEP09J6_ORACLE_DYNAMICS_ABLATION Compare joint DNN and oracle dynamics.
% D0 updates eta and theta at every bearing measurement. D1 uses the true
% residual function in propagation and updates eta at every measurement.
% Both use sigmaBearing=0.01 deg and otherwise identical Step 09-J.6 data.

if nargin < 1, makePlots = true; end
addpath(genpath(pwd));
[cfg0,seed,meta] = config_step09J6_seed101_operational();
cfg0.meas.type = "bearing";
cfg0.meas.sigmaBearing = deg2rad(0.01);
if cfg0.dim == 2
    cfg0.meas.R = cfg0.meas.sigmaBearing^2;
else
    cfg0.meas.R = cfg0.meas.sigmaBearing^2*eye(2);
end
cfg0.dnn.adaptQThetaEnabled = false;
cfg0.dnn.adaptQEpsilonEnabled = false;

cfgDNN = cfg0;
cfgDNN.step.name = "step09J6_D0_joint_DNN_EKF_001deg";
cfgDNN.dnn.predictionResidualSource = "GS_composite";
cfgDNN.gs.enabled = true;
cfgDNN.gs.compositeMode = "additive";

cfgOracle = cfg0;
cfgOracle.step.name = "step09J6_D1_oracle_dynamics_001deg";
cfgOracle.estimator.type = "oracle_residual_EKF";
cfgOracle.dnn.predictionResidualSource = "oracle";
% Keep the same MLP initializer so pre-measurement RNG consumption matches.
cfgOracle.gs.enabled = false;
cfgOracle.gs.bootstrapUpload = false;
cfgOracle.gs.useNonlocalBranchCovariance = false;
cfgOracle.gs.uploadMode = "none";
cfgOracle.gs.broadcastMode = "none";

fprintf("Step 09-J.6 oracle-dynamics ablation\n");
fprintf("sigma_b=0.01 deg, T=%.1f s, dt=%.4g s\n",cfg0.T,cfg0.dt);
fprintf("theta0Std=%.3e; every measurement updates the active estimator.\n", ...
    meta.thetaInitStd);
fprintf("\nD0 joint DNN-EKF...\n");
rng(seed); resDNN = simulate_GS_DNN_EKF(cfgDNN);
fprintf("D1 oracle dynamics...\n");
rng(seed); resOracle = simulateLocalDNNEKF(cfgOracle);
assert(all(isfinite(resDNN.xhat(:))) && all(isfinite(resOracle.xhat(:))), ...
    "A non-finite state estimate was produced.");

m0 = metrics(resDNN,cfg0.dim); m1 = metrics(resOracle,cfg0.dim);
caseName = ["D0 joint DNN-EKF";"D1 oracle dynamics"];
residualSource = ["GS_composite";"oracle"];
positionRMSE = [m0.positionRMSE;m1.positionRMSE];
finalPositionRMSE = [m0.finalPositionRMSE;m1.finalPositionRMSE];
losDirectionRMSE = [m0.losDirectionRMSE;m1.losDirectionRMSE];
finalLosDirectionRMSE = [m0.finalLosDirectionRMSE;m1.finalLosDirectionRMSE];
crossTrackRMSE = [m0.crossTrackRMSE;m1.crossTrackRMSE];
velocityRMSE = [m0.velocityRMSE;m1.velocityRMSE];
finalVelocityRMSE = [m0.finalVelocityRMSE;m1.finalVelocityRMSE];
meanNIS = [m0.meanNIS;m1.meanNIS];
meanThetaUpdateNorm = [m0.meanThetaUpdateNorm;m1.meanThetaUpdateNorm];
summary = table(caseName,residualSource,positionRMSE,finalPositionRMSE, ...
    losDirectionRMSE,finalLosDirectionRMSE,crossTrackRMSE,velocityRMSE, ...
    finalVelocityRMSE,meanNIS,meanThetaUpdateNorm);
disp(summary);

figures = gobjects(0);
if makePlots, figures = makeFigure(resDNN,resOracle,cfg0.dim); end
out = struct("summary",summary,"resDNN",resDNN,"resOracle",resOracle, ...
    "cfgDNN",cfgDNN,"cfgOracle",cfgOracle,"seed",seed, ...
    "sigmaBearingDeg",0.01,"figures",figures);
end

function m = metrics(res,dim)
s = series(res,dim); N = numel(res.time); idx = max(1,round(.9*N)):N;
m.positionRMSE = rmsall(s.p);
m.finalPositionRMSE = rmsall(s.p(idx,:));
m.losDirectionRMSE = rmsall(s.los);
m.finalLosDirectionRMSE = rmsall(s.los(idx,:));
m.crossTrackRMSE = rmsall(s.cross);
m.velocityRMSE = rmsall(s.v);
m.finalVelocityRMSE = rmsall(s.v(idx,:));
x = res.NIS(isfinite(res.NIS)); m.meanNIS = mean(x,"omitnan");
if isfield(res,"thetaUpdateNorm")
    m.meanThetaUpdateNorm = mean(res.thetaUpdateNorm(:),"omitnan");
else
    m.meanThetaUpdateNorm = NaN;
end
end

function y = rmsall(x)
y = sqrt(mean(x(:).^2,"omitnan"));
end

function s = series(res,dim)
[~,N,Nw] = size(res.xhat);
xt = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
ep = res.xhat(1:dim,:,:) - xt(1:dim,:,:);
iv = dim+(1:dim); ev = res.xhat(iv,:,:) - xt(iv,:,:);
rt = repmat(reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
rho = rt-res.watcherR; u = rho./max(sqrt(sum(rho.^2,1)),realmin);
s.p = reshape(sqrt(sum(ep.^2,1)),N,Nw);
s.v = reshape(sqrt(sum(ev.^2,1)),N,Nw);
s.los = reshape(sum(ep.*u,1),N,Nw);
s.cross = sqrt(max(s.p.^2-s.los.^2,0));
end

function f = makeFigure(a,b,dim)
sa=series(a,dim); sb=series(b,dim); L={'joint DNN-EKF','oracle dynamics'};
f=figure('Name','Step 09-J.6 joint DNN versus oracle dynamics');
tiledlayout(2,2,'TileSpacing','compact');
nexttile; plotPair(a.time,mean(sa.p,2,"omitnan"),b.time,mean(sb.p,2,"omitnan"));
ylabel('mean position error [m]'); title('Target position'); legend(L,'Location','best');
nexttile; plotPair(a.time,sqrt(mean(sa.los.^2,2,"omitnan")),b.time,sqrt(mean(sb.los.^2,2,"omitnan")));
ylabel('RMS LOS error [m]'); title('Range-like position error');
nexttile; plotPair(a.time,mean(sa.cross,2,"omitnan"),b.time,mean(sb.cross,2,"omitnan"));
ylabel('mean cross-track error [m]'); title('LOS-transverse position error');
nexttile; plotPair(a.time,mean(sa.v,2,"omitnan"),b.time,mean(sb.v,2,"omitnan"));
ylabel('mean velocity error [m/s]'); title('Target velocity');
end

function plotPair(t1,y1,t2,y2)
plot(t1,y1,'LineWidth',1.2); hold on; plot(t2,y2,'LineWidth',1.2);
grid on; xlabel('time [s]');
end
