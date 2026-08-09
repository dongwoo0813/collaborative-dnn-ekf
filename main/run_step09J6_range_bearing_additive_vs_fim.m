function [out09j6rb,diag09j6rb,plots09j6rb] = ...
    run_step09J6_range_bearing_additive_vs_fim( ...
    makePlots,communicationArchitecture,simulationTime,seedOverride)
%RUN_STEP09J6_RANGE_BEARING_ADDITIVE_VS_FIM Compare GS composition modes.
% Both cases receive range+bearing at every measurement update.
% Defaults: sigmaRange=1 m and sigmaBearing=0.01 deg (both 1 sigma).
% The existing bearing-FIM gate is unchanged; only the measurement model
% differs from the bearing-only Step 09-J.6 operational comparison.

if nargin < 1, makePlots = true; end
if nargin < 2 || isempty(communicationArchitecture)
    communicationArchitecture = "gs";
end
if nargin < 3 || isempty(simulationTime)
    simulationTime = 200;
end
if nargin < 4
    seedOverride = [];
end
addpath(genpath(pwd));
[cfgBase,seed,meta] = config_step09J6_seed101_operational();
if ~isempty(seedOverride), seed = double(seedOverride); end
cfgBase.T = simulationTime;
cfgBase.time = 0:cfgBase.dt:cfgBase.T;
cfgBase.N = numel(cfgBase.time);
cfgBase.communication.architecture = string(communicationArchitecture);
cfgBase.gs.uploadMode = "after_measurement_update";
cfgBase.gs.fimGate.accumulationMode = "cumulative_sum";
cfgBase.gs.fimGate.normalizeTrace = false;

sigmaRange = 1.0;
sigmaBearingDeg = 0.01;
cfgBase.meas.type = "range_bearing";
cfgBase.meas.sigmaRange = sigmaRange;
cfgBase.meas.sigmaBearing = deg2rad(sigmaBearingDeg);
if cfgBase.dim == 2
    cfgBase.meas.R = diag([sigmaRange^2,cfgBase.meas.sigmaBearing^2]);
elseif cfgBase.dim == 3
    cfgBase.meas.R = diag([sigmaRange^2, ...
        cfgBase.meas.sigmaBearing^2,cfgBase.meas.sigmaBearing^2]);
else
    error("Unsupported cfg.dim=%d.",cfgBase.dim);
end

cfgAdd = cfgBase;
cfgAdd.step.name = "step09J6_range_bearing_additive";
cfgAdd.gs.compositeMode = "additive";
cfgAdd.dnn.predictionResidualSource = "GS_composite";

cfgFIM = cfgBase;
cfgFIM.step.name = "step09J6_range_bearing_bearing_FIM_gated";
cfgFIM.gs.compositeMode = "bearing_fim_gated";
cfgFIM.dnn.predictionResidualSource = "GS_composite";

fprintf("Step 09-J.6 range+bearing: additive versus global geometry fusion\n");
fprintf("communication architecture=%s\n",string(communicationArchitecture));
fprintf("sigma_r=%.3f m, sigma_b=%.4f deg, T=%.1f s, dt=%.4g s\n", ...
    sigmaRange,sigmaBearingDeg,cfgBase.T,cfgBase.dt);
fprintf("theta0Std=%.3e; every measurement updates eta and theta.\n", ...
    meta.thetaInitStd);

fprintf("\nRunning additive GS...\n");
rng(seed); resAdd = simulate_GS_DNN_EKF(cfgAdd);
fprintf("Running bearing-FIM-gated GS...\n");
rng(seed); resFIM = simulate_GS_DNN_EKF(cfgFIM);
assert(all(isfinite(resAdd.xhat(:))) && all(isfinite(resFIM.xhat(:))), ...
    "A non-finite kinematic estimate was produced.");
assert(all(isfinite(resAdd.dnnResidual(:))) && ...
    all(isfinite(resFIM.dnnResidual(:))), ...
    "A non-finite DNN residual was produced.");

mAdd = summarizeCase(resAdd,cfgBase.dim);
mFIM = summarizeCase(resFIM,cfgBase.dim);
caseName = ["additive";"bearing-FIM-gated"];
positionRMSE = [mAdd.positionRMSE;mFIM.positionRMSE];
finalPositionRMSE = [mAdd.finalPositionRMSE;mFIM.finalPositionRMSE];
exactRangeRMSE = [mAdd.exactRangeRMSE;mFIM.exactRangeRMSE];
finalExactRangeRMSE = [mAdd.finalExactRangeRMSE;mFIM.finalExactRangeRMSE];
crossTrackRMSE = [mAdd.crossTrackRMSE;mFIM.crossTrackRMSE];
velocityRMSE = [mAdd.velocityRMSE;mFIM.velocityRMSE];
residualVectorRMSE = [mAdd.residualVectorRMSE;mFIM.residualVectorRMSE];
finalResidualVectorRMSE = [mAdd.finalResidualVectorRMSE; ...
    mFIM.finalResidualVectorRMSE];
meanCosine = [mAdd.meanCosine;mFIM.meanCosine];
finalCosine = [mAdd.finalCosine;mFIM.finalCosine];
meanNormRatio = [mAdd.meanNormRatio;mFIM.meanNormRatio];
meanNIS = [mAdd.meanNIS;mFIM.meanNIS];
performanceSummary = table(caseName,positionRMSE,finalPositionRMSE, ...
    exactRangeRMSE,finalExactRangeRMSE,crossTrackRMSE,velocityRMSE, ...
    residualVectorRMSE,finalResidualVectorRMSE,meanCosine,finalCosine, ...
    meanNormRatio,meanNIS);
disp(performanceSummary);

out09j6rb = struct("resGSAdd",resAdd,"resGSFIM",resFIM, ...
    "cfgGSAdd",cfgAdd,"cfgGSFIM",cfgFIM,"seed",seed, ...
    "thetaInitStd",meta.thetaInitStd,"sigmaRange",sigmaRange, ...
    "sigmaBearingDeg",sigmaBearingDeg, ...
    "performanceSummary",performanceSummary);
diag09j6rb = run_step09J6_estimate_norm_cosine_diagnostic( ...
    out09j6rb,[0 25 50 100 150 200],false);
plots09j6rb = struct();
if makePlots
    plots09j6rb = plot_step09J6_operational_results(out09j6rb);
end
end

function m = summarizeCase(res,dim)
[~,N,Nw] = size(res.xhat); idxFinal = max(1,round(.9*N)):N;
xt = repmat(reshape(res.etaTrue,2*dim,N,1),1,1,Nw);
ep = res.xhat(1:dim,:,:) - xt(1:dim,:,:);
iv = dim+(1:dim); ev = res.xhat(iv,:,:) - xt(iv,:,:);
rt = repmat(reshape(res.etaTrue(1:dim,:),dim,N,1),1,1,Nw);
rhoTrue = rt-res.watcherR;
rhoHat = res.xhat(1:dim,:,:)-res.watcherR;
rangeTrue = reshape(sqrt(sum(rhoTrue.^2,1)),N,Nw);
rangeHat = reshape(sqrt(sum(rhoHat.^2,1)),N,Nw);
rangeError = rangeHat-rangeTrue;
u = rhoTrue./max(reshape(rangeTrue,1,N,Nw),realmin);
losProjection = reshape(sum(ep.*u,1),N,Nw);
pNorm = reshape(sqrt(sum(ep.^2,1)),N,Nw);
crossNorm = sqrt(max(pNorm.^2-losProjection.^2,0));
vNorm = reshape(sqrt(sum(ev.^2,1)),N,Nw);

dTrue = repmat(reshape(res.trueResidual,dim,N,1),1,1,Nw);
dHat = res.dnnResidual;
dErr = reshape(sqrt(sum((dHat-dTrue).^2,1)),N,Nw);
tNorm = reshape(sqrt(sum(dTrue.^2,1)),N,Nw);
hNorm = reshape(sqrt(sum(dHat.^2,1)),N,Nw);
dotValue = reshape(sum(dHat.*dTrue,1),N,Nw);
floorValue = max(1e-12,1e-3*median(tNorm(:),"omitnan"));
valid = tNorm>floorValue & hNorm>floorValue;
cosine = NaN(N,Nw);
cosine(valid) = dotValue(valid)./(tNorm(valid).*hNorm(valid));
ratio = NaN(N,Nw); useRatio = tNorm>floorValue;
ratio(useRatio) = hNorm(useRatio)./tNorm(useRatio);

m.positionRMSE = rmsall(pNorm);
m.finalPositionRMSE = rmsall(pNorm(idxFinal,:));
m.exactRangeRMSE = rmsall(rangeError);
m.finalExactRangeRMSE = rmsall(rangeError(idxFinal,:));
m.crossTrackRMSE = rmsall(crossNorm);
m.velocityRMSE = rmsall(vNorm);
m.residualVectorRMSE = rmsall(dErr);
m.finalResidualVectorRMSE = rmsall(dErr(idxFinal,:));
m.meanCosine = mean(cosine(:),"omitnan");
m.finalCosine = mean(cosine(idxFinal,:),"all","omitnan");
m.meanNormRatio = mean(ratio(:),"omitnan");
nis = res.NIS(isfinite(res.NIS)); m.meanNIS = mean(nis,"omitnan");
end

function y = rmsall(x)
y = sqrt(mean(x(:).^2,"omitnan"));
end
