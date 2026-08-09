addpath(genpath(pwd));
rehash;

cfg = config_step04_GS_DNN_EKF();

cfg.dim = 2;
cfg.Nw  = 4;

cfg.gs.compositeMode = "gated_additive";
cfg.gs.nonlocalWeightMode = "none";

cfg.gate.mode = "tight_frame_2d_rt";
cfg.gate.minRange = 1e-12;

eta = [3; 4; 0.1; -0.2];

nPhi = numel(featureBlock(1, eta, cfg));
nTheta = cfg.dim * nPhi;

watcher = struct();
watcher.localBranchID = 1;
watcher.nTheta = nTheta;

emptyRecord = struct();
emptyRecord.theta = [];
emptyRecord.active = false;
emptyRecord.usedInPrediction = false;
emptyRecord.status = "empty";
emptyRecord.isStale = false;

watcher.gsBranches = repmat(emptyRecord, cfg.Nw, 1);

thetaLocal = 1e-3 * (1:nTheta).';

for j = 2:cfg.Nw
    watcher.gsBranches(j).theta = thetaLocal;
    watcher.gsBranches(j).active = true;
    watcher.gsBranches(j).usedInPrediction = true;
    watcher.gsBranches(j).status = "valid";
    watcher.gsBranches(j).isStale = false;
end

[d0, JetaComp, ~, ~] = evaluateWatcherCompositeResidual( ...
    watcher, eta, thetaLocal, cfg);

h = 1e-6;
nEta = numel(eta);
Jfd = zeros(cfg.dim, nEta);

for k = 1:nEta
    e = zeros(nEta, 1);
    e(k) = h;

    dPlus = evaluateWatcherCompositeResidual( ...
        watcher, eta + e, thetaLocal, cfg);

    dMinus = evaluateWatcherCompositeResidual( ...
        watcher, eta - e, thetaLocal, cfg);

    Jfd(:, k) = (dPlus - dMinus) / (2*h);
end

fprintf("||JetaComp - Jfd||_F = %.3e\n", norm(JetaComp - Jfd, "fro"));

disp("Analytic JetaComp = ");
disp(JetaComp);

disp("Finite-difference Jfd = ");
disp(Jfd);