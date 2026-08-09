addpath(genpath(pwd));
rehash;

cfg = config_step04_GS_DNN_EKF();

cfg.dim = 2;
cfg.Nw  = 4;

cfg.gs.compositeMode = "local_full_plus_gated_nonlocal";
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

[dHybrid, JHybrid, branchContribHybrid, branchUsedHybrid] = ...
    evaluateWatcherCompositeResidual(watcher, eta, thetaLocal, cfg);

dManual = zeros(cfg.dim, 1);
JManual = zeros(cfg.dim, 2*cfg.dim);

for j = 1:cfg.Nw

    if j == watcher.localBranchID
        theta_j = thetaLocal;
    else
        theta_j = watcher.gsBranches(j).theta;
    end

    phi_j = featureBlock(j, eta, cfg);
    W_j = reshape(theta_j, cfg.dim, numel(phi_j));

    dRaw_j = W_j * phi_j;
    JRaw_j = W_j * featureJacobianEta(j, eta, cfg);

    if j == watcher.localBranchID

        % Hybrid mode keeps the local branch full.
        dManual = dManual + dRaw_j;
        JManual = JManual + JRaw_j;

    else

        % Hybrid mode gates only nonlocal branches.
        [B_j, dB_dEta] = branchGateMatrix(j, eta, cfg);

        JGate_j = zeros(cfg.dim, 2*cfg.dim);

        for kEta = 1:(2*cfg.dim)
            JGate_j(:, kEta) = dB_dEta(:, :, kEta) * dRaw_j;
        end

        dManual = dManual + B_j * dRaw_j;
        JManual = JManual + B_j * JRaw_j + JGate_j;

    end

end

disp("branchUsedHybrid = ");
disp(branchUsedHybrid.');

fprintf("||dHybrid - dManual||_2      = %.3e\n", norm(dHybrid - dManual));
fprintf("||JHybrid - JManual||_F      = %.3e\n", norm(JHybrid - JManual, "fro"));
fprintf("||sum branchContrib - dHybrid||_2 = %.3e\n", ...
    norm(sum(branchContribHybrid, 2) - dHybrid));