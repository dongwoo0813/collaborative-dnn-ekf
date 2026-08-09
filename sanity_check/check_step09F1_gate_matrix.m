addpath(genpath(pwd));

cfg = config_step04_GS_DNN_EKF();

cfg.dim = 2;
cfg.Nw  = 4;

cfg.gate.mode = "tight_frame_2d_rt";
cfg.gate.minRange = 1e-12;

eta = [3; 4; 0.1; -0.2];

Bsum = zeros(cfg.dim);

for j = 1:cfg.Nw
    Bj = branchGateMatrix(j, eta, cfg);

    fprintf("j = %d\n", j);
    disp(Bj);

    Bsum = Bsum + Bj;
end

disp("sum_j B_j = ");
disp(Bsum);

fprintf("||sum_j B_j - I||_F = %.3e\n", norm(Bsum - eye(cfg.dim), "fro"));