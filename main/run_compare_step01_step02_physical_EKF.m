%{
Script:
    run_compare_step01_step02_physical_EKF.m

Purpose:
    Compare Step 01 and Step 02 physical EKF simulations.

    Step 01:
        The target truth dynamics do not include unknown residual
        acceleration.

    Step 02:
        The target truth dynamics include unknown residual acceleration,
        but the physical EKF prediction model still ignores it.

    This script compares how much the physical EKF tracking performance
    degrades when the truth model contains unmodeled residual dynamics.

Inputs:
    None directly.

    This script uses:
        config_step01_physical_EKF.m
        config_step02_residual_physical_EKF.m

Outputs:
    results01 - Step 01 simulation results.
    results02 - Step 02 simulation results.
    metrics01 - Step 01 physical EKF metrics.
    metrics02 - Step 02 physical EKF metrics.

Main equations:
    Step 01 truth model:

        dot r_t = v_t,

        dot v_t = a_nom.

    Step 02 truth model:

        dot r_t = v_t,

        dot v_t = a_nom + d_unk(eta,t).

    Both simulations use the same physical EKF prediction model:

        dot r_hat_t = v_hat_t,

        dot v_hat_t = a_nom.

    The tracking degradation is measured using

        RMSE_r = sqrt(1/N sum_k ||r_hat(k) - r_true(k)||^2),

        RMSE_v = sqrt(1/N sum_k ||v_hat(k) - v_true(k)||^2).

Notes:
    - The same random seed is used for both simulations so that initial EKF
      perturbations and measurement noise are comparable.
    - Watcher control remains disabled.
    - This is not yet a DNN-EKF comparison.
    - This comparison creates the baseline motivation for adding DNN-EKF.
%}

clear; clc; close all;

% Add project root and all subfolders to path.
thisFile = mfilename("fullpath");
mainDir = fileparts(thisFile);
projectRoot = fileparts(mainDir);

addpath(genpath(projectRoot));
rehash;

% Use the same random seed for a fair comparison.
rngSeed = 1;

% -------------------------------------------------------------------------
% Step 01: physical EKF without unknown residual truth dynamics
% -------------------------------------------------------------------------
rng(rngSeed);

cfg01 = config_step01_physical_EKF();

results01 = simulatePhysicalEKF(cfg01);
metrics01 = computePhysicalEKFMetrics(results01, cfg01);

% -------------------------------------------------------------------------
% Step 02: physical EKF with unknown residual truth dynamics
% -------------------------------------------------------------------------
rng(rngSeed);

cfg02 = config_step02_residual_physical_EKF();

results02 = simulatePhysicalEKF(cfg02);
metrics02 = computePhysicalEKFMetrics(results02, cfg02);

% -------------------------------------------------------------------------
% Print comparison
% -------------------------------------------------------------------------
fprintf("\n");
fprintf("============================================================\n");
fprintf(" Step 01 vs Step 02 Physical EKF Comparison\n");
fprintf("============================================================\n");
fprintf("Random seed: %d\n", rngSeed);
fprintf("Number of watchers: %d\n", cfg01.Nw);
fprintf("Dimension: %d\n", cfg01.dim);
fprintf("\n");

fprintf("Step 01: residual disabled\n");
fprintf("Step 02: residual enabled, residualAmp = %.3e\n", cfg02.truth.residualAmp);
fprintf("\n");

fprintf("Mean position RMSE:\n");
fprintf("    Step 01 = %.6f\n", metrics01.meanPosRMSE);
fprintf("    Step 02 = %.6f\n", metrics02.meanPosRMSE);
fprintf("    Ratio   = %.6f\n", safeRatio(metrics02.meanPosRMSE, metrics01.meanPosRMSE));
fprintf("\n");

fprintf("Mean velocity RMSE:\n");
fprintf("    Step 01 = %.6f\n", metrics01.meanVelRMSE);
fprintf("    Step 02 = %.6f\n", metrics02.meanVelRMSE);
fprintf("    Ratio   = %.6f\n", safeRatio(metrics02.meanVelRMSE, metrics01.meanVelRMSE));
fprintf("\n");

for i = 1:cfg01.Nw
    fprintf("Watcher %d:\n", i);
    fprintf("    Position RMSE: Step 01 = %.6f, Step 02 = %.6f, Ratio = %.6f\n", ...
        metrics01.posRMSE(i), metrics02.posRMSE(i), ...
        safeRatio(metrics02.posRMSE(i), metrics01.posRMSE(i)));

    fprintf("    Velocity RMSE: Step 01 = %.6f, Step 02 = %.6f, Ratio = %.6f\n", ...
        metrics01.velRMSE(i), metrics02.velRMSE(i), ...
        safeRatio(metrics02.velRMSE(i), metrics01.velRMSE(i)));
end

fprintf("============================================================\n\n");

% -------------------------------------------------------------------------
% Plot RMSE comparison
% -------------------------------------------------------------------------
watcherLabels = categorical("Watcher " + string(1:cfg01.Nw));
watcherLabels = reordercats(watcherLabels, "Watcher " + string(1:cfg01.Nw));

figure;
bar(watcherLabels, [metrics01.posRMSE, metrics02.posRMSE]);
grid on;
xlabel("Watcher");
ylabel("Position RMSE");
title("Position RMSE Comparison: Step 01 vs Step 02");
legend("Step 01: no residual", "Step 02: residual truth", "Location", "best");

figure;
bar(watcherLabels, [metrics01.velRMSE, metrics02.velRMSE]);
grid on;
xlabel("Watcher");
ylabel("Velocity RMSE");
title("Velocity RMSE Comparison: Step 01 vs Step 02");
legend("Step 01: no residual", "Step 02: residual truth", "Location", "best");

% -------------------------------------------------------------------------
% Optional residual check
% -------------------------------------------------------------------------
if isfield(results01, "trueResidual") && isfield(results02, "trueResidual")
    fprintf("Residual log check:\n");
    fprintf("    max |Step 01 residual| = %.6e\n", max(abs(results01.trueResidual(:))));
    fprintf("    max |Step 02 residual| = %.6e\n", max(abs(results02.trueResidual(:))));
    fprintf("\n");
end

function r = safeRatio(a, b)
%{
Function:
    safeRatio

Purpose:
    Compute a / b while avoiding division by very small denominators.

Inputs:
    a - Numerator.
    b - Denominator.

Outputs:
    r - Ratio a / b if b is sufficiently nonzero.
        NaN otherwise.
%}

    if abs(b) < 1e-12
        r = NaN;
    else
        r = a / b;
    end

end