function cfg = config_step02_residual_physical_EKF()
%{
Function:
    config_step02_residual_physical_EKF.m

Purpose:
    Create the configuration for Step 02 residual-mismatch physical EKF
    simulation.

    This configuration starts from the Step 01 physical EKF baseline and
    turns on an unknown residual acceleration in the target truth dynamics.

    The physical EKF prediction model still does not know this residual.
    Therefore, this configuration creates a model mismatch:

        truth model:
            dot v_t = a_nom + d_unk

        EKF prediction model:
            dot v_t = a_nom

Inputs:
    None.

Outputs:
    cfg - Simulation configuration structure.

Main equations:
    The target truth dynamics use

        dot r_t = v_t,

        dot v_t = a_nom(r_t,v_t,t) + d_unk(eta,t),

    where

        d_unk(eta,t) = trueResidual(t,eta,cfg).

    The baseline physical EKF still predicts with

        dot r_t = v_t,

        dot v_t = a_nom(r_t,v_t,t).

Notes:
    - This file intentionally reuses config_step01_physical_EKF.m.
    - Step 01 remains unchanged.
    - The residual amplitude should be small enough that the EKF does not
      immediately diverge, but large enough to show model mismatch.
    - This step prepares the simulation for the later DNN-EKF residual
      approximation.
%}

    % Start from the runnable Step 01 physical EKF baseline.
    cfg = config_step01_physical_EKF();

    % Step label for bookkeeping.
    cfg.step.name = "step02_residual_physical_EKF";
    
    cfg.truth.useResidual = true;
     
    % Select the truth residual model.
    %
    % Options:
    %   "smooth_synthetic" : simple sinusoidal debugging residual
    %   "branchwise"       : hidden branch-wise residual using
    %                        d_unk(eta) = sum_j W_j^star phi_j(eta)
    cfg.truth.residualModel = "branchwise";
    
    % Residual acceleration amplitude.
    %
    % For the branchwise model, this value is used indirectly to scale the
    % hidden truth branch weights.
    cfg.truth.residualAmp = 1e-4;

end