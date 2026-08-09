    function etaDot = targetTruthDynamics(t, eta, cfg)
%{
Function:
    targetTruthDynamics.m

Purpose:
    Compute the true target dynamics used in simulation.

    The target physical state is

        eta = [r_t; v_t],

    where r_t is the target position and v_t is the target velocity.

    In Step 01, the truth dynamics use only the nominal double-integrator
    model.

    In Step 02, an unknown residual acceleration can be added to the truth
    dynamics by setting

        cfg.truth.useResidual = true.

Inputs:
    t      - Current simulation time.
             Type: scalar.
             Units: seconds.

    eta    - True target physical state.
             Size: 2*cfg.dim x 1.
             Definition:
                 eta = [r_t; v_t].

    cfg    - Simulation configuration structure.
             Required fields:
                 cfg.dim

             Optional fields:
                 cfg.truth.useResidual
                 cfg.truth.residualAmp

Outputs:
    etaDot - Time derivative of the true target state.
             Size: 2*cfg.dim x 1.

Main equations:
    The target truth dynamics are

        dot r_t = v_t,

        dot v_t = a_nom(r_t,v_t,t) + d_unk(eta,t),

    where

        a_nom(r_t,v_t,t)

    is computed by targetNominalDynamics.m.

    In Step 01,

        d_unk(eta,t) = 0.

    In Step 02, if cfg.truth.useResidual = true,

        d_unk(eta,t) = trueResidual(t,eta,cfg).

Notes:
    - The physical EKF prediction model should not directly use
      trueResidual.m.
    - This file creates the model mismatch needed before adding the DNN-EKF.
    - Keeping the residual behind cfg.truth.useResidual allows Step 01 to
      remain unchanged.
%}

    dim = cfg.dim;

    r_t = eta(1:dim);
    v_t = eta(dim+1:2*dim);

    aNom = targetNominalDynamics(t, r_t, v_t, cfg);

    % Default Step 01 behavior: no unknown residual acceleration.
    aUnk = zeros(dim,1);

    % Step 02 behavior: add unknown residual only when explicitly enabled.
    if isfield(cfg, "truth") && isfield(cfg.truth, "useResidual")
        if cfg.truth.useResidual
            aUnk = trueResidual(t, eta, cfg);
        end
    end

    etaDot = [v_t; aNom + aUnk];

end