function Jphi = featureJacobianEta(i, eta, cfg)
%{
Function:
    featureJacobianEta.m

Purpose:
    Compute the analytical Jacobian of the fixed feature vector phi_i(eta)
    with respect to the physical target state

        eta = [r_t; v_t].

    The feature vector is defined in featureBlock.m as

        phi_i(eta)
        =
        [ 1;
          rBar;
          vBar;
          rBar' * rBar;
          sin(psi_i + sum(rBar));
          cos(psi_i + sum(vBar));
          rBar' * vBar ],

    where

        rBar = r_t / rScale,

        vBar = v_t / vScale,

        psi_i = 2*pi*(i-1)/cfg.Nw.

Inputs:
    i      - Branch index.
             Type: positive integer.
             Range:
                 1 <= i <= cfg.Nw.

    eta    - Target physical state.
             Size: 2*cfg.dim x 1.
             Definition:
                 eta = [r_t; v_t].

    cfg    - Simulation configuration structure.
             Required fields:
                 cfg.dim
                 cfg.Nw

             Optional fields:
                 cfg.dnn.rScale
                 cfg.dnn.vScale

Outputs:
    Jphi   - Feature Jacobian with respect to eta.
             Size: nPhi x 2*cfg.dim.

             Row q contains

                 partial phi_q / partial eta.

Main equations:
    The local DNN branch output is

        d_i(eta; theta_i) = W_i phi_i(eta).

    Therefore, the derivative of the branch output with respect to eta is

        partial d_i / partial eta
            = W_i * partial phi_i / partial eta.

    This function returns

        partial phi_i / partial eta.

Notes:
    - This analytical Jacobian replaces finite-difference differentiation
      for the DNN feature part.
    - It is exact for the current fixed feature definition.
    - If featureBlock.m is changed later, this file must be updated
      consistently.
%}

    dim = cfg.dim;

    r = eta(1:dim);
    v = eta(dim+1:2*dim);

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "rScale")
        rScale = cfg.dnn.rScale;
    else
        rScale = 1000;
    end

    if isfield(cfg, "dnn") && isfield(cfg.dnn, "vScale")
        vScale = cfg.dnn.vScale;
    else
        vScale = 1;
    end

    rBar = r / rScale;
    vBar = v / vScale;

    psi = 2*pi*(i-1)/cfg.Nw;

    nPhi = 2*dim + 5;
    nEta = 2*dim;

    Jphi = zeros(nPhi, nEta);

    idxR = 1:dim;
    idxV = dim + (1:dim);

    row = 1;

    % ---------------------------------------------------------------------
    % phi_1 = 1
    % ---------------------------------------------------------------------
    Jphi(row, :) = zeros(1, nEta);
    row = row + 1;

    % ---------------------------------------------------------------------
    % phi = rBar = r / rScale
    % ---------------------------------------------------------------------
    for d = 1:dim
        Jphi(row, idxR(d)) = 1 / rScale;
        row = row + 1;
    end

    % ---------------------------------------------------------------------
    % phi = vBar = v / vScale
    % ---------------------------------------------------------------------
    for d = 1:dim
        Jphi(row, idxV(d)) = 1 / vScale;
        row = row + 1;
    end

    % ---------------------------------------------------------------------
    % phi = rBar' * rBar
    %
    % d/dr (rBar' rBar) = 2 rBar' / rScale
    % ---------------------------------------------------------------------
    Jphi(row, idxR) = 2 * rBar' / rScale;
    row = row + 1;

    % ---------------------------------------------------------------------
    % phi = sin(psi + sum(rBar))
    %
    % d/dr = cos(psi + sum(rBar)) * 1/rScale * ones(1,dim)
    % ---------------------------------------------------------------------
    argR = psi + sum(rBar);
    Jphi(row, idxR) = cos(argR) * ones(1,dim) / rScale;
    row = row + 1;

    % ---------------------------------------------------------------------
    % phi = cos(psi + sum(vBar))
    %
    % d/dv = -sin(psi + sum(vBar)) * 1/vScale * ones(1,dim)
    % ---------------------------------------------------------------------
    argV = psi + sum(vBar);
    Jphi(row, idxV) = -sin(argV) * ones(1,dim) / vScale;
    row = row + 1;

    % ---------------------------------------------------------------------
    % phi = rBar' * vBar
    %
    % d/dr = vBar' / rScale
    % d/dv = rBar' / vScale
    % ---------------------------------------------------------------------
    Jphi(row, idxR) = vBar' / rScale;
    Jphi(row, idxV) = rBar' / vScale;

end