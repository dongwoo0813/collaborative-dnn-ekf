function dHat_i = branchOutput(i, eta, theta_i, cfg)
%{
Function:
    branchOutput.m

Purpose:
    Compute the residual acceleration contribution of DNN branch i.

    In the current fixed-feature DNN-EKF design, branch i has the form

        d_i(eta; theta_i) = W_i phi_i(eta),

    where

        phi_i(eta)

    is computed by featureBlock.m and

        theta_i = vec(W_i)

    is the trainable output-layer parameter vector estimated by the EKF.

Inputs:
    i        - Branch index.
               Type: positive integer.
               Range: 1 <= i <= cfg.Nw.

    eta      - Target physical state.
               Size: 2*cfg.dim x 1.
               Definition:
                   eta = [r_t; v_t].

    theta_i  - Branch parameter vector.
               Size: cfg.dim*nPhi x 1.
               Definition:
                   theta_i = vec(W_i),
                   where W_i has size cfg.dim x nPhi.

    cfg      - Simulation configuration structure.
               Required fields:
                   cfg.dim
                   cfg.Nw

Outputs:
    dHat_i   - Branch residual acceleration output.
               Size: cfg.dim x 1.

Main equations:
    The fixed-feature branch model is

        d_i(eta; theta_i) = W_i phi_i(eta),

    where

        W_i = reshape(theta_i, cfg.dim, nPhi).

    The feature dimension is inferred from

        nPhi = length(featureBlock(i,eta,cfg)).

Notes:
    - This function is linear in theta_i.
    - This makes the EKF Jacobian with respect to theta_i simple.
    - The branch output has the same dimension as target acceleration.
    - This function does not modify theta_i.
%}

    dim = cfg.dim;

    phi_i = featureBlock(i, eta, cfg);
    nPhi = numel(phi_i);

    expectedLength = dim * nPhi;

    if numel(theta_i) ~= expectedLength
        error("theta_i has wrong length. Expected %d, got %d.", ...
            expectedLength, numel(theta_i));
    end

    W_i = reshape(theta_i, dim, nPhi);

    dHat_i = W_i * phi_i;

end