function B_i = branchJacobianTheta(i, eta, cfg)
%{
Function:
    branchJacobianTheta.m

Purpose:
    Compute the Jacobian of DNN residual branch i with respect to its
    parameter vector theta_i.

    The branch residual model is

        d_i(eta; theta_i) = W_i phi_i(eta),

    where

        theta_i = vec(W_i),

    and W_i has size

        cfg.dim x nPhi.

Inputs:
    i      - Branch index.
             Type: positive integer.
             Range: 1 <= i <= cfg.Nw.

    eta    - Target physical state.
             Size: 2*cfg.dim x 1.
             Definition:
                 eta = [r_t; v_t].

    cfg    - Simulation configuration structure.
             Required fields:
                 cfg.dim
                 cfg.Nw

Outputs:
    B_i    - Jacobian of branch output with respect to theta_i.
             Size: cfg.dim x (cfg.dim*nPhi).

Main equations:
    Branch output:

        d_i = W_i phi_i.

    Since

        theta_i = vec(W_i),

    the Jacobian is

        partial d_i / partial theta_i
        =
        kron(phi_i^T, I_dim).

    Therefore,

        B_i = kron(phi_i', eye(cfg.dim)).

Notes:
    - This Jacobian is exact because the branch model is linear in theta_i.
    - This function does not include the derivative with respect to eta.
    - The derivative with respect to eta will be handled separately later
      when building the augmented DNN-EKF prediction Jacobian.
    - This function is used for covariance propagation and for mapping
      parameter uncertainty into residual acceleration uncertainty.
%}

    dim = cfg.dim;

    phi_i = featureBlock(i, eta, cfg);

    B_i = kron(phi_i', eye(dim));

end