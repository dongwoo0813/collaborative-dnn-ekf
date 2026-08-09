function phi = featureBlock(i, eta, cfg)
%{
Function:
    featureBlock.m

Purpose:
    Compute the fixed feature vector for DNN residual branch i.

    In the collaborative DNN-EKF architecture, each watcher/branch i uses a
    local residual model of the form

        d_i(eta; theta_i) = W_i phi_i(eta),

    where

        phi_i(eta)

    is a fixed nonlinear feature vector and

        theta_i = vec(W_i)

    contains the trainable output-layer weights.

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

             Optional fields:
                 cfg.dnn.rScale
                 cfg.dnn.vScale

Outputs:
    phi    - Fixed feature vector for branch i.
             Size: nPhi x 1.

             The current design uses

                 nPhi = 2*cfg.dim + 5.

Main equations:
    The branch residual model will be

        d_i(eta; theta_i) = W_i phi_i(eta),

    where

        W_i in R^{cfg.dim x nPhi}.

    The normalized variables are

        r_bar = r_t / rScale,

        v_bar = v_t / vScale.

    The feature vector is

        phi_i(eta)
        =
        [ 1;
          r_bar;
          v_bar;
          ||r_bar||^2;
          sin(psi_i + 1^T r_bar);
          cos(psi_i + 1^T v_bar);
          r_bar^T v_bar ],

    where

        psi_i = 2*pi*(i-1)/cfg.Nw.

Notes:
    - This function does not contain trainable parameters.
    - Branch-to-branch diversity is introduced through psi_i.
    - The scaling rScale and vScale help avoid numerically large features.
    - Later, branchOutput.m will compute W_i * phi_i(eta).
    - Later, branchJacobianTheta.m will use the fact that the model is
      linear in theta_i.
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

    phi = [1;
            rBar;
            vBar;
            rBar' * rBar;
            sin(psi + sum(rBar));
            cos(psi + sum(vBar));
            rBar' * vBar ];

end