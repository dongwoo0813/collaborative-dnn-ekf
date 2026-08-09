function aNom = targetNominalDynamics(t, r, v, cfg)
    %{
    Function:
        targetNominalDynamics.m
    
    Purpose:
        Compute the nominal target acceleration used by the truth model and,
        later, by the EKF prediction model.
    
        In Step 01, the target is modeled as a simple double integrator with
        zero nominal acceleration:
    
            r_dot = v,
            v_dot = a_nom,
    
        where
    
            a_nom = 0.
    
    Inputs:
        t     - Current simulation time.
                Type: scalar.
                Units: seconds.
    
        r     - Target position vector.
                Size: cfg.dim x 1.
    
        v     - Target velocity vector.
                Size: cfg.dim x 1.
    
        cfg   - Simulation configuration structure.
                Required fields:
                    cfg.dim
    
    Outputs:
        aNom  - Nominal target acceleration.
                Size: cfg.dim x 1.
    
    Main equations:
        Step 01 nominal dynamics:
    
            a_nom(r,v,t) = 0.
    
        Later, this can be replaced by orbital dynamics such as
    
            a_nom(r) = -mu r / ||r||^3.
    
    Notes:
        - Keep this file dimension-generic.
        - Do not hard-code 2D or 3D state indices here.
        - Unknown residual acceleration should be added in trueResidual.m later,
          not inside this nominal model.
    %}
    
    dim = cfg.dim;
    
    aNom = zeros(dim,1);

end