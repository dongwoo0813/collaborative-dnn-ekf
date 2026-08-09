function xNext = propagateRK4(f, t, x, dt)
%{
Function:
    propagateRK4.m

Purpose:
    Propagate a continuous-time state over one time step using the classical
    fourth-order Runge-Kutta method.

Inputs:
    f      - Function handle for the continuous-time dynamics.
             Expected form:
                 xdot = f(t, x).

    t      - Current time.
             Type: scalar.

    x      - Current state.
             Size: n x 1.

    dt     - Time step.
             Type: scalar.

Outputs:
    xNext  - Propagated state after one time step.
             Size: n x 1.

Main equations:
    The RK4 update is

        k1 = f(t, x),

        k2 = f(t + dt/2, x + dt k1/2),

        k3 = f(t + dt/2, x + dt k2/2),

        k4 = f(t + dt,   x + dt k3),

        x_{k+1} = x_k + dt/6 (k1 + 2k2 + 2k3 + k4).

Notes:
    - This function is generic and can be used for target dynamics,
      watcher dynamics, and later controlled dynamics.
%}
    
    k1 = f(t, x);
    k2 = f(t + 0.5*dt, x + 0.5*dt*k1);
    k3 = f(t + 0.5*dt, x + 0.5*dt*k2);
    k4 = f(t + dt,     x + dt*k3);
    
    xNext = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);

end