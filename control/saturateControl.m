function uSat = saturateControl(u, umax)
%{
Function:
    saturateControl.m

Purpose:
    Saturate a control vector by its Euclidean norm.

Inputs:
    u     - Control vector.
            Size: n x 1.

    umax  - Maximum allowed norm.
            Type: nonnegative scalar.

Outputs:
    uSat  - Saturated control vector.
            Size: n x 1.

Main equation:
    If ||u|| <= umax, then

        uSat = u.

    If ||u|| > umax, then

        uSat = umax * u / ||u||.

Notes:
    - This can be used for both translational thrust and attitude torque.
%}

    if umax <= 0
        uSat = zeros(size(u));
        return;
    end

    nrm = norm(u);

    if nrm > umax
        uSat = umax * u / nrm;
    else
        uSat = u;
    end

end