function check_joint_sensitivity_fim_ct_dt()
%CHECK_JOINT_SENSITIVITY_FIM_CT_DT Compare CT and sampled-data FIM forms.
%
% Scalar constant acceleration observed through position:
%   xdot = [0 1;0 0]x + [0;1]theta,  y=[1 0]x+noise.
% Initial position/velocity are nuisance parameters. As dt decreases, the
% continuous and discrete marginal theta information must agree when
% Rc=R*dt.

    addpath(genpath(pwd));
    nEta=2; nTheta=1;
    A=[0 1;0 0]; B=[0;1]; H=[1 0];
    dt=0.01; T=2; R=0.01; Rc=R*dt;
    F=[1 dt;0 1]; L=[0.5*dt^2;dt];
    ct=initJointSensitivityFIM(nEta,nTheta);
    ds=initJointSensitivityFIM(nEta,nTheta);
    for k=1:round(T/dt)
        ct=propagateJointSensitivityFIMContinuous( ...
            ct,A,B,H,Rc,dt,2);
        ds=updateJointSensitivityFIMDiscrete( ...
            ds,F,L,H,R,true,dt);
    end
    mct=marginalizeJointParameterFIM(ct);
    mds=marginalizeJointParameterFIM(ds);
    relativeError=norm(mct.information-mds.information,'fro')/ ...
        max(norm(mct.information,'fro'),eps);
    fprintf('Continuous marginal information = %.12g\n',mct.information);
    fprintf('Discrete marginal information   = %.12g\n',mds.information);
    fprintf('Relative error                  = %.3e\n',relativeError);
    assert(relativeError<3e-2, ...
        'Continuous/discrete FIM consistency check failed.');
    fprintf('Continuous/discrete joint-sensitivity FIM check PASSED.\n');
end
