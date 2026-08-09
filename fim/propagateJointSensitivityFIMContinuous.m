function state = propagateJointSensitivityFIMContinuous( ...
    state,Aeta,Btheta,H,Rc,dt,numSubsteps)
%PROPAGATEJOINTSENSITIVITYFIMCONTINUOUS Continuous-time sensitivity/FIM.
%
% Model over this integration interval:
%   delta eta dot = Aeta delta eta + Btheta delta theta
%   y              = H delta eta + white noise, E[v(t)v(tau)']=Rc delta(t-tau)
%
% Differential equations:
%   Phi dot = Aeta Phi
%   S dot   = Aeta S + Btheta
%   I dot   = Psi' Rc^{-1} Psi,
%   Psi     = H [Phi,S].
%
% Aeta, Btheta, H, and Rc are held at the supplied local linearization over
% dt. RK4 integrates the coupled equations. For a sampled sensor with
% per-sample covariance R and sample interval dtSample, Rc=R*dtSample gives
% the same information rate under a zero-order-hold approximation.

    if nargin < 7 || isempty(numSubsteps)
        numSubsteps = 4;
    end
    validateattributes(dt,{'numeric'},{'scalar','real','finite','positive'});
    validateattributes(numSubsteps,{'numeric'}, ...
        {'scalar','integer','positive'});
    [Aeta,Btheta,H,Rc] = validateInputs(state,Aeta,Btheta,H,Rc);
    RcInv = pinvSymmetricPSD(Rc);
    h = dt/numSubsteps;
    Y = packState(state);
    for k=1:numSubsteps
        k1 = rhs(Y,Aeta,Btheta,H,RcInv,state.nEta,state.nTheta);
        k2 = rhs(Y+0.5*h*k1,Aeta,Btheta,H,RcInv, ...
            state.nEta,state.nTheta);
        k3 = rhs(Y+0.5*h*k2,Aeta,Btheta,H,RcInv, ...
            state.nEta,state.nTheta);
        k4 = rhs(Y+h*k3,Aeta,Btheta,H,RcInv, ...
            state.nEta,state.nTheta);
        Y = Y+(h/6)*(k1+2*k2+2*k3+k4);
    end
    state = unpackState(state,Y);
    state.information = projectPSD(state.information);
    state.elapsedTime = state.elapsedTime+dt;
end

function dY = rhs(Y,A,B,H,Rinv,nEta,nTheta)
    nq = nEta+nTheta;
    nPhi = nEta*nEta;
    nS = nEta*nTheta;
    Phi = reshape(Y(1:nPhi),nEta,nEta);
    S = reshape(Y(nPhi+(1:nS)),nEta,nTheta);
    Z = [Phi,S];
    Psi = H*Z;
    PhiDot = A*Phi;
    SDot = A*S+B;
    IDot = Psi.'*Rinv*Psi;
    IDot = IDot(1:nq,1:nq);
    dY = [PhiDot(:);SDot(:);IDot(:)];
end

function Y = packState(state)
    Y = [state.PhiEta(:);state.STheta(:);state.information(:)];
end

function state = unpackState(state,Y)
    nEta=state.nEta; nTheta=state.nTheta; nq=nEta+nTheta;
    nPhi=nEta*nEta; nS=nEta*nTheta;
    state.PhiEta=reshape(Y(1:nPhi),nEta,nEta);
    state.STheta=reshape(Y(nPhi+(1:nS)),nEta,nTheta);
    state.information=reshape(Y(nPhi+nS+(1:nq*nq)),nq,nq);
end

function [A,B,H,R] = validateInputs(state,A,B,H,R)
    nEta=state.nEta; nTheta=state.nTheta;
    if any(size(A)~=[nEta nEta]), error('Aeta has incompatible size.'); end
    if any(size(B)~=[nEta nTheta]), error('Btheta has incompatible size.'); end
    if size(H,2)~=nEta, error('H must have nEta columns.'); end
    if any(size(R)~=[size(H,1) size(H,1)]), ...
        error('Rc has incompatible size.'); end
    if any(~isfinite([A(:);B(:);H(:);R(:)])), ...
        error('Continuous FIM inputs must be finite.'); end
end

function Ainv = pinvSymmetricPSD(A)
    A=0.5*(A+A.');
    if min(eig(A))<=0, error('Rc must be positive definite.'); end
    Ainv=pinv(A);
end

function A = projectPSD(A)
    A=0.5*(A+A.'); [V,D]=eig(A); d=real(diag(D));
    scale=max(max(abs(d)),1); d(d<scale*1e-12)=0;
    A=V*diag(d)*V.'; A=0.5*(A+A.');
end
