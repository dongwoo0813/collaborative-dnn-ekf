function state = updateJointSensitivityFIMDiscrete( ...
    state,Feta,Ltheta,H,R,measurementAvailable,dt)
%UPDATEJOINTSENSITIVITYFIMDISCRETE Discrete sensitivity and sampled FIM.
%
% Propagation:
%   Phi_{k+1} = Feta_k Phi_k
%   S_{k+1}   = Feta_k S_k + Ltheta_k
%
% Measurement information at k+1:
%   Psi_{k+1} = H_{k+1}[Phi_{k+1},S_{k+1}]
%   I_{k+1}   = I_k + Psi' R^{-1} Psi.
%
% Ltheta is the discrete state sensitivity d eta_{k+1}/d theta with eta_k
% held fixed. For an augmented transition F_aug, it is its eta-theta block.

    if nargin < 6
        measurementAvailable = true;
    end
    if nargin < 7 || isempty(dt)
        dt = 1;
    end
    validateattributes(dt,{'numeric'},{'scalar','real','finite','positive'});
    nEta=state.nEta; nTheta=state.nTheta;
    if any(size(Feta)~=[nEta nEta]), error('Feta has incompatible size.'); end
    if any(size(Ltheta)~=[nEta nTheta]), ...
        error('Ltheta has incompatible size.'); end
    if size(H,2)~=nEta, error('H must have nEta columns.'); end
    if any(size(R)~=[size(H,1) size(H,1)]), ...
        error('R has incompatible size.'); end
    if any(~isfinite([Feta(:);Ltheta(:);H(:);R(:)])), ...
        error('Discrete FIM inputs must be finite.'); end

    state.PhiEta=Feta*state.PhiEta;
    state.STheta=Feta*state.STheta+Ltheta;
    if logical(measurementAvailable)
        if min(eig(0.5*(R+R.')))<=0, error('R must be positive definite.'); end
        Z=[state.PhiEta,state.STheta];
        Psi=H*Z;
        state.information=state.information+Psi.'*(R\Psi);
        state.information=projectPSD(state.information);
        state.numMeasurements=state.numMeasurements+1;
    end
    state.elapsedTime=state.elapsedTime+dt;
end

function A = projectPSD(A)
    A=0.5*(A+A.'); [V,D]=eig(A); d=real(diag(D));
    scale=max(max(abs(d)),1); d(d<scale*1e-12)=0;
    A=V*diag(d)*V.'; A=0.5*(A+A.');
end
