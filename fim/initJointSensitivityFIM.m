function state = initJointSensitivityFIM(nEta,nTheta,priorInformation)
%INITJOINTSENSITIVITYFIM Initialize a finite-horizon joint FIM state.
%
% The unknown vector at the beginning of one information window is
%
%     q = [eta(t0); theta].
%
% PhiEta = d eta(t)/d eta(t0),  STheta = d eta(t)/d theta.
% Reset this structure at the beginning of every finite window.

    validateattributes(nEta,{'numeric'},{'scalar','integer','positive'});
    validateattributes(nTheta,{'numeric'},{'scalar','integer','positive'});
    nq = nEta+nTheta;
    if nargin < 3 || isempty(priorInformation)
        priorInformation = zeros(nq);
    end
    if any(size(priorInformation)~=[nq nq]) || ...
            any(~isfinite(priorInformation(:)))
        error('priorInformation must be finite (nEta+nTheta)-square.');
    end
    state = struct();
    state.nEta = nEta;
    state.nTheta = nTheta;
    state.PhiEta = eye(nEta);
    state.STheta = zeros(nEta,nTheta);
    state.information = projectPSD(priorInformation);
    state.elapsedTime = 0;
    state.numMeasurements = 0;
end

function A = projectPSD(A)
    A = 0.5*(A+A.');
    [V,D] = eig(A);
    d = real(diag(D));
    scale = max(max(abs(d)),1);
    d(d<scale*1e-12)=0;
    A = V*diag(d)*V.';
    A = 0.5*(A+A.');
end
