function result = marginalizeJointParameterFIM(state)
%MARGINALIZEJOINTPARAMETERFIM Remove initial kinematic-state information.
%
% For I(q), q=[eta(t0);theta], the equivalent FIM for theta with eta(t0)
% treated as an unknown nuisance parameter is
%
%   I_theta|eta = I_tt-I_te pinv(I_ee) I_et.

% The pseudoinverse is required for angle-only windows in which the initial
% kinematic state is not fully observable.

    nEta=state.nEta; nTheta=state.nTheta;
    I=0.5*(state.information+state.information.');
    ie=1:nEta; it=nEta+(1:nTheta);
    Ieta=I(ie,ie);
    Itheta=I(it,it)-I(it,ie)*pinv(Ieta)*I(ie,it);
    Itheta=projectPSD(Itheta);
    eigenvalues=sort(real(eig(Itheta)),'ascend');
    scale=max(eigenvalues(end),1);
    tol=scale*1e-10;
    rankValue=sum(eigenvalues>tol);
    if eigenvalues(1)>tol
        conditionNumber=eigenvalues(end)/eigenvalues(1);
        worstParameterSigma=1/sqrt(eigenvalues(1));
    else
        conditionNumber=Inf;
        worstParameterSigma=Inf;
    end
    result=struct('information',Itheta,'eigenvalues',eigenvalues, ...
        'effectiveRank',rankValue,'conditionNumber',conditionNumber, ...
        'worstParameterSigma',worstParameterSigma, ...
        'elapsedTime',state.elapsedTime, ...
        'numMeasurements',state.numMeasurements);
end

function A = projectPSD(A)
    A=0.5*(A+A.'); [V,D]=eig(A); d=real(diag(D));
    scale=max(max(abs(d)),1); d(d<scale*1e-12)=0;
    A=V*diag(d)*V.'; A=0.5*(A+A.');
end
