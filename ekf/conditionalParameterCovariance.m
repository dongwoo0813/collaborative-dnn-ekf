function PthetaGivenEta = conditionalParameterCovariance(P,idxEta,idxTheta)
%CONDITIONALPARAMETERCOVARIANCE Gaussian P(theta | eta).
%
%   P(theta|eta) = Ptt-Pte*pinv(Pee)*Pet.
%
% This removes the part of parameter uncertainty that can be explained by
% the contemporaneous kinematic-state uncertainty.  It is the covariance
% counterpart of eliminating eta as a nuisance variable in a joint FIM.

    Pee = symmetrize(P(idxEta,idxEta));
    Ptt = symmetrize(P(idxTheta,idxTheta));
    Pte = P(idxTheta,idxEta);
    PthetaGivenEta = Ptt-Pte*pinv(Pee)*Pte.';
    PthetaGivenEta = projectPSD(PthetaGivenEta);
end

function A = symmetrize(A)
    A=0.5*(A+A.');
end

function A = projectPSD(A)
    A=symmetrize(A);
    [V,D]=eig(A);
    d=real(diag(D));
    scale=max(max(abs(d)),1);
    d(d<scale*1e-13)=0;
    A=V*diag(d)*V.';
    A=symmetrize(A);
end
