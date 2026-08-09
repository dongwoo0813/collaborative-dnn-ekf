function [B,diagInfo] = computeOutputInformationWeights( ...
    watcher,branchUsed,JthetaRawAll,cfg)
%COMPUTEOUTPUTINFORMATIONWEIGHTS Residual-output precision fusion weights.
%
% Each branch is interpreted as a local expert estimating the same global
% residual acceleration.  For branch j,
%
%   Sigma_d,j = Jtheta_j P(theta_j|eta_j) Jtheta_j' + Sigma_epsilon,j,
%   Omega_d,j = inv(Sigma_d,j),
%   B_j       = inv(sum_l Omega_d,l) Omega_d,j.
%
% No LOS-projector EMA, branch-count multiplier, or residual prior is used.
% If the combined precision is not full rank, this function fails rather
% than silently supplying an unobservable direction with a hidden prior.

    dim=cfg.dim; Nw=cfg.Nw;
    branchUsed=logical(branchUsed(:));
    if numel(branchUsed)~=Nw || ~any(branchUsed)
        error('computeOutputInformationWeights:BadBranchSet', ...
            'branchUsed must select at least one of cfg.Nw branches.');
    end
    if size(JthetaRawAll,1)~=dim || size(JthetaRawAll,3)~=Nw
        error('computeOutputInformationWeights:BadJthetaSize', ...
            'JthetaRawAll has incompatible dimensions.');
    end

    minVariance=getNumeric(cfg,'minOutputVariance',1e-20);
    if minVariance<=0
        error('computeOutputInformationWeights:BadVarianceFloor', ...
            'minOutputVariance must be strictly positive.');
    end
    approxVariance=getNumeric(cfg,'approxErrorVariance',0);
    rankTolerance=getNumeric(cfg,'rankTolerance',1e-10);
    requireFullRank=getLogical(cfg,'requireFullRank',true);
    Sigma=zeros(dim,dim,Nw);
    Omega=zeros(dim,dim,Nw);
    Pconditional=cell(Nw,1);

    for j=1:Nw
        if ~branchUsed(j), continue; end
        if j==watcher.localBranchID
            Pj=conditionalParameterCovariance( ...
                watcher.P,watcher.idxEta,watcher.idxTheta);
        else
            rec=watcher.gsBranches(j);
            if isfield(rec,'PthetaConditional') && ...
                    ~isempty(rec.PthetaConditional) && ...
                    all(isfinite(rec.PthetaConditional(:)))
                Pj=0.5*(rec.PthetaConditional+rec.PthetaConditional.');
            else
                Pj=0.5*(rec.Ptheta+rec.Ptheta.');
            end
        end
        J=JthetaRawAll(:,:,j);
        Sj=J*Pj*J.'+approxVariance*eye(dim);
        Sj=0.5*(Sj+Sj.');
        [V,D]=eig(Sj);
        e=real(diag(D));
        e=max(e,minVariance);
        Sj=V*diag(e)*V.';
        Oj=V*diag(1./e)*V.';
        Sigma(:,:,j)=0.5*(Sj+Sj.');
        Omega(:,:,j)=0.5*(Oj+Oj.');
        Pconditional{j}=Pj;
    end

    OmegaSum=sum(Omega,3);
    OmegaSum=0.5*(OmegaSum+OmegaSum.');
    eigSum=sort(real(eig(OmegaSum)),'ascend');
    threshold=max(eigSum(end),1)*rankTolerance;
    effectiveRank=sum(eigSum>threshold);
    if requireFullRank && effectiveRank<dim
        error('computeOutputInformationWeights:RankDeficient', ...
            ['Combined output precision has rank %d < %d. ', ...
             'The configured watchers do not observe every residual direction.'], ...
            effectiveRank,dim);
    end

    B=zeros(dim,dim,Nw);
    if effectiveRank==dim
        for j=1:Nw
            if branchUsed(j), B(:,:,j)=OmegaSum\Omega(:,:,j); end
        end
    else
        OmegaSumInv=pinv(OmegaSum);
        for j=1:Nw
            if branchUsed(j), B(:,:,j)=OmegaSumInv*Omega(:,:,j); end
        end
    end
    sumB=sum(B,3);
    diagInfo=struct('enabled',true,'mode',"output_information_fusion", ...
        'B',B,'OmegaBars',Omega,'OmegaOutput',Omega, ...
        'SigmaOutput',Sigma,'OmegaSigma',OmegaSum, ...
        'PthetaConditional',{Pconditional}, ...
        'eigenvaluesOmegaSigma',eigSum, ...
        'effectiveRank',effectiveRank, ...
        'minEigOmegaSigma',eigSum(1), ...
        'maxEigOmegaSigma',eigSum(end), ...
        'condOmegaSigma',conditionFromEigenvalues(eigSum,threshold), ...
        'sumGate',sumB, ...
        'sumGateIdentityError',norm(sumB-eye(dim),'fro'), ...
        'minOutputVariance',minVariance, ...
        'approxErrorVariance',approxVariance);
end

function value=getNumeric(cfg,name,defaultValue)
    value=defaultValue;
    if isfield(cfg,'gs') && isfield(cfg.gs,'outputInfoFusion') && ...
            isfield(cfg.gs.outputInfoFusion,name)
        value=cfg.gs.outputInfoFusion.(name);
    end
    validateattributes(value,{'numeric'},{'scalar','real','finite','nonnegative'});
end

function value=getLogical(cfg,name,defaultValue)
    value=defaultValue;
    if isfield(cfg,'gs') && isfield(cfg.gs,'outputInfoFusion') && ...
            isfield(cfg.gs.outputInfoFusion,name)
        value=logical(cfg.gs.outputInfoFusion.(name));
    end
end

function c=conditionFromEigenvalues(e,tol)
    if e(1)>tol, c=e(end)/e(1); else, c=Inf; end
end
