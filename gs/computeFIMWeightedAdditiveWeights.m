function [W,diagInfo] = computeFIMWeightedAdditiveWeights( ...
    watcher,branchUsed,cfg)
%COMPUTEFIMWEIGHTEDADDITIVEWEIGHTS Per-branch geometry information weights.
%
% Each shared DNN is an additive function block, not a redundant estimate
% of the entire residual.  Consequently the weights are normalized within
% each branch rather than across branches:
%
%   W_j = Omega_j/lambda_max(Omega_j),
%   d   = sum_j W_j d_j.
%
% There is deliberately no constraint sum_j W_j=I.  For an instantaneous
% 2-D bearing geometry Omega_j is proportional to I-u_j*u_j', so this rule
% reduces exactly to the LOS-transverse projector.  The accumulated
% OmegaBar metadata allows a second direction to enter continuously through
% its relative eigenvalue lambda_2/lambda_1.

    dim=cfg.dim;
    Nw=cfg.Nw;
    branchUsed=logical(branchUsed(:));
    if numel(branchUsed)~=Nw
        error('computeFIMWeightedAdditiveWeights:BadBranchSet', ...
            'branchUsed must have cfg.Nw elements.');
    end

    relativeFloor=getRelativeFloor(cfg);
    OmegaBars=zeros(dim,dim,Nw);
    W=zeros(dim,dim,Nw);
    lambdaMax=zeros(Nw,1);
    normalizedEigenvalues=zeros(dim,Nw);
    effectiveRank=zeros(Nw,1);

    for j=1:Nw
        if ~branchUsed(j), continue; end
        Omega=readBranchInformation(watcher,j,dim);
        Omega=projectPSD(Omega,relativeFloor);
        [V,D]=eig(Omega);
        eigenvalues=max(real(diag(D)),0);
        [eigenvalues,order]=sort(eigenvalues,'descend');
        V=V(:,order);
        scale=eigenvalues(1);
        OmegaBars(:,:,j)=V*diag(eigenvalues)*V.';
        lambdaMax(j)=scale;
        if scale>0
            relative=eigenvalues/scale;
            relative(relative<relativeFloor)=0;
            W(:,:,j)=V*diag(relative)*V.';
            normalizedEigenvalues(:,j)=relative;
            effectiveRank(j)=nnz(relative>0);
        end
        W(:,:,j)=0.5*(W(:,:,j)+W(:,:,j).');
    end

    OmegaSum=sum(OmegaBars,3);
    OmegaSum=0.5*(OmegaSum+OmegaSum.');
    eigSum=sort(max(real(eig(OmegaSum)),0),'ascend');
    positive=eigSum(eigSum>max(eigSum(end),1)*relativeFloor);
    if isempty(positive)
        condSum=Inf;
    else
        condSum=eigSum(end)/positive(1);
    end
    sumW=sum(W,3);
    diagInfo=struct('enabled',true,'mode',"fim_weighted_additive", ...
        'B',W,'W',W,'OmegaBars',OmegaBars,'OmegaSigma',OmegaSum, ...
        'lambdaMaxByBranch',lambdaMax, ...
        'normalizedEigenvalues',normalizedEigenvalues, ...
        'effectiveRankByBranch',effectiveRank, ...
        'sumGate',sumW,'sumWeight',sumW, ...
        'sumGateIdentityError',norm(sumW-eye(dim),'fro'), ...
        'sumWeightIdentityDeviation',norm(sumW-eye(dim),'fro'), ...
        'minEigOmegaSigma',eigSum(1), ...
        'condOmegaSigma',condSum, ...
        'relativeEigenvalueFloor',relativeFloor);
end

function Omega=readBranchInformation(watcher,branchID,dim)
    if branchID==watcher.localBranchID
        if isfield(watcher,'OmegaBar') && ~isempty(watcher.OmegaBar)
            Omega=watcher.OmegaBar;
        else
            Omega=zeros(dim);
        end
    elseif isfield(watcher,'gsBranches') && ...
            numel(watcher.gsBranches)>=branchID && ...
            isfield(watcher.gsBranches(branchID),'OmegaBar') && ...
            ~isempty(watcher.gsBranches(branchID).OmegaBar)
        Omega=watcher.gsBranches(branchID).OmegaBar;
    else
        Omega=zeros(dim);
    end
    if any(size(Omega)~=[dim dim]) || any(~isfinite(Omega(:)))
        error('computeFIMWeightedAdditiveWeights:BadInformation', ...
            'Branch %d OmegaBar must be a finite %d-by-%d matrix.', ...
            branchID,dim,dim);
    end
end

function A=projectPSD(A,relativeFloor)
    A=0.5*(double(A)+double(A).');
    [V,D]=eig(A);
    d=max(real(diag(D)),0);
    scale=max(d);
    if scale>0, d(d<scale*relativeFloor)=0; end
    A=V*diag(d)*V.';
    A=0.5*(A+A.');
end

function value=getRelativeFloor(cfg)
    value=1e-10;
    if isfield(cfg,'gs') && isfield(cfg.gs,'fimWeightedAdditive') && ...
            isfield(cfg.gs.fimWeightedAdditive,'relativeEigenvalueFloor')
        value=cfg.gs.fimWeightedAdditive.relativeEigenvalueFloor;
    end
    validateattributes(value,{'numeric'}, ...
        {'scalar','real','finite','nonnegative','<',1});
end
