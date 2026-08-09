function out=check_output_information_fusion()
%CHECK_OUTPUTINFORMATIONFUSION Verify matrix information-weight algebra.

    cfg=struct();
    cfg.dim=2;
    cfg.Nw=3;
    cfg.gs.outputInfoFusion.minOutputVariance=1e-20;
    cfg.gs.outputInfoFusion.approxErrorVariance=0;
    cfg.gs.outputInfoFusion.rankTolerance=1e-12;
    cfg.gs.outputInfoFusion.requireFullRank=true;

    watcher=struct();
    watcher.localBranchID=1;
    watcher.idxEta=1:4;
    watcher.idxTheta=5:6;
    watcher.P=blkdiag(eye(4),diag([4,9]));
    blank=struct('Ptheta',eye(2),'PthetaConditional',eye(2));
    watcher.gsBranches=repmat(blank,3,1);
    watcher.gsBranches(2).PthetaConditional=diag([1,16]);
    watcher.gsBranches(3).PthetaConditional=diag([25,1]);

    JthetaRawAll=repmat(eye(2),1,1,3);
    branchUsed=true(3,1);
    [B,diagInfo]=computeOutputInformationWeights( ...
        watcher,branchUsed,JthetaRawAll,cfg);

    Sigma=cat(3,diag([4,9]),diag([1,16]),diag([25,1]));
    Omega=zeros(2,2,3);
    for j=1:3, Omega(:,:,j)=inv(Sigma(:,:,j)); end %#ok<MINV>
    OmegaSum=sum(Omega,3);
    Bexpected=zeros(2,2,3);
    for j=1:3, Bexpected(:,:,j)=OmegaSum\Omega(:,:,j); end

    assert(norm(B-Bexpected,'fro')<1e-12, ...
        'Information weights do not match the closed-form result.');
    assert(diagInfo.sumGateIdentityError<1e-12, ...
        'Information weights do not sum to identity.');
    assert(diagInfo.effectiveRank==cfg.dim, ...
        'Combined output precision should be full rank.');

    d=[1,3,-2;2,-1,4];
    fused=zeros(2,1);
    rhs=zeros(2,1);
    for j=1:3
        fused=fused+B(:,:,j)*d(:,j);
        rhs=rhs+Omega(:,:,j)*d(:,j);
    end
    direct=OmegaSum\rhs;
    assert(norm(fused-direct)<1e-12, ...
        'Weighted output does not equal information-form fusion.');

    out=struct('B',B,'Sigma',Sigma,'Omega',Omega, ...
        'OmegaSum',OmegaSum,'fused',fused, ...
        'sumGateIdentityError',diagInfo.sumGateIdentityError);
    fprintf(['Output-information fusion sanity check passed: ', ...
        '||sum B-I||_F=%.3e.\n'],diagInfo.sumGateIdentityError);
end
