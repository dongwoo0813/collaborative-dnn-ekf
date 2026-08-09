function out=check_fim_weighted_additive()
%CHECK_FIM_WEIGHTED_ADDITIVE Verify LOS projector and additive semantics.

    cfg=struct();
    cfg.dim=2;
    cfg.Nw=2;
    cfg.gs.fimWeightedAdditive.relativeEigenvalueFloor=1e-12;
    watcher=struct();
    watcher.localBranchID=1;

    % Branch 1: LOS u=[1;0], hence I-u*u'=diag([0,1]).
    u=[1;0];
    watcher.OmegaBar=eye(2)-u*u.';
    record=struct('OmegaBar',diag([4,1]));
    watcher.gsBranches=repmat(record,2,1);
    watcher.gsBranches(2).OmegaBar=diag([4,1]);

    [W,diagInfo]=computeFIMWeightedAdditiveWeights( ...
        watcher,true(2,1),cfg);
    expected1=diag([0,1]);
    expected2=diag([1,0.25]);
    assert(norm(W(:,:,1)-expected1,'fro')<1e-12, ...
        'Single-bearing weight must equal I-u*u''.');
    assert(norm(W(:,:,2)-expected2,'fro')<1e-12, ...
        'Relative FIM eigenvalues were not preserved.');

    d1=[2;3];
    d2=[5;7];
    additive=W(:,:,1)*d1+W(:,:,2)*d2;
    expected=expected1*d1+expected2*d2;
    assert(norm(additive-expected)<1e-12);
    assert(norm(sum(W,3)-eye(2),'fro')>1e-3, ...
        'Weights must not be normalized across additive branches.');

    [Omega1,~]=updateOmegaBar(zeros(2),u,0.02,false, ...
        "cumulative_sum");
    [Omega2,~]=updateOmegaBar(Omega1,[0;1],0.02,false, ...
        "cumulative_sum");
    assert(norm(Omega1-expected1,'fro')<1e-12, ...
        'Cumulative mode must add the instantaneous LOS projector.');
    assert(norm(Omega2-eye(2),'fro')<1e-12, ...
        'Orthogonal LOS histories must accumulate to full 2-D support.');

    out=struct('W',W,'sumW',sum(W,3),'additive',additive, ...
        'diag',diagInfo);
    fprintf(['FIM-weighted additive sanity check passed: ', ...
        'W1=I-u*u'', and ||sum W-I||_F=%.3e (intentionally nonzero).\n'], ...
        diagInfo.sumWeightIdentityDeviation);
end
