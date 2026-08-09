function check_step09J6_estimate_norm_cosine_diagnostic()
% Targeted dimension, norm-ratio, cosine, and zero-norm check for Step 09-J.6.
    t = (0:2).';
    dTrue = [1 0 0; 0 2 0];
    dAdd = zeros(2,3,2);
    dFIM = zeros(2,3,2);
    dAdd(:,:,1) = [1 0 0; 0 4 0];
    dAdd(:,:,2) = [2 0 0; 0 2 0];
    dFIM(:,:,1) = [-1 0 0; 0 2 0];
    dFIM(:,:,2) = [0 0 0; 1 2 0];
    resAdd = struct("time", t, "trueResidual", dTrue, "dnnResidual", dAdd);
    resFIM = struct("time", t, "trueResidual", dTrue, "dnnResidual", dFIM);
    out = struct("resGSAdd", resAdd, "resGSFIM", resFIM);
    diag = run_step09J6_estimate_norm_cosine_diagnostic(out, [0 2], false);
    assert(isequal(size(diag.dHatAdd), [2 3 2]));
    assert(abs(diag.normRatioAdd(2,1) - 2) < 1e-12);
    assert(abs(diag.cosineAdd(1,1) - 1) < 1e-12);
    assert(abs(diag.cosineFIM(1,1) + 1) < 1e-12);
    assert(isnan(diag.cosineAdd(3,1)), "Zero truth must produce NaN cosine.");
    assert(all(isfinite(diag.windowSummaryTable.meanNormRatioAdd)));
    fprintf("Step 09-J.6 estimate norm/cosine check PASSED.\n");
end
