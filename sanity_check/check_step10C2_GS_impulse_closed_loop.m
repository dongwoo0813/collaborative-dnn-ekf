function report = check_step10C2_GS_impulse_closed_loop(out,requireImpulse)
%CHECK_STEP10C2_GS_IMPULSE_CLOSED_LOOP Validate live controller comparisons.

    if nargin < 2, requireImpulse = true; end

    required = ["resLocal","resAdd","resOutputInformation","summary"];
    assert(all(isfield(out,required)),"Step 10-C.2 output is incomplete.");
    results = {out.resLocal,out.resAdd,out.resOutputInformation};
    for i = 1:numel(results)
        res = results{i};
        if requireImpulse
            assert(any(res.controllerActive(:)), ...
                "Closed-loop case %d did not execute an impulse.",i);
            assert(any(vecnorm(res.watcherU,2,1) > 0,"all"), ...
                "Closed-loop case %d logged no nonzero watcher force.",i);
        else
            assert(~any(vecnorm(res.watcherU,2,1) > 0,"all"), ...
                "Coast case %d unexpectedly logged a nonzero force.",i);
        end
    end
    assert(all(isfinite(out.summary.positionRMSE)) && ...
        all(isfinite(out.summary.operationalResidualRMSE)), ...
        "Closed-loop metrics must be finite.");
    report = struct("passed",true, ...
        "impulseSteps",out.summary.impulseSteps, ...
        "meanFinalDeltaV",out.summary.meanFinalDeltaV);
    fprintf("Step 10-C.2 closed-loop check passed: impulse steps [%s].\n", ...
        num2str(report.impulseSteps.'));
end
