function out = debug_step04d_Qnonlocal_components(results)
%{
Function:
    debug_step04d_Qnonlocal_components

Purpose:
    Print and return diagnostic information for the Step 04b nonlocal
    GS-branch covariance injection.

What this checks:
    This function checks why Qnonlocal may be numerically small by inspecting
    the final watcher diagnostics stored in

        watcher.lastNonlocalCovInjection.

    It does not change the filter. It only reports the current magnitude of
    the nonlocal covariance terms.

Inputs:
    results - Output structure from simulate_GS_DNN_EKF or from a comparison
              script containing results.watchersFinal.

Outputs:
    out - Struct containing per-watcher diagnostic table.

Notes:
    This is a debugging/interpretation helper. It should be used before adding
    any artificial nonlocal covariance scaling. In the final method, the
    magnitude of Qnonlocal should come from Ptheta_GS, covariance aging,
    Young coefficients, Sres, betaDNN, and the discrete-time mapping M.
%}

    if ~isfield(results, "watchersFinal") || isempty(results.watchersFinal)
        error("results.watchersFinal is missing. Run a GS DNN-EKF simulation first.");
    end

    watchers = results.watchersFinal;
    Nw = numel(watchers);

    watcherID = (1:Nw).';
    numActiveNonlocal = NaN(Nw, 1);
    traceSdNonlocal = NaN(Nw, 1);
    traceQnonlocal = NaN(Nw, 1);
    normQnonlocalFro = NaN(Nw, 1);
    maxEigQnonlocal = NaN(Nw, 1);
    minEigQnonlocal = NaN(Nw, 1);

    for i = 1:Nw

        if ~isfield(watchers(i), "lastNonlocalCovInjection")
            continue;
        end

        info = watchers(i).lastNonlocalCovInjection;

        if isfield(info, "numActiveNonlocal")
            numActiveNonlocal(i) = info.numActiveNonlocal;
        end

        if isfield(info, "traceSdNonlocal")
            traceSdNonlocal(i) = info.traceSdNonlocal;
        end

        if isfield(info, "traceQnonlocal")
            traceQnonlocal(i) = info.traceQnonlocal;
        end

        if isfield(info, "Qnonlocal") && ~isempty(info.Qnonlocal)
            Q = 0.5 * (info.Qnonlocal + info.Qnonlocal');
            normQnonlocalFro(i) = norm(Q, "fro");

            eigQ = eig(Q);
            maxEigQnonlocal(i) = max(eigQ);
            minEigQnonlocal(i) = min(eigQ);
        end

    end

    T = table( ...
        watcherID, ...
        numActiveNonlocal, ...
        traceSdNonlocal, ...
        traceQnonlocal, ...
        normQnonlocalFro, ...
        minEigQnonlocal, ...
        maxEigQnonlocal);

    fprintf("\n============================================================\n");
    fprintf("Step 04d Qnonlocal component diagnostics\n");
    fprintf("============================================================\n");
    disp(T);

    fprintf("\nMean trace(SdNonlocal) = %.6e\n", mean(traceSdNonlocal, "omitnan"));
    fprintf("Mean trace(Qnonlocal)  = %.6e\n", mean(traceQnonlocal, "omitnan"));
    fprintf("Mean ||Qnonlocal||_F   = %.6e\n", mean(normQnonlocalFro, "omitnan"));

    out = struct();
    out.table = T;
    out.meanTraceSdNonlocal = mean(traceSdNonlocal, "omitnan");
    out.meanTraceQnonlocal = mean(traceQnonlocal, "omitnan");
    out.meanNormQnonlocalFro = mean(normQnonlocalFro, "omitnan");

end