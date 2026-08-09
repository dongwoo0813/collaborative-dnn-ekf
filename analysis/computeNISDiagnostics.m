function nisStats = computeNISDiagnostics(results, cfg)
%{
Function:
    computeNISDiagnostics.m

Purpose:
    Compute basic NIS consistency diagnostics for each watcher.

    The Normalized Innovation Squared is

        NIS_k = nu_k' S_k^{-1} nu_k.

    For a consistent EKF with measurement dimension nz,

        E[NIS_k] approximately equals nz,

    and the percentage of samples exceeding the 95% chi-square bound
    should be approximately 5%.

Inputs:
    results - simulation results structure containing results.NIS
    cfg     - simulation configuration structure

Outputs:
    nisStats - structure containing ANIS, exceedance rate, max NIS, and
               number of valid measurement updates per watcher
%}

    if ~isfield(results, "NIS")
        error("results.NIS does not exist. Run the simulation with NIS logging first.");
    end

    Nw = cfg.Nw;

    if cfg.dim == 2
        nz = 1;
        chi2_95 = 3.8415;
        chi2_99 = 6.6349;
    elseif cfg.dim == 3
        nz = 2;
        chi2_95 = 5.9915;
        chi2_99 = 9.2103;
    else
        error("Unsupported cfg.dim. Use cfg.dim = 2 or cfg.dim = 3.");
    end

    ANIS = NaN(Nw,1);
    exceed95Rate = NaN(Nw,1);
    exceed99Rate = NaN(Nw,1);
    maxNIS = NaN(Nw,1);
    numValid = zeros(Nw,1);

    for i = 1:Nw

        nis_i = results.NIS(:,i);

        if isfield(results, "measAvail")
            avail_i = results.measAvail(:,i);
            nis_i = nis_i(avail_i);
        end

        nis_i = nis_i(isfinite(nis_i));

        numValid(i) = numel(nis_i);

        if isempty(nis_i)
            continue;
        end

        ANIS(i) = mean(nis_i);
        exceed95Rate(i) = mean(nis_i > chi2_95);
        exceed99Rate(i) = mean(nis_i > chi2_99);
        maxNIS(i) = max(nis_i);

    end

    nisStats.nz = nz;
    nisStats.chi2_95 = chi2_95;
    nisStats.chi2_99 = chi2_99;
    nisStats.ANIS = ANIS;
    nisStats.exceed95Rate = exceed95Rate;
    nisStats.exceed99Rate = exceed99Rate;
    nisStats.maxNIS = maxNIS;
    nisStats.numValid = numValid;

    fprintf("\n=== NIS Diagnostics ===\n");
    fprintf("Measurement dimension nz: %d\n", nz);
    fprintf("Expected ANIS if consistent: %.3f\n", nz);
    fprintf("Chi-square 95%% bound: %.4f\n", chi2_95);
    fprintf("Chi-square 99%% bound: %.4f\n\n", chi2_99);

    fprintf("%10s %12s %12s %12s %12s %12s\n", ...
        "Watcher", "Valid", "ANIS", ">95 Rate", ">99 Rate", "Max NIS");

    for i = 1:Nw
        fprintf("%10d %12d %12.4f %12.4f %12.4f %12.4f\n", ...
            i, numValid(i), ANIS(i), exceed95Rate(i), exceed99Rate(i), maxNIS(i));
    end

    fprintf("=======================\n\n");

end