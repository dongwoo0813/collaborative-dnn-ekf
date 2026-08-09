function result = run_step10A4_multiseed_ablation(seeds,makePlots,desiredBaseline,simulationTime)
%RUN_STEP10A4_MULTISEED_ABLATION Repeat Step 10-A.3 over random seeds.
%
% Example:
%   result = run_step10A4_multiseed_ablation([101 202 303],false,100,600);

    if nargin < 1 || isempty(seeds), seeds = [101 202 303]; end
    if nargin < 2, makePlots = false; end
    if nargin < 3, desiredBaseline = 100; end
    if nargin < 4, simulationTime = 600; end
    seeds = double(seeds(:).');
    allRows = table();
    runs = cell(1,numel(seeds));
    for iseed = 1:numel(seeds)
        fprintf("\nStep 10-A.4 seed %g/%g\n",iseed,numel(seeds));
        runs{iseed} = run_step10A3_trajectory_ablation( ...
            makePlots,desiredBaseline,simulationTime,seeds(iseed));
        rows = runs{iseed}.summary;
        if isempty(allRows)
            allRows = rows;
        else
            allRows = [allRows; rows]; %#ok<AGROW>
        end
    end

    metricNames = {"finalPositionRMSE","finalRangeRMSE", ...
        "finalVelocityRMSE","finalResidualVectorRMSE", ...
        "meanLOSChangeOverSigma","meanSelectedInformationMinEig", ...
        "meanDeltaV"};
    combos = unique(allRows(:,{'trajectory','caseName'}),'rows');
    nCombo = height(combos);
    meanTable = combos;
    stdTable = combos;
    for im = 1:numel(metricNames)
        name = metricNames{im};
        means = NaN(nCombo,1); stds = NaN(nCombo,1);
        for ic = 1:nCombo
            mask = string(allRows.trajectory)==string(combos.trajectory(ic)) & ...
                string(allRows.caseName)==string(combos.caseName(ic));
            values = allRows.(name)(mask);
            means(ic) = mean(values,"omitnan");
            stds(ic) = std(values,0,"omitnan");
        end
        meanTable.("mean_"+name) = means;
        stdTable.("std_"+name) = stds;
    end

    result = struct("seeds",seeds,"runs",{runs},"allRows",allRows, ...
        "meanTable",meanTable,"stdTable",stdTable);
    fprintf("\nStep 10-A.4 multi-seed mean table\n");
    disp(meanTable);
    fprintf("Step 10-A.4 multi-seed standard-deviation table\n");
    disp(stdTable);
end
