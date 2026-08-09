function figs = plot_spiral_position_information_results(ablation)
%PLOT_SPIRAL_POSITION_INFORMATION_RESULTS Plot truth against case estimates.
%   FIGS = PLOT_SPIRAL_POSITION_INFORMATION_RESULTS(ABLATION) makes one
%   six-panel figure per scenario.  Each coloured state/acceleration curve
%   is the mean of the four watcher-local estimates for that ablation case.
%   This is a representative common estimate for visual comparison; the
%   RMSE summary in ABLATION is still computed over all local copies.

    if isfield(ablation,'nearParallel')
        figs = struct;
        figs.nearParallel = plotOne(ablation.nearParallel);
        figs.wellConditioned = plotOne(ablation.wellConditioned);
    else
        figs = plotOne(ablation);
    end
end

function fig = plotOne(a)
    cases = {a.noManeuver,a.localRadial,a.positionInformation};
    labels = ["Shared WLS: no maneuver", ...
              "Shared WLS: local radial", ...
              "Shared WLS: history position information"];
    if isfield(a,'hybridPosition')
        cases{end+1} = a.hybridPosition;
        labels(end+1) = "Shared WLS: hybrid position";
    end
    colors = lines(numel(cases));
    componentNames = ["r_x [m]","r_y [m]", ...
                      "v_x [m/s]","v_y [m/s]", ...
                      "a_x [m/s^2]","a_y [m/s^2]"];
    stateIndices = [1 2 3 4];

    scenario = string(a.summary.scenario(1));
    fig = figure('Name',"State and acceleration estimates: "+scenario);
    tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

    for q = 1:6
        nexttile; hold on;
        truth = cases{1}.dTrue(1,:); % overwritten below for state panels
        if q <= 4
            truth = cases{1}.etaTrue(stateIndices(q),:);
        else
            truth = cases{1}.dTrue(q-4,:);
        end
        plot(cases{1}.time,truth,'k','LineWidth',1.5,'DisplayName','truth');

        for k = 1:numel(cases)
            r = cases{k};
            if q <= 4
                estimate = mean(squeeze(r.xhat(stateIndices(q),:,:)),2).';
            else
                estimate = mean(squeeze(r.dHat(q-4,:,:)),2).';
            end
            plot(r.time,estimate,'Color',colors(k,:),'LineWidth',1.15, ...
                'DisplayName',labels(k));
        end
        grid on; ylabel(componentNames(q));
        if q == 1
            title("State and acceleration estimates — "+replace(scenario,"_"," "));
            legend('Location','best');
        end
        if q >= 5, xlabel('time [s]'); end
    end
end
