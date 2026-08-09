function study = run_additive_vector_representation_ablation(seeds,T,dt,watcherCounts,scenario,makePlots)
%RUN_ADDITIVE_VECTOR_REPRESENTATION_ABLATION Test whether adding sub-DNNs
% expands the shared residual representation.
%   The experiment holds each seed's truth, bearing-noise realization,
%   formation subset, and nominal-derived local-radial maneuver schedule
%   fixed.  It compares the existing directional-WLS reconstruction against
%   a shared vector-valued additive model
%
%       d_hat(eta) = sum_j W_out,j * phi_j(eta).
%
%   Each phi_j is a distinct frozen two-layer 3-neuron backbone and each
%   W_out,j is adapted only by watcher j's local EKF.  Therefore increasing
%   N adds six output-head parameters and three new nonlinear features; the
%   N-branch class contains the previous class by setting the new head to 0.
%
%   This is deliberately an architecture/representation ablation.  It uses
%   the current instantaneous parameter cache, rather than event-triggered
%   communication, so a later communication study can isolate the effects
%   of trigger period, delay, and packet loss.

    if nargin < 1 || isempty(seeds), seeds = 101:105; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = .1; end
    if nargin < 4 || isempty(watcherCounts), watcherCounts = 1:4; end
    if nargin < 5 || isempty(scenario), scenario = "well_conditioned"; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end
    seeds = reshape(seeds,1,[]); watcherCounts = reshape(watcherCounts,1,[]);
    validateattributes(watcherCounts,{'numeric'},{'integer','>=',1,'<=',4});

    labels = ["Shared directional WLS"; "Shared additive-vector DNN"];
    metrics = ["positionRMSE","velocityRMSE","accelerationRMSE", ...
        "finalPositionRMSE"];
    raw = nan(numel(watcherCounts),numel(seeds),numel(labels),numel(metrics));

    for w = 1:numel(watcherCounts)
        for s = 1:numel(seeds)
            common = {seeds(s),T,false,scenario,dt,true,"local_radial", ...
                true,watcherCounts(w),"parameter_covariance"};
            % Suppress the per-run toy summary: this runner reports the
            % aggregate table after all seeds and watcher counts finish.
            directional = runToyQuiet(common,"directional_wls");
            additive = runToyQuiet(common,"additive_vector");
            results = {directional.sharedAdditive,additive.sharedAdditive};
            for c = 1:numel(labels)
                r = results{c};
                e = r.xhat(1:4,:,:)-repmat(r.etaTrue,1,1,size(r.xhat,3));
                de = r.dHat-repmat(r.dTrue,1,1,size(r.dHat,3));
                raw(w,s,c,1) = sqrt(mean(vecnorm(e(1:2,:,:),2,1).^2,'all'));
                raw(w,s,c,2) = sqrt(mean(vecnorm(e(3:4,:,:),2,1).^2,'all'));
                raw(w,s,c,3) = sqrt(mean(vecnorm(de,2,1).^2,'all'));
                last = round(.9*numel(r.time)):numel(r.time);
                raw(w,s,c,4) = sqrt(mean(vecnorm(e(1:2,last,:),2,1).^2,'all'));
            end
        end
    end

    nRows = numel(watcherCounts)*numel(labels); row = 0;
    summary = table(zeros(nRows,1),strings(nRows,1), ...
        repmat(string(scenario),nRows,1),zeros(nRows,1),zeros(nRows,1), ...
        zeros(nRows,1),zeros(nRows,1),zeros(nRows,1),zeros(nRows,1), ...
        zeros(nRows,1),zeros(nRows,1),zeros(nRows,1),zeros(nRows,1), ...
        'VariableNames',{'nWatchers','architecture','scenario','nSeeds', ...
        'parameterCount','positionRMSEMean','positionRMSEStd', ...
        'velocityRMSEMean','velocityRMSEStd','accelerationRMSEMean', ...
        'accelerationRMSEStd','finalPositionRMSEMean','finalPositionRMSEStd'});
    for w = 1:numel(watcherCounts)
        for c = 1:numel(labels)
            row = row+1; values = reshape(raw(w,:,c,:),numel(seeds),numel(metrics));
            summary.nWatchers(row) = watcherCounts(w);
            summary.architecture(row) = labels(c); summary.nSeeds(row) = numel(seeds);
            summary.parameterCount(row) = 6*watcherCounts(w);
            summary.positionRMSEMean(row) = mean(values(:,1));
            summary.positionRMSEStd(row) = std(values(:,1),0);
            summary.velocityRMSEMean(row) = mean(values(:,2));
            summary.velocityRMSEStd(row) = std(values(:,2),0);
            summary.accelerationRMSEMean(row) = mean(values(:,3));
            summary.accelerationRMSEStd(row) = std(values(:,3),0);
            summary.finalPositionRMSEMean(row) = mean(values(:,4));
            summary.finalPositionRMSEStd(row) = std(values(:,4),0);
        end
    end
    study = struct('seeds',seeds,'watcherCounts',watcherCounts, ...
        'scenario',string(scenario),'maneuverObjective',"local_radial", ...
        'communication',"instantaneous parameter cache (not event-triggered)", ...
        'raw',raw,'metricNames',metrics,'summary',summary);
    disp(summary);

    if makePlots
        fig = figure('Name',"Additive-vector representation ablation: "+string(scenario));
        tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
        titles = ["position RMSE","velocity RMSE", ...
            "acceleration approximation RMSE","final position RMSE"];
        for q = 1:numel(metrics)
            nexttile; hold on;
            for c = 1:numel(labels)
                y = squeeze(mean(raw(:,:,c,q),2)); e = squeeze(std(raw(:,:,c,q),0,2));
                errorbar(watcherCounts,y,e,'-o','LineWidth',1.2, ...
                    'DisplayName',labels(c));
            end
            grid on; xticks(watcherCounts); xlabel('number of watchers');
            ylabel(metrics(q)); title(titles(q));
            if q == 1, legend('Location','best'); end
        end
        study.figure = fig;
    end
end

function out = runToyQuiet(common,architecture)
%RUNTOYQUIET Keep a Monte-Carlo study's command window readable.
    evalc('out = run_toy_distributed_additive_dnn_ekf(common{:},architecture);');
end
