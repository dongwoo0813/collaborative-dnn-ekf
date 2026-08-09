function study = run_common_state_factor_packet_ablation( ...
    seed,T,dt,nWatchers,hiddenLayerCount,makePlots)
%RUN_COMMON_STATE_FACTOR_PACKET_ABLATION Test raw replay versus factor packets.
%
% The factor-packet case is deliberately an intermediate reference: it has
% one common GS state and full all-layer theta, but watchers transmit local
% bearing normal-equation packets rather than raw angle samples.  It is not
% yet the final owner-block distributed learning architecture.

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 600; end
    if nargin < 3 || isempty(dt), dt = 0.1; end
    if nargin < 4 || isempty(nWatchers), nWatchers = 4; end
    if nargin < 5 || isempty(hiddenLayerCount), hiddenLayerCount = 3; end
    if nargin < 6 || isempty(makePlots), makePlots = true; end

    rawReplay = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "centralized_common_dnn_periodic",[],[],[],[],60);
    factorPacket = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "centralized_common_dnn_factor_packet_60s");
    hybrid = run_toy_block_structured_global_head_dnn_ekf( ...
        seed,T,false,nWatchers,dt,hiddenLayerCount, ...
        "hybrid_canonical_factor_joint_gs_60s");
    assert(isequal(rawReplay.truth,factorPacket.truth), ...
        'All packet ablation cases must use identical truth.');

    study.rawReplay60s = rawReplay.sharedBlock;
    study.commonStateFactorPacket60s = factorPacket.sharedBlock;
    study.hybrid60s = hybrid.sharedBlock;
    study.truth = rawReplay.truth; study.trueAcceleration = rawReplay.trueAcceleration;
    study.cfg = rawReplay.cfg;
    names = ["Centralized raw-bearing replay: 60 s"; ...
        "Common-state factor packet MAP: 60 s"; ...
        "Previous hybrid owner/factor GS: 60 s"];
    cases = {study.rawReplay60s,study.commonStateFactorPacket60s,study.hybrid60s};
    vals = zeros(3,5);
    for c = 1:3
        r = cases{c};
        vals(c,1:4) = [r.positionRMSE,r.velocityRMSE,r.accelerationRMSE,r.finalPositionRMSE];
        vals(c,5) = r.parameterUploads;
    end
    study.summary = table(names,vals(:,1),vals(:,2),vals(:,3),vals(:,4),vals(:,5), ...
        'VariableNames',{'caseName','positionRMSE','velocityRMSE', ...
        'accelerationRMSE','finalPositionRMSE','parameterUploads'});
    disp(study.summary);
    if makePlots
        study.figures = plotPacketStudy(study,cases,names);
    else
        study.figures = struct;
    end
end

function figs = plotPacketStudy(study,cases,names)
    t = study.cfg.time; colors = lines(3);
    figs.errors = figure('Name','Common-state factor-packet GS ablation');
    tiledlayout(3,1,'TileSpacing','compact');
    fields = {'positionError','velocityError','accelerationError'};
    units = {'position RMSE [m]','velocity RMSE [m/s]','acceleration RMSE [m/s^2]'};
    for row = 1:3
        nexttile; hold on; grid on;
        for c = 1:3
            plot(t,cases{c}.(fields{row}),'LineWidth',1.15,'Color',colors(c,:));
        end
        ylabel(units{row});
        if row == 1, legend(names,'Location','best'); end
    end
    xlabel('time [s]');
end
