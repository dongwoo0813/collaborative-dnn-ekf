function study = run_additive_vector_offline_representation_ablation(seed,watcherCounts,makePlots)
%RUN_ADDITIVE_VECTOR_OFFLINE_REPRESENTATION_ABLATION
% Measure only the function-class benefit of summing fixed sub-DNNs.
%
% For N branches this fits, offline,
%
%   d_hat_N(eta) = sum_{j=1}^N W_out,j phi_j(eta)
%
% to the known toy residual field with ridge least squares.  The held-out
% points are never used in the fit.  Consequently this script answers a
% narrower question than the online EKF study: does adding a branch make
% the *available representation* richer before bearing-only estimation,
% parameter allocation, communication, and maneuvering are involved?
%
% Example:
%   rep = run_additive_vector_offline_representation_ablation(101,1:4,true);
%   rep.summary

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(watcherCounts), watcherCounts = 1:4; end
    if nargin < 3 || isempty(makePlots), makePlots = true; end
    watcherCounts = reshape(watcherCounts,1,[]);
    % This is an offline representation test, not the 4-camera EKF toy.
    % It can therefore examine a larger bank of independently frozen
    % sub-DNN features without changing the physical watcher formation.
    validateattributes(watcherCounts,{'numeric'},{'integer','>=',1,'<=',8});

    cfg = representationConfig(seed);
    [etaTrain,dTrain] = spiralSamples(cfg,"train");
    [etaTest,dTest] = spiralSamples(cfg,"test");
    nCases = numel(watcherCounts);
    trainRMSE = zeros(nCases,1); testRMSE = zeros(nCases,1);
    theta = cell(nCases,1); dTestHat = cell(nCases,1);

    for q = 1:nCases
        nWatchers = watcherCounts(q);
        PhiTrain = stackedFeatures(etaTrain,cfg,nWatchers);
        PhiTest = stackedFeatures(etaTest,cfg,nWatchers);

        % W_N = D Phi' (Phi Phi' + lambda I)^(-1), equivalently the
        % ridge-LS minimizer of ||D-W_N Phi||_F^2+lambda||W_N||_F^2.
        W = dTrain*PhiTrain'/(PhiTrain*PhiTrain' + ...
            cfg.ridge*eye(size(PhiTrain,1)));
        trainError = dTrain-W*PhiTrain;
        testError = dTest-W*PhiTest;
        trainRMSE(q) = sqrt(mean(sum(trainError.^2,1)));
        testRMSE(q) = sqrt(mean(sum(testError.^2,1)));
        theta{q} = W;
        dTestHat{q} = W*PhiTest;
    end

    summary = table(watcherCounts(:),6*watcherCounts(:), ...
        trainRMSE,testRMSE, ...
        'VariableNames',{'nWatchers','onlineHeadParameterCount', ...
        'trainAccelerationRMSE','holdoutAccelerationRMSE'});
    study = struct('seed',seed,'watcherCounts',watcherCounts, ...
        'description',"Offline ridge-LS representation test; no EKF, bearing noise, maneuver, or communication.", ...
        'ridge',cfg.ridge,'etaTrain',etaTrain,'dTrain',dTrain, ...
        'etaTest',etaTest,'dTest',dTest,'theta', {theta}, ...
        'dTestHat',{dTestHat},'summary',summary);
    disp(summary);

    if makePlots
        fig = figure('Name','Offline additive-vector representation ablation');
        tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
        nexttile; hold on;
        plot(watcherCounts,trainRMSE,'-o','LineWidth',1.4, ...
            'DisplayName','train RMSE');
        plot(watcherCounts,testRMSE,'-s','LineWidth',1.4, ...
            'DisplayName','independent holdout RMSE');
        grid on; xticks(watcherCounts); xlabel('number of sub-DNN branches');
        ylabel('acceleration RMSE [m/s^2]');
        title('Additive-vector representation error'); legend('Location','best');

        % Show the largest model: this makes it easy to inspect whether it
        % captures both acceleration components on an unseen spiral.
        [~,idx] = max(watcherCounts); t = 1:size(dTest,2);
        nexttile; hold on;
        plot(t,dTest(1,:),'k','LineWidth',1.5,'DisplayName','truth d_x');
        plot(t,dTestHat{idx}(1,:),'--','LineWidth',1.2, ...
            'DisplayName',sprintf('N=%d estimate d_x',watcherCounts(idx)));
        plot(t,dTest(2,:),'Color',[.2 .2 .2],'LineWidth',1.5, ...
            'DisplayName','truth d_y');
        plot(t,dTestHat{idx}(2,:),':','LineWidth',1.7, ...
            'DisplayName',sprintf('N=%d estimate d_y',watcherCounts(idx)));
        grid on; xlabel('held-out sample index'); ylabel('acceleration [m/s^2]');
        title('Largest additive model on held-out states'); legend('Location','best');
        study.figure = fig;
    end
end

function cfg = representationConfig(seed)
    cfg.seed = seed; cfg.radiusGoal = 100; cfg.radialRate = .30;
    cfg.angularRate = .012; cfg.velocityGain = .035; cfg.scale = 2;
    cfg.ridge = 1e-7; cfg.inputScale = [100;100;.8;.8];
    cfg.nPhi = 3;
    baseW1 = [1 0 .20 0; 0 1 0 .20; 0 0 .70 .70];
    cfg.W1 = zeros(3,4,8); cfg.W2 = zeros(3,3,8);
    for j = 1:8
        rng(cfg.seed+100*j);
        cfg.W1(:,:,j) = baseW1+.18*randn(3,4);
        cfg.W2(:,:,j) = eye(3)+.12*randn(3,3);
    end
end

function [eta,d] = spiralSamples(cfg,kind)
    switch string(kind)
        case "train"
            rhoGrid = linspace(2,cfg.radiusGoal,17);
            angleGrid = linspace(0,2*pi,25); angleGrid(end) = [];
            velocityOffsets = [0 .08 -.08; 0 .05 -.05];
            eta = zeros(4,numel(rhoGrid)*numel(angleGrid)*size(velocityOffsets,2));
            k = 0;
            for rho = rhoGrid
                for angle = angleGrid
                    r = rho*[cos(angle);sin(angle)]; v = desiredVelocity(r,cfg);
                    for dv = velocityOffsets
                        k = k+1; eta(:,k) = [r;v+dv];
                    end
                end
            end
        case "test"
            % A different, randomly phased polar grid prevents accidental
            % interpolation-only reporting on the warm-start training set.
            rng(cfg.seed+999); n = 900;
            rho = 2+(cfg.radiusGoal-2)*rand(1,n);
            angle = 2*pi*rand(1,n);
            eta = zeros(4,n);
            for k = 1:n
                r = rho(k)*[cos(angle(k));sin(angle(k))];
                eta(:,k) = [r;desiredVelocity(r,cfg)+.10*randn(2,1)];
            end
        otherwise
            error('Unknown sample set.');
    end
    d = zeros(2,size(eta,2));
    for k = 1:size(eta,2), d(:,k) = trueResidual(eta(:,k),cfg); end
end

function Phi = stackedFeatures(eta,cfg,nWatchers)
    Phi = zeros(cfg.nPhi*nWatchers,size(eta,2));
    for k = 1:size(eta,2)
        for j = 1:nWatchers
            rows = (j-1)*cfg.nPhi+(1:cfg.nPhi);
            Phi(rows,k) = branchFeatures(eta(:,k),cfg,j);
        end
    end
end

function d = trueResidual(eta,cfg)
    d = cfg.velocityGain*(desiredVelocity(eta(1:2),cfg)-eta(3:4));
end

function v = desiredVelocity(r,cfg)
    rho = max(norm(r),.25); uR = r/rho; uT = [-uR(2);uR(1)];
    v = cfg.radialRate*(1-rho/cfg.radiusGoal)*uR + cfg.angularRate*rho*uT;
end

function phi = branchFeatures(eta,cfg,branch)
    input = eta./cfg.inputScale;
    h1 = tanh(cfg.W1(:,:,branch)*input);
    phi = tanh(cfg.W2(:,:,branch)*h1);
end
