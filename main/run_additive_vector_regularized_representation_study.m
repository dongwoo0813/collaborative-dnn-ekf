function study = run_additive_vector_regularized_representation_study(seeds,watcherCounts,ridgeGrid,makePlots)
%RUN_ADDITIVE_VECTOR_REGULARIZED_REPRESENTATION_STUDY
% Monte-Carlo offline additive-vector representation study with validation
% selection of the ridge coefficient.  This prevents a conclusion based on
% one lucky frozen-backbone realization or one arbitrary regularization.
%
% For every seed and N, fit W_N(lambda) to the training set, choose lambda
% with an independent validation set, and report only the final independent
% test error:
%
%   lambda_N^* = argmin_lambda RMSE_val(W_N(lambda)),
%   RMSE_test(N) = RMSE_test(W_N(lambda_N^*)).
%
% Example:
%   study = run_additive_vector_regularized_representation_study( ...
%       101:110,1:8,logspace(-10,-3,8),true);
%   study.summary

    if nargin < 1 || isempty(seeds), seeds = 101:110; end
    if nargin < 2 || isempty(watcherCounts), watcherCounts = 1:8; end
    if nargin < 3 || isempty(ridgeGrid), ridgeGrid = logspace(-10,-3,8); end
    if nargin < 4 || isempty(makePlots), makePlots = true; end
    seeds = reshape(seeds,1,[]); watcherCounts = reshape(watcherCounts,1,[]);
    ridgeGrid = reshape(ridgeGrid,1,[]);
    validateattributes(watcherCounts,{'numeric'},{'integer','>=',1,'<=',8});
    validateattributes(ridgeGrid,{'numeric'},{'positive','finite'});

    nSeed = numel(seeds); nN = numel(watcherCounts);
    trainRMSE = nan(nN,nSeed); validationRMSE = nan(nN,nSeed);
    testRMSE = nan(nN,nSeed); chosenRidge = nan(nN,nSeed);

    for s = 1:nSeed
        cfg = representationConfig(seeds(s));
        [etaTrain,dTrain] = spiralSamples(cfg,"train");
        [etaValidation,dValidation] = spiralSamples(cfg,"validation");
        [etaTest,dTest] = spiralSamples(cfg,"test");
        for q = 1:nN
            nWatchers = watcherCounts(q);
            PhiTrain = stackedFeatures(etaTrain,cfg,nWatchers);
            PhiValidation = stackedFeatures(etaValidation,cfg,nWatchers);
            PhiTest = stackedFeatures(etaTest,cfg,nWatchers);
            validationGrid = nan(size(ridgeGrid));
            Wgrid = cell(size(ridgeGrid));
            for r = 1:numel(ridgeGrid)
                lambda = ridgeGrid(r);
                Wgrid{r} = dTrain*PhiTrain'/(PhiTrain*PhiTrain' + ...
                    lambda*eye(size(PhiTrain,1)));
                validationGrid(r) = accelerationRMSE(dValidation-Wgrid{r}*PhiValidation);
            end
            [validationRMSE(q,s),best] = min(validationGrid);
            W = Wgrid{best}; chosenRidge(q,s) = ridgeGrid(best);
            trainRMSE(q,s) = accelerationRMSE(dTrain-W*PhiTrain);
            testRMSE(q,s) = accelerationRMSE(dTest-W*PhiTest);
        end
    end

    summary = table(watcherCounts(:),6*watcherCounts(:), ...
        mean(trainRMSE,2),std(trainRMSE,0,2), ...
        mean(validationRMSE,2),std(validationRMSE,0,2), ...
        mean(testRMSE,2),std(testRMSE,0,2), ...
        median(chosenRidge,2), ...
        'VariableNames',{'nWatchers','onlineHeadParameterCount', ...
        'trainRMSEMean','trainRMSEStd','validationRMSEMean','validationRMSEStd', ...
        'testRMSEMean','testRMSEStd','medianSelectedRidge'});
    study = struct('seeds',seeds,'watcherCounts',watcherCounts, ...
        'ridgeGrid',ridgeGrid, ...
        'description',"Offline train-validation-test representation study; no EKF or watcher geometry.", ...
        'trainRMSE',trainRMSE,'validationRMSE',validationRMSE, ...
        'testRMSE',testRMSE,'chosenRidge',chosenRidge,'summary',summary);
    disp(summary);

    if makePlots
        fig = figure('Name','Regularized additive-vector representation study');
        tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
        nexttile; hold on;
        errorbar(watcherCounts,mean(trainRMSE,2),std(trainRMSE,0,2),'-o', ...
            'LineWidth',1.3,'DisplayName','train');
        errorbar(watcherCounts,mean(validationRMSE,2),std(validationRMSE,0,2),'-s', ...
            'LineWidth',1.3,'DisplayName','validation (selected)');
        errorbar(watcherCounts,mean(testRMSE,2),std(testRMSE,0,2),'-d', ...
            'LineWidth',1.3,'DisplayName','independent test');
        grid on; xticks(watcherCounts); xlabel('number of sub-DNN branches');
        ylabel('acceleration RMSE [m/s^2]');
        title(sprintf('Representation error across %d backbone seeds',nSeed));
        legend('Location','best');
        nexttile;
        semilogy(watcherCounts,chosenRidge','-o','LineWidth',1.0); grid on;
        xticks(watcherCounts); xlabel('number of sub-DNN branches');
        ylabel('selected ridge \lambda'); title('Validation-selected regularization');
        study.figure = fig;
    end
end

function cfg = representationConfig(seed)
    cfg.seed = seed; cfg.radiusGoal = 100; cfg.radialRate = .30;
    cfg.angularRate = .012; cfg.velocityGain = .035; cfg.inputScale = [100;100;.8;.8];
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
            eta = zeros(4,numel(rhoGrid)*numel(angleGrid)*size(velocityOffsets,2)); k = 0;
            for rho = rhoGrid
                for angle = angleGrid
                    r = rho*[cos(angle);sin(angle)]; v = desiredVelocity(r,cfg);
                    for dv = velocityOffsets
                        k = k+1; eta(:,k) = [r;v+dv];
                    end
                end
            end
        case {"validation","test"}
            if string(kind) == "validation", rng(cfg.seed+777); else, rng(cfg.seed+999); end
            n = 900; rho = 2+(cfg.radiusGoal-2)*rand(1,n); angle = 2*pi*rand(1,n);
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
            Phi((j-1)*cfg.nPhi+(1:cfg.nPhi),k) = branchFeatures(eta(:,k),cfg,j);
        end
    end
end

function value = accelerationRMSE(error)
    value = sqrt(mean(sum(error.^2,1)));
end

function d = trueResidual(eta,cfg)
    d = cfg.velocityGain*(desiredVelocity(eta(1:2),cfg)-eta(3:4));
end

function v = desiredVelocity(r,cfg)
    rho = max(norm(r),.25); uR = r/rho; uT = [-uR(2);uR(1)];
    v = cfg.radialRate*(1-rho/cfg.radiusGoal)*uR + cfg.angularRate*rho*uT;
end

function phi = branchFeatures(eta,cfg,branch)
    h1 = tanh(cfg.W1(:,:,branch)*(eta./cfg.inputScale));
    phi = tanh(cfg.W2(:,:,branch)*h1);
end
