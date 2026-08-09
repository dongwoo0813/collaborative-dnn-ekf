function result = run_four_watcher_acceleration_fusion_sanity( ...
    makePlots,simulationTime,dt,seed)
%RUN_FOUR_WATCHER_ACCELERATION_FUSION_SANITY
% Simple four-watcher EKF sanity experiment for additive versus geometry
% fusion of local target-acceleration estimates.
%
% Each watcher is placed at the east, west, north, or south position and
% runs an EKF with state x=[r;v;a].  Every watcher estimates the FULL target
% acceleration.  Therefore, summing all local acceleration estimates is an
% intentionally informative test: it is not the statistically appropriate
% fusion rule for four redundant full-state estimates.  A mean estimate is
% included as a reference, while the geometry estimate uses
%
%   B_j = (sum_l Omega_l + eps I)\Omega_j,
%   Omega_j = I-u_j*u_j',
%   a_geo = sum_j B_j*a_j.
%
% The truth trajectory is
%   r_x = R0*cos(w*t) + v0_x*t + 0.5*a0_x*t^2
%   r_y = R0*sin(w*t) + v0_y*t + 0.5*a0_y*t^2,
% with v and a obtained by differentiation.  Range and bearing are both
% measured by every watcher.
%
% Example:
%   out = run_four_watcher_acceleration_fusion_sanity(true,200,0.1,101);

if nargin < 1 || isempty(makePlots), makePlots = true; end
if nargin < 2 || isempty(simulationTime), simulationTime = 200; end
if nargin < 3 || isempty(dt), dt = 0.1; end
if nargin < 4 || isempty(seed), seed = 101; end

rng(seed);
dim = 2;
Nw = 4;
time = (0:dt:simulationTime).';
N = numel(time);

% Truth parameters.  R0 is the circular component amplitude; v0 and a0
% are the linear drift and constant quadratic-drift coefficients.
R0 = 100;
w = 0.025;
v0 = [0.08; -0.04];
a0 = [2.0e-3; -1.0e-3];

truth = zeros(6,N);
tau = time.';
truth(1,:) = R0*cos(w*tau) + v0(1)*tau + 0.5*a0(1)*tau.^2;
truth(2,:) = R0*sin(w*tau) + v0(2)*tau + 0.5*a0(2)*tau.^2;
truth(3,:) = -R0*w*sin(w*tau) + v0(1) + a0(1)*tau;
truth(4,:) =  R0*w*cos(w*tau) + v0(2) + a0(2)*tau;
truth(5,:) = -R0*w^2*cos(w*tau) + a0(1);
truth(6,:) = -R0*w^2*sin(w*tau) + a0(2);

% Cardinal watcher locations: west, east, south, north.
watcherR = [ -1000, 1000, 0, 0; ...
                  0,    0,-1000,1000 ];

sigmaRange = 1.0;
sigmaBearing = deg2rad(0.01);
Rmeas = diag([sigmaRange^2,sigmaBearing^2]);

% Constant-acceleration EKF model with acceleration random walk.
F = [eye(2),dt*eye(2),0.5*dt^2*eye(2); ...
     zeros(2),eye(2),dt*eye(2); ...
     zeros(2),zeros(2),eye(2)];
G = [0.5*dt^2*eye(2);dt*eye(2);eye(2)];
qJerk = 2e-5;
Q = (qJerk^2)*G*G.';

xhat = zeros(6,N,Nw);
P = zeros(6,6,Nw);
x0Error = [20;-15;0.08;-0.05;0.02;-0.02];
P0 = diag([20^2,20^2,0.1^2,0.1^2,0.02^2,0.02^2]);
for j = 1:Nw
    xhat(:,1,j) = truth(:,1) + x0Error + ...
        [5*randn(2,1);0.02*randn(2,1);0.005*randn(2,1)];
    P(:,:,j) = P0;
end

localAcceleration = zeros(2,N,Nw);
geometryWeight = zeros(2,2,Nw,N);
additiveAcceleration = zeros(2,N);
geometryAcceleration = zeros(2,N);
meanAcceleration = zeros(2,N);

% Initial local acceleration records.
for j = 1:Nw
    localAcceleration(:,1,j) = xhat(5:6,1,j);
end
[additiveAcceleration(:,1),geometryAcceleration(:,1), ...
    meanAcceleration(:,1),geometryWeight(:,:,:,1)] = fuseAccelerations( ...
    localAcceleration(:,1,:),truth(1:2,1),watcherR,1e-9);

for k = 1:N-1
    t = time(k+1);
    for j = 1:Nw
        % Prediction.
        xPred = F*xhat(:,k,j);
        PPred = F*P(:,:,j)*F.' + Q;

        % Range+bearing measurement from the fixed cardinal watcher.
        rho = truth(1:2,k+1)-watcherR(:,j);
        rangeTrue = norm(rho);
        bearingTrue = atan2(rho(2),rho(1));
        z = [rangeTrue + sigmaRange*randn; ...
            wrapAngle(bearingTrue + sigmaBearing*randn)];

        rhoHat = xPred(1:2)-watcherR(:,j);
        rangeHat = max(norm(rhoHat),1e-9);
        bearingHat = atan2(rhoHat(2),rhoHat(1));
        innovation = [z(1)-rangeHat; wrapAngle(z(2)-bearingHat)];

        uHat = rhoHat/rangeHat;
        bearingGradient = [-uHat(2);uHat(1)]/rangeHat;
        H = [uHat.' zeros(1,4); bearingGradient.' zeros(1,4)];
        S = H*PPred*H.' + Rmeas;
        K = (PPred*H.')/S;
        xPost = xPred + K*innovation;
        PPost = (eye(6)-K*H)*PPred*(eye(6)-K*H).' + K*Rmeas*K.';
        PPost = 0.5*(PPost+PPost.');

        xhat(:,k+1,j) = xPost;
        P(:,:,j) = PPost;
        localAcceleration(:,k+1,j) = xPost(5:6);
    end

    [additiveAcceleration(:,k+1),geometryAcceleration(:,k+1), ...
        meanAcceleration(:,k+1),geometryWeight(:,:,:,k+1)] = ...
        fuseAccelerations(localAcceleration(:,k+1,:),truth(1:2,k+1), ...
        watcherR,1e-9);
end

trueAcceleration = truth(5:6,:);
result = struct();
result.time = time;
result.truth = truth;
result.trueAcceleration = trueAcceleration;
result.watcherR = watcherR;
result.localState = xhat;
result.localAcceleration = localAcceleration;
result.additiveAcceleration = additiveAcceleration;
result.geometryAcceleration = geometryAcceleration;
result.meanAcceleration = meanAcceleration;
result.geometryWeight = geometryWeight;
result.measurement = struct('rangeSigma',sigmaRange, ...
    'bearingSigmaDeg',rad2deg(sigmaBearing));
result.parameters = struct('R0',R0,'w',w,'v0',v0,'a0',a0, ...
    'dt',dt,'simulationTime',simulationTime,'seed',seed);

result.summary = makeSummary(result);
fprintf('Four-watcher acceleration fusion sanity test\n');
fprintf('range sigma=%.3f m, bearing sigma=%.4f deg, T=%.1f s, dt=%.3g s\n', ...
    sigmaRange,rad2deg(sigmaBearing),simulationTime,dt);
disp(result.summary);

if makePlots
    result.figure = makePlotsForResult(result);
end
end

function [aAdd,aGeo,aMean,B] = fuseAccelerations(localA,targetR,watcherR,epsilon)
Nw = size(watcherR,2);
dim = 2;
Omega = zeros(dim,dim,Nw);
for j = 1:Nw
    rho = targetR-watcherR(:,j);
    u = rho/max(norm(rho),realmin);
    Omega(:,:,j) = eye(dim)-u*u.';
end
OmegaSum = sum(Omega,3)+epsilon*eye(dim);
B = zeros(dim,dim,Nw);
for j = 1:Nw
    B(:,:,j) = OmegaSum\Omega(:,:,j);
end
aAdd = sum(localA,3);
aGeo = zeros(dim,1);
for j = 1:Nw
    aGeo = aGeo+B(:,:,j)*localA(:,:,j);
end
aMean = aAdd/Nw;
end

function summary = makeSummary(result)
truthA = result.trueAcceleration;
cases = {result.additiveAcceleration,result.geometryAcceleration, ...
    result.meanAcceleration};
names = ["additive-sum";"geometry-aware";"simple-mean"];
rmse = zeros(3,1); finalRMSE = zeros(3,1);
for i = 1:3
    e = cases{i}-truthA;
    en = sqrt(sum(e.^2,1));
    rmse(i) = sqrt(mean(en.^2));
    idx = max(1,round(0.9*numel(result.time))):numel(result.time);
    finalRMSE(i) = sqrt(mean(en(idx).^2));
end
summary = table(names,rmse,finalRMSE, ...
    'VariableNames',{'caseName','accelerationRMSE','finalAccelerationRMSE'});
end

function fig = makePlotsForResult(result)
t = result.time;
truthA = result.trueAcceleration;
fig = figure('Name','Four-watcher acceleration fusion sanity','Color','w');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
nexttile;
plot(t,truthA(1,:),'k','LineWidth',1.4); hold on;
plot(t,result.additiveAcceleration(1,:),'b');
plot(t,result.geometryAcceleration(1,:),'r--');
plot(t,result.meanAcceleration(1,:),'g-.');
title('Acceleration x'); ylabel('m/s^2'); grid on;
legend('truth','additive sum','geometry-aware','simple mean','Location','best');
nexttile;
plot(t,truthA(2,:),'k','LineWidth',1.4); hold on;
plot(t,result.additiveAcceleration(2,:),'b');
plot(t,result.geometryAcceleration(2,:),'r--');
plot(t,result.meanAcceleration(2,:),'g-.');
title('Acceleration y'); ylabel('m/s^2'); grid on;
nexttile;
plot(t,vecnorm(result.additiveAcceleration-truthA,2,1),'b'); hold on;
plot(t,vecnorm(result.geometryAcceleration-truthA,2,1),'r--');
plot(t,vecnorm(result.meanAcceleration-truthA,2,1),'g-.');
title('Acceleration error norm'); xlabel('time [s]'); ylabel('m/s^2'); grid on;
nexttile;
B11 = squeeze(mean(result.geometryWeight(1,1,:,:),3));
B22 = squeeze(mean(result.geometryWeight(2,2,:,:),3));
plot(t,B11,'b'); hold on;
plot(t,B22,'r--');
title('Mean diagonal geometry weights'); xlabel('time [s]'); grid on;
legend('mean B(1,1)','mean B(2,2)','Location','best');
end

function angle = wrapAngle(angle)
angle = atan2(sin(angle),cos(angle));
end
