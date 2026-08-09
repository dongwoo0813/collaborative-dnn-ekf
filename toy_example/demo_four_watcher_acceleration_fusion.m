%% demo_four_watcher_acceleration_fusion.m
% Four independent angle-only [position; velocity; acceleration] EKFs
% followed by geometry-aware acceleration fusion.
%
% Purpose
%   1) Simulate a 2-D target with sinusoidal acceleration.
%   2) Place four stationary watchers approximately 1000 m away.
%   3) Run one angle-only EKF per watcher.
%   4) Compare:
%        - individual local EKF acceleration estimates,
%        - arithmetic mean of the full local estimates,
%        - equal-weight geometry-aware transverse fusion,
%        - covariance-weighted transverse BLUE.
%
% State
%   x = [r_x; r_y; v_x; v_y; a_x; a_y].
%
% Measurement of watcher i
%   z_i = atan2(r_y-r_{w_i,y}, r_x-r_{w_i,x}) + noise.
%
% Geometry projection
%   u_i  = (r_t-r_{w_i}) / ||r_t-r_{w_i}||,
%   Pi_i = I - u_i*u_i'.
%
% Covariance-weighted transverse BLUE
%   y_i = Pi_i*aHat_i,
%   S_i = Pi_i*P_{a,i}*Pi_i',
%   aHat_f = pinv(sum(Pi_i'*pinv(S_i)*Pi_i)) ...
%           * sum(Pi_i'*pinv(S_i)*y_i).
%
% Run command
%   demo_four_watcher_acceleration_fusion
%
% Notes
%   - sigmaBearingDeg = 0.05 means 0.05 degrees, not radians.
%   - The BLUE implementation neglects unknown cross-covariances between
%     local filters. The fused acceleration is therefore used as an output
%     only and is not fed back into the local EKFs.
%   - The initial uncertainty is deliberately larger along each watcher's
%     LOS direction to represent the weak radial information of a local
%     angle-only estimator.

clear; clc; close all;
rng(4, 'twister');

%% Simulation settings
dt = 0.5;                  % s
simulationTime = 200;      % s
time = 0:dt:simulationTime;
numSteps = numel(time);
stateDim = 6;
spaceDim = 2;
numWatchers = 4;

sigmaBearingDeg = 0.05;    % deg
sigmaBearing = deg2rad(sigmaBearingDeg);
R = sigmaBearing^2;

% Acceleration random-walk/jerk tuning used by each local EKF.
sigmaJerk = 1e-2;        % m/s^3

% Set true only for a diagnostic that isolates fusion geometry. The normal
% implementation should use false and construct LOS from estimated states.
useTruePositionForGeometry = false;

%% Watcher positions: approximately 1000 m from the target region
watcherPosition = [ ...
     1000,     0;
    -1000,     0;
        0,  1000;
        0, -1000]';        % 2 x 4

%% True target motion
xTrue = zeros(stateDim, numSteps);
xTrue(:,1) = [0; 0; 0.5; -0.25; targetAcceleration(time(1))];

for k = 2:numSteps
    aPrevious = targetAcceleration(time(k-1));
    aCurrent  = targetAcceleration(time(k));
    aAverage  = 0.5*(aPrevious + aCurrent);

    xTrue(1:2,k) = xTrue(1:2,k-1) ...
        + dt*xTrue(3:4,k-1) + 0.5*dt^2*aAverage;
    xTrue(3:4,k) = xTrue(3:4,k-1) + dt*aAverage;
    xTrue(5:6,k) = aCurrent;
end

%% Constant-acceleration EKF prediction model
I2 = eye(spaceDim);
Z2 = zeros(spaceDim);

F = [I2, dt*I2, 0.5*dt^2*I2;
     Z2, I2,    dt*I2;
     Z2, Z2,    I2];

% White jerk input over one sample.
Gjerk = [(dt^3/6)*I2;
         (dt^2/2)*I2;
          dt*I2];
Q = sigmaJerk^2*(Gjerk*Gjerk') + 1e-14*eye(stateDim);

%% Initialize four local EKFs
xHat = zeros(stateDim, numSteps, numWatchers);
P = zeros(stateDim, stateDim, numWatchers);

for i = 1:numWatchers
    initialLOS = xTrue(1:2,1) - watcherPosition(:,i);
    initialLOS = initialLOS/norm(initialLOS);
    initialTransverse = [-initialLOS(2); initialLOS(1)];
    localBasis = [initialLOS, initialTransverse];

    % Larger initial error along LOS, smaller error transverse to LOS.
    positionError = localBasis*[50*randn; 5*randn];
    velocityError = localBasis*[0.30*randn; 0.05*randn];
    accelerationError = localBasis*[0.020*randn; 0.002*randn];

    xHat(:,1,i) = xTrue(:,1) ...
        + [positionError; velocityError; accelerationError];

    Pposition = localBasis*diag([80^2, 10^2])*localBasis';
    Pvelocity = localBasis*diag([0.50^2, 0.10^2])*localBasis';
    Pacceleration = localBasis*diag([0.030^2, 0.003^2])*localBasis';

    P(:,:,i) = blkdiag(Pposition, Pvelocity, Pacceleration);
end

%% Logs
localAcceleration = zeros(spaceDim, numSteps, numWatchers);
meanAcceleration = zeros(spaceDim, numSteps);
equalGeometryAcceleration = zeros(spaceDim, numSteps);
blueAcceleration = zeros(spaceDim, numSteps);
blueCovariance = zeros(spaceDim, spaceDim, numSteps);
geometryMinEigenvalue = zeros(1, numSteps);
geometryCondition = zeros(1, numSteps);
geometryRank = zeros(1, numSteps);

%% Local filtering and acceleration fusion
for k = 1:numSteps
    % Local EKF prediction.
    if k > 1
        for i = 1:numWatchers
            xHat(:,k,i) = F*xHat(:,k-1,i);
            P(:,:,i) = F*P(:,:,i)*F' + Q;
            P(:,:,i) = 0.5*(P(:,:,i) + P(:,:,i)');
        end
    end

    % Independent angle-only measurement update at each watcher.
    for i = 1:numWatchers
        trueRelativePosition = xTrue(1:2,k) - watcherPosition(:,i);
        trueBearing = atan2(trueRelativePosition(2), ...
                            trueRelativePosition(1));
        measuredBearing = trueBearing + sigmaBearing*randn;

        estimatedRelativePosition = xHat(1:2,k,i) ...
            - watcherPosition(:,i);
        dx = estimatedRelativePosition(1);
        dy = estimatedRelativePosition(2);
        rangeSquared = max(dx^2 + dy^2, 1e-12);

        predictedBearing = atan2(dy, dx);

        H = zeros(1, stateDim);
        H(1) = -dy/rangeSquared;
        H(2) =  dx/rangeSquared;

        innovation = wrapToPiLocal(measuredBearing - predictedBearing);
        priorCovariance = P(:,:,i);
        innovationVariance = H*priorCovariance*H' + R;
        KalmanGain = priorCovariance*H'/innovationVariance;

        xHat(:,k,i) = xHat(:,k,i) + KalmanGain*innovation;

        % Joseph-form covariance update.
        identityState = eye(stateDim);
        P(:,:,i) = (identityState - KalmanGain*H)*priorCovariance ...
                 *(identityState - KalmanGain*H)' ...
                 + KalmanGain*R*KalmanGain';
        P(:,:,i) = 0.5*(P(:,:,i) + P(:,:,i)');

        localAcceleration(:,k,i) = xHat(5:6,k,i);
    end

    % Arithmetic mean of the four full local acceleration estimates.
    accelerationSum = zeros(spaceDim,1);
    positionSum = zeros(spaceDim,1);
    for i = 1:numWatchers
        accelerationSum = accelerationSum + xHat(5:6,k,i);
        positionSum = positionSum + xHat(1:2,k,i);
    end
    meanAcceleration(:,k) = accelerationSum/numWatchers;

    if useTruePositionForGeometry
        commonTargetPosition = xTrue(1:2,k);
    else
        commonTargetPosition = positionSum/numWatchers;
    end

    % Equal-weight geometry fusion and covariance-weighted BLUE.
    geometryMatrix = zeros(spaceDim);
    geometryRightHandSide = zeros(spaceDim,1);
    blueInformation = zeros(spaceDim);
    blueInformationVector = zeros(spaceDim,1);

    for i = 1:numWatchers
        los = commonTargetPosition - watcherPosition(:,i);
        los = los/norm(los);
        projection = eye(spaceDim) - los*los';

        localA = xHat(5:6,k,i);
        localPa = P(5:6,5:6,i);

        % Equal geometry weighting.
        geometryMatrix = geometryMatrix + projection;
        geometryRightHandSide = geometryRightHandSide ...
            + projection*localA;

        % Covariance-weighted transverse BLUE.
        transverseCovariance = projection*localPa*projection';
        transverseInformation = pinvPSD(transverseCovariance, 1e-12);

        blueInformation = blueInformation ...
            + projection'*transverseInformation*projection;
        blueInformationVector = blueInformationVector ...
            + projection'*transverseInformation*projection*localA;
    end

    equalGeometryAcceleration(:,k) = ...
        pinvPSD(geometryMatrix, 1e-12)*geometryRightHandSide;

    blueCovariance(:,:,k) = pinvPSD(blueInformation, 1e-12);
    blueAcceleration(:,k) = blueCovariance(:,:,k)*blueInformationVector;

    geometryEigenvalues = eig(0.5*(geometryMatrix + geometryMatrix'));
    geometryEigenvalues = sort(real(geometryEigenvalues));
    geometryRank(k) = sum(geometryEigenvalues > 1e-10);
    geometryMinEigenvalue(k) = geometryEigenvalues(1);
    geometryCondition(k) = geometryEigenvalues(end) ...
        / max(geometryEigenvalues(1), 1e-12);
end

%% Error metrics
trueAcceleration = xTrue(5:6,:);

localRMSE = zeros(numWatchers,1);
for i = 1:numWatchers
    localError = localAcceleration(:,:,i) - trueAcceleration;
    localRMSE(i) = sqrt(mean(sum(localError.^2,1)));
end

meanError = meanAcceleration - trueAcceleration;
equalGeometryError = equalGeometryAcceleration - trueAcceleration;
blueError = blueAcceleration - trueAcceleration;

meanRMSE = sqrt(mean(sum(meanError.^2,1)));
equalGeometryRMSE = sqrt(mean(sum(equalGeometryError.^2,1)));
blueRMSE = sqrt(mean(sum(blueError.^2,1)));

fprintf('\nFour local angle-only [r;v;a] EKFs\n');
fprintf('Bearing noise standard deviation: %.4f deg\n', sigmaBearingDeg);
fprintf('Watcher distance scale: approximately 1000 m\n\n');

for i = 1:numWatchers
    fprintf('Watcher %d local acceleration RMSE: %.6e m/s^2\n', ...
        i, localRMSE(i));
end
fprintf('Full-vector arithmetic mean RMSE:  %.6e m/s^2\n', meanRMSE);
fprintf('Equal geometry fusion RMSE:       %.6e m/s^2\n', ...
    equalGeometryRMSE);
fprintf('Transverse BLUE fusion RMSE:      %.6e m/s^2\n', blueRMSE);
fprintf('Median geometry condition number: %.3f\n', ...
    median(geometryCondition));
fprintf('Minimum geometry rank:            %.0f / %d\n\n', ...
    min(geometryRank), spaceDim);

%% Plots
figure('Name','Watcher geometry and target trajectory');
plot(xTrue(1,:), xTrue(2,:), 'LineWidth', 1.5); hold on;
scatter(watcherPosition(1,:), watcherPosition(2,:), 70, 'filled');
scatter(xTrue(1,1), xTrue(2,1), 70, 'filled');
axis equal; grid on;
xlabel('x position (m)'); ylabel('y position (m)');
title('Four-watchers geometry and target trajectory');
legend('Target trajectory','Watchers','Target initial position', ...
    'Location','best');

figure('Name','Acceleration estimates');
subplot(2,1,1);
plot(time, trueAcceleration(1,:), 'k', 'LineWidth', 1.8); hold on;
plot(time, meanAcceleration(1,:), '--', 'LineWidth', 1.0);
plot(time, equalGeometryAcceleration(1,:), '-.', 'LineWidth', 1.1);
plot(time, blueAcceleration(1,:), 'LineWidth', 1.2);
grid on; ylabel('a_x (m/s^2)');
title('Target acceleration estimation');
legend('Truth','Full local mean','Equal geometry','Transverse BLUE', ...
    'Location','best');

subplot(2,1,2);
plot(time, trueAcceleration(2,:), 'k', 'LineWidth', 1.8); hold on;
plot(time, meanAcceleration(2,:), '--', 'LineWidth', 1.0);
plot(time, equalGeometryAcceleration(2,:), '-.', 'LineWidth', 1.1);
plot(time, blueAcceleration(2,:), 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('a_y (m/s^2)');
legend('Truth','Full local mean','Equal geometry','Transverse BLUE', ...
    'Location','best');

figure('Name','Acceleration error norms');
semilogy(time, max(vecnorm(meanError,2,1),1e-12), '--', ...
    'LineWidth', 1.0); hold on;
semilogy(time, max(vecnorm(equalGeometryError,2,1),1e-12), '-.', ...
    'LineWidth', 1.1);
semilogy(time, max(vecnorm(blueError,2,1),1e-12), ...
    'LineWidth', 1.2);
grid on;
xlabel('Time (s)'); ylabel('||acceleration error||_2 (m/s^2)');
title('Acceleration fusion error');
legend('Full local mean','Equal geometry','Transverse BLUE', ...
    'Location','best');

figure('Name','Fusion geometry diagnostics');
subplot(2,1,1);
plot(time, geometryMinEigenvalue, 'LineWidth', 1.2);
grid on; ylabel('\lambda_{min}(\Sigma \Pi_i)');
title('Geometry diagnostics');
subplot(2,1,2);
plot(time, geometryCondition, 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('Condition number');

%% Local functions
function acceleration = targetAcceleration(t)
% Smooth deterministic target acceleration in the inertial x-y frame.
acceleration = [0.010*sin(2*pi*t/80);
                0.008*cos(2*pi*t/120)];
end

function wrappedAngle = wrapToPiLocal(angle)
% Toolbox-independent wrapping to [-pi, pi).
wrappedAngle = mod(angle + pi, 2*pi) - pi;
end

function matrixPseudoinverse = pinvPSD(matrix, relativeTolerance)
% Symmetric pseudoinverse for a positive-semidefinite matrix.
% Eigenvalues smaller than relativeTolerance*max(eigenvalue) are discarded.
matrix = 0.5*(matrix + matrix');
[eigenvectors, eigenvalueMatrix] = eig(matrix);
eigenvalues = real(diag(eigenvalueMatrix));
maximumEigenvalue = max(eigenvalues);

if maximumEigenvalue <= 0
    matrixPseudoinverse = zeros(size(matrix));
    return;
end

threshold = relativeTolerance*maximumEigenvalue;
inverseEigenvalues = zeros(size(eigenvalues));
valid = eigenvalues > threshold;
inverseEigenvalues(valid) = 1./eigenvalues(valid);

matrixPseudoinverse = eigenvectors*diag(inverseEigenvalues)*eigenvectors';
matrixPseudoinverse = 0.5*(matrixPseudoinverse + matrixPseudoinverse');
end
