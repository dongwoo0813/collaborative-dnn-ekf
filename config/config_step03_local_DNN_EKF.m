function cfg = config_step03_local_DNN_EKF()
%{
Function:
    config_step03_local_DNN_EKF.m

Purpose:
    Create the configuration for Step 03 local block DNN-EKF simulation.

    This configuration starts from the Step 02 residual-mismatch physical
    EKF configuration and adds local DNN-EKF parameter settings.

    In Step 03, each watcher estimates an augmented local state

        X_i = [eta_i; theta_i],

    where

        eta_i = [r_t; v_t]

    is the physical target state estimate and theta_i is the local DNN
    branch parameter vector.

    There is still no ground-station sharing and no peer-to-peer sharing in
    this step.

Inputs:
    None.

Outputs:
    cfg - Simulation configuration structure.

Main equations:
    The local branch residual model is

        d_i(eta; theta_i) = W_i phi_i(eta),

    where

        theta_i = vec(W_i),

        W_i in R^{dim x nPhi}.

    Therefore, the number of parameters in one local branch is

        nTheta_i = dim * nPhi.

    The local DNN-EKF state dimension is

        nX_i = 2*dim + nTheta_i.

Notes:
    - This config does not activate GS or P2P communication.
    - The truth residual is branch-wise:
          d_true(eta) = sum_j W_j^star phi_j(eta).
    - The local estimator currently uses only its own branch:
          d_hat_i(eta) = W_i phi_i(eta).
    - Control remains disabled.
%}

    % Start from Step 02 residual-mismatch configuration.
    cfg = config_step02_residual_physical_EKF();

    % Step label.
    cfg.step.name = "step03_local_DNN_EKF";

    % Estimator and communication mode.
    cfg.estimator.type = "local_DNN_EKF";
    cfg.comm.mode = "none";

    % Keep watcher control disabled for now.
    cfg.watchers.motionMode = "prescribed";
    cfg.control.translationMode = "none";
    cfg.control.attitudeMode = "none";

    % ---------------------------------------------------------------------
    % Truth residual model
    % ---------------------------------------------------------------------
    cfg.truth.useResidual = true;
    cfg.truth.residualModel = "branchwise";

    % Residual acceleration scale.
    %
    % For the branchwise truth model, this scales the hidden true branch
    % weights inside truthBranchWeights.m.
    cfg.truth.residualAmp = 1e-4;

    % ---------------------------------------------------------------------
    % Fixed-feature DNN branch settings
    % ---------------------------------------------------------------------

    % Feature scaling. These values should match featureBlock.m.
    %
    % The features should be dimensionless or similarly scaled. This is
    % important because a scalar Ptheta0 or thetaSigmaSS assumes comparable
    % parameter scales.
    cfg.dnn.rScale = 1000;
    cfg.dnn.vScale = 1;

    % Current featureBlock.m uses
    %
    %   phi = [1;
    %          rBar;
    %          vBar;
    %          rBar' rBar;
    %          sin(...);
    %          cos(...);
    %          rBar' vBar]
    %
    % Therefore,
    %
    %   nPhi = 1 + dim + dim + 1 + 1 + 1 + 1
    %        = 2*dim + 5.
    cfg.dnn.nPhi = 2*cfg.dim + 5;

    % Number of trainable output-layer parameters per branch.
    %
    %   W_i has size dim x nPhi,
    %   theta_i = vec(W_i).
    cfg.dnn.nThetaPerBranch = cfg.dim * cfg.dnn.nPhi;

    % ---------------------------------------------------------------------
    % DNN residual usage switch
    % ---------------------------------------------------------------------
    % true:
    %   Use local DNN residual in prediction:
    %
    %       dot v = a_nom + d_i(eta;theta_i)
    %
    % false:
    %   Keep theta in the augmented EKF, but do not inject the DNN residual
    %   into physical prediction:
    %
    %       dot v = a_nom
    %
    % This is useful for debugging whether the learned DNN residual improves
    % or degrades target tracking.
    cfg.dnn.useResidualInPrediction = true;


    % Conservative trust factor for injecting the learned DNN residual into the
    % physical prediction model.
    %
    % betaDNN = 1.0 gives full DNN residual injection.
    % betaDNN = 0.0 recovers the physical prediction model.
    %
    % Start small because local-only branches may not represent the full
    % branchwise truth residual.
    cfg.dnn.predictionResidualSource = "oracle"; % What residual should I use.
    cfg.dnn.residualInjectionGain = 1.0;

    % ---------------------------------------------------------------------
    % Initial DNN parameter estimate and covariance
    % ---------------------------------------------------------------------
    % Previous value Ptheta0 = 1e-6 corresponds to std(theta)=1e-3, which is
    % too large for the current residualAmp = 1e-4 case.
    %
    % The hidden truth branch weights are roughly O(1e-5) per component.
    % Therefore, start theta near zero with a conservative initial covariance.
    cfg.dnn.theta0_std = 0*1e-5;

    % Initial parameter covariance.
    %
    % std(theta) = 2e-5.
    %
    % This still lets the filter move the weights, but prevents the DNN from
    % immediately producing unrealistically large residual accelerations.
    cfg.dnn.Ptheta0 = (2e-5)^2;

    % Numerical covariance floor.
    cfg.dnn.minCovDiag = 1e-14;

    % ---------------------------------------------------------------------
    % DNN parameter dynamics
    % ---------------------------------------------------------------------
    % Options:
    %   "random_walk" : dot theta = 0, covariance grows through qTheta
    %   "FOGM"        : dot theta = -(1/tauTheta)(theta-thetaMean) + noise
    %
    % FOGM is used here as weak regularization to prevent unbounded parameter
    % drift while still behaving nearly constant over short measurement
    % blackouts.
    cfg.dnn.thetaDynamics = "FOGM";

    % FOGM correlation time [s].
    cfg.dnn.thetaTau = 10000;

    % FOGM mean. Zero mean acts as weak regularization.
    cfg.dnn.thetaMean = 0;

    % FOGM steady-state theta standard deviation.
    %
    % This defines the base parameter process noise:
    %
    %   Qtheta_base = thetaSigmaSS^2 * (1 - exp(-2*dt/thetaTau)) I.
    %
    % Start conservatively. If the DNN barely learns, increase this to 5e-5.
    cfg.dnn.thetaSigmaSS = 2e-5;

    % Random-walk spectral density. Used only if thetaDynamics = "random_walk".
    cfg.dnn.qTheta = 1e-12;

    % ---------------------------------------------------------------------
    % Adaptive covariance matching for Qtheta
    % ---------------------------------------------------------------------
    % The adaptive law uses
    %
    %   Qtheta = gammaTheta * Qtheta_base.
    %
    % gammaTheta is updated in DNN_EKF_Update_Local.m using innovation
    % covariance matching:
    %
    %   ratio = trace(S_hat) / trace(S_model).
    %
    % Because your NIS is already close to consistent, this adaptation should
    % be conservative. Its main role is to slowly adjust the allowed DNN
    % parameter mobility, not to aggressively force NIS matching.
    % Adapt both Qtheta and Qepsilon,c from every valid measurement update.
    cfg.dnn.adaptQThetaEnabled = true;

    % Initial Qtheta multiplier.
    cfg.dnn.gammaTheta0 = 1.0;

    % Clamp gammaTheta.
    %
    % Keep this narrow at first. If gammaTheta grows too much, the DNN can
    % again overfit the bearing innovation with unrealistic acceleration.
    cfg.dnn.gammaThetaMin = 0.1;
    cfg.dnn.gammaThetaMax = 10.0;

    % EWMA settings for empirical innovation covariance.
    cfg.dnn.cmAlphaS = 0.01;
    cfg.dnn.cmAlphaMu = 0.01;

    % Burn-in before adapting gammaTheta.
    %
    % Your covariance matching ratio plot showed early saturation near the
    % upper ratio clamp. A longer burn-in prevents the early transient from
    % immediately inflating Qtheta.
    cfg.dnn.cmBurnInMeas = 200;

    % Apply adaptation every cmAdaptInt available measurements.
    cfg.dnn.cmAdaptInt = 10;

    % Deadband on log(trace(S_hat)/trace(S_model)).
    %
    % Small deviations around ratio = 1 are ignored.
    cfg.dnn.cmEpsLog = 0.05;

    % Conservative multiplicative adaptation gain.
    cfg.dnn.cmGainTheta = 5e-3;

    % Clamp trace ratio before taking log.
    cfg.dnn.cmRatioMin = 0.2;
    cfg.dnn.cmRatioMax = 10.0;

    % Continuous-time DNN approximation-error white acceleration noise:
    %   E[epsilon(t)epsilon(tau)'] = QepsilonC delta(t-tau),
    %   QepsilonC = gammaEpsilon*qEpsilonC0*I [m^2/s^3].
    % This base value preserves the former q_acc=1e-6 per-step covariance
    % at dt=0.01 s, while making the underlying intensity independent of dt.
    cfg.dnn.qEpsilonC0 = 1e-8;
    cfg.dnn.adaptQEpsilonEnabled = true;
    cfg.dnn.gammaEpsilon0 = 1.0;
    cfg.dnn.gammaEpsilonMin = 1e-2;
    cfg.dnn.gammaEpsilonMax = 1e2;
    cfg.dnn.cmGainEpsilon = 2e-2;

end
