function metrics = computePhysicalEKFMetrics(results, cfg)
%{
Function:
    computePhysicalEKFMetrics.m

Purpose:
    Compute numerical tracking metrics for the Step 01 physical EKF baseline.

Inputs:
    results - Simulation result structure.
              Required fields:
                  results.time
                  results.etaTrue
                  results.xhat
                  results.Pdiag

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.dim
                  cfg.Nw

Outputs:
    metrics - Metrics structure.
              Fields:
                  metrics.posRMSE
                  metrics.velRMSE
                  metrics.meanPosRMSE
                  metrics.meanVelRMSE
                  metrics.finalPosError
                  metrics.finalVelError

Main equations:
    For watcher i,

        RMSE_{r,i}
        =
        sqrt( 1/N sum_k ||r_hat_i(k) - r_true(k)||^2 ).

    Similarly,

        RMSE_{v,i}
        =
        sqrt( 1/N sum_k ||v_hat_i(k) - v_true(k)||^2 ).

Notes:
    - These metrics are for physical target tracking only.
    - DNN approximation metrics will be added later after trueResidual.m and
      branchOutput.m are introduced.
%}

    dim = cfg.dim;
    Nw = cfg.Nw;
    N = numel(results.time);
    
    etaTrue = results.etaTrue;
    xhat = results.xhat;
    
    posRMSE = zeros(Nw,1);
    velRMSE = zeros(Nw,1);
    finalPosError = zeros(Nw,1);
    finalVelError = zeros(Nw,1);
    
    for i = 1:Nw
        posErrSqSum = 0;
        velErrSqSum = 0;
    
        for k = 1:N
            e = xhat(:,k,i) - etaTrue(:,k);
    
            er = e(1:dim);
            ev = e(dim+1:2*dim);
    
            posErrSqSum = posErrSqSum + er' * er;
            velErrSqSum = velErrSqSum + ev' * ev;
        end
    
        posRMSE(i) = sqrt(posErrSqSum / N);
        velRMSE(i) = sqrt(velErrSqSum / N);
    
        eFinal = xhat(:,end,i) - etaTrue(:,end);
        finalPosError(i) = norm(eFinal(1:dim));
        finalVelError(i) = norm(eFinal(dim+1:2*dim));
    end
    
    metrics.posRMSE = posRMSE;
    metrics.velRMSE = velRMSE;
    metrics.meanPosRMSE = mean(posRMSE);
    metrics.meanVelRMSE = mean(velRMSE);
    metrics.finalPosError = finalPosError;
    metrics.finalVelError = finalVelError;
    
    fprintf("\n=== Step 01 Physical EKF Metrics ===\n");
    for i = 1:Nw
        fprintf("Watcher %d: pos RMSE = %.6f, vel RMSE = %.6f\n", ...
            i, posRMSE(i), velRMSE(i));
    end
    fprintf("Mean position RMSE = %.6f\n", metrics.meanPosRMSE);
    fprintf("Mean velocity RMSE = %.6f\n", metrics.meanVelRMSE);
    fprintf("====================================\n\n");

end