function plotMeasurementAvailability(results, cfg)
%{
Function:
    plotMeasurementAvailability.m

Purpose:
    Plot the measurement availability history for all watcher spacecraft.

    This function visualizes the binary measurement-availability indicator

        delta_i^m(k) in {0,1},

    where

        delta_i^m(k) = 1

    means watcher i has a valid measurement at time step k, and

        delta_i^m(k) = 0

    means watcher i performs prediction only at time step k.

Inputs:
    results - Simulation result structure.
              Required fields:
                  results.time      - time vector, size 1 x N or N x 1
                  results.measAvail - measurement availability log,
                                      size N x cfg.Nw

    cfg     - Simulation configuration structure.
              Required fields:
                  cfg.Nw

Outputs:
    None.
    This function creates a MATLAB figure.

Main equations:
    The measurement availability indicator is

        delta_i^m(k) =
            1, if watcher i has a valid measurement at time k,
            0, otherwise.

    In the EKF loop, the update is performed only when

        delta_i^m(k) = 1.

    If

        delta_i^m(k) = 0,

    the watcher performs prediction only.

Notes:
    - This measurement-availability indicator is separate from the
      communication trigger used later for DNN parameter sharing.
    - In Step 01, fovAvailable.m may return true at every step, so this plot
      may show all ones.
    - Later, this function will be useful for visualizing FOV dropout,
      blackout intervals, camera pointing constraints, and intermittent
      measurements.
%}
    
    time = results.time(:);
    measAvail = results.measAvail;
    
    Nw = cfg.Nw;
    
    if size(measAvail, 2) ~= Nw
        error("results.measAvail must have size N x cfg.Nw.");
    end
    
    figure;
    hold on; grid on;
    
    for i = 1:Nw
        y = double(measAvail(:,i));
    
        % Offset each watcher vertically so that the binary histories are easy
        % to distinguish.
        yOffset = y + (i-1)*1.3;
    
        stairs(time, yOffset, "LineWidth", 1.2);
    end
    
    xlabel("Time [s]");
    ylabel("Measurement availability");
    title("Watcher Measurement Availability");
    
    yticks((0:Nw-1)*1.3 + 0.5);
    yticklabels("Watcher " + string(1:Nw));
    
    ylim([-0.3, (Nw-1)*1.3 + 1.3]);

end