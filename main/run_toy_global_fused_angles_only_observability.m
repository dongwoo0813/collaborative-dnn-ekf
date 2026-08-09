function out = run_toy_global_fused_angles_only_observability(seed,T,makePlots,dt)
%RUN_TOY_GLOBAL_FUSED_ANGLES_ONLY_OBSERVABILITY Global bearing-information toy.
% One global EKF estimates x=[r_x r_y v_x v_y a_x a_y]'.  All four bearing
% measurements are fused sequentially; each watcher nevertheless decides
% its own transverse pulse-pair maneuver from its own LOS-projected global
% covariance.  No range measurement is used.

    if nargin < 1 || isempty(seed), seed = 101; end
    if nargin < 2 || isempty(T), T = 1800; end
    if nargin < 3 || isempty(makePlots), makePlots = true; end
    if nargin < 4 || isempty(dt), dt = 0.5; end
    addpath(genpath(pwd)); cfg = configGlobalToy(seed,T,dt);
    fprintf('Toy global-fused angles-only EKF: seed=%d, T=%.0f s, dt=%.2f s\n',seed,T,dt);
    rng(seed); out.coast = simulateGlobalToy(cfg,false);
    rng(seed); out.active = simulateGlobalToy(cfg,true);
    out.cfg = cfg; out.summary = summarizeGlobalToy(out); disp(out.summary);
    if makePlots, out.figures = plot_toy_global_fused_angles_only_observability(out,true); end
end

function cfg = configGlobalToy(seed,T,dt)
    cfg.seed=seed; cfg.T=T; cfg.dt=dt; cfg.time=0:dt:T; cfg.N=numel(cfg.time); cfg.Nw=4;
    cfg.r0=[500;0]; cfg.v0=[0;.2];
    cfg.watcherR0=1000*[1 0 -1 0;0 1 0 -1]; cfg.watcherV0=repmat(cfg.v0,1,cfg.Nw);
    cfg.sigmaBearing=deg2rad(.05);
    cfg.P0=diag([50^2 50^2 .1^2 .1^2 (3e-4)^2 (3e-4)^2]); cfg.qJerk=2e-11;
    cfg.control.decisionDt=5; cfg.control.firstDecisionTime=20; cfg.control.cooldown=100;
    cfg.control.halfBurn=15; cfg.control.acceleration=2e-2;
    cfg.control.initialRatio=1.15; cfg.control.growthRatio=1.50;
end

function res = simulateGlobalToy(cfg,active)
    N=cfg.N; dt=cfg.dt; Nw=cfg.Nw; xTrue=zeros(6,N); xTrue(:,1)=[cfg.r0;cfg.v0;truthA(0)];
    xhat=zeros(6,N); xhat(:,1)=[cfg.r0+[35;-25];cfg.v0+[.04;-.03];zeros(2,1)]; P=cfg.P0;
    rw=zeros(2,N,Nw); vw=zeros(2,N,Nw); uw=zeros(2,N,Nw);
    metric=nan(N,Nw); referenceLog=nan(N,Nw); threshold=nan(N,Nw); trigger=false(N,Nw);
    referenceMetric=nan(Nw,1);
    eventStart=nan(Nw,1); lastEnd=-inf(Nw,1); lastMetric=nan(Nw,1); nextDecision=cfg.control.firstDecisionTime*ones(Nw,1); events=zeros(Nw,1);
    for i=1:Nw, rw(:,1,i)=cfg.watcherR0(:,i); vw(:,1,i)=cfg.watcherV0(:,i); end
    for k=1:N-1
        t=cfg.time(k);
        for i=1:Nw
            eventActive=active && isfinite(eventStart(i)) && t<eventStart(i)+2*cfg.control.halfBurn;
            if active && ~eventActive && t>=nextDecision(i)-eps
                q=radialVariance(xhat(:,k),P,rw(:,k,i)); metric(k,i)=q;
                if ~isfinite(referenceMetric(i)), referenceMetric(i)=q; end
                initialThreshold=cfg.control.initialRatio*referenceMetric(i);
                if isfinite(lastMetric(i)), qThreshold=max(initialThreshold,cfg.control.growthRatio*lastMetric(i)); else, qThreshold=initialThreshold; end
                threshold(k,i)=qThreshold/referenceMetric(i); referenceLog(k,i)=referenceMetric(i);
                if q>qThreshold && t>=lastEnd(i)+cfg.control.cooldown
                    eventStart(i)=t; lastMetric(i)=q; events(i)=events(i)+1; trigger(k,i)=true;
                    eventActive=true;
                end
                nextDecision(i)=t+cfg.control.decisionDt;
            end
            if eventActive
                elapsed=t-eventStart(i); phase=1; if elapsed>=cfg.control.halfBurn, phase=-1; end
                uw(:,k,i)=phase*cfg.control.acceleration*transverseDirection(xhat(:,k),rw(:,k,i));
                if elapsed+dt>=2*cfg.control.halfBurn, lastEnd(i)=eventStart(i)+2*cfg.control.halfBurn; end
            end
            rw(:,k+1,i)=rw(:,k,i)+vw(:,k,i)*dt+.5*uw(:,k,i)*dt^2;
            vw(:,k+1,i)=vw(:,k,i)+uw(:,k,i)*dt;
        end
        xTrue(:,k+1)=propagateTruth(xTrue(:,k),t,dt);
        [xp,Pp]=predictGlobal(xhat(:,k),P,cfg);
        for i=1:Nw
            z=bearing(xTrue(1:2,k+1),rw(:,k+1,i))+cfg.sigmaBearing*randn;
            [xp,Pp]=updateGlobal(xp,Pp,z,rw(:,k+1,i),cfg);
        end
        xhat(:,k+1)=xp; P=Pp;
    end
    res=struct('time',cfg.time,'xTrue',xTrue,'xhat',xhat,'P',P,'watcherR',rw,'watcherV',vw,'watcherU',uw, ...
        'metric',metric,'reference',referenceLog,'threshold',threshold,'trigger',trigger,'eventCount',events);
end

function x=propagateTruth(x,t,dt)
    a0=truthA(t); a1=truthA(t+dt); x(1:2)=x(1:2)+x(3:4)*dt+.5*a0*dt^2; x(3:4)=x(3:4)+.5*(a0+a1)*dt; x(5:6)=a1;
end
function a=truthA(t), a=[1.4e-4+2.5e-5*sin(2*pi*t/900);-1e-4+2e-5*cos(2*pi*t/760)]; end
function [xp,Pp]=predictGlobal(x,P,cfg)
    d=cfg.dt; I=eye(2); Z=zeros(2); F=[I d*I .5*d^2*I;Z I d*I;Z Z I]; q=cfg.qJerk;
    Q=kron(q*[d^5/20 d^4/8 d^3/6;d^4/8 d^3/3 d^2/2;d^3/6 d^2/2 d],I);
    xp=F*x; Pp=.5*(F*P*F'+Q+(F*P*F'+Q)');
end
function [x,P]=updateGlobal(x,P,z,rw,cfg)
    rho=x(1:2)-rw; r2=max(rho'*rho,1); h=atan2(rho(2),rho(1)); H=[-rho(2)/r2 rho(1)/r2 zeros(1,4)];
    nu=wrap(z-h); S=H*P*H'+cfg.sigmaBearing^2; K=P*H'/S; x=x+K*nu;
    I=eye(6); P=(I-K*H)*P*(I-K*H)'+K*(cfg.sigmaBearing^2)*K'; P=.5*(P+P');
end
function q=radialVariance(x,P,rw), e=x(1:2)-rw; e=e/max(norm(e),eps); q=max(e'*P(1:2,1:2)*e,0); end
function d=transverseDirection(x,rw), e=x(1:2)-rw; e=e/max(norm(e),eps); d=[-e(2);e(1)]; end
function b=bearing(r,rw), b=atan2(r(2)-rw(2),r(1)-rw(1)); end
function a=wrap(a), a=mod(a+pi,2*pi)-pi; end
function tab=summarizeGlobalToy(out)
    names=["Global fused / coast";"Global fused / local maneuvers"]; cases={out.coast,out.active}; tab=table();
    for j=1:2
        r=cases{j}; er=vecnorm(r.xhat(1:2,:)-r.xTrue(1:2,:),2,1); ev=vecnorm(r.xhat(3:4,:)-r.xTrue(3:4,:),2,1); ea=vecnorm(r.xhat(5:6,:)-r.xTrue(5:6,:),2,1);
        row=table(names(j),sqrt(mean(er.^2)),sqrt(mean(ev.^2)),sqrt(mean(ea.^2)),sqrt(mean(er(round(.9*numel(er)):end).^2)), ...
            'VariableNames',{'caseName','positionRMSE','velocityRMSE','accelerationRMSE','finalPositionRMSE'});
        if isempty(tab), tab=row; else, tab=[tab;row]; end %#ok<AGROW>
    end
end
