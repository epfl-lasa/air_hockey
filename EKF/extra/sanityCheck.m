function [] = sanityCheck()

    data = get_data();
    estimated = get_estimatedState(data);
    get_plot(estimated,data);
    disp('test');

end

function [data] = get_data()

    data.dt = 1/100;
    data.simTime = 2.5;
    data.tspan   = 0:data.dt:data.simTime;
    data.numSteps = length(data.tspan);
    data.m = 1; % Mass
    data.n = 4;
    data.q = [0.1; 0.1; 0.1; 0.1].^2; % Process Noise
    data.r = [0.1; 0.1].^2;   % Measurement Noise

    data.measurements = NaN(4,data.numSteps);

    x1 = 0;
    dx1 = 0;
    x2 = 0;
    dx2 = 1;

    for i = 1:data.numSteps

        data.processNoise(:,i) = data.q.*randn(data.n,1);
        data.measurementNoise(:,i) = data.r.*randn(2,1);

        f = 1;
        x1 = x1 + data.dt*dx1               + data.processNoise(1,i);
        dx1 = dx1 + (data.dt/data.m)*f      + data.processNoise(2,i);
        x2 = x2 + data.dt*dx2               + data.processNoise(3,i);
        dx2 = dx2                           + data.processNoise(4,i);

        data.measurements(:,i) = [x1;dx1;x2;dx2];
        data.measuredStates(:,i) = [x1,;x2]  + data.measurementNoise(:,i);
        data.x1(i) = x1;
        data.dx1(i) = dx1;
        data.x2(i) = x2;
        data.dx2(i) = dx2;

    end

end

function [estimated] = get_estimatedState(data)

    estimated.dt = 1/500;
    estimated.simTime = 2.5;
    estimated.tspan   = 0:estimated.dt:estimated.simTime;
    estimated.m = data.m; % Mass
    estimated.n = data.n;
    estimated.numSteps = length(estimated.tspan);

    estimated.state = NaN(data.n,data.numSteps);

    x = data.measurements(:,1);
    u = 0;

    % Initialize the filter
    estimated.r = data.r;
    R = diag(estimated.r);
    estimated.q = data.q;
    Q = diag(estimated.q);
    P = eye(data.n);

    % Observation matrix Jacobian
    H = [1,0,0,0;...
         0,0,1,0;];
    
    measCount = 1;

    for k = 1:estimated.numSteps
    
        % State transition matrix Jacobian
        F = get_F(estimated,x,u);
    
        % Predict
        x_pred = get_f(estimated,x,u); % Predicted State estimate
        P_pred = F*P*F' + Q;
    
        % Update if new measurement
        if estimated.tspan(k) > data.tspan(measCount+1)
            S = H*P_pred*H' + R; % (assumes constant R)
            K = P_pred*H'*inv(S);
            z = data.measuredStates(:,measCount);
            y = z - get_h(x_pred);
            x = x_pred + K*y;
            measCount = measCount + 1;
            P = (eye(estimated.n) - K*H)*P_pred;
        else
            x = x_pred;
        end
    
        % Save values
        estimated.state(:,k) = x;      % Save estimate
        [estimated.x1(k),estimated.dx1(k),estimated.x2(k),estimated.dx2(k)] = get_statesFromX(x);
        estimated.P(k,:,:) = P;
    
    end
end


    function [x_new] = get_f(estimated,x,u)

        [x1,dx1,x2,dx2] = get_statesFromX(x);

        f = 1;
        x1 = x1 + estimated.dt*dx1;
        dx1 = dx1 + (estimated.dt/estimated.m)*f;
        x2 = x2 + estimated.dt*dx2;
        dx2 = dx2;

        x_new = [x1;dx1;x2;dx2];

    end


    function [z] = get_h(x)
        [x1,dx1,x2,dx2] = get_statesFromX(x);
        z = [x1;x2];
    end

    function [F] = get_F(estimated,x,u)

        F = [1,     estimated.dt,    0,        0;...
             0,     1                0,        0;...
             0,     0,               1,        estimated.dt;...
             0,     0,               0,        1];

    end

    function [x1,dx1,x2,dx2] = get_statesFromX(x)
        x1 = x(1);
        dx1 = x(2);
        x2 = x(3);
        dx2 = x(4);
    end

    function [X1,dX1,X2,dX2] = get_statesFromX_vec(X)
        X1 = x(:,1);
        dX1 = x(:,2);
        X2 = x(:,3);
        dX2 = x(:,4);
    end

    function [] = get_plot(estimated,data)


        figure('position',[332 138 560 705]);
        ax(1) = subplot(3,1,1);
        % Dashed
        plot(data.tspan,data.x1,'LineWidth',3,DisplayName="Measured x1 Position"); hold on;
        plot(estimated.tspan,estimated.x1,'--','LineWidth',3,DisplayName="Estimated x1 Position"); hold on;
        plot(data.tspan,data.x2,'LineWidth',3,DisplayName="Measured x2 Position"); hold on;
        plot(estimated.tspan,estimated.x2,'--','LineWidth',3,DisplayName="Estimated x2 Position"); hold on;

%         % Dotted
%         plot(data.tspan,data.x1,'.','markersize',25,'LineWidth',3,DisplayName="Measured x1 Position"); hold on;
%         plot(estimated.tspan,estimated.x1,'.','markersize',15,'LineWidth',3,DisplayName="Estimated x1 Position"); hold on;
%         plot(data.tspan,data.x2,'.','markersize',25,'LineWidth',3,DisplayName="Measured x2 Position"); hold on;
%         plot(estimated.tspan,estimated.x2,'.','markersize',15,'LineWidth',3,DisplayName="Estimated x2 Position"); hold on;

        ylabel("Position (m)");
        xlabel("Time (s)");
        legend('location','southeast'); box off; set(gca,'linewidth',2.5,'fontsize', 16);

        ax(2) = subplot(3,1,2);
        % Dashed
        plot(data.tspan,data.dx1,'.','markersize',10,'LineWidth',3,DisplayName="Measured x1 Velocity"); hold on;
        plot(estimated.tspan,estimated.dx1,'--','LineWidth',3,DisplayName="Estimated x1 Velocity"); hold on;
        plot(data.tspan,data.dx2,'LineWidth',3,DisplayName="Measured x2 Velocity"); hold on;
        plot(estimated.tspan,estimated.dx2,'--','LineWidth',3,DisplayName="Estimated x2 Velocity"); hold on;

%         % Dotted
%         plot(data.tspan,data.dx1,'.','markersize',25,'LineWidth',3,DisplayName="Measured x1 Velocity"); hold on;
%         plot(estimated.tspan,estimated.dx1,'.','markersize',15,'LineWidth',3,DisplayName="Estimated x1 Velocity"); hold on;
%         plot(data.tspan,data.dx2,'.','markersize',25,'LineWidth',3,DisplayName="Measured x2 Velocity"); hold on;
%         plot(estimated.tspan,estimated.dx2,'.','markersize',15,'LineWidth',3,DisplayName="Estimated x2 Velocity"); hold on;

        ylabel("Velocity (m/s)");
        xlabel("Time (s)");
        legend('location','southeast'); box off; set(gca,'linewidth',2.5,'fontsize', 16);
            
        
    end
