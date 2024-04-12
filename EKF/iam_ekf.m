classdef iam_ekf < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        
        % Kalman Fitler
        n % number of state
        q % std of process
        Q % covariance of process
        r % std of measurement

        R % covariance of measurement
        s_init % initial state
        x_init % initial state + noise
        P_init % initial state covariance

        % Energy consraints
%         eMax
        e_pred
        eCurrent
        e_bool % True: energy less than min, False: energy greater than max no more force

        estimateState % x_hat
        Xf_pred % predicted final position (m)
        
        % pull in from data
        mu
        m
        sigma_2
        dt
        g
        restit % Resitution factor (unit less)

        d_offset

        otherVars_kal

        % Saving figures
        saveFigBool
        images_path
        dataType

    end

    methods
        function this = iam_ekf()
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            dbstop if error
            this.saveFigBool = true;
            this.images_path = 'images/';
            data = dataToy(); % Create data structure from toy model
%             data = dataSim('data/log_box_traj_mu001.txt');
%             data = dataReal('../log_real_traj.txt');
            this.init(data); % Initalize Kalman filter parameters
            [this.estimateState,this.Xf_pred,this.otherVars_kal] = this.get_estimateState(data);
            get_plots(this,data);
            disp('test');

        end

        function [] = init(this,data)

            % Pull in mu, m, sigma_2,dt
            this.mu = data.mu;
            this.m = data.m;
            this.sigma_2 = data.sigma_2;
            this.dt = data.dt;
            this.g = data.g;
            this.dataType = data.dataType;
            this.restit = data.restit;

            if strcmp(this.dataType,'sim')
                this.d_offset = 0.15;
            elseif strcmp(this.dataType,'toy')
                this.d_offset = 0;
            end


            % Initialize the filter
            this.n = data.n;      
            this.q = data.q;
            this.r = data.r;
            this.R = diag(this.r);
            this.Q = diag(this.q);                          
            this.P_init = eye(this.n);                           
            this.e_pred = data.sigma_1*data.sigma_2*sqrt(2*pi);
            this.e_bool = 1;

        end

        function [estimateState,Xf_pred,otherVars_kal] = get_estimateState(this, data)
            
            estimateState = NaN(this.n,data.numSteps);
            Xf_pred = NaN(1,data.numSteps);

            measCount = 1;
            x = data.X_init;
            u = data.flux;
            P = this.P_init;

            % Observation matrix Jacobian
            H = [1,0,0,0,0;...
                 0,0,1,0,0;]; 

            for k = 1:data.numSteps

              % State transition matrix Jacobian
              F = this.get_F(x,u);

              % Predict
              x_pred = this.get_f(x,u); % Predicted State estimate
              P_pred = F*P*F' + this.Q;

              % Update if new measurement
              if data.t_kal(k) > data.t(measCount+1)
                  S = H*P_pred*H' + this.R; % (assumes constant R)
                  K = P_pred*H'*inv(S);
                  z = data.measuredStates(:,measCount);
                  y = z - this.get_h(x_pred);
                  x = x_pred + K*y;
                  measCount = measCount + 1;
                  P = (eye(this.n) - K*H)*P_pred;
              else
                  x = x_pred;
              end

              % Check energy tank
              if x(end) >= this.e_pred
                  this.e_bool = false;
              end
              
              % Save values
              estimateState(:,k) = x;                  % Save estimate
              Xf_pred(k) = this.get_Xf_pred(x,u);        % Save Xf_pred
              [otherVars_kal.f_ext(k),otherVars_kal.f_f(k)] = this.get_force(x,u);
              otherVars_kal.P(k,:,:) = P; 
              otherVars_kal.e_bool(k) = this.e_bool;
              otherVars_kal.d(k) = this.get_d(x);

            end
            
        end

        % Get state transition 
        function [X_next] = get_f(this,x,u)

            % Extract State
            [x_o, dx_o, x_ee, dx_ee, E] = this.get_statesfromX(x);
            [f_ext, f_f] = this.get_force(x,u);
            x_o_next =   x_o + this.dt*dx_o + (this.dt^2/this.m) * (f_ext+f_f);
            dx_o_next =  dx_o + this.dt * (1/this.m) * (f_ext+f_f); 
            x_ee_next =  x_ee + this.dt*dx_ee;
            dx_ee_next = dx_ee;

            if E >= this.e_pred
                E_next = this.e_pred;
            else
                E_next = E + this.dt*abs(dx_o*f_ext);
            end

            X_next = [x_o_next;...
                      dx_o_next;...
                      x_ee_next;...
                      dx_ee_next;...
                      E_next];
        end

        % Get outputs
        function [hVec] = get_h(this,x)
            [x_o, dx_o, x_ee, dx_ee, E] = this.get_statesfromX(x);
            hVec = [x_o;x_ee];
        end

        % Get linearized state transition
        function F = get_F(this,x,u)

            % (CHECK) SIGN ERRO PROPIGATION OF d = x_o-x_ee

            % Extract State
            [x_o, dx_o, x_ee, dx_ee, E] = this.get_statesfromX(x);
            [f_ext,f_f] = this.get_force(x,u);
            par_f_par_x_o = -1 * (-this.get_d(x))/this.sigma_2^2 * f_ext;
            par_f_par_x_ee = -1 * par_f_par_x_o; % Same except for negative sign

            par_dx_o_par_x_o =  this.dt * (1/this.m) * par_f_par_x_o;
            par_dx_o_par_x_ee = this.dt * (1/this.m) * par_f_par_x_ee; 
            par_dx_o_par_dx_ee = 0;
            par_dx_o_par_E = 0;

            par_E_par_x_o =   this.dt*f_ext + this.dt*x_o*par_f_par_x_o;
            par_E_par_x_ee =  this.dt*x_o*par_f_par_x_ee; 

            % Put in matrix from
            F = [1,                this.dt,           0,                 0,                  0;... 
                 par_dx_o_par_x_o, 1,                 par_dx_o_par_x_ee, par_dx_o_par_dx_ee, par_dx_o_par_E;...
                 0,                0,                 1,                 this.dt,            0;...
                 0,                0,                 0,                 1,                  0;...
                 par_E_par_x_o,    0,                 par_E_par_x_ee,    0,                  1];
        
        end

        % Get forces acting on box
        function [f_ext,f_f] = get_force(this,x,u)
            
            [x_o, dx_o, x_ee, dx_ee, E] = this.get_statesfromX(x);

            % 0.2 object position prevents second hit problem for now
            if this.e_bool %&& (x_o < 0.2)
                f_ext = this.get_sigma_1(u)*exp(-this.get_d(x)^2/(2*this.sigma_2^2));
            else
                f_ext = 0;
            end

            % Friction acts on box
            if abs(dx_o) > 0
                f_f = -abs(dx_o)*this.mu*this.m*this.g; 
            else
                f_f = 0;
            end

        end

        % Get sigma_1
        function [sigma_1] = get_sigma_1(this,u)
            flux = u;
            sigma_1 = this.m * (1+this.mu)^2 * flux^2 / (2*this.sigma_2*sqrt(2*pi)); % (FIX) Assumed mu

        end
        
        % Get final box position estimate
        function [Xf_pred] = get_Xf_pred(this,x,u)

            [x_o, dx_o, x_ee, dx_ee, E] = this.get_statesfromX(x);
%             Xf_pred = x_o + (this.m * dx_o^2 / (2 * this.mu * this.g)); % (fix) Assumed mu

            x_o_init = 0; % (fix) Assumed x_o_init = 0
            flux = u;
            dx_o_postContact = flux*(1+this.restit); % (fix) Assumed mu
            Xf_pred_flux = x_o_init + (dx_o_postContact^2 / (2 * this.mu * this.g)); % (fix) Assumed mu
            Xf_pred_init = x_o + (dx_o^2 / (2 * this.mu * this.g)); % (fix) Assumed mu
            
            alpha = E/this.e_pred;
            Xf_pred = (1-alpha) * (Xf_pred_flux) + alpha * (Xf_pred_init);

        end

        function [d] = get_d(this,x)
            [x_o, dx_o, x_ee, dx_ee, E] = this.get_statesfromX(x);
            d = x_ee - x_o + this.d_offset;
        end

        function [x_o, dx_o, x_ee, dx_ee, E] = get_statesfromX(this,x)

            x_o = x(1);
            dx_o = x(2);
            x_ee = x(3);
            dx_ee = x(4);
            E = x(5);

        end

        function [X_o, dX_o, X_ee, dX_ee, E] = get_statesfromX_vec(this)

            X_o = this.estimateState(1,:);
            dX_o = this.estimateState(2,:);
            X_ee = this.estimateState(3,:);
            dX_ee = this.estimateState(4,:);
            E = this.estimateState(5,:);

        end

        % make plots
        function [] = get_plots(this,data)
            
            [X_o, dX_o, X_ee, dX_ee, E] = this.get_statesfromX_vec();
                        
            % Show Results
            figure('position',[332 138 560 705]);
            ax(1) = subplot(3,1,1);
            plot(data.t,data.x_o,'LineWidth',3,DisplayName="Measured Box Position"); hold on;
            plot(data.t_kal,X_o,'LineWidth',3,DisplayName="Estimated Box Position"); hold on;
            plot(data.t_kal,this.Xf_pred,":",'LineWidth',3,DisplayName="Predicted Distance");

%             plot(data.t_kal,this.Xf_pred,":k",'LineWidth',3,DisplayName="Predicted Final Position");
            ylabel("Position (m)");
            xlabel("Time (s)");
            legend('location','southeast'); box off; set(gca,'linewidth',2.5,'fontsize', 16);
            
            ax(2) = subplot(3,1,2);
            plot(data.t,data.dx_o,'LineWidth',3,DisplayName="Measured Box Speed"); hold on;
            plot(data.t_kal,dX_o,'LineWidth',3,DisplayName="Estimated Box Speed"); hold on;
%             plot(data.t_kal,data.trueStates(5,:),'LineWidth',3,DisplayName="Speed of ee (m/s)");
            ylabel("Speed (m/s)");
            xlabel("Time (s)");
            legend; box off; set(gca,'linewidth',2.5,'fontsize', 16);
            linkaxes(ax,'x');

            ax(3) = subplot(3,1,3);
            if strcmp(data.dataType, 'toy') % If toy model
                plot(data.t,data.otherVars_data.f_ext,'LineWidth',3,DisplayName="Measured Force"); hold on;
                plot(data.t_kal,this.otherVars_kal.f_ext,'LineWidth',3,DisplayName="Estimated Force"); hold on;
            
            elseif strcmp(data.dataType, 'sim')
                for i = 1:length(X_o)
                    [f_ext(i),f_f(i)] = this.get_force(this.estimateState(:,i),data.flux);
                end

                plot(data.t_kal,f_ext,'LineWidth',3,DisplayName="Estimated Force"); hold on;
            end
            ylabel("Force (N)");
            xlabel("Time (s)");
            legend; box off; set(gca,'linewidth',2.5,'fontsize', 16);
            linkaxes(ax,'x');

            if this.saveFigBool
                    saveas(gcf,[this.images_path,'/',this.dataType,'-SummaryPlot.png']);
            end

            figure('position',[2074 337 560 705]);
            % End-effector position (X_ee)
            ax(1) = subplot(3,1,1);
            plot(data.t,data.x_ee,'LineWidth',3,DisplayName="Measured d"); hold on; % (FIX) Harded coded offset
            plot(data.t_kal,X_ee,'LineWidth',3,DisplayName="Estimated d"); hold on;
            box off; set(gca,'linewidth',2.5,'fontsize', 16);
            ylabel('$X_{ee} \textrm{(m)}$','interpreter','latex','fontsize',25);
            xlabel('Time $\textrm{(s)}$','interpreter','latex','fontsize',25);

            % Distance (d)
            ax(2) = subplot(3,1,2);

            plot(data.t,data.x_ee-data.x_o + this.d_offset,'LineWidth',3,DisplayName="Measured d"); hold on; % (FIX) Harded coded offset
            plot(data.t_kal,this.otherVars_kal.d,'LineWidth',3,DisplayName="Estimated d"); hold on;
            box off; set(gca,'linewidth',2.5,'fontsize', 16);
            ylabel('$d \textrm{(m)}$','interpreter','latex','fontsize',25);
            xlabel('Time $\textrm{(s)}$','interpreter','latex','fontsize',25);

            % Energy (E)
            ax(3) = subplot(3,1,3);
%             plot(data.t, data.otherVars_data.E,'linewidth',2.5,DisplayName="Measured Energy"); hold on;
            plot(data.t_kal, E,'linewidth',2.5,DisplayName="Estimated Energy"); hold on;
            plot(data.t_kal, this.e_pred*ones(data.numSteps,1),'--k','linewidth',1,DisplayName="Max Energy");
            box off; set(gca,'linewidth',2.5,'fontsize', 16);
            ylabel('$E \textrm{(J)}$','interpreter','latex','fontsize',25);
            xlabel('Time $\textrm{(s)}$','interpreter','latex','fontsize',25);
            legend('location','southeast');

            if this.saveFigBool
                    saveas(gcf,[this.images_path,'/',this.dataType,'-ContactModeling.png']);
            end

%             figure;
%             plot(data.x_ee-data.x_o,data.otherVars_data.f_ext); hold on;
%             plot(X_ee-X_o,this.otherVars_kal.f);
% 

%             figure;plot(exp(-(X_ee-X_o).^2./(2*this.sigma_2^2)));

            % The problem is with the X_o and X_ee estiamte

%             figure; plot(X_ee-X_o,f_ext); hold on;
% 
%             figure; plot(X_ee-X_o);
%             figure; plot(data.x_ee-data.x_o);
% 
%             figure; plot(X_ee);
%             figure; plot(data.x_ee);
% 
%             figure;plot(data.t_kal,this.otherVars_kal.e_bool);

%             figure('position',[923 294 560 420]);
%             plot(data.t_kal, this.otherVars_kal.P(:,1,1),'linewidth',2.5,DisplayName="Covariance Box Position"); hold on;
%             plot(data.t_kal, this.otherVars_kal.P(:,1,1),'linewidth',2.5,DisplayName="Covariance Box Velocity"); hold on;
%             plot(data.t_kal, this.otherVars_kal.P(:,1,1),'linewidth',2.5,DisplayName="Covariance EE Position"); hold on;
%             plot(data.t_kal, this.otherVars_kal.P(:,1,1),'linewidth',2.5,DisplayName="Covariance Box Position"); hold on;
%             box off; set(gca,'linewidth',2.5,'fontsize', 16);
%             ylabel('$au$','interpreter','latex','fontsize',25);
%             xlabel('Time $\textrm{(s)}$','interpreter','latex','fontsize',25);
%             legend('location','southeast');
    
        end


    end
end










