classdef iam_ekf < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        
        % Kalman Fitler
        n % number of state
        n_boxModel % number of states in box model
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

        fullModel 
        % estimateState % x_hat
        % Xf_pred % predicted final position (m)
        % otherVars_kal

        boxModel
        % estimateState % x_hat
        % Xf_pred % predicted final position (m)
        % otherVars_kal


        % pull in from data
        m
        sigma_2
        dt
        g

        d_offset


        % Saving figures
        saveFigBool
        images_path
        dataType

    end

    methods
        function this = iam_ekf()
            close all
            dbstop if error
            this.saveFigBool = true;
            this.images_path = 'images/';
%             data = dataToy(); % Create data structure from toy model
%             data = dataSim('data/log_box_traj_mu001.txt');
%             data = dataReal('data/hit_1-IIWA_7.csv');
%             this.init(data); % Initalize Kalman filter parameters
%             [this.fullModel] = this.get_estimateState_fullModel(data);
%             [this.boxModel] = this.get_estimateState_boxModel(data);

            % Change to run mulitple trials with diffrent coefficients
            initialCoeffients = [0.5;0.5];
            for i = 1:10
%                 data = dataToy(initialCoeffients(:,i));
%                 data = dataSim('data/log_box_traj_mu001.txt');
                data = dataReal('data/hit_1-IIWA_7.csv');

                this.init(data);
                [this.fullModel{i}] = this.get_estimateState_fullModel(data);
                initialCoeffients(:,i+1) = this.fullModel{i}.estimateState([end-1:end],end);
            end
            [this.boxModel] = this.get_estimateState_boxModel(data);

            trialDex = 1;
            get_plots(this,data,trialDex);

            figure;plot(initialCoeffients');
            disp('test');

        end

        function [] = init(this,data)

            % Pull in mu, m, sigma_2,dt
            this.m = data.m;
            this.sigma_2 = data.sigma_2;
            this.dt = data.dt;
            this.g = data.g;
            this.dataType = data.dataType;

            if strcmp(this.dataType,'sim')
                this.d_offset = 0.15;
            elseif strcmp(this.dataType,'toy')
                this.d_offset = 0;
            end


            % Initialize the filter
            this.n = data.n;   
            this.n_boxModel = 4;
            this.q = data.q;
            this.r = data.r;
            this.R = diag(this.r);
            this.Q = diag(this.q);                          
            this.P_init = eye(this.n);                           
            this.e_pred = data.sigma_1*data.sigma_2*sqrt(2*pi);
            this.e_bool = 1;

        end

        function [fullModel] = get_estimateState_fullModel(this, data)
            
            estimateState = NaN(this.n,data.numSteps);
            Xf_pred = NaN(1,data.numSteps);

            measCount = 1;
            x = data.X_init;
            u = data.flux;
            P = this.P_init;

            % Observation matrix Jacobian
            H = [1,0,0,0,0,0,0;...
                 0,0,1,0,0,0,0;]; 

            for k = 1:data.numSteps

              % State transition matrix Jacobian
              F = this.get_F(x,u,'full');

              % Predict
              x_pred = this.get_f(x,u,'full'); % Predicted State estimate
              P_pred = F*P*F' + this.Q;

              % Update if new measurement
              if data.t_kal(k) > data.t(measCount+1)
                  S = H*P_pred*H' + this.R; % (assumes constant R)
                  K = P_pred*H'*inv(S);
                  z = data.measuredStates(:,measCount);
                  y = z - this.get_h(x_pred,'full');
                  x = x_pred + K*y;
                  measCount = measCount + 1;
                  P = (eye(this.n) - K*H)*P_pred;
              else
                  x = x_pred;
              end

              % Check energy tank
              [~, ~, ~, ~, E,~,~] = this.get_statesfromX(x);
              if E >= this.e_pred
                  this.e_bool = false;
              end
              
              % Save values
              estimateState(:,k) = x;                  % Save estimate
              [Xf_pred(k),Xf_pred_flux(k),Xf_pred_int(k)] = this.get_Xf_pred(x,u,'full');        % Save Xf_pred
              [otherVars_kal.f_ext(k),otherVars_kal.f_f(k)] = this.get_force(x,u);
              otherVars_kal.P(k,:,:) = P; 
              otherVars_kal.e_bool(k) = this.e_bool;
              otherVars_kal.d(k) = this.get_d(x);
              otherVars_kal.Xf_pred_flux(k) = Xf_pred_flux(k);
              otherVars_kal.Xf_pred_int(k) = Xf_pred_int(k);
            end

            % Create output Structure fullModel
            fullModel.estimateState = estimateState;
            fullModel.Xf_pred = Xf_pred;
            fullModel.otherVars_kal = otherVars_kal;
            
        end

        function [boxModel] = get_estimateState_boxModel(this, data)
            
            estimateState = NaN(this.n_boxModel,data.numSteps);
            Xf_pred = NaN(1,data.numSteps);

            measCount = 1;
            x = data.X_init([1:2,end-1:end]);
            u = data.flux;
            P = this.P_init(1:this.n_boxModel,1:this.n_boxModel); % (CHECK) Assumed identidy P_init
            Q_box = diag(this.q([1:2,end-1:end]));

            % Observation matrix Jacobian
            H = [1,0,0,0]; 

            for k = 1:data.numSteps

              % State transition matrix Jacobian
              F = this.get_F(x,u,'box');

              % Predict
              x_pred = this.get_f(x,u,'box'); % Predicted State estimate
              P_pred = F*P*F' + Q_box;

              % Update if new measurement
              if data.t_kal(k) > data.t(measCount+1)
                  S = H*P_pred*H' + this.R(1); % (NOTE) hard code first R
                  K = P_pred*H'*inv(S);
                  z = data.measuredStates(1,measCount); % (NOTE) hard code first measure
                  y = z - this.get_h(x_pred,'box');
                  x = x_pred + K*y;
                  measCount = measCount + 1;
                  P = (eye(this.n_boxModel) - K*H)*P_pred;
              else
                  x = x_pred;
              end

              % Save values
              estimateState(:,k) = x;                  % Save estimate
              [Xf_pred(k)] = this.get_Xf_pred(x,u,'box');        % Save Xf_pred
              otherVars_kal.P(k,:,:) = P; 
            end

            % Create output Structure fullModel
            boxModel.estimateState = estimateState;
            boxModel.Xf_pred = Xf_pred;
            boxModel.otherVars_kal = otherVars_kal;
            
        end

        % Get state transition 
        function [X_next] = get_f(this,x,u,model)

            if strcmp(model,'full')
                % Extract State
                [x_o, dx_o, x_ee, dx_ee, E, mu, restit] = this.get_statesfromX(x);
                [f_ext, f_f] = this.get_force(x,u);
                x_o_next =   x_o + this.dt*dx_o + 0.5*(this.dt^2/this.m) * (f_ext+f_f);
                dx_o_next =  dx_o + this.dt * (1/this.m) * (f_ext+f_f);
                x_ee_next =  x_ee + this.dt*dx_ee;
                dx_ee_next = dx_ee;

                if E >= this.e_pred
                    E_next = this.e_pred;
                else
                    E_next = E + this.dt*abs(dx_o*f_ext);
                end

                mu_next = mu;
                restit_next = restit;

                X_next = [x_o_next;...
                      dx_o_next;...
                      x_ee_next;...
                      dx_ee_next;...
                      E_next;...
                      mu_next;...
                      restit_next];

            elseif strcmp(model,'box')

                % Extract State
                [x_o, dx_o, mu, restit] = this.get_statesfromX_boxModel(x);
                x_o_next =   x_o + this.dt*dx_o;
                dx_o_next =  dx_o;
                mu_next = mu;
                restit_next = restit;

                X_next = [x_o_next;...
                          dx_o_next;...
                          mu_next;...
                          restit_next];
                
            end



        end

        % Get outputs
        function [hVec] = get_h(this,x,model)
            if strcmp(model,'full')
            [x_o, dx_o, x_ee, dx_ee, E, mu, restit] = this.get_statesfromX(x);
            hVec = [x_o;x_ee];
            elseif strcmp(model,'box')
                [x_o, dx_o, mu, restit] = this.get_statesfromX_boxModel(x);
                hVec = [x_o];
            end
        end

        % Get linearized state transition
        function F = get_F(this,x,u,model)
            
            if strcmp(model,'full')
                % (CHECK) SIGN ERRO PROPIGATION OF d = x_o-x_ee

                % Extract State
                [x_o, dx_o, x_ee, dx_ee, E, mu, restit] = this.get_statesfromX(x);
                [f_ext,f_f] = this.get_force(x,u);

                par_fext_par_x_o = -1 * (-this.get_d(x))/this.sigma_2^2 * f_ext;
                par_fext_par_x_ee = -1 * par_fext_par_x_o; % Same except for negative sign

                par_x_o_par_x_o = 1 + 0.5*this.dt^2/this.m * par_fext_par_x_o;
                par_x_o_par_x_ee = 0.5*this.dt^2/this.m * par_fext_par_x_ee;

                par_dx_o_par_x_o =  this.dt * (1/this.m) * par_fext_par_x_o;
                par_dx_o_par_x_ee = this.dt * (1/this.m) * par_fext_par_x_ee;

                par_fext_par_mu = 0;
                par_fext_par_restit = this.m*(1+restit)/(this.sigma_2 * sqrt(2*pi)) * exp(-this.get_d(x)^2/(2*this.sigma_2^2));

                par_ff_par_mu = -sign(dx_o)*this.m*this.g;
                par_ff_par_restit = 0;

                par_xo_par_mu =     0.5*this.dt^2/this.m * (par_fext_par_mu + par_ff_par_mu);
                par_xo_par_restit = 0.5*this.dt^2/this.m * (par_fext_par_restit + par_ff_par_restit);

                par_dxo_par_mu = this.dt/this.m * (par_fext_par_mu + par_ff_par_mu);
                par_dxo_par_restit = this.dt/this.m * (par_fext_par_restit + par_ff_par_restit);

                % Put in matrix from
                % [x_o, dx_o, x_ee, dx_ee, E, mu, restit]
                F = [par_x_o_par_x_o, this.dt,           par_x_o_par_x_ee,  0,          0,      par_xo_par_mu,     par_xo_par_restit;...
                    par_dx_o_par_x_o, 1,                 par_dx_o_par_x_ee, 0,          0,      par_dxo_par_mu,    par_dxo_par_restit;...
                    0,                0,                 1,                 this.dt,    0,      0,                 0;...
                    0,                0,                 0,                 1,          0,      0,                 0;...
                    0,                0,                 0,                 0,          1,      0,                 0;...
                    0,                0,                 0,                 0,          0,      1,                 0;...
                    0,                0,                 0,                 0,          0,      0,                 1];

                % (FIX) Add partial terms for mu and restit
            elseif strcmp(model,'box')
                F = [1, this.dt,   0,   0;...
                     0,       1,   0,   0;...
                     0,       0,   1,   0;...
                     0,       0,   0,   1];
            end
        
        end

        % Get forces acting on box
        function [f_ext,f_f] = get_force(this,x,u)
            
            [x_o, dx_o, x_ee, dx_ee, E, mu, restit] = this.get_statesfromX(x);

            if this.e_bool 
                f_ext = this.get_sigma_1(x,u)*exp(-this.get_d(x)^2/(2*this.sigma_2^2));
            else
                f_ext = 0;
            end

            % Friction acts on box
            if abs(dx_o) > 0
                f_f = -sign(dx_o)* mu *this.m * this.g; 
            else
                f_f = 0;
            end

        end

        % Get sigma_1
        function [sigma_1] = get_sigma_1(this,x,u)
            flux = u;
            [x_o, dx_o, x_ee, dx_ee, E, mu, restit] = this.get_statesfromX(x);
            sigma_1 = this.m * (1+restit)^2 * flux^2 / (2*this.sigma_2*sqrt(2*pi));

        end
        
        % Get final box position estimate
        function [Xf_pred,Xf_pred_flux,Xf_pred_int] = get_Xf_pred(this,x,u,model)

            if strcmp(model,'full')
                [x_o, dx_o, x_ee, dx_ee, E, mu, restit] = this.get_statesfromX(x);

                % Xf_pred = x_o + (this.m * dx_o^2 / (2 * this.mu * this.g)); % (fix) Assumed mu

                x_o_init = 0; % (fix) Assumed x_o_init = 0
                flux = u;
                dx_o_postContact = flux*(1+restit);
                Xf_pred_flux = x_o_init + (dx_o_postContact^2 / (2 * mu * this.g));
                Xf_pred_int = x_o + (dx_o^2 / (2 * mu * this.g));

                alpha = E/this.e_pred;
                Xf_pred = (1-alpha) * (Xf_pred_flux) + alpha * (Xf_pred_int);

            elseif strcmp(model,'box')
                [x_o, dx_o, mu, restit] = this.get_statesfromX_boxModel(x);
                Xf_pred = x_o + (dx_o^2 / (2 * mu * this.g));

            end


        end

        function [d] = get_d(this,x)
            [x_o, dx_o, x_ee, dx_ee, E, mu, restit] = this.get_statesfromX(x);
            d = x_ee - x_o + this.d_offset;
        end

        function [x_o, dx_o, x_ee, dx_ee, E, mu, restit] = get_statesfromX(this,x)

            x_o = x(1);
            dx_o = x(2);
            x_ee = x(3);
            dx_ee = x(4);
            E = x(5);
            mu = x(6);
            restit = x(7);

        end

        function [x_o, dx_o, mu, restit] = get_statesfromX_boxModel(this,x)

            x_o = x(1);
            dx_o = x(2);
            mu = x(3);
            restit = x(4);

        end

        function [X_o, dX_o, X_ee, dX_ee, E, mu, restit] = get_statesfromX_vec(this,trialDex)

            X_o = this.fullModel{trialDex}.estimateState(1,:);
            dX_o = this.fullModel{trialDex}.estimateState(2,:);
            X_ee = this.fullModel{trialDex}.estimateState(3,:);
            dX_ee = this.fullModel{trialDex}.estimateState(4,:);
            E = this.fullModel{trialDex}.estimateState(5,:);
            mu = this.fullModel{trialDex}.estimateState(6,:);
            restit = this.fullModel{trialDex}.estimateState(7,:);

        end

        function [X_o, dX_o, X_ee, dX_ee, E, mu, restit] = get_statesfromX_boxModel_vec(this)

            X_o = this.boxModel.estimateState(1,:);
            dX_o = this.boxModel.estimateState(2,:);
            mu = this.boxModel.estimateState(3,:);
            restit = this.boxModel.estimateState(4,:);

        end

        % make plots
        function [] = get_plots(this,data,trialDex)
            
            [X_o, dX_o, X_ee, dX_ee, E, mu , restit] = this.get_statesfromX_vec(trialDex);
                        
            % Show Results
            figure('position',[332 138 560 705]);
            ax(1) = subplot(3,1,1);
            plot(data.t,data.x_o,'LineWidth',3,DisplayName="Measured Box Position"); hold on;
            plot(data.t_kal,X_o,'LineWidth',3,DisplayName="Estimated Box Position"); hold on;
            plot(data.t_kal,this.fullModel{trialDex}.Xf_pred,":",'LineWidth',3,DisplayName="Predicted Distance (Full Model)");
%             plot(data.t_kal,this.fullModel.otherVars_kal.Xf_pred_flux,":r",'LineWidth',3,DisplayName="Predicted Distance (Flux)");
%             plot(data.t_kal,this.fullModel.otherVars_kal.Xf_pred_int,":g",'LineWidth',3,DisplayName="Predicted Distance (int)");
            plot(data.t_kal,this.boxModel.Xf_pred,"--",'LineWidth',3,DisplayName="Predicted Distance (Box Model)");

%             plot(data.t_kal,this.Xf_pred,":k",'LineWidth',3,DisplayName="Predicted Final Position");
            ylabel("Position (m)");
            xlabel("Time (s)");
            legend('location','southeast'); box off; set(gca,'linewidth',2.5,'fontsize', 16);
            
            ax(2) = subplot(3,1,2);
%             plot(data.t,data.dx_o,'LineWidth',3,DisplayName="Measured Box Speed"); hold on;
            plot(data.t_kal,dX_o,'LineWidth',3,DisplayName="Estimated Box Speed"); hold on;
%             plot(data.t_kal,data.trueStates(5,:),'LineWidth',3,DisplayName="Speed of ee (m/s)");
            ylabel("Speed (m/s)");
            xlabel("Time (s)");
            legend; box off; set(gca,'linewidth',2.5,'fontsize', 16);
            linkaxes(ax,'x');

            ax(3) = subplot(3,1,3);
            if strcmp(data.dataType, 'toy') % If toy model
                plot(data.t,data.otherVars_data.f_ext,'LineWidth',3,DisplayName="Measured Force"); hold on;
                plot(data.t_kal,this.fullModel{trialDex}.otherVars_kal.f_ext,'LineWidth',3,DisplayName="Estimated Force"); hold on;
            
            elseif strcmp(data.dataType, 'sim')
                for i = 1:length(X_o)
                    [f_ext(i),f_f(i)] = this.get_force(this.fullModel{trialDex}.estimateState(:,i),data.flux);
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
            grid on;

            % Distance (d)
            ax(2) = subplot(3,1,2);

            plot(data.t,data.x_ee-data.x_o + this.d_offset,'LineWidth',3,DisplayName="Measured d"); hold on; % (FIX) Harded coded offset
            plot(data.t_kal,this.fullModel{trialDex}.otherVars_kal.d,'LineWidth',3,DisplayName="Estimated d"); hold on;
            box off; set(gca,'linewidth',2.5,'fontsize', 16);
            ylabel('$d \textrm{(m)}$','interpreter','latex','fontsize',25);
            xlabel('Time $\textrm{(s)}$','interpreter','latex','fontsize',25);
            grid on;

            % Energy (E)
            ax(3) = subplot(3,1,3);
%             plot(data.t, data.otherVars_data.E,'linewidth',2.5,DisplayName="Measured Energy"); hold on;
            plot(data.t_kal, E,'linewidth',2.5,DisplayName="Estimated Energy"); hold on;
            plot(data.t_kal, this.e_pred*ones(data.numSteps,1),'--k','linewidth',1,DisplayName="Max Energy");
            box off; set(gca,'linewidth',2.5,'fontsize', 16);
            ylabel('$E \textrm{(J)}$','interpreter','latex','fontsize',25);
            xlabel('Time $\textrm{(s)}$','interpreter','latex','fontsize',25);
            legend('location','southeast');
            grid on;

            if this.saveFigBool
                    saveas(gcf,[this.images_path,'/',this.dataType,'-ContactModeling.png']);
            end
            linkaxes(ax,'x');

            % Coefficients
            figure;
            plot(data.t_kal, mu,'linewidth',2.5,DisplayName="\mu"); hold on;
            plot(data.t_kal, restit,'linewidth',2.5,DisplayName="\epsilon");
            box off; set(gca,'linewidth',2.5,'fontsize', 16);
            ylim([0 1]);
            ylabel('Coefficient (au)','interpreter','latex','fontsize',25);
            xlabel('Time $\textrm{(s)}$','interpreter','latex','fontsize',25);
            legend('location','southeast');
            grid on;



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










