classdef dataToy < dataClass
    % This function inherests the class variables from the general data
    % Class. It creates the data set from the "Toy" model

    methods

        function this = dataToy()

            this.init();
            this.check_variablesDefined();

        end

        function [] = init(this)

            this.dataType = 'toy';

            this.m = 0.363;
            this.flux = 1; %m/s,
            this.m_ee = 2;
            this.mu = 0.3;
            this.g = 9.81;
            this.restit = 0.7;

            this.r = [0.001; 0.001].^2;
            this.q = [0.01; 0.03; 0.01; 0.03; 0.01].^2;

            this.V_EE = this.flux * (1 + (this.m/this.m_ee));
            this.X_init = [0; 0; -0.25; this.V_EE; 0]; % [X_o; dX_o; X_ee; dX_ee; E]
            this.n = length(this.X_init);

            % Call get sim_data
            [sim_data,this.processNoise,this.measurementNoise,this.otherVars_data] = this.get_dataToy();

            % Simulate box's trajectory and track it
            % Specifying all parameters, boh for simulation (_sim) and for the Extended Kalman Filter
            this.dt = 0.001;
            % Initalize Sampling
            this.kalTime = sim_data(1,end);
            this.t_kal   = this.dt:this.dt:this.kalTime;
            this.numSteps = length(this.t_kal);

            this.t = sim_data(1,:);
            this.x_o = sim_data(3,:); % X position of the box
            this.dx_o = (1./diff(this.t)).*diff(this.x_o);
            this.dx_o(end+1) = this.dx_o(end);
            this.x_ee = sim_data(6,:); % (CHECK)

            this.sigma_2 = 0.01; % Standard deviation
            this.sigma_1 = this.m*((1+this.restit)^2 * this.flux^2) / (2 * this.sigma_2 * sqrt(2*pi));

            % Define Measured States
            this.measuredStates(1,:) = this.x_o; % X position of the box
            this.measuredStates(2,:) = this.x_ee; % End-effector position

        end

        function [sim_data,processNoise,measurementNoise,otherVars_data] = get_dataToy(this)

            % NO REFRENCE TO "this" OBJECT

            % Initalize Sampling
            dt = 1/200;
            simTime = 2.5; 
            tspan   = 0:dt:simTime;
            numSteps = length(tspan);
           
            x = this.X_init; % initial state
            sim_data = NaN(7,numSteps);
            sim_data(1,1) = tspan(1);
            sim_data(3,1) = x(1); 
            sim_data(6,1) = x(3);

            % (REPLACE) WITH WILL DEFINED F_TRUE(X) LATER
            
            % Obtain True States and Measurements
            % Propagate the constant velocity model, and generate the measurements with noise. 
            %Note that the total force transmitted through impact is equal to :
            %Integral Fh over d = sigma_1 * sigma_2 * sqrt(2*pi)
            impulseOccured = false;
            for i = 2:length(tspan)

                processNoise(:,i) = 0.*this.q.*randn(this.n,1);
                measurementNoise(:,i) = 0.*this.r.*randn(2,1);

                x_o = x(1);
                dx_o = x(2);
                x_ee = x(3);
                dx_ee = x(4);
                E = x(5);
        
                d = x_ee-x_o; % (FIX) use same function later
                f_impact = 0;
                if ~impulseOccured && (d>=0) % Impart impulse
                    v_o_plus = (1+this.restit)*this.flux; %  (FIX) assumed mu
                    f_ext = (1/dt)*v_o_plus*this.m;
                    impulseOccured = true;
                elseif abs(dx_o) > 0
                    f_ext = -sign(dx_o)*this.mu*this.m*this.g; % Friction acts on box

                else
                    f_ext = 0;  % No force acts on box
                end

                x_o_next =   x_o  + dt*dx_o + 0.5*(dt^2/this.m) * f_ext     + processNoise(1,i);
                dx_o_next =  dx_o + dt * (1/this.m) * f_ext             + processNoise(2,i);
                x_ee_next =  x_ee + dt*dx_ee                            + processNoise(3,i);
                
                if ~impulseOccured
                    dx_ee_next = this.X_init(4)                                      + processNoise(4,i);
                elseif impulseOccured && x_ee < this.X_init(3)
                    dx_ee_next = 0                                                   + processNoise(4,i);
                elseif impulseOccured
                    dx_ee_next = 0                                      + processNoise(4,i);
                end

                E_next = E + dt*abs(dx_o*f_impact)                      + processNoise(5,i);

                X_next = [x_o_next;...
                          dx_o_next;...
                          x_ee_next;...
                          dx_ee_next;...
                          E_next];
                        
                x = X_next; % + sqrt(this.processNoise)*randn(length(x),1); % (CHECK) Add process noise
                sim_data(:,i) = [tspan(i);...
                                 nan;...
                                 x(1)              + measurementNoise(1,i);...
                                 nan;...
                                 nan;...
                                 x(3)              + measurementNoise(2,i);...
                                 nan]; % (CHECK) Add measurement noise
                otherVars_data.f_ext(i) = f_ext;
                otherVars_data.E(i) = x(end);

            end
        end

    end
end

