classdef dataSim < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties

        % Time values for EKF
        dt % time step (s)
        % Set in spesific data function
        kalTime % kalman filter time (s)
        t_kal % vector of time steps
        numSteps % Length of t_kal

        q % Process Noise variance
        r % Measurement noise variance

        % Measurements sampled with time t with iregulare sampling
        t % Time sample of measurements
        measuredStates % measured states values
        processNoise
        measurementNoise

        m % Mass of box in toy model (kg) 
        m_ee % end-effector mass (kg)
        mu % coefficient of friction (unit less)
        g % gravitational coefficient (N/kg)
        restit % Resitution factor (unit less)
        n % length of state vector

        flux % flux of hit (m/s)
        
        V_EE % end-effector speed required to achieve target flux (m/s)

        sigma_2 % standard deviation of ?? (??)
        sigma_1 % max amplitude of the hitting force (N)

        % Measured quantities
        x_o % object position (m)
        dx_o % object speed (m) (estimated by finite diffrence)
        x_ee % end-effector position (m)

        X_init % Initial State for EKF
        
        fileName
        dataType % 'sim'

    end

    methods

        function this = dataSim(fileName)

            this.fileName = fileName;
            this.dataType = 'sim';
            this.init();

        end

        function [] = init(this)

            delimiterIn = ' ';
            sim_data = importdata(this.fileName,delimiterIn)';

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

            this.m = 0.363;
            this.m_ee = 2;
            this.mu = 0.3;
            this.g = 9.81;
            this.restit = 0.7;

            this.r = [0.001; 0.001].^2;
            this.q = [0.01; 0.03; 0.01; 0.03; 0.01].^2;

            this.flux = 1; %m/s,
            this.V_EE = this.flux * (1 + (this.m/this.m_ee))/(1+this.restit);
            this.X_init = [this.x_o(1); 0; this.x_ee(1); 0; 0];
            this.n = length(this.X_init);

            this.sigma_2 = 0.005; % Standard deviation
            this.sigma_1 = this.m*((1+this.restit)^2 * this.flux^2) / (2 * this.sigma_2 * sqrt(2*pi));

            % Define Measured States
            this.measuredStates(1,:) = this.x_o; % X position of the box
            this.measuredStates(2,:) = this.x_ee; % End-effector position

        end

    end
end










