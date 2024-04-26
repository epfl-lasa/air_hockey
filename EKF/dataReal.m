classdef dataReal < dataClass
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    methods

        function this = dataReal(fileName,initialCoeff)

            this.dataType = 'real';
            this.fileName = fileName;
            this.init(initialCoeff);
            this.check_variablesDefined();

        end

        function [this] = init(this,initialCoeff)

            delimiterIn = ',';
%             sim_data = importdata(this.fileName,delimiterIn)';
            sim_data = readtable(this.fileName);

            mu_initial = initialCoeff(1);
            restit_initial = initialCoeff(2);

            this.t = sim_data{:,'time'}- sim_data{1,'time'};
            this.x_o = sim_data{:,'x_o'} - sim_data{1,'x_o'}; % X position of the box
            this.dx_o = (1./diff(this.t)).*diff(this.x_o);
            this.dx_o(end+1) = this.dx_o(end);
            this.x_ee = sim_data{:,'x_eef'}; % (CHECK)

            % Look at variance
%             windoww = 50;
%             count = 1;
%             rangee = 1+windoww:length(this.t)-windoww;
%             for i = rangee
%                 x_o_var(count) = var(sim_data{i-windoww:i+windoww,'x_o'});
%                 count = count + 1;
%             end

%             figure; plot(this.t(rangee),x_o_var);

            % Simulate box's trajectory and track it
            % Specifying all parameters, boh for simulation (_sim) and for the Extended Kalman Filter
            this.dt = 0.001;
            % Initalize Sampling
            this.kalTime = this.t(end);
            this.t_kal   = this.dt:this.dt:this.kalTime;
            this.numSteps = length(this.t_kal);

            this.m = 0.363;
            this.m_ee = 2;
            this.mu = 0.3;
            this.g = 9.81;
            this.restit = 0.7;

            this.r = [0.01; 0.01].^2;
            this.q = [0.005; 0.01; 0.005; 0.01; 0.01; 0.01; 0.01].^2;

            this.flux = 0.98; %m/s,
            this.V_EE = this.flux * (1 + (this.m/this.m_ee))/(1+this.restit);
            this.X_init = [this.x_o(1); 0; this.x_ee(1); 0; 0; mu_initial; restit_initial];
            this.n = length(this.X_init);

            this.sigma_2 = 0.005; % Standard deviation
            this.sigma_1 = this.m*((1+this.restit)^2 * this.flux^2) / (2 * this.sigma_2 * sqrt(2*pi));

            % Define Measured States
            this.measuredStates(1,:) = this.x_o; % X position of the box
            this.measuredStates(2,:) = this.x_ee; % End-effector position

            this.otherVars_data = nan;
            this.processNoise = nan;
            this.measurementNoise = nan;

        end

    end
end










