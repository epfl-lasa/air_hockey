classdef dataReal < dataClass
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    methods

        function this = dataReal(fileName,initCoeff)

            this.dataType = 'real';
            this.fileName = fileName;
            this.init(initCoeff);
            this.check_variablesDefined();

        end

        function [this] = init(this,initCoeff)

            delimiterIn = ',';
%             sim_data = importdata(this.fileName,delimiterIn)';
            sim_data = readtable(this.fileName);

            mu_initial = initCoeff(1);
            restit_initial = initCoeff(2);

%             % Interpolate data remove frequency drop around 0.35-0.37 seconds
%             dexDrop = find(1./diff(sim_data{:,'time'})<150);
%             sim_data{:,'time'} = sim_data{:,'time'} - sim_data{1,'time'};
%             n = size(sim_data,1);
%             t_end = sim_data{end,'time'};
%             time = [0:1/150:t_end]';
%             x_o = interp1(sim_data{[1:dexDrop-2,dexDrop+1:n],'time'},sim_data{[1:dexDrop-2,dexDrop+1:n],'x_o'},time,'linear');
%             x_eef = interp1(sim_data{[1:dexDrop-2,dexDrop+1:n],'time'},sim_data{[1:dexDrop-2,dexDrop+1:n],'x_eef'},time,'linear');
%             clear sim_data
%             sim_data = table(time,x_o,x_eef);
% 
%             % Fitler Data for jumps
%             cf = 20; % cutoff freqnency
%             Fs = 150;
%             [b,a] = butter(4,cf/(Fs/2)); % make filter
%             sim_data{:,'x_o'} = filtfilt(b,a,sim_data{:,'x_o'}); % apply fitler

            this.t = sim_data{:,'time'}- sim_data{1,'time'};
            this.x_o = sim_data{:,'x_o'} - sim_data{1,'x_o'}; % X position of the box
            x = sim_data{:,'x_o'} - sim_data{1,'x_o'}; % X position of the box
            y = sim_data{:,'y_o'} - sim_data{1,'y_o'}; % X position of the box
            this.x_o = sqrt(x.^2+y.^2);
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
%             this.r = [0.02; 0.02].^2; 
%             this.q = [0.005; 0.01; 0.005; 0.01; 0.01; 0.01; 0.01].^2;

            this.r = [0.001; 0.001].^2;
            this.q = [0.01; 0.03; 0.01; 0.03; 0.01; 0.01; 0.01].^2;

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










