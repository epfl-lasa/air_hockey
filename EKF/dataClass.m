classdef dataClass < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        
        % Time values for EKF
        dt % time step (s)
        kalTime % kalman filter time (s)
        t_kal % vector of time steps kalman filter (s)
        numSteps % Length of t_kal

        q % Process Noise variance
        r % Measurement noise variance

        % Measurements sampled with time t with iregulare sampling
        t % Time sample of measurements (s)
        measuredStates % measured states values

        processNoise % Process noise added to singles
        measurementNoise % Measurement noise added to singles

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
        dx_o % object speed (m/s) (estimated by finite diffrence)
        x_ee % end-effector position (m)

        X_init % Initial State for EKF
        
        fileName % File name if ata is imported
        dataType % 'toy'

        otherVars_data % This is a structure with f_ext and E used for exporting simulation states fromt the toy model. 

    end

    methods

        function [] = check_variablesDefined(this)
            thisFieldNames = fieldnames(this);
            for i = 1:length(thisFieldNames)
                if isempty(this.(thisFieldNames{i})) && ~strcmp(thisFieldNames{i},'fileName')
                    error(['Empty field: ',thisFieldNames{i}]);
                end
            end

        end

    end
end










