function [I_ionic, delta_state_vars] = compute_chan_I_state_delta(state_vars, neuron_code)
% compute_ion_channel_I(state_vars, neuron_type) computes total inherent
% membrane channel current, and infinitesimal change in all state variables
% for N neurons in a population, except the membrane voltage
%       Inputs:
%       state_vars: state variables, N-by-M matrix, each row is [V, m_na,
%       h_na, m_nap, h_nap, m_k, m_cal, h_cal, m_kca, ca_concentration].
%       Voltage is in mV;
%       neuron_type: string, neuron type to determine maximal conductance
%       rand_seed: nonnegative integer to set up the random number
%       generator for consistent randomized leakage potentials
%       Outputs:
%       I_ions: total inherent membrane channel current, N-by-1 matrix
%       delta_state_vars: change in state variables in one time step,
%       N-by-M matrix

% ----- Constant Parameters ----- 
    num_of_neurons = size(state_vars, 1);
    % Maximal conductances in nS
    g_na = 0; g_nap = g_na; g_k = g_na; g_cal = g_na; g_kca = g_na; g_l = g_na;
    if neuron_code == 4 % population pre_I
        g_na = 170;
        g_nap = 5.0; 
        g_k = 180;
        g_l = 2.5;
    elseif neuron_code == 2 % population post_I
        g_na = 400;
        g_k = 250;
        g_l = 6.0;
    else
        g_na = 400;
        g_k = 250;
        g_cal = 0.05;
        g_kca = 3.0;
        g_l = 6.0;
    end
    % Reversal potentials in mV
    rng(neuron_code);
    E_na = 55;
    E_k = -94;
    E_ca = 13.27.*log(4./state_vars(:, end));
    if neuron_code == 4 % population pre_I
        E_l = -68 + 1.36.*randn(num_of_neurons, 1);
    else
        E_l = -60 + 1.2.*randn(num_of_neurons, 1);
    end

    % Total ionic current (N-by-1 column vector)
    I_ionic = g_na.*state_vars(:,2).^3.*state_vars(:,3).*(state_vars(:,1)-E_na) + ...
        g_nap.*state_vars(:,4).*state_vars(:,5).*(state_vars(:,1)-E_na) + ...
        g_k.*state_vars(:,6).^4.*(state_vars(:,1)-E_k) + ...
        g_cal.*state_vars(:,7).*state_vars(:,8).*(state_vars(:,1)-E_ca) + ...
        g_kca.*state_vars(:,9).^2.*(state_vars(:,1)-E_k) + ...
        g_l.*(state_vars(:,1)-E_l);

    % Ion gating state transfer variables 
    delta_state_vars = zeros(size(state_vars));
    params = [43.8, 6, 14, 0.252; ...
              67.5, 10.8, 12.8, 8.456; ...
              47.1, 3.1, 6.2, 1; ...
              60, 9, 9, 5000;...
              nan, nan, nan, nan; ...
              27.4, 5.7, nan, 0.5;...
              52.4, 5.2, nan, 18; ...
              nan, nan, nan, nan];
    for i=1:size(params, 1)
        if i==5 % potassium rectifier
            close_to_open_rate = 0.01.*(state_vars(:,1)+44)./(1-exp(-(state_vars(:,1)+44)./5));
            open_to_close_rate = 0.17.*exp(-(state_vars(:,1)+49)./40);
            equi_state = close_to_open_rate./(close_to_open_rate + open_to_close_rate);
            time_constant = 1./(close_to_open_rate + open_to_close_rate);
        elseif i==8 % calcium-dependent potassium
            close_to_open_rate = 1.25*10^8.*state_vars(:,end).^2; 
            open_to_close_rate = 2.5;
            equi_state = close_to_open_rate./(close_to_open_rate + open_to_close_rate);
            time_constant = 1000./(close_to_open_rate + open_to_close_rate);
        else
            equi_state = 1./(1+exp(-(state_vars(:,1)+params(i,1))./params(i,2)));
            if ~isnan(params(i,3))
                time_constant = params(i,4)./cosh((state_vars(:,1)+params(i,1))./params(i,3));
            else
                time_constant = repmat(params(i,4), num_of_neurons, 1);
            end
        end
        delta_state_vars(:, i+1) = (equi_state - state_vars(:, i+1))./time_constant; 
        if any(isnan(delta_state_vars(:, i+1))) || any(isinf(delta_state_vars(:, i+1)))
            pause(1);
        end
    end
    
    % Intracellular calcium concentration
    buffering_prob = 0.03./(state_vars(:, end) + 0.03 + 0.001);
    delta_state_vars(:, end) = 5.18*10^(-8).*(1-buffering_prob)...
        .*10^(-15)*g_cal.*state_vars(:,7).*state_vars(:,8).*(state_vars(:,1)-E_ca) ...
        + (5*10^(-5)-state_vars(:, end))./500;
    
    % Leave the voltage update to the caller
    delta_state_vars(:, 1) = zeros(num_of_neurons, 1); 
end
