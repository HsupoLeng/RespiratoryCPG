function new_state_vars = sim_unifm_HH_pop_exp_euler(t, state_vars, spikes, synaptic_weights, external_drives, drive_weights, leakage_voltages, neuron_code, num_of_neurons, sim_time_step)
% sim_unifm_HH_pop_exp_euler(t, y, I_in) computes the infinitesimal update to the state
% variables of N neurons modelled by Hudgkin-Huxley model
%   Input: 
%       t: timestamp, scalar;
%       state_vars: state variables, N-by-M matrix, each row is [V, m_na,
%       h_na, m_nap, h_nap, m_k, m_cal, h_cal, m_kca, ca_concentration]. 
%       Voltage is in mV;
%       spikes: timestamps of spikes coming from pre-synaptic populations,
%       K-by-1 cell matrix, where K is the total number of populations;
%       synaptic_weights: synaptic weight between populations, K-by-1
%       matrix, where K is the total number of populations;
%       external_drives: drives from other brain regions to this population
%       in the breathing CPG, L-by-1 matrix, where L is the total
%       number of drive sources
%       drive_weights: weight of each drive source to this population,
%       L-by-1 matrix. Weight is zero if no connection to the source.
%       Weight could be either positive or negative otherwise. 
%       leakage_voltages: leakage voltage of each neuron in the population
%       neuron_code: integer, neuron type to determine maximal conductance
%       num_of_neurons: number of neurons in the population
%       sim_time_step: simulation time step
%   Output:
%       new_state_vars: updated state variables


% Constant Parameters
C  = 36;    % capacitance (in pF)

state_vars = reshape(state_vars, num_of_neurons, []);
new_state_vars = zeros(size(state_vars));

% Compute exitatory and inhibitory synaptic current for neurons in the
% population. Identical for each neuron
% Time constants are in ms
% Voltages are in mV
time_constant_syn_ext = 5; 
time_constant_syn_inh = 15;
g_ext = 1.0; g_inh = g_ext; g_ext_dr = g_ext; g_inh_dr = g_ext; 
E_syn_ext = -10;
E_syn_inh = -75;

all_spikes_arr = horzcat(spikes{:});
num_of_spikes_per_pop = cellfun(@length, spikes);
synaptic_weights_ext = repelem(max(synaptic_weights, 0), num_of_spikes_per_pop);
synaptic_weights_inh = repelem(max(-synaptic_weights, 0), num_of_spikes_per_pop);
g_synExts_elems = exp(-(t - all_spikes_arr)./time_constant_syn_ext).*synaptic_weights_ext;
g_synInhs_elems = exp(-(t - all_spikes_arr)./time_constant_syn_inh).*synaptic_weights_inh;

% Alternative method of solving membrane voltage ODE: exponential Euler
% method
g_na = 0; g_nap = g_na; g_k = g_na; g_cal = g_na; g_kca = g_na; g_l = g_na;
if neuron_code == 1 % poulation aug_E
    g_na = 400;
    g_k = 250;
    g_cal = 0.05;
    g_kca = 3.0;
    g_l = 6.0;
elseif neuron_code == 4 % population pre_I
    g_na = 170;
    g_nap = 5.0; 
    g_k = 180;
    g_l = 2.5;
elseif neuron_code == 5 % population early_I_1
    g_na = 400;
    g_k = 250;
    g_cal = 0.05;
    g_kca = 3.5;
    g_l = 6.0;
elseif neuron_code == 6 % population ramp_I
    g_na = 400;
    g_k = 250;
    g_l = 6.0;
else
    g_na = 400;
    g_k = 250;
    g_cal = 0.05;
    g_kca = 6.0;
    g_l = 6.0;
end
% Reversal potentials in mV
E_na = 55;
E_k = -94;
E_ca = 13.27.*log(4./state_vars(:, end));

% Compute new membrane potentials
g_total = g_na.*state_vars(:,2).^3.*state_vars(:,3) + ...
        g_nap.*state_vars(:,4).*state_vars(:,5) + ...
        g_k.*state_vars(:,6).^4 + ...
        g_cal.*state_vars(:,7).*state_vars(:,8) + ...
        g_kca.*state_vars(:,9).^2 + ...
        g_l + ...
        g_ext.*sum(g_synExts_elems) + ...
        g_ext_dr*max(drive_weights, 0)'*external_drives + ...
        g_inh.*sum(g_synInhs_elems) + ...
        g_inh_dr*max(-drive_weights, 0)'*external_drives;
act = g_na.*state_vars(:,2).^3.*state_vars(:,3).*E_na + ...
        g_nap.*state_vars(:,4).*state_vars(:,5).*E_na + ...
        g_k.*state_vars(:,6).^4.*E_k + ...
        g_cal.*state_vars(:,7).*state_vars(:,8).*E_ca + ...
        g_kca.*state_vars(:,9).^2.*E_k + ...
        g_l.*leakage_voltages + ...
        (g_ext.*sum(g_synExts_elems) + g_ext_dr*max(drive_weights, 0)'*external_drives).*E_syn_ext + ...
        (g_inh.*sum(g_synInhs_elems) + g_inh_dr*max(-drive_weights, 0)'*external_drives).*E_syn_inh;
z = exp(-sim_time_step.*g_total./C);
new_state_vars(:,1) = state_vars(:,1).*z + (act./g_total).*(1-z);

% Compute new gating variables
params = [nan, nan, nan, nan, 0; ... % This line is a place-holder
          43.8, 6, 14, 0.252, g_na; ...
          67.5, 10.8, 12.8, 8.456, g_na; ...
          47.1, 3.1, 6.2, 1, g_nap; ...
          60, 9, 9, 5000, g_nap;...
          nan, nan, nan, nan, g_k; ...
          27.4, 5.7, nan, 0.5, g_cal;...
          52.4, 5.2, nan, 18, g_cal; ...
          nan, nan, nan, nan, g_kca];
    for i=2:(size(state_vars, 2)-1)
        if params(i, end) == 0 
            continue;
        end
        if i == 3 || i==5 || i == 8 % For the h's, we need to invert the sign in the formula for their equilibrium values
            is_close_gate = true;
        else 
            is_close_gate = false;
        end
        if i==6 % potassium rectifier
            close_to_open_rate = 0.01.*(state_vars(:,1)+44)./(1-exp(-(state_vars(:,1)+44)./5));
            open_to_close_rate = 0.17.*exp(-(state_vars(:,1)+49)./40);
            equi_state = close_to_open_rate./(close_to_open_rate + open_to_close_rate);
            time_constant = 1./(close_to_open_rate + open_to_close_rate);
        elseif i==9 % calcium-dependent potassium
            close_to_open_rate = 1.25*10^8.*(state_vars(:,end).^2); 
            open_to_close_rate = 2.5;
            equi_state = close_to_open_rate./(close_to_open_rate + open_to_close_rate);
            if neuron_code==1
                kca_time_constant_scale = 8.0;
            elseif neuron_code==2
                kca_time_constant_scale = 6.0;
            elseif neuron_code==3
                kca_time_constant_scale = 0.1;
            elseif neuron_code==5 
                kca_time_constant_scale = 8.0;
            elseif neuron_code==7
                kca_time_constant_scale = 2.0; 
            else 
                kca_time_constant_scale = nan;
            end
            time_constant = kca_time_constant_scale.*1000./(close_to_open_rate + open_to_close_rate);
        else
            equi_state = 1./(1+exp(((-1)^(1+is_close_gate)).*(state_vars(:,1)+params(i,1))./params(i,2)));
            if ~isnan(params(i,3))
                time_constant = params(i,4)./cosh((state_vars(:,1)+params(i,1))./params(i,3));
            else
                time_constant = repmat(params(i,4), num_of_neurons, 1);
            end
        end

        z = exp(-sim_time_step./time_constant);
        new_state_vars(:,i) = state_vars(:,i).*z + equi_state.*(1-z);
    end
    
% Intracellular calcium concentration
buffering_prob = 0.03./(state_vars(:, end) + 0.03 + 0.001);
equi_state = 500.*(- 2*10^(-5).*(1-buffering_prob)...
    .*g_cal.*state_vars(:,7).*state_vars(:,8).*(state_vars(:,1)-E_ca)) ...
    + 5*10^(-5);
z = exp(-sim_time_step/500);
new_state_vars(:, end) = state_vars(:, end).*z + equi_state.*(1-z);
new_state_vars = reshape(new_state_vars, [], 1);
end


