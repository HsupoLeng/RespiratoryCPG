function delta_state_vars = sim_unifm_HH_pop(t, state_vars, spikes, synaptic_weights, external_drives, drive_weights, leakage_voltages, neuron_code, num_of_neurons)
% HHModel(t, y, I_in) computes the infinitesimal update to the state
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
%       neuron_type: string, neuron type to determine maximal conductance
%   Output:
%       delta_state_vars: infinitesimal change in the state variables


% Constant Parameters
C  = 36;    % capacitance (in pF)

state_vars = reshape(state_vars, num_of_neurons, []);
[I_ionic, delta_state_vars] = compute_chan_I_state_delta(state_vars, leakage_voltages, neuron_code);

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
I_synInhs_elems = exp(-(t - all_spikes_arr)./time_constant_syn_inh).*synaptic_weights_inh;
I_synExt = (g_ext.*sum(g_synExts_elems) + g_ext_dr*max(drive_weights, 0)'*external_drives).*(state_vars(:,1)-E_syn_ext);
I_synInh = (g_inh.*sum(I_synInhs_elems) + g_inh_dr*max(-drive_weights, 0)'*external_drives).*(state_vars(:,1)-E_syn_inh);

delta_state_vars(:,1) = -(I_ionic + I_synExt + I_synInh)./C;
delta_state_vars = reshape(delta_state_vars, [], 1);
end


