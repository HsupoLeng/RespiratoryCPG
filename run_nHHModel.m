clc;

% Experiment parameters. Time are all in second
dt = 0.1;
T = 60;
N = 50; % Number of neurons per population
M = 10; % Number of state variables per neuron

% Structure of the breathing CPG
neuron_pops = struct();
neuron_pops(1).name = 'aug_E'; neuron_pops(1).code = 1;
neuron_pops(2).name = 'post_I'; neuron_pops(2).code = 2;
neuron_pops(3).name = 'post_I_e'; neuron_pops(3).code = 3;
neuron_pops(4).name = 'pre_I'; neuron_pops(4).code = 4;
neuron_pops(5).name = 'early_I_1'; neuron_pops(5).code = 5;
neuron_pops(6).name = 'ramp_I'; neuron_pops(6).code = 6;
neuron_pops(7).name = 'early_I_2'; neuron_pops(7).code = 7;
neuron_codes = {neuron_pops(:).code}';

state_vars_all_pop = rand(length(neuron_pops), N, M);
state_vars_all_pop_cell = num2cell(state_vars_all_pop, [2,3]);

spikes_all_pop = cell(length(neuron_pops), 1);

% Synaptic weights correspond to the sequence in neuron_pops definition
neuron_pops(1).synaptic_weights = [0, -0.32, 0, 0, -0.115, 0, 0];
neuron_pops(2).synaptic_weights = [-0.01, 0, 0, 0, -0.04, 0, 0];
neuron_pops(3).synaptic_weights = [-0.15, 0, 0, 0, -0.2, 0, 0];
neuron_pops(4).synaptic_weights = [-0.025, -0.225, 0, 0.03, 0, 0, 0];
neuron_pops(5).synaptic_weights = [-0.145, -0.4, 0, 0.034, 0, 0, 0];
neuron_pops(6).synaptic_weights = [-2, -1, 0, 0.06, 0, 0, 0];
neuron_pops(7).synaptic_weights = [-0.25, -1, 0, 0, 0, 0, 0];
synaptic_weights_all_pop_cell = {neuron_pops(:).synaptic_weights}';

% External drives and their weights are in the sequence of:
% pons, RTN_to_BotC, pre_BotC
for i=1:length(neuron_pops)
    neuron_pops(i).external_drives = ones(3, 1);
end
external_drives_all_pop_cell = {neuron_pops(:).external_drives}';

neuron_pops(1).drive_weights = [0.4, 1, 0]';
neuron_pops(2).drive_weights = [1.5, 0.1, 0]';
neuron_pops(3).drive_weights = [1, 0.1, 0]';
neuron_pops(4).drive_weights = [0.55, 0.13, 0.3]';
neuron_pops(5).drive_weights = [1.1, 0.7, 0]';
neuron_pops(6).drive_weights = [2, 0, 0]';
neuron_pops(7).drive_weights = [1.7, 0, 0]';
drive_weights_all_pop_cell = {neuron_pops(:).drive_weights}';

sim_time_step = 0.001;
for t = 0:dt:T
    ts = t:sim_time_step:t+dt;
    voltage_all_pop_snapshot = zeros(length(ts), length(neuron_pops), N);
    for i=1:length(ts)
        for j=1:length(neuron_pops)
            delta_state_vars = sim_unifm_HH_pop(ts(i), state_vars_all_pop_cell{j}, spikes_all_pop, ...
                synaptic_weights_all_pop_cell{j}, external_drives_all_pop_cell{j}, ...
                drive_weights_all_pop_cell{j}, neuron_codes{j}, N);
            if any(isnan(delta_state_vars))
                pause(1);
            end
            delta_state_vars = reshape(delta_state_vars, N, []);
            state_vars_all_pop_cell{j} = squeeze(state_vars_all_pop_cell{j}) ...
                + delta_state_vars.*sim_time_step;
            voltage_all_pop_snapshot(i, j, :) = delta_state_vars(:, 1); 
        end
    end
V = y(:,    1:N );
m = y(:,  N+1:2*N);
n = y(:,2*N+1:3*N);
h = y(:,3*N+1:4*N);
plot(t,V)
loc findpeak(V)

% Save t_peak

end