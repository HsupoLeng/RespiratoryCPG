clc;

% Experiment parameters. Time are all in ms
dt = 30;
T = 4*10^3;
sim_time_segment_seq = 0:dt:T;
N = 50; % Number of neurons per population
M = 10; % Number of state variables per neuron
warning('off', 'signal:findpeaks:largeMinPeakHeight');

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
state_vars_all_pop(:, :, 1) = state_vars_all_pop(:, :, 1) - 60;
state_vars_all_pop(:, :, [2,4,6,7,9]) = state_vars_all_pop(:, :, [2,4,6,7,9]) .* 0.3;
state_vars_all_pop(:, :, [3, 8]) = max(state_vars_all_pop(:, :, [3, 8]), 0.4);
state_vars_all_pop(:, :, 5) = min(state_vars_all_pop(:, :, 5), 0.1);
state_vars_all_pop(:, :, end) = state_vars_all_pop(:, :, end) .* 10^(-5);
state_vars_all_pop_cell = num2cell(state_vars_all_pop, [2,3]);
state_vars_all_pop_cell = cellfun(@squeeze, state_vars_all_pop_cell, ...
    'UniformOutput', false);

spike_times_all_pop_cell = cell(length(neuron_pops), 1);
spike_times_all_neuron_history = cell(length(sim_time_segment_seq), length(neuron_pops), N);
activity_avg_rate_all_pop = zeros(length(sim_time_segment_seq), length(neuron_pops));

% Synaptic weights correspond to the sequence in neuron_pops definition
neuron_pops(1).synaptic_weights = [0, -0.32, 0, 0, -0.115, 0, 0];
neuron_pops(2).synaptic_weights = [-0.01, 0, 0, 0, -0.04, 0, 0];
neuron_pops(3).synaptic_weights = [-0.15, 0, 0, 0, -0.2, 0, 0];
neuron_pops(4).synaptic_weights = [-0.025, -0.225, 0, 0.03, 0, 0, 0];
neuron_pops(5).synaptic_weights = [-0.145, -0.4, 0, 0.034, 0, 0, 0];
neuron_pops(6).synaptic_weights = [-2, -1, 0, 0.06, 0, 0, -0.275];
neuron_pops(7).synaptic_weights = [-0.25, -1, 0, 0, 0, 0, 0];
synaptic_weights_all_pop_cell = {neuron_pops(:).synaptic_weights}';

% External drives and their weights are in the sequence of:
% pons, RTN_to_BotC, pre_BotC
for i=1:length(neuron_pops)
    neuron_pops(i).external_drives = 2.*ones(3, 1);
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

sim_time_step = 0.01; % Time step 0.01ms
for segment_ind=1:length(sim_time_segment_seq)
    segment_start_t = sim_time_segment_seq(segment_ind);
    ts = segment_start_t:sim_time_step:segment_start_t+dt;
    ts = ts(1:end-1);
    voltage_all_pop_snapshot = zeros(length(ts), length(neuron_pops), N);
    tic;
    for i=1:length(ts)
        t = ts(i);
        for j=1:length(neuron_pops)
            delta_state_vars = sim_unifm_HH_pop(t, state_vars_all_pop_cell{j}, spike_times_all_pop_cell, ...
                synaptic_weights_all_pop_cell{j}, external_drives_all_pop_cell{j}, ...
                drive_weights_all_pop_cell{j}, neuron_codes{j}, N);
 
            state_vars_all_pop_cell{j} = state_vars_all_pop_cell{j} ...
                + delta_state_vars.*sim_time_step;
            
            state_vars_all_pop_cell{j}(:, 4) = min(state_vars_all_pop_cell{j}(:, 4), 1);

            voltage_all_pop_snapshot(i, j, :) = state_vars_all_pop_cell{j}(:, 1); 
        end
        
        %Alternative cell array implementation: (not much acceleration...
        %{     
        sim_unifm_HH_pop_wrapper = @(state_vars, synaptic_weights, external_drives, ...
            drive_weights, neuron_code, num_neuron) ...
            sim_unifm_HH_pop(t, state_vars, spike_times_all_pop_cell, synaptic_weights, ...
            external_drives, drive_weights, neuron_code, num_neuron);
        delta_state_vars_cell = cellfun(sim_unifm_HH_pop_wrapper, ...
            state_vars_all_pop_cell, synaptic_weights_all_pop_cell, ...
            external_drives_all_pop_cell, drive_weights_all_pop_cell, ...
            neuron_codes, num2cell(N.*ones(size(neuron_pops)))', ...
            'UniformOutput', false);
        delta_state_vars_cell = cellfun(@times, delta_state_vars_cell, ...
            num2cell(sim_time_step.*ones(size(delta_state_vars_cell))), ...
            'UniformOutput', false);
        state_vars_all_pop_cell = cellfun(@plus, state_vars_all_pop_cell, ...
            delta_state_vars_cell, 'UniformOutput', false);
        for j=1:length(state_vars_all_pop_cell)
            voltage_all_pop_snapshot(i, j, :) = state_vars_all_pop_cell{j}(:, 1);
        end
        %}
    end
    
    voltage_all_pop_snapshot_cell = num2cell(voltage_all_pop_snapshot, 1);
    [~, spike_time_inds_cell] = cellfun(@(v) findpeaks(v, 'minpeakheight', -30), voltage_all_pop_snapshot_cell, 'UniformOutput', false);
    spike_times_all_neuron_cell = cellfun(@(t_inds) ts(t_inds), spike_time_inds_cell, 'UniformOutput', false);
    spike_times_all_neuron_cell = squeeze(spike_times_all_neuron_cell);
    for i=1:size(spike_times_all_neuron_cell, 1)
        spike_times_all_pop_cell{i} = cell2mat(spike_times_all_neuron_cell(i, :));
    end
    spike_times_all_neuron_history(segment_ind, :, :) = spike_times_all_neuron_cell;
    activity_avg_rate_all_pop(segment_ind, :) = cellfun(@length, spike_times_all_pop_cell).*(1000/dt)./N;
    fprintf('Iteration %d: ', segment_ind);toc;
    % Example: visualize all neurons' acitivities in pre_I population
    %{
    figure(1);
    clf; 
    hold on;
    for i=1:size(voltage_all_pop_snapshot, 3)
        plot(voltage_all_pop_snapshot(:, 4, i));
    end
    hold off;
    %}
end