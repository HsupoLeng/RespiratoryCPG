clc;

% Experiment parameters. Time are all in ms
sim_time_step = 0.1; % Time step 0.01ms
T = 30*10^3;
sim_time_seq = 0:sim_time_step:T;
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

% Initialize state variables
state_vars_all_pop = rand(length(neuron_pops), N, M);
state_vars_all_pop([1:3,5:7], :, 1) = state_vars_all_pop([1:3, 5:7], :, 1) - 70;
state_vars_all_pop(4, :, 1) = state_vars_all_pop(4, :, 1) - 50;
state_vars_all_pop([1:3,5:7], :, [2,4,6,7,9]) = min(state_vars_all_pop([1:3,5:7], :, [2,4,6,7,9]), 0.1);
state_vars_all_pop(4, :, [2,4,6,7,9]) = min(state_vars_all_pop(4, :, [2,4,6,7,9]), 0.3);
state_vars_all_pop([1:3, 5:7], :, [3,5,8]) = min(state_vars_all_pop([1:3,5:7], :, [3,5,8]), 0.3);
state_vars_all_pop(4, :, [3,5,8]) = max(state_vars_all_pop(4, :, [3,5,8]), 0.4);
state_vars_all_pop(:, :, end) = state_vars_all_pop(:, :, end) .* 10^(-5);
state_vars_all_pop_cell = num2cell(state_vars_all_pop, [2,3]);
state_vars_all_pop_cell = cellfun(@squeeze, state_vars_all_pop_cell, ...
    'UniformOutput', false);

spike_times_all_pop_cell = cell(length(neuron_pops), 1);
spike_times_all_neuron_history_cell = cell(length(neuron_pops), N);
spike_trains_all_neuron_history = zeros(length(sim_time_seq), length(neuron_pops), N);
%activity_avg_rate_all_pop = zeros(length(sim_time_segment_seq), length(neuron_pops));

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

% Leakage reversal potentials
for i=1:length(neuron_pops)
    rng(neuron_pops(i).code);
    if i==4
        E_l = -68 + 1.36.*randn(N, 1);
    else
        E_l = -60 + 1.2.*randn(N, 1);
    end
    neuron_pops(i).leakage_voltages = E_l;
end
leakage_voltages_all_pop_cell = {neuron_pops(:).leakage_voltages}';
pool = gcp('nocreate');
delete(pool);
parpool('local');
for i=1:length(sim_time_seq)
    t = sim_time_seq(i);

    parfor j=1:length(neuron_pops)        
        [ts, state_vars_at_ts] = ode15s(@(t, state_vars) sim_unifm_HH_pop(t, state_vars, spike_times_all_pop_cell, ...
            synaptic_weights_all_pop_cell{j}, external_drives_all_pop_cell{j}, ...
            drive_weights_all_pop_cell{j}, leakage_voltages_all_pop_cell{j}, ...
            neuron_codes{j}, N), t:0.01:t+sim_time_step, ...
            state_vars_all_pop_cell{j});
        state_vars_all_pop_cell{j} = reshape(state_vars_at_ts(end, :), N, []);
        %{
        delta_state_vars = sim_unifm_HH_pop(t, state_vars_all_pop_cell{j}, spike_times_all_pop_cell, ...
            synaptic_weights_all_pop_cell{j}, external_drives_all_pop_cell{j}, ...
            drive_weights_all_pop_cell{j}, leakage_voltages_all_pop_cell{j}, ...
            neuron_codes{j}, N)
        delta_state_vars = reshape(delta_state_vars, N, []);
        state_vars_all_pop_cell{j} = state_vars_all_pop_cell{j} + ...
            delta_state_vars.*sim_time_step;
        %}
        %state_vars_all_pop_cell{j}(:, 4) = max(min(state_vars_all_pop_cell{j}(:, 4), 1), 0);
        %state_vars_all_pop_cell{j}(:, 5) = max(min(state_vars_all_pop_cell{j}(:, 5), 1), 0);
        
        spike_trains_all_neuron_history(i, j, :) = state_vars_all_pop_cell{j}(:, 1); 
    end
    
    % Find new spike every 2ms. Find the spike in a 30ms window
    if mod(i, 2/sim_time_step)
        continue;
    end
    
    spike_search_window = max(1, i-(30/sim_time_step)):i;
    if length(spike_search_window) < 3
        spike_times_new_all_neuron_cell = cell(size(spike_times_all_neuron_history_cell));
    else
        spike_trains_all_neuron_cell = squeeze(num2cell(spike_trains_all_neuron_history(spike_search_window, :, :), 1));
        [~, spike_time_inds_cell] = cellfun(@(v) findpeaks(v, 'MinPeakHeight', -20, 'MinPeakProminence', 30), spike_trains_all_neuron_cell, 'UniformOutput', false);
        spike_time_inds_cell = squeeze(spike_time_inds_cell);
        spike_times_all_neuron_cell = cellfun(@(t_inds) sim_time_seq(min(spike_search_window)-1 + t_inds), spike_time_inds_cell, 'UniformOutput', false);
        spike_times_new_all_neuron_cell = cellfun(@(spike_times, spike_times_history) setdiff(spike_times, spike_times_history), ...
            spike_times_all_neuron_cell, spike_times_all_neuron_history_cell, 'UniformOutput', false);
    end
    spike_times_all_neuron_history_cell = cellfun(@(spike_times_history, new_spike_times) horzcat(spike_times_history, new_spike_times), ...
        spike_times_all_neuron_history_cell, spike_times_new_all_neuron_cell, 'UniformOutput', false);
    for k=1:size(spike_times_all_neuron_history_cell, 1)
        spike_times_all_pop_cell{k} = cell2mat(spike_times_all_neuron_history_cell(k, :));
    end
   
    % Example: visualize all neurons' acitivities in pre_I population
    %{
    if ~mod(i, 50/sim_time_step) 
        figure(1);
        clf; 
        hold on;
        for k=1:size(spike_trains_all_neuron_history, 3)
            plot(spike_trains_all_neuron_history(i-(50/sim_time_step)+1:i, 4, k));
        end
        hold off;
        fprintf('Now at %f second\n', t/1000);
    end
    %}
end
toc;
figure(1);
hold on;
for i=1:length(neuron_pops)
    plot_neuron_ind = 1;
    subplot(length(neuron_pops), 1, i);
    spike_times = spike_times_all_neuron_history_cell{i,plot_neuron_ind};
    plot([spike_times;spike_times], [ones(size(spike_times));zeros(size(spike_times))], 'k-');
    ylim([-1, 2]);
    set(gca,'TickDir','out') % draw the tick marks on the outside
    set(gca,'YTick', []) ;
    set(gca,'YColor',get(gcf,'Color'));
    title(strrep(neuron_pops(i).name, '_', '-'));
end
hold off;

