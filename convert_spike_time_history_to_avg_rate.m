function convert_spike_time_history_to_avg_rate(spike_times_all_neuron_one_pop)
    all_spikes = horzcat(spike_times_all_neuron_one_pop{:});
    T = max(all_spikes+1000); %Time unit in ms;
    bin_size = 30;
    num_of_neurons = length(spike_times_all_neuron_one_pop);
    time_seq = 0:bin_size:T;
    avg_rate_hist = arrayfun(@(timestamp) sum(bitand(all_spikes<timestamp, all_spikes>timestamp-bin_size))...
        .*1000./(bin_size*num_of_neurons), time_seq(2:end));
    plot(time_seq, [0, avg_rate_hist]);
end