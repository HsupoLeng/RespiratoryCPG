function plot_pop_avg_activity(spike_times_all_neuron_history_cell, neuron_pops)
    neuron_codes = [2,1,5,4,6];
    x_ax_lim_for_all = 0;
    for i=1:length(neuron_codes)
        subplot(length(neuron_codes), 1, i);
        x_ax_lim = convert_spike_time_history_to_avg_rate(spike_times_all_neuron_history_cell(neuron_codes(i), :));
        x_ax_lim_for_all = max(x_ax_lim, x_ax_lim_for_all);
        ylabel(strrep(neuron_pops(neuron_codes(i)).name, '_', '-'));
        set(get(gca,'YLabel'),'Rotation',0, 'HorizontalAlignment', 'right');
        set(gca, 'xtick', []);
        set(gca, 'xticklabel', []);
        set(gca, 'ytick', []);
        set(gca, 'yticklabel', []);
        ylim([0, 100]);
        box off;
    end
    for i=1:length(neuron_codes)
        subplot(length(neuron_codes), 1, i);
        xlim([0, x_ax_lim_for_all]);
    end
end