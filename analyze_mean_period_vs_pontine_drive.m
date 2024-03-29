avg_periods = zeros(6, 1);
load('all_pop_40s_1ms_findpeak_drive_0_1dot7_1dot3.mat');
[~, avg_rate] = convert_spike_time_history_to_avg_rate(spike_times_all_neuron_history_cell(4, :));
[~, period_starts] = findpeaks(avg_rate, 'MinPeakHeight', 45, 'MinPeakDistance', 20);
periods = diff(period_starts);
avg_periods(1) = mean(periods) * 30/1000;
load('spike_times_40s_1ms_findpeak_drive_0dot3_1dot7_1dot3.mat');
[~, avg_rate] = convert_spike_time_history_to_avg_rate(spike_times_all_neuron_history_cell(4, :));
[~, period_starts] = findpeaks(avg_rate, 'MinPeakHeight', 60, 'MinPeakDistance', 20);
periods = diff(period_starts);
avg_periods(2) = mean(periods) * 30/1000;
load('spike_times_40s_1ms_findpeak_drive_0dot5_1dot7_1dot3.mat');
[~, avg_rate] = convert_spike_time_history_to_avg_rate(spike_times_all_neuron_history_cell(4, :));
[~, period_starts] = findpeaks(avg_rate, 'MinPeakHeight', 60, 'MinPeakDistance', 20);
periods = diff(period_starts);
avg_periods(3) = mean(periods) * 30/1000;
load('spike_times_40s_1ms_findpeak_drive_0dot7_1dot7_1dot3.mat');
[~, avg_rate] = convert_spike_time_history_to_avg_rate(spike_times_all_neuron_history_cell(4, :));
[~, period_starts] = findpeaks(avg_rate, 'MinPeakHeight', 60, 'MinPeakDistance', 20);
periods = diff(period_starts);
avg_periods(4)= mean(periods) * 30/1000;
load('all_pop_40s_1ms_findpeak_drive_1_1dot7_1dot3_sustained_activity.mat');
[~, avg_rate] = convert_spike_time_history_to_avg_rate(spike_times_all_neuron_history_cell(4, :));
[~, period_starts] = findpeaks(avg_rate, 'MinPeakHeight', 70, 'MinPeakDistance', 20);
periods = diff(period_starts);
avg_periods(5)= mean(periods) * 30/1000;
load('spike_times_40s_1ms_findpeak_drive_1dot1_1dot7_1dot3.mat');
[~, avg_rate] = convert_spike_time_history_to_avg_rate(spike_times_all_neuron_history_cell(4, :));
[~, period_starts] = findpeaks(avg_rate, 'MinPeakHeight', 70, 'MinPeakDistance', 20);
periods = diff(period_starts);
avg_periods(6) = mean(periods) * 30/1000;