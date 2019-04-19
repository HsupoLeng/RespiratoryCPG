drive_settings = 0:0.1:1.2;
drive_settings(end) = 1.15;
for i=1:length(drive_settings)
    run_nHHModel(drive_settings(i));
end