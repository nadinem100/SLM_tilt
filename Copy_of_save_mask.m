function Copy_of_save_mask(phase_mask, filename, x_pixels, y_pixels, scal)
    path = '/Users/nadinemeister/Library/CloudStorage/Box-Box/EndresLab/z_Second Experiment/Code/SLM simulation/nadine/DMD_SLM/phase_masks';
    full_filename = fullfile(path, sprintf('%s_noblaze.bmp', filename));
    imwrite(phase_mask / (2 * pi), full_filename);
    
    fprintf('Phase mask saved to: %s\n', full_filename);

    %% GS parameters
    blazedgrating_freq = [1/7, 0]; % [x,y]
    
    %% Add Blazed Grating
    xx = 1: x_pixels;% * scal;
    yy = 1: y_pixels;% * scal;
    x_size_mm = 15.56;
    y_size_mm = 9.22;

    grating_x = 2 * pi * xx * blazedgrating_freq(1);
    grating_y = 2 * pi * yy * blazedgrating_freq(2);
    grating = grating_x + grating_y.';
    grating = mod(grating, 2 * pi);
    
    full_blaze_filename = fullfile(path, sprintf('%s_blazepd%d.bmp', filename, 1 / blazedgrating_freq(1)));
    imwrite(mod((phase_mask + grating), 2 * pi) / (2 * pi), full_blaze_filename);

    fprintf('Phase mask saved to: %s\n', full_blaze_filename);
end
