function phase_mask = modularized_WGS(filename)
    params = initialize_parameters();
    [A_in, pad, X, Y, X0, Y0, x, y] = setup_grids(params);
    [spacing, A_single] = calculate_spacing(A_in, params);
    [A_target, coordinates, onesmatrix, height_corr2] = create_full_target_array(A_in, params.n_horiz, params.n_vert, spacing, params, A_single);
    phase_mask = gerchberg_saxton(A_in, A_target, pad, params);
    %phase_mask = gerchberg_saxton_z(A_in, x, y, A_target, pad, params, coordinates, onesmatrix, height_corr2);
    plot_results(A_in, phase_mask, params.x_pixels, params.y_pixels, A_single);
    %save_mask(phase_mask, filename, params.x_pixels, params.y_pixels, params.scal);
end

function params = initialize_parameters()
    params.n_horiz = 100; params.n_vert = 100;
    params.x_pixels = 4000; params.y_pixels = 2464;
    params.pixel_um = 3.74; params.waist_um = 9/2*1e3;
    params.spacing_factor = 4/(2*0.77);
    params.iterations = 50; params.useFilter = 0; params.useGPU = 0;
    params.scal = 4;
    params.box1=1;
end

function [A_in, pad, X, Y, X0, Y0, x, y] = setup_grids(params)
    % Set up the required coordinate grids
    x0 = 1:params.x_pixels;
    y0 = 1:params.y_pixels;
    x = 1:params.scal*params.x_pixels;
    y = 1:params.scal*params.y_pixels;
    [X0, Y0] = meshgrid(x0, y0);
    [X, Y] = meshgrid(x, y);
    
    % Input beam waist setup
    waist_in = params.waist_um / params.pixel_um;
    A_in = zeros(length(y), length(x));
    A_in((params.scal-1)*params.y_pixels/2+1:(params.scal+1)*params.y_pixels/2, ...
        (params.scal-1)*params.x_pixels/2+1:(params.scal+1)*params.x_pixels/2) = ...
        exp(-((X0-ceil(params.x_pixels/2)).^2 + (Y0-ceil(params.y_pixels/2)).^2) / (waist_in^2));
    
    pad = abs(X - params.scal * params.x_pixels / 2) < params.x_pixels / 2 & ...
          abs(Y - params.scal * params.y_pixels / 2) < params.y_pixels / 2;
end

function [spacing, A_single] = calculate_spacing(A_in, params)
    A_single = fftshift(fft2(ifftshift(A_in)));
    [max_val, max_idx] = max(abs(A_single(:)).^2);
    stop_idx = max_idx;
    val = max_val;
    
    while val >= exp(-2) * max_val
        stop_idx = stop_idx + 1;
        val = abs(A_single(stop_idx)).^2;
    end
    
    spacing = ceil(params.spacing_factor * 2 * (stop_idx - max_idx));
end


function [A_target, coordinates, onesmatrix, height_corr2] = create_full_target_array(A_in, n_horiz, n_vert, spacing, params, A_single)
    % Generate the target array for GS algorithm
    A_target = zeros(size(A_in));
    hh = [1:n_horiz]';
    vv = [1:n_vert]';
    spacing_h = spacing; 
    spacing_v = spacing*params.y_pixels/params.x_pixels;
    spacing_global = 1/2* (n_horiz+2);
    h_offset = round(spacing*spacing_global);
    v_offset = round(-spacing/2);
    [~,center_idx] = max(abs(A_single(:)));
    [center_row,center_col] = ind2sub(size(A_single),center_idx);

    h_offset_curr = round(spacing_h*(hh-n_horiz/2));
    v_offset_curr = round(spacing_v*(vv-n_vert/2));
    target_rows = repmat(center_row + v_offset + v_offset_curr, n_horiz, 1);
    target_cols = repelem(center_col + h_offset + h_offset_curr, n_vert);
    
    target_idx = sub2ind(size(A_target),target_rows,target_cols); % Indices of tweezer centers
    height_corr = ones(1,n_horiz*n_vert);
    height_corr = reshape(height_corr,[n_horiz*n_vert,1]);
    A_target(target_idx) = sqrt(height_corr.^(-1)); %For WGS feedback

    % part 2 added for coordinates
    tweezlist = [];
    coordinates = [];
    tweezer_coor = zeros((2*params.box1+1)^2);   
    tweezer_idx = 1:n_horiz*n_vert;

    for idx = tweezer_idx
        v_coor = [target_rows(idx)-params.box1:target_rows(idx)+params.box1];
        h_coor = [target_cols(idx)-params.box1:target_cols(idx)+params.box1];            
    
        refrow = target_rows(idx);
        refcol = target_cols(idx);
                
        if A_target(refrow,refcol) ~= 0                        
            tweezlist = cat(1, tweezlist, [target_rows(idx),target_cols(idx)] ); %switched cols rows
        
            tweezer_rows = repmat(v_coor', 2*params.box1+1, 1);
            tweezer_cols = repelem(h_coor', 2*params.box1+1);
            tweezer_coor(:,idx) = sub2ind(size(A_target),tweezer_rows,tweezer_cols);    
            coordinates = cat(1, coordinates, tweezer_coor(:,idx));    
        
        end
    end

    %part 3 added for onesmatrix
    height_corr2 = repelem( height_corr, (2*params.box1+1)^2);
    onesmatrix = ones(size(height_corr2));


end


function phase_mask = gerchberg_saxton(A_in, A_target, pad, params)
    % Perform GS algorithm for phase retrieval
    psi = 2 * pi * rand(size(A_in));
    for ii = 1:params.iterations
        A_mod = A_in .* exp(1i * psi);
        A_out = fftshift(fft2(ifftshift(A_mod)));
        psi_out = angle(A_out);
        A_iterated = A_target .* exp(1i * psi_out);
        A_new = fftshift(ifft2(ifftshift(A_iterated)));
        psi = mod(angle(A_new), 2 * pi) .* pad;
    end
    phase_mask = psi;
end

function phase_mask = gerchberg_saxton_z(A_in, x, y, A_target, pad, params, coordinates, onesmatrix, height_corr2)
    psi = zeros(length(y),length(x));
    psi0 = 2*pi*rand(params.y_pixels,params.x_pixels);
    psi((params.scal-1)*params.y_pixels/2+1:(params.scal+1)*params.y_pixels/2, (params.scal-1)*params.x_pixels/2+1:(params.scal+1)*params.x_pixels/2) = psi0;

    N_cutoff = ceil(params.iterations)/2; %idk about this one
    k = 2*pi/(689*10^(-9));

    % generating the new plane we are trying to create tweezers on
    dx=params.pixel_um; % dx = 0.2;
    dy=dx; % dy = 0.2;
    x_len = params.x_pixels * params.pixel_um;
    y_len = params.y_pixels * params.pixel_um;
    xpp = (-x_len/2*params.scal:dx:x_len/2*params.scal-dx); %-30:dx:30;
    ypp =  (-y_len/2*params.scal:dy:(y_len)/2*params.scal-dy); %30:dy:30;
    [XX,YY] = meshgrid(xpp,ypp);
    zss = linspace(-1,1,size(XX,2));
    zss = linspace(0.00001,0.00001,size(XX,2));
    zzz = 3E-3*repmat(zss,size(XX,1),1);
    f = 200E-3;

    weight = 1;
    weight1back = 1;
    g = ones(length(y),length(x));
    B_target = abs(A_target);
    Gg = 0.6;
    
    for ii = 1:params.iterations 
    %while std(weight) >0.001
        A_mod = A_in.*exp(1i*psi);
        A_out = fftshift(fft2(ifftshift(A_mod)));
        if ii <= N_cutoff %|| (std(weight) == weight2back == weight1back && std(weight)>0.000001 && N_cutoff>0)
            %if ii<= iterations*85/100
                psi_out = mod(angle(A_out),2*pi);
            %end
        end
        
        B_out = abs(fftshift(fft2(A_mod.*exp((-1i*k*zzz.*(XX.^2+YY.^2)./(2*f.*(f-zzz))))))) ;
        
        new_g = zeros(length(y),length(x));
        
        if ii > 1        
            B_out(coordinates) = B_out(coordinates)./(onesmatrix - Gg.*(onesmatrix - sqrt(height_corr2))); % B_out(coordinates)./sqrt(height_corr2)
            
            % if useFilter
            %     B_out = B_out./(filter_2d); %.* took out sqrt
            % end
        end
        
        B_mean = mean(mean(B_out(coordinates))); % Mean of the amplitudes at target tweezer centers
        B_out0 = B_out(coordinates);
    
        B_out0 = reshape(B_out0, (2*params.box1+1)^2, length(coordinates)/((2*params.box1+1)^2) ); % n_horiz*n_vert 
        B_out_box = mean(B_out0,1);
    
        weight2back = weight1back;
        weight1back = std(weight);
        weight = B_mean./B_out_box; %HOW STRONG/WEAK A TWEEZEER IS COMPARED TO THE MEAN/IDEAL
    
        new_g(coordinates) = repelem(weight', (2*params.box1+1)^2).*g(coordinates);
    
        % Weight factor, rescale the amplitudes according to tweezer centers
       fprintf('error signal = %0.5f\n',std(weight));
    %    errorsig14400_hmupd(times,ii)= std(weight);
    
        g = new_g;
        A_iterated = fftshift(ifft2(ifftshift( (fft2(fftshift(g.*B_target )).*exp((1i*k.*zzz.*(XX.^2+YY.^2)./(2*f.*(f-zzz)))) )  ))).*exp(1i*psi_out); % Only the center pixels of each tweezer are left
        A_new = fftshift(ifft2(ifftshift(A_iterated) ));
        psi = mod(angle(A_new),2*pi);
        psi = psi.*pad;
    end
    toc
    
    phase_mask = psi((params.scal-1)*params.y_pixels/2+1:(params.scal+1)*params.y_pixels/2, (params.scal-1)*params.x_pixels/2+1:(params.scal+1)*params.x_pixels/2);
    phase_mask = mod(phase_mask, 2*pi);
end

function save_mask(phase_mask, filename, x_pixels, y_pixels, scal)
    path = 'C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\phase_masks';
    full_filename = fullfile(path, sprintf('%s_noblaze.bmp', filename));
    imwrite(phase_mask / (2 * pi), full_filename);
    
    fprintf('Phase mask saved to: %s\n', full_filename);

    %% GS parameters
    blazedgrating_freq = [1/14,0]; % [x,y]
    %% Add Blazed grating
    xx= 1: x_pixels*scal;
    yy= 1: y_pixels*scal;
    x_size_mm = 15.56;
    y_size_mm = 9.22;

    grating_x = 2*pi*xx*blazedgrating_freq(1);
    grating_y = 2*pi*yy*blazedgrating_freq(2);

    grating = grating_x + grating_y.'; 
    grating = mod(grating, 2*pi);
    
    full_blaze_filename = fullfile(path, sprintf('%s_blazepd%d.bmp', filename, 1/blazedgrating_freq(1)));
    imwrite(mod((phase_mask+grating), 2*pi)/(2*pi), full_blaze_filename)

    fprintf('Phase mask saved to: %s\n', full_blaze_filename);

end

function plot_results(A_in, psi, x_pixels, y_pixels, A_single)
    A_mod = A_in.*exp(1i*mod(psi,2*pi));
    A_out = fftshift(fft2((A_mod)));

    figure(1);
    subplot(1,2,1);
    imagesc(abs(A_in).^2);
    pbaspect([x_pixels y_pixels 1])
    title('I_{in}');

    subplot(1,2,2);
    imagesc(psi);
    pbaspect([x_pixels y_pixels 1])
    title('Phase');
    % colorbar;

    figure(2);
    subplot(1,2,1);
    imagesc(abs(A_single).^2);
    pbaspect([1 1 1])
    title('I_{target}');

    subplot(1,2,2);
    imagesc(abs(A_out).^2);
    pbaspect([1 1 1])
    title('I_{out}');
    
    figure(3);
        subplot(1,2,1);
    imagesc(abs(A_out).^2);
    pbaspect([1 1 1])
    title('I_{target}');

end

