%%
global n_horiz n_vert x_pixels y_pixels pixel_um waist_um spacing_factor
global steer_x_factor steer_y_factor iterations height_corr loop
global magnification ROI_size identification_threshold exposureTimeus
global x0 y0 width height


%% Tweezer parameters
n_horiz = 50; n_vert = 50;
x_pixels = 4000;
y_pixels = 2464;
%scaling=2;
waist_um = 9/2*1e3; pixel_um = 3.74; % Physical beam size *10^(-6)
spacing_um = 4;
spacing_factor = 7; %spacing_factor * w0 is spacing between final tweezers
%lambda_um = 1.06;
%focal_len_um= ;
%steer_x_factor = -7; steer_y_factor = 8; %mot steering away from 0 order
%% This first part comes from the calculation in wgs_initial

%numbers for Holoeye GAEA-2.1 SLM (https://holoeye.com/products/spatial-light-modulators/gaea-2-phase-only/)
fill_factor = 0.90; % I suppose this is the fractional area that is effective
reflectivity = 0.60;
spacing_global = 1/2* (n_horiz+2);
%% GS parameters
useFilter = 0;
useGPU = 0;
iterations = 10;
gain = 1.0; 
height_corr = ones(1,n_horiz*n_vert);
pos_x_feedback_master = zeros(1,n_horiz*n_vert);
pos_y_feedback_master = zeros(1,n_horiz*n_vert);
first_run = 1;
position_correction = 0;
%%
scal = 4;
x_pixel_size = scal; % Large pixel size gives finer grid
y_pixel_size = scal;
x0 = 1:x_pixels;
y0 = 1:y_pixels;

x = 1:x_pixel_size*x_pixels; % Actual coordinate of the grid
y = 1:y_pixel_size*y_pixels;
%[X,Y] = meshgrid(x,y);
[X0,Y0] = meshgrid(x0,y0); % SLM grid
[X,Y] = meshgrid(x,y); % WGS grid
R = round(sqrt( (X-length(x)/2).^2 + (Y-length(y)/2).^2 )); % polar

waist_in = waist_um/pixel_um; % Input beam waist determines the size of individual tweezer
A_in = zeros(length(y),length(x));
A_in((scal-1)*y_pixels/2+1:(scal+1)*y_pixels/2, (scal-1)*x_pixels/2+1:(scal+1)*x_pixels/2) = ...
    exp(-( (X0-ceil(x_pixels/2)).^2 + (Y0-ceil(y_pixels/2)).^2)/(waist_in^2)); % Amplitude of E field
pad = abs(X-scal*x_pixels/2) < x_pixels/2 & abs(Y-scal*y_pixels/2) < y_pixels/2;
if useGPU
    A_in = gpuArray(A_in);
    pad = gpuArray(pad);
end

A_single = fftshift(fft2(ifftshift(A_in)));
[max_val,max_idx] = max(abs(A_single(:)).^2);
val = max_val;
stop_idx = max_idx;
while val >= exp(-2)*max_val %1/e^2 radius
   stop_idx = stop_idx + 1;
   val = abs(A_single(stop_idx)).^2;
end


spacing = ceil(spacing_factor*2*(stop_idx - max_idx)); %round(0.75*(2)*ceil(spacing_factor*(stop_idx - max_idx)))
% 2*(stop_idx - max_idx) = beam diameter (in pixels on the atom plane) of a single tweezer

%% Creating target array

spacing_h = spacing; spacing_v = spacing;
spacing_v = spacing*y_pixels/x_pixels;
h_offset = 0; %round(spacing*spacing_global); %maybe this should also be round(-spacing/2)
v_offset = round(-spacing/2);

% find center of whole array by finding the center of the single tweezer
[~,center_idx] = max(abs(A_single(:)));
[center_row,center_col] = ind2sub(size(A_single),center_idx);

A_target = zeros(size(A_single));
if useGPU
A_target = gpuArray(A_target);
end

% define tweezers index vectors
hh = [1:n_horiz]';
vv = [1:n_vert]';
h_offset_curr = round(spacing_h*(hh-n_horiz/2));
v_offset_curr = round(spacing_v*(vv-n_vert/2));
target_rows = repmat(center_row + v_offset + v_offset_curr, n_horiz, 1);
target_cols = repelem(center_col + h_offset + h_offset_curr, n_vert);

target_idx = sub2ind(size(A_target),target_rows,target_cols); % Indices of tweezer centers

height_corr = reshape(height_corr,[n_horiz*n_vert,1]);
A_target(target_idx) = sqrt(height_corr.^(-1)); %For WGS feedback

% box1 = stop_idx - max_idx - 3; % Fine tune this to optimize uniformity,-3
box1 = 1; %Changing this would make the PGS fail
tweezer_idx = 1:n_horiz*n_vert;
coordinates = [];
height_corr2 = repelem(height_corr.^(-1), (2*box1+1)^2);
tweezer_coor = zeros((2*box1+1)^2);
if useGPU
tweezer_coor = gpuArray(tweezer_coor);
end
for idx = tweezer_idx
    v_coor = [target_rows(idx)-box1:target_rows(idx)+box1];
    h_coor = [target_cols(idx)-box1:target_cols(idx)+box1];
    tweezers = A_target(v_coor,h_coor);
    tweezer_rows = repmat(v_coor', 2*box1+1, 1);
    tweezer_cols = repelem(h_coor', 2*box1+1);
    tweezer_coor(:,idx) = sub2ind(size(A_target),tweezer_rows,tweezer_cols);
	coordinates = cat(1, coordinates, tweezer_coor(:,idx));
end

%% GS parameters initialization

if first_run
    psi0 = 2*pi*rand(y_pixels,x_pixels); % Random phase initialization
    N_cutoff = inf; %ceil(iterations)/2; % Fixed phase beyond this number of iteration
else
    psi0 = phase_mask_master; %Start with the same phase as the run before
    N_cutoff = inf; % 0; % Fixed phase beyond this number of iteration
end
g = ones(length(y),length(x)); % Weight factor used in the GS feedback loop
if useGPU
    g = gpuArray(g);
    A_in = gpuArray(A_in);
    A_target = gpuArray(A_target);
end
B_target = abs(A_target);
n_tweezers = n_vert*n_horiz;

psi = zeros(length(y),length(x));
if useGPU
psi = gpuArray(psi);

end
psi((scal-1)*y_pixels/2+1:(scal+1)*y_pixels/2, (scal-1)*x_pixels/2+1:(scal+1)*x_pixels/2) = psi0;


% %% Initial GS 
% 
% tic
% for ii = 1:iterations 
% 
%     A_mod = A_in.*exp(1i*psi);
%     A_out = fftshift(fft2(ifftshift(A_mod)));
% 
%     if ii <= N_cutoff
%         psi_out = mod(angle(A_out),2*pi);
%     end
% 
%     B_out = abs(A_out);
% 
% %     if useFilter
% %         B_out = B_out.*sqrt(filter_2d);
% %     end
%     if useFilter
%         %SLM_bandwidth;
%         filter_2d = Efficiency_map(X,Y,0.95); %0.95 is 0th order efficiency
%         if useGPU
%             filter_2d = gpuArray(filter_2d);
%         end
%     end
%     % XLv
% 
%     new_g = zeros(length(y),length(x));
%     if useGPU
%     new_g = gpuArray(new_g);
%     end
% 
%     if ii > 1        
%         B_out(coordinates) = B_out(coordinates)./sqrt(height_corr2);
%         if useFilter
%             B_out = B_out.*sqrt(filter_2d);
%         end
%     end
% 
%     B_mean = mean(mean(B_out(coordinates))); % Mean of the amplitudes at target tweezer centers
%     B_out0 = B_out(coordinates);
% 
%     B_out0 = reshape(B_out0, (2*box1+1)^2, n_vert*n_horiz);
%     B_out_box = mean(B_out0,1);
%     weight = B_mean./B_out_box;
%     new_g(coordinates) = repelem(weight', (2*box1+1)^2).*g(coordinates);
% 
%     % Weight factor, rescale the amplitudes according to tweezer centers
%     fprintf('error signal = %0.5f\n',std(weight));
% 
%     g = new_g;
%     A_iterated = g.*B_target.*exp(1i*psi_out); % Only the center pixels of each tweezer are left
%     A_new = fftshift(ifft2(ifftshift(A_iterated)));
%     psi = mod(angle(A_new),2*pi);
%     psi = psi.*pad;
% 
% end
% toc
% phase_mask = psi((scal-1)*y_pixels/2+1:(scal+1)*y_pixels/2, (scal-1)*x_pixels/2+1:(scal+1)*x_pixels/2);
% phase_mask = mod(phase_mask, 2*pi);
goal_plane_tilt = 5; %degrees

y_len= 9.22E-3;
x_len=15.56E-3;
dx=x_len/x_pixels; % dx = 0.2;
dy=y_len/y_pixels; % dy = 0.2;
xpp = (-x_len/2*scal:dx:x_len/2*scal-dx); %-30:dx:30;
ypp =  (-y_len/2*scal:dy:(y_len)/2*scal-dy); %30:dy:30;
[XX,YY] = meshgrid(xpp,ypp);


f = 200E-3;
%[XX,YY] = meshgrid(x,y);
k = 2*pi/(689*10^(-9));
zss = linspace(1,-1,size(XX,2));
% zss = linspace(0,0,size(XX,2));
zzz = 3E-3*repmat(zss,size(XX,1),1);
weight = 1;
weight1back = 1;
weight2back = 1;

tic

onesmatrix = ones(size(height_corr2));
Gg = 0.6; %0.608
for ii = 1:iterations 
%while std(weight) >0.001
    A_mod = A_in.*exp(1i*psi);
    A_out = fftshift(fft2(ifftshift(A_mod)));
    if ii <= N_cutoff %|| (std(weight) == weight2back == weight1back && std(weight)>0.000001 && N_cutoff>0)
        %if ii<= iterations*85/100
            psi_out = mod(angle(A_out),2*pi);
        %end
    end
    
    B_out = abs(fftshift(fft2(A_mod.*exp((-1i*k*zzz.*(XX.^2+YY.^2)./(2*f.*(f-zzz))))))) ;
    
%     if useFilter
%         B_out = B_out.*sqrt(filter_2d);
%     end

    
    new_g = zeros(length(y),length(x));
    if useGPU
    new_g = gpuArray(new_g);
    end
    
    if ii > 1        
        B_out(coordinates) = B_out(coordinates)./(onesmatrix - Gg.*(onesmatrix - sqrt(height_corr2))); % B_out(coordinates)./sqrt(height_corr2)
        
        if useFilter
            B_out = B_out./(filter_2d); %.* took out sqrt
        end
    end
    
    B_mean = mean(mean(B_out(coordinates))); % Mean of the amplitudes at target tweezer centers
    B_out0 = B_out(coordinates);

    B_out0 = reshape(B_out0, (2*box1+1)^2, length(coordinates)/((2*box1+1)^2) ); % n_horiz*n_vert 
    B_out_box = mean(B_out0,1);

    weight2back = weight1back;
    weight1back = std(weight);
    weight = B_mean./B_out_box;

%check convergence time
%     if ii == N_cutoff-2
%         if std(weight)>0.00001
%             ii=ii-1;
%         else
%             toc
%             timelistoc100(times) = toc;
%         end
 %   end

    new_g(coordinates) = repelem(weight', (2*box1+1)^2).*g(coordinates);

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
%%
% timelistoc14400_hmupd(times) = toc;
redSLM = 1;
x_pixels1 = x_pixels/redSLM;
y_pixels1 = y_pixels/redSLM;
phase_mask = psi((scal-1)*y_pixels1/2+1:(scal+1)*y_pixels1/2, (scal-1)*x_pixels1/2+1:(scal+1)*x_pixels1/2);
phase_mask = mod(phase_mask, 2*pi);

filename = sprintf('try20z1_n%dx%d_iter%d_spac%d_angle%d', n_horiz, n_vert, iterations, spacing_factor, goal_plane_tilt);
save_mask(phase_mask, filename, x_pixels, y_pixels, scal)