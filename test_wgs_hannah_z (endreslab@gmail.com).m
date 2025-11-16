%%% XLv 2022-11-21 integrating more precise optical field simulation
%%
global n_horiz n_vert x_pixels y_pixels pixel_um waist_um spacing_factor
global steer_x_factor steer_y_factor iterations height_corr loop
global magnification ROI_size identification_threshold exposureTimeus
global x0 y0 width height


%% Tweezer parameters
n_horiz = 100; n_vert = 100;
x_pixels = 4000;
y_pixels = 2464;
%scaling=2;
waist_um = 9/2*1e3; pixel_um = 3.74; % Physical beam size *10^(-6)
spacing_um = 4;
spacing_factor = spacing_um/(2*0.77);%2.5; %2.5;
%lambda_um = 1.06;
%focal_len_um= ;
%steer_x_factor = -7; steer_y_factor = 8; %mot steering away from 0 order
%% This first part comes from the calculation in wgs_initial

%numbers for HSP Meadowlark SLM
fill_factor = 0.957; % 0.98; % I suppose this is the fractional area that is effective
reflectivity = 0.88; % 0.97; % This is called Light Utilization Efficiency in the manual
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

%spacing= 21;

spacing = ceil(spacing_factor*2*(stop_idx - max_idx)); %round(0.75*(2)*ceil(spacing_factor*(stop_idx - max_idx)))

%% Creating target array

spacing_h = spacing; spacing_v = spacing;
spacing_v = spacing*y_pixels/x_pixels;
h_offset = round(spacing*spacing_global);
v_offset = round(-spacing/2);
[~,center_idx] = max(abs(A_single(:)));
[center_row,center_col] = ind2sub(size(A_single),center_idx);

A_target = zeros(size(A_single));
if useGPU
A_target = gpuArray(A_target);
end

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

iterations = 10;

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

y_len=17.7E-3;
x_len=10.6E-3;
dx=x_len/x_pixels; % dx = 0.2;
dy=y_len/y_pixels; % dy = 0.2;
xpp = (-x_len/2*scal:dx:x_len/2*scal-dx); %-30:dx:30;
ypp =  (-y_len/2*scal:dy:(y_len)/2*scal-dy); %30:dy:30;
[XX,YY] = meshgrid(xpp,ypp);


f = 200E-3;
%[XX,YY] = meshgrid(x,y);
k = 2*pi/(689*10^(-9));
zss = linspace(-1,1,size(XX,2));
zss = linspace(0,0,size(XX,2));
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
%% Plots
plot_results=1;
if plot_results
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
%%
y_len=17.7E-3;
x_len=10.6E-3;
dx=x_len/x_pixels; % dx = 0.2;
dy=y_len/y_pixels; % dy = 0.2;
dz = 5E-7; %1E-7;
x = (-x_len/2*scal:dx:x_len/2*scal-dx); %-30:dx:30;
y =  (-y_len/2*scal:dy:(y_len)/2*scal-dy); %30:dy:30;
z = -(40*dz):dz:(40*dz);

apperture = 10.4E-3;
f = 7.9E-3;
lambda = 1060E-9;
k = 2*pi/lambda;
dk = 2*pi/(max(x)-min(x));
dxprime = f*dk/(apperture*k);
xprime = cumsum(dxprime*ones(length(x),1)) - dxprime;
xprime = xprime-(max(xprime)-min(xprime))/2;

[X,Y] = meshgrid(x,y);

w_in = waist_um ; %(0.3:0.01:2);
% depth = nan(1,length(w_in));
% waist_out = nan(1,length(w_in));
% radial_trap_frequency = nan(1,length(w_in));
% axial_trap_frequency = nan(1,length(w_in));

% intensityfitfun = @(p,xy) p(2)*exp(-2*(x.^2)/p(1)^2);
% p_guess = [5, 353.4];
% y_zero_idx = find(y==0);
%for ii = 1:length(w_in)
    %A_in = A_in ; %sqrt(2/(pi*w_in(ii)^2))*exp(-(X.^2 + Y.^2)/w_in(ii)^2);
   A_in_apperture = A_mod; %A_out;

%     if A_in_apperture((X.^2 + Y.^2) > apperture/2)
%         disp('yes')
%     end
    A_in_apperture((X.^2 + Y.^2)> apperture/2) = 0;
    %A_in_apperture((X.^2 + Y.^2)< (0.3/76)*10.6E-3/2) = 0; 
%     A_out = fftshift(fft2(A_in_apperture));
%     I_out = A_out.*conj(A_out);
%     figure()
%     imagesc(I_out); axis image;
%%
%      [depth(ii),max_idx] = max(I_out(:));
%      [max_row,max_col] = ind2sub(size(I_out),max_idx);
%  %     depth(ii) = max(max(I_out));
%      
%      p_fit = nlinfit(x,I_out(y_zero_idx,:),intensityfitfun,p_guess);
%      p_guess = p_fit;
%      
%      waist_out(ii) = p_fit(1);
%      
%      max_col_d2 = gradient(gradient(I_out(:,max_col)));
%      radial_trap_frequency(ii) = sqrt(max_col_d2(max_row));
%      radial_trap_frequency(ii) = sqrt(depth(ii)/waist_out(ii)^2);

%    I_out_zmax = nan(1,length(z));

    A_out_FP= fftshift(fft2(A_in_apperture));
    I_zeroOut = A_out_FP.*conj( A_out_FP);
    I_zeroOut = I_zeroOut(2260:2330,3880:3960);
    [I_maxFP, X_max] = max(max( A_out_FP.*conj( A_out_FP)));

    I_zeroOut=A_out_FP.*conj( A_out_FP);
    I_zeroOut=I_zeroOut(2230:2370,3860:4110); %4um: 2280:2320,3870:3940
figure()
imagesc(I_zeroOut);
    %%
   % zz=0E-6;
    slice_Iout= zeros(length(y),length(z));

    for zz = 1:length(z)
        zzz=z(zz);
        %A_out = fftshift(fft2(A_in_apperture.*exp(-1i*k*zz*(X.^2+Y.^2)/(2*f^2))));
        A_out = fftshift(fft2(A_in_apperture.*exp(-1i*k*zzz*(X.^2+Y.^2)/(2*f*(f-zzz)))));
        I_out = A_out.*(conj(A_out)); %abs(A_out).^2;
        slice_Iout(:,zz)=I_out(:,X_max);
    %    I_out_zmax(zz) = max(max(I_out));
    end
    

    %%
    figure()
    A_prop2=A_mod.*exp(-1i*k*10*(X.^2+Y.^2)/(2*f^2));
    imagesc(A_prop2.*conj(A_prop2))
    slice_Iout45=slice_Iout;
    %%
 figure()
 xl=[-4,4];
 yl=[0,15];
 imagesc(yl,xl,slice_Iout(2220:2350,:)'); axis image;
 colorbar
 
 %%
 
 save_mask(phase_mask, filename, x_pixels, y_pixels, scal)