%% previous parameter initializations from meadowlark that im adjusting to holoeye parameters
x_pixels = 1024;
y_pixels = 768;
redSLM = 1;
scal = 1;
waist_um = 9.22*1e3;
pixel_um = 3.64;
odd_tw = 0;
array_offset_top = 0;
quadrantspacing_horiz = 0;
quadrantspacing_vert = 0;

spacing_factor= 7.55*1.0614 / 2 ;
spacing = ceil(spacing_factor*2*(stop_idx - max_idx));

%% SLM parameters initialization and beam size

plot_results = 1;

%scal = 4;

x_pixels1 = x_pixels/redSLM;
y_pixels1 = y_pixels/redSLM;

x_pixel_size = scal; % Large pixel size gives finer grid
y_pixel_size = scal;
x0 = 1:x_pixels1;
y0 = 1:y_pixels1;

x = 1:x_pixel_size*x_pixels1; % Actual coordinate of the grid
y = 1:y_pixel_size*y_pixels1;
[X,Y] = meshgrid(x,y);
[X0,Y0] = meshgrid(x0,y0); % SLM grid
[X,Y] = meshgrid(x,y); % WGS grid
R = round(sqrt( (X-length(x)/2).^2 + (Y-length(y)/2).^2 )); % polar
%waist_um = 4.2*1e3; %20; % Physical beam size and SLM pixel pitch
%pixel_um = 8;
waist_in = waist_um/pixel_um; % Input beam waist determines the size of individual tweezer
A_in = zeros(length(y),length(x));
A_in((scal-1)*y_pixels1/2+1:(scal+1)*y_pixels1/2, (scal-1)*x_pixels1/2+1:(scal+1)*x_pixels1/2) = ...
    exp(-( (X0-ceil(x_pixels1/2)).^2 + (Y0-ceil(y_pixels1/2)).^2)/(waist_in^2)); % Amplitude of E field
pad = abs(X-scal*x_pixels1/2) < x_pixels1/2 & abs(Y-scal*y_pixels1/2) < y_pixels1/2;
if useGPU
    A_in = gpuArray(A_in);
    pad = gpuArray(pad);
end

A_single = fftshift(fft2(ifftshift(A_in)));
[max_val,max_idx] = max(abs(A_single(:)).^2);
val = max_val;
stop_idx = max_idx;
while val > exp(-4)*max_val
   stop_idx = stop_idx + 1;
   val = abs(A_single(stop_idx)).^2;
end
% spacing = round(0.75*(2)*ceil(2.5*(stop_idx - max_idx)));
%% Creating target array

spacing_h = spacing; spacing_v = spacing;
spacing_v = spacing*y_pixels1/x_pixels1;
h_offset = round(-spacing/2); %round(spacing*spacing_global)
v_offset = round(-spacing/2);%round(-spacing/2);
if odd_tw ==1
    h_offset = 0;
    v_offset = 0;
end
[~,center_idx] = max(abs(A_single(:)));
[center_row,center_col] = ind2sub(size(A_single),center_idx);

A_target = zeros(size(A_single));
if useGPU
A_target = gpuArray(A_target);
end

% hh = cat( 2, cat(2,[1:n_horiz/3], [ (n_horiz/3+1+quadrantspacing_horiz*2) : (n_horiz*2/3+quadrantspacing_horiz*2)] , [(n_horiz*2/3+quadrantspacing_horiz*4+1) : (n_horiz+quadrantspacing_horiz*4) ]))';
% vv = cat( 2, cat(2, [1:n_vert/3], [ (n_vert/3+1+quadrantspacing_vert*2)  : (n_vert*2/3+quadrantspacing_vert*2)]  ,   [(n_vert*2/3+quadrantspacing_vert*4+1)  : (n_vert+quadrantspacing_vert*4) ]))';

if odd_tw ==1
    hh = cat(2,[1:n_horiz])';
    vv = cat(2,[1:n_vert])';
else
    hh = cat(2,[1:n_horiz/2 + array_offset_top] , [ n_horiz/2+1+quadrantspacing_horiz*2 + array_offset_top : n_horiz+quadrantspacing_horiz*2 ] )' ;
    vv = cat(2, [1:n_vert/2] , [ n_vert/2+1+quadrantspacing_vert*2  : n_vert+quadrantspacing_vert*2 ]  )' ;
end

h_offset_curr = round(spacing_h*(hh-(n_horiz/2 + quadrantspacing_horiz))); %*2
v_offset_curr = round(spacing_v*(vv-(n_vert/2 + quadrantspacing_vert))); %*2

if odd_tw ==1
    h_offset_curr = round(spacing_h*(hh-(round(n_horiz/2)-1 ))); %*2
    v_offset_curr = round(spacing_v*(vv-(round(n_vert/2)-1) ) ); %*2
end

target_rows = repmat(center_row + v_offset + v_offset_curr , n_horiz, 1);
target_cols = repelem(center_col + h_offset + h_offset_curr , n_vert);

target_rows_ref = repmat(v_offset_curr , n_horiz, 1);
target_cols_ref = repelem(h_offset_curr , n_vert);
if odd_tw ==1
    target_cols = target_cols';
    target_cols_ref = target_cols_ref';
end
disp(size(target_rows_ref))
disp(size(target_cols_ref))
%add angle option 23-11-12 

target_rows = round(target_cols_ref.*sin(angle_adjust)+target_rows_ref.*cos(angle_adjust)+ repmat( repmat(center_row+v_offset, n_horiz, 1) , n_vert, 1));
target_cols = round(target_cols_ref.*cos(angle_adjust)-target_rows_ref.*sin(angle_adjust)+ repmat( repmat(center_col+h_offset , n_vert,1), n_horiz,1));

%added to check!!! 10/06
% if pos_adjust ==1
%     target_rows = target_rows_previous + round(pos_gain_y.*flip(tw_diff_y'));
%     target_cols = target_cols_previous + round(pos_gain_x.*tw_diff_x');
% else
%     target_rows = target_rows + round(pos_gain_y.*flip(tw_diff_y'));
%     target_cols = target_cols + round(pos_gain_x.*tw_diff_x');
% end
% if pos_final ==1
%     target_rows = target_rows_previous;
%     target_cols = target_cols_previous;
% end
% if onetweezer == 1
%     target_rows = (center_row);
%     target_cols = (center_col);
% end

% if center_high ==1 %making single tweezer with array
%     target_rows = cat(1, target_rows, center_row);
%     target_cols = cat(1, target_cols, center_col);
% end

  if first_run
    target_idx = sub2ind(size(A_target),target_rows,target_cols); % Indices of tweezer centers
    %if center_high ==0
        height_corr = reshape(height_corr,[n_horiz*n_vert,1]);
    % else
    %     height_corr = reshape(height_corr,[n_horiz*n_vert+1,1]);
    %     target_idx = cat(1,target_idx,sub2ind(size(A_target),center_row,center_col) );
    % end
  else
      target_idx = sub2ind(size(A_target),tweezlist(:,1),tweezlist(:,2));
      height_corr = reshape(height_corr,[length(tweezlist),1]);

  end

A_target(target_idx) = sqrt(height_corr); %For WGS feedback ****deleted .^(-1)****

if ifcircle == 1
A_target( ( ((X-length(x)/2)*length(y)/length(x) ).^2 + ((Y-length(y)/2) ).^2 ) > circrad^2 ) = 0 ;
end

%[target_rows,target_cols] = ind2sub(10000,10000);

% box1 = stop_idx - max_idx - 3; % Fine tune this to optimize uniformity,-3
box1 = 1; %Changing this would make the PGS fail
if onetweezer == 1
    box1 = 0;
end

tweezer_idx = 1:n_horiz*n_vert;

% if center_high
%     tweezer_idx = 1:n_horiz*n_vert+1;
% end

if useFilter
    diffx = -300; %blazedgrating(1)*20; %2040/30;
    diffy = 400; %blazedgrating(2)*20; %2040/30;
    edge = 500;
    filter_2d = 1.5 + sqrt( (center_col +diffx - X).^2 + (center_row +diffy - Y).^2 )/3000 + Y/6000 ; %0.95 is 0th order efficiency
   % filter_2d(X.^2+Y.^2 > edge^2) = 1;
    if useGPU

       filter_2d = gpuArray(filter_2d);
    end
else
    filter_2d = ones(size(X));
    if useGPU
       filter_2d = gpuArray(filter_2d);
    end
end

if first_run
tweezlist = [];
coordinates = [];
tweezer_coor = zeros((2*box1+1)^2);

if useGPU
tweezer_coor = gpuArray(tweezer_coor);
end


for idx = tweezer_idx
    % if onetweezer == 1
    %     v_coor = [center_col] ;
    %     h_coor = [center_row] ;
    % else
        % if center_high ==1
        %     if idx~=10001
        %     v_coor = [target_rows(idx)-box1:target_rows(idx)+box1];
        %     h_coor = [target_cols(idx)-box1:target_cols(idx)+box1];
        %     else
        %         v_coor = [center_row-box1:center_row+box1];
        %         h_coor = [center_col-box1:center_col+box1];
        %     end
        % else
        v_coor = [target_rows(idx)-box1:target_rows(idx)+box1];
        h_coor = [target_cols(idx)-box1:target_cols(idx)+box1];            
        % end
    % end
    
    % if center_high==1
    %     if idx ==10001
    %      refrow = center_row;
    %      refcol = center_col;
    %     else
    %     refrow = target_rows(idx);
    %     refcol = target_cols(idx);
    %     end
    % else
    refrow = target_rows(idx);
    refcol = target_cols(idx);
    % end
    tweezers_full_idx = A_target(v_coor,h_coor);
    
    if A_target(refrow,refcol) ~= 0 

    tweezers = A_target(v_coor,h_coor);
    
    if first_run
        %if center_high ==0
        tweezlist = cat(1, tweezlist, [target_rows(idx),target_cols(idx)] ); %switched cols rows
        % else
        % if idx == 10001
        %     tweezlist = cat(1, tweezlist, [center_row,center_col] );
        % else
        %     tweezlist = cat(1, tweezlist, [target_rows(idx),target_cols(idx)] );
        % end
        %end
    end
    tweezer_rows = repmat(v_coor', 2*box1+1, 1);
    tweezer_cols = repelem(h_coor', 2*box1+1);
%     if idx == 10001
%         if center_high ==1
%             tweezer_coor(:,idx) = sub2ind(size(A_target),center_row,center_col);
%             coordinates = cat(1, coordinates, tweezer_coor(:,idx)); 
%         else
%             tweezer_coor(:,idx) = sub2ind(size(A_target),tweezer_rows,tweezer_cols);    
%             coordinates = cat(1, coordinates, tweezer_coor(:,idx)); 
%         end
        
%     else
        tweezer_coor(:,idx) = sub2ind(size(A_target),tweezer_rows,tweezer_cols);    
        coordinates = cat(1, coordinates, tweezer_coor(:,idx));    
%     end

    
    end
end


    height_corr = ones( length(coordinates)/(2*box1+1)^2 , 1);
    
    if initialize_w_loadingcurve == 1
        idxx_min = 0;
        distmin = 200;
        for idxx = 1: length(tweezlist)
             dist_hc = sqrt(((center_row -tweezlist(idxx,1))*(center_col/center_row))^2 + (center_col - tweezlist(idxx, 2))^2) ;
             adjj = (max(tweezlist(:,2))-center_col)/tan(3.2535*pi/180);
             twangle = atan(dist_hc/adjj)*180/pi;
             height_corr(idxx) = 0.1767*exp(0.7255*twangle) ;
             if dist_hc<distmin
                 idxx_min = idxx;
             end
        end


    end

    % if center_high ==1
    %     %coord = repmat(29493601, (2*box1+1)^2, 1);
    %     %coordinates = cat(1,coordinates, coord); %adding coordinates from "one_tweezer"
    %     height_corr = ones( length(coordinates)/(2*box1+1)^2-1 , 1);
    %     height_corr = cat(1, height_corr, center_weight);
    % end
end
height_corr(height_corr>= 6.0) = 6.0; %updated to try to prevent really low signal outcomes.
height_corr(height_corr<= 0.08) = 0.08; %updated to try to prevent really high signal outcomes.

if center_high == 1
    height_corr(idxx_min) = center_weight;
end

height_corr2 = repelem( height_corr, (2*box1+1)^2); % deleted  .^(-1)   ones( length(coordinates)/(2*box1+1)^2 , 1).^(-1) 


%% GS parameters initialization

if first_run || pos_adjust == 1
    psi0 = 2*pi*rand(y_pixels1,x_pixels1); % Random phase initialization
    N_cutoff = ceil(iterations)/2; % Fixed phase beyond this number of iteration
else
    psi0 = phase_mask_master; %Start with the same phase as the run before
    N_cutoff = 0; % Fixed phase beyond this number of iteration
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
psi((scal-1)*y_pixels1/2+1:(scal+1)*y_pixels1/2, (scal-1)*x_pixels1/2+1:(scal+1)*x_pixels1/2) = psi0;

%iterations = 50;


%% Initial GS 
weight = 1;
weight1back = 1;
weight2back = 1;
tic

y_len=17.7E-3;
x_len=10.6E-3;
dx=x_len/x_pixels; % dx = 0.2;
dy=y_len/y_pixels; % dy = 0.2;
xpp = (-x_len/2*scal:dx:x_len/2*scal-dx); %-30:dx:30;
ypp =  (-y_len/2*scal:dy:(y_len)/2*scal-dy); %30:dy:30;
[XX,YY] = meshgrid(xpp,ypp);


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

% timelistoc14400_hmupd(times) = toc;

phase_mask = psi((scal-1)*y_pixels1/2+1:(scal+1)*y_pixels1/2, (scal-1)*x_pixels1/2+1:(scal+1)*x_pixels1/2);
phase_mask = mod(phase_mask, 2*pi);

save_idx = 2;
path='C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\phase_masks';
%    imwrite(phase_mask,[path, sprintf('phase%d.bmp',save_idx)]);
imwrite(phase_mask/(2*pi),'C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\phase_masks\nofancy_noblaze_try2.bmp')



%% GS parameters
blazedgrating = [7,0];%[2,0]; % [x,y]
useFilter = 1;
useGPU = 1;
iterations = 200;
gain = 1.0;
%% Add Blazed grating
xx= 1: x_pixels;
yy= 1: y_pixels;

line_pair_per_mm_x = blazedgrating(1); %35
mm_per_line_pair_x = 1/line_pair_per_mm_x;
grating_x = 2*pi*xx*(x_size_mm/x_pixels)/mm_per_line_pair_x;

line_pair_per_mm_y = blazedgrating(2);
mm_per_line_pair_y = 1/line_pair_per_mm_y;
grating_y = 2*pi*yy*(y_size_mm/y_pixels)/mm_per_line_pair_y;

grating = grating_x + grating_y.'; 
grating = mod(grating,2*pi);

imwrite((phase_mask+grating)/(2*pi),'C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\phase_masks\nofancy_blazepd7_try2.bmp')

%% Plots

if plot_results
    A_mod = A_in.*exp(1i*mod(psi,2*pi));
    A_out = fftshift(fft2((A_mod)));

    figure(1);
    subplot(1,2,1);
    imagesc(abs(A_in).^2);
    pbaspect([x_pixels1 y_pixels1 1])
    title('I_{in}');

    subplot(1,2,2);
    imagesc(psi);
    pbaspect([x_pixels1 y_pixels1 1])
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


%% Performance

% I_out = abs(A_out).^2;
% box2 = stop_idx - max_idx;
% box3 = stop_idx - max_idx + 3;
% 
% for idx = tweezer_idx
%     tweezers2(:,:,idx) = ...
%         I_out(target_rows(idx)-box2:target_rows(idx)+box2,target_cols(idx)-box2:target_cols(idx)+box2);
%     I_tweezer(idx) = sum(sum(tweezers2(:,:,idx)));
% end
% Uniformity = 1 - (max(I_tweezer) - min(I_tweezer))/(max(I_tweezer) + min(I_tweezer)); 
% Nonuniformity = std(I_tweezer) / mean(I_tweezer);
% 
% signal_sum = ...
%     sum(sum(sum(I_out(target_rows(idx)-box3:target_rows(idx)+box3,target_cols(idx)-box3:target_cols(idx)+box3))));
% total_sum = sum(sum(I_out));
% bg = mean(min(I_out));
% Efficiency = signal_sum / ( total_sum - bg*scal^2*x_pixels*y_pixels );
% 
% fprintf('Uniformity = %0.4f\n',Uniformity);
% fprintf('Fractional std = %0.2f %%\n',1e2*Nonuniformity);
% fprintf('Efficiency = %0.2f %%\n',1e2*Efficiency);
% phase_mask = psi((scal-1)*y_pixels/2+1:(scal+1)*y_pixels/2, (scal-1)*x_pixels/2+1:(scal+1)*x_pixels/2);
% phase_mask = mod(phase_mask, 2*pi);

%% Position correction loop 
% 
% if position_correction
%     pos_y_feedback_master = zeros(n_tweezers,1);
%     pos_x_feedback_master = zeros(n_tweezers,1);
%     wgs_position_correction_tweezers_module();
%     psi0 = phase_mask;
%     N_cutoff = 0;
%     iterations = 20;
% 
%     loop = 5;
%     list_dist_x_dev_slm = [];
%     list_dist_y_dev_slm = [];
%     list_pos_atom_460_x_slm = [];
%     list_pos_atom_1_x_slm = [];
%     list_pos_atom_1056_x_slm = [];
%     list_pos_atom_460_y_slm = [];
%     list_pos_atom_1_y_slm = [];
%     list_pos_atom_1056_y_slm = [];
%     feedback_list_pos_atom_460_x_slm = [];
%     feedback_list_pos_atom_1_x_slm = [];
%     feedback_list_pos_atom_1056_x_slm = [];
%     feedback_list_pos_atom_460_y_slm = [];
%     feedback_list_pos_atom_1_y_slm = [];
%     feedback_list_pos_atom_1056_y_slm = [];
%     for i = 1:loop
%         wgs_position_correction_GS_module()
%         wgs_position_correction_tweezers_module()
%     end   
% end
% 
if first_run
    phase_mask_master = phase_mask;
end
% 
% %% Performance
% 
% if position_correction
%     x_axis = 1:length(feedback_list_pos_atom_1_x_slm);
%     label_x = 'Iteration';
% 
%     hold off
%     figure()
%     hold on
%     plot(x_axis,feedback_list_pos_atom_1056_x_slm,'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); 
%     plot(x_axis,feedback_list_pos_atom_460_x_slm,'o-r'); pbaspect([(1+sqrt(5))/2 1 1]);
%     plot(x_axis,feedback_list_pos_atom_1_x_slm,'o-g'); pbaspect([(1+sqrt(5))/2 1 1]);
%     xlabel(label_x); ylabel('Position along X (number of pixels)');
%     legend('trap 200','trap 100','trap 1');
%     grid on;
%     hold off
% 
%     figure()
%     hold on
%     plot(x_axis,feedback_list_pos_atom_1056_y_slm,'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); 
%     plot(x_axis,feedback_list_pos_atom_460_y_slm,'o-r'); pbaspect([(1+sqrt(5))/2 1 1]);
%     plot(x_axis,feedback_list_pos_atom_1_y_slm,'o-g'); pbaspect([(1+sqrt(5))/2 1 1]);
%     legend('trap 200','trap 100','trap 1');
%     grid on;
%     hold off
% 
%     figure()
%     semilogy(x_axis,list_dist_x_dev_slm,'ko-', x_axis,list_dist_y_dev_slm, 'o-r'); pbaspect([(1+sqrt(5))/2 1 1]); 
%     xlabel(label_x); ylabel('Standard deviation (% of the mean value)');
%     grid on
%     legend('x direction','y direction');
% end