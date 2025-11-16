clear
%cd('C:\Users\Gandalf\Box\EndresLab\z_Second Experiment\Code\SLM simulation\Large Array HM');

 %% Define global variables
 global n_horiz n_vert x_pixels y_pixels pixel_um waist_um spacing_global spacing %spacing_factor move_tweezer_one
 global iterations height_corr loop useGPU useFilter % steer_x_factor steer_y_factor
 %global magnification ROI_size identification_threshold exposureTimeus
 global x0 y0 %width height

%% SLM parameters

SLM_name = 'MeadowlarkHSP';

% if ~libisloaded('Blink_C_wrapper')
%     loadlibrary('Blink_C_wrapper.dll', 'Blink_C_wrapper.h','includepath','C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\Large Array HM\Blink OverDrive Plus\SDK');
% end
% % This loads the image generation functions
% if ~libisloaded('ImageGen')
%     loadlibrary('ImageGen.dll', 'ImageGen.h');
% end

%% only run for PCI
% Basic parameters for calling Create_SDK

bit_depth = 12; %for the 1920 use 12, for the small 512x512 use 8
num_boards_found = libpointer('uint32Ptr', 0);
constructed_okay = libpointer('int32Ptr', 0);is_nematic_type = 1;
RAM_write_enable = 1;
use_GPU = 0;
max_transients = 10;
wait_For_Trigger = 0; % This feature is user-settable; use 1 for 'on' or 0 for 'off'
flip_immediate = 0; % Only supported on the 1024
timeout_ms = 5000; %5000

%%
%Both pulse options can be false, but only one can be true. You either generate a pulse when the new image begins loading to the SLM
%or every 1.184 ms on SLM refresh boundaries, or if both are false no output pulse is generated.
OutputPulseImageFlip = 0;
RGB = 0;
OutputPulseImageRefresh = 0; %only supported on 1920x1152, FW rev 1.8. 

% - This regional LUT file is only used with Overdrive Plus, otherwise it should always be a null string
reg_lut = libpointer('string'); %This parameter is specific to the small 512 with Overdrive, do not edit

% % Call the constructor
% if strcmp(SLM_name,'MeadowlarkHSP')==1
%   calllib('Blink_C_wrapper', 'Create_SDK', bit_depth, num_boards_found, constructed_okay, is_nematic_type, RAM_write_enable, use_GPU, max_transients, reg_lut);
% 
% end
%%
% constructed okay return of 1 is success

% if constructed_okay.value ~= 1  
%     disp(calllib('Blink_C_wrapper', 'Get_last_error_message'));
% end

% if num_boards_found.value > 0 
board_number = 1;
disp('Blink SDK was successfully constructed');
fprintf('Found %u SLM controller(s)\n', num_boards_found.value);

% if strcmp(SLM_name,'MeadowlarkHSP')==1
% calib_file = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\0thorder06-27-2022_slm6175at1060.lut'; %done on 06/27/2022, locally saved on Gandalf
% calllib('Blink_C_wrapper', 'Load_LUT_file', board_number, calib_file);
pixel_um = 9.2; 
x_size_mm = 17.6;
y_size_mm = 10.7;
% if ~libisloaded('ImageGen')
%     loadlibrary('ImageGen.dll', 'ImageGen.h');
% end
height_SLM = 768; %calllib('Blink_C_wrapper', 'Get_image_height', board_number);
width_SLM = 1024; %calllib('Blink_C_wrapper', 'Get_image_width', board_number);
depth_SLM = 2^8; %768; %calllib('Blink_C_wrapper', 'Get_image_depth', board_number);
x_pixels = width_SLM;
y_pixels = height_SLM;
Bytes = depth_SLM/8;
Image = libpointer('uint8Ptr', zeros(width_SLM*height_SLM*Bytes*3,1));
WFC = libpointer('uint8Ptr', zeros(width_SLM*height_SLM,1));
CenterX = width_SLM/2;
CenterY = height_SLM/2;
RGB = true;
% elseif strcmp(SLM_name, 'MeadowlarkEd')==1
%     calib_file = 'C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\LUT Files\slm6130_at1064_HDMI.lut';
%     calllib('Blink_C_wrapper', 'Load_lut', calib_file);
%     if ~libisloaded('ImageGen')
%         loadlibrary('ImageGen.dll', 'ImageGen.h');
%     end
%     y_pixels = calllib('Blink_C_wrapper', 'Get_Width');
%     x_pixels = calllib('Blink_C_wrapper', 'Get_Height');
%     depth_SLM = calllib('Blink_C_wrapper', 'Get_Depth');
%     width_SLM = x_pixels;
%     height_SLM = y_pixels;
%     WFC = libpointer('uint8Ptr', zeros(width_SLM*height_SLM*3,1));
%     CenterX = width_SLM/2;
%     CenterY = height_SLM/2;
%     RGB = true;
%     pixel_um = 8; 
%     x_size_mm = 15.36;
%     y_size_mm = 9.6;
% 
% else
% 
%     warning('Unrecognized SLM.')
% end
% end

%% Tweezer parameters

n_horiz = 10; n_vert = 10; %largest n_vert: ~160 maybe both numbers need to be even?
%mag=3; %magnification of imaging system

spacing_global = 1/2* (n_horiz+2);
%waist_um = 4.6*1e3; % Physical beam size

waist_um = 9/2*1e3; % Physical beam size *10^(-6)

%% Spacing

scal = 1;
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
ain=size(A_in);
A_single = fftshift(fft2(ifftshift(A_in)));
[max_val,max_idx] = max(abs(A_single(:)).^2);
val = max_val;
stop_idx = max_idx;
while val >= exp(-2)*max_val %1/e^2 radius
   stop_idx = stop_idx + 1;
   val = abs(A_single(stop_idx)).^2;
end

difflimBW=4;
spacing_factor= 7.55*1.0614 / 2 ; %6*4.05/4.5 *2;
spacingexp=spacing_factor*0.77*2;
%spacing_um= 10; %4.5;
%spacing_factor = spacing_um/(2*difflimBW); % diffraction limited beam waist ~4um
spacing = ceil(spacing_factor*2*(stop_idx - max_idx));


%% Defocusing

x0 = 1:x_pixels;
y0 = 1:y_pixels;
x0 = x0 - mean(x0);
y0 = y0 - mean(y0);
epsilon = 0; %defocusing 4
defocus = 1/1000*(x0.*x0 + y0'.*y0')*(x_size_mm/x_pixels)*epsilon;

%% Aberrations correction parameters

 xc = 0; yc = 0; % will be calibrated 8/12/22
%  xc = x_pixels/2;
%  yc = y_pixels/2;
radius = y_pixels/2; % 9/2*1e3/(pixel_um); %x_pixels; %  % this is the aperture of the pupil (objective)


zer = zeros(15,1);


%%
zer(2)= 0;
zer(3)= 0;
zer(4) = 0; % 0.2; %0.025;
zer(5) = 0; %0.2;
% zer(6) = -0.25; %0.125; %-0.125; 
% zer(7) = 0; %-0.25; %-1.5; %0.1; 
% zer(8) = -0.1; %-0.15;
% %zer(13) = 0;
zer(9) = 0.2;
%  zer(10) = 0.25;
% zer(11) = 0.1;
zer(12) = -0.25;
%zer(13) =  0.2;
%zer(14) = -0;
%wavefront = Zernike(radius, zer, x_pixels, y_pixels, xc, yc);


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

% Basler parameters: included in imageBasler_for_feedback

%% Tweezers initialization
% timelistoc14400_hmupd = [];
% errorsig14400_hmupd = [];
% for times = 1:6
height_corr = ones(1,n_horiz*n_vert);
%quadrantspacing_factor = 3;
waist_um = 9/2*1e3;
quadrantspacing_vert = 0;
quadrantspacing_horiz = 0;
ifcircle = 0;
circrad = 2690; %500; %2662

useFilter = 0;

redSLM = 1; %reduce SLM pixel usage

pos_x_feedback_master = zeros(1,n_horiz*n_vert);
pos_y_feedback_master = zeros(1,n_horiz*n_vert);
first_run = 1;
position_correction = 0;
odd_tw = 0;
goal_plane_tilt = 5; %degrees
f = 200E-3;
k = 2*pi/(1060*10^(-9));
zss = linspace(-1,1,size(xx,2));
zzz = 3E-3*repmat(zss,size(xx,1),1);
array_offset_top = 0;
angle_adjust = 0;
onetweezer = 0;
initialize_w_loadingcurve = 0;
center_high = 0;
pos_adjust = 0;
% pos_gain_y = 0;
% pos_gain_x = 0;
useGPU = 0;
wgs_position_correction_initial;
height_corr = reshape(height_corr,[1, length(coordinates)/(2*box1+1)^2 ] ) ; %Could be included in the wgs function

% end

%% SLM phase mask output
%zer=zeros(15,1);
wavefront = mod(Zernike(radius, zer, x_pixels, y_pixels, xc, yc),2*pi);

mask1 = gather(phase_mask); %zeros(size(wavefront)); %gather(phase_mask); %zeros(size(wavefront));
%mask1(y_pixels/4+1:y_pixels*3/4, x_pixels/4+1:x_pixels*3/4) = gather(phase_mask);
gratingmask = zeros(size(wavefront));
%gratingmask(y_pixels/4+1:y_pixels*3/4, x_pixels/4+1:x_pixels*3/4) = grating(y_pixels/4+1:y_pixels*3/4, x_pixels/4+1:x_pixels*3/4);
gratingmask = grating;
mask_for_display = mod( mask1 + wavefront + gratingmask , 2*pi ); %+defocus  gather(phase_mask)

%imagesc(mask_for_display)
%imagesc(wavefront)
%pbaspect([x_pixels,y_pixels,1])
mask_for_display = reshape(mask_for_display.', 1, numel(mask_for_display))'; %change
mask_for_display = mask_for_display * 256/2/pi;
ImageOne = libpointer('uint8Ptr', zeros(x_pixels*y_pixels*Bytes,1));
ImageOne.Value = mask_for_display;
% calllib('Blink_C_wrapper', 'Write_image', board_number, ImageOne, width_SLM*height_SLM*depth_SLM/8, wait_For_Trigger, flip_immediate, OutputPulseImageFlip, OutputPulseImageRefresh, timeout_ms);

pause(0.5)
nb_image = 5;

%mask_for_display_origin = mask_for_display;
%% input corner pixels to align
exposureTimeus = 350; 

% stablist = [];
% for stability = 1:10
height_corr = ones(1,length(tweezlist));
first_run = 1;
%imageBasler_for_correction_account_fluctuations;
gain = 1;
threshold = 0.1;%0.1;
% imageBasler_for_feedback;

% stablist(stability)= 1e2*std(signal_sums)/mean(signal_sums);
% end
%%
figure()
% plot(stablist/100)
axis([1 10 0 .02])
xlabel('Measurement');
ylabel('Non-uniformity')

title('Stabiliity of Uniformity, 2500 tweezers')
%%

% figure()
% 
% plot(height_corr)

% mask = mod(zeros(y_pixels,x_pixels),1);
% dispOnSLM(mask,99);

%% correct focus gradient aberrations
% f=10E-2; f = f/(pixel_um*10^(-6))*scal;
% lambda = 1060E-9; lambda = lambda/(pixel_um*10^(-6))*scal;
% 
% height_corr = ones(1,n_horiz*n_vert);
% pos_x_feedback_master = zeros(1,n_horiz*n_vert);
% pos_y_feedback_master = zeros(1,n_horiz*n_vert);
% position_correction = 0;
% 
% 
% first_run = 1;
% zcorrect = 1;
% iterations = 100;
% zpl = 0*[1 1 1 ; 1 1 1 ; 1 1 1 ]*10E-6; %indexes for z planes of different focal planes
% zpl = zpl/(pixel_um*10^(-6))*scal;
% wgs_include_z
%% Tweezer height correction

%mask_for_display = mask_for_display_origin ;
% height_corr = ones(1,n_horiz*n_vert);
%threshold = 0.10;

loop = 10;
% height_corr = ones(1,n_horiz*n_vert);
gain_pos = 1;
nb_image = 5;

gain = 1 ;

list_deviation = [];
list_dist_x_dev = [];
list_dist_y_dev = [];
list_pos_atom_1_x = [];
list_pos_atom_460_x = [];
list_pos_atom_1056_x = [];
list_pos_atom_1_y = [];
list_pos_atom_460_y = [];
list_pos_atom_1056_y = [];
list_dist_x_dev_slm = [];
list_dist_y_dev_slm = [];
list_pos_atom_1_x_slm = [];
list_pos_atom_460_x_slm = [];
list_pos_atom_1056_x_slm = [];
list_pos_atom_1_y_slm = [];
list_pos_atom_460_y_slm = [];
list_pos_atom_1056_y_slm = [];
feedback_list_pos_atom_460_x_slm = [];
feedback_list_pos_atom_1_x_slm = [];
feedback_list_pos_atom_1056_x_slm = [];
feedback_list_pos_atom_460_y_slm = [];
feedback_list_pos_atom_1_y_slm = [];
feedback_list_pos_atom_1056_y_slm = [];

first_run = 0;
waist_in =  9/2*1e3/9.2;
waist_um = 9/2*1e3;

unifiter900_quarterSLM= [];
for index = 1:loop
      % height_corr = ones(1,n_horiz*n_vert);
    if index > 10
        gain_pos = 0;
    end
    waist_um = 9/2*1e3;
    wgs_position_correction_initial;
    height_corr = reshape(height_corr,[1,length(tweezlist)]);
    %    zer(5) = -0.08  + index*0.02;
    x0 = 1:x_pixels;
    y0 = 1:y_pixels;
    x0 = x0 - mean(x0);
    y0 = y0 - mean(y0);

%mask_for_display = gather(phase_mask)+wavefront;

mask1 = zeros(size(wavefront));
% mask1(y_pixels/4+1:y_pixels*3/4, x_pixels/4+1:x_pixels*3/4) = gather(phase_mask); %(y_pixels/4+1:y_pixels*3/4, x_pixels/4+1:x_pixels*3/4)
mask1 = gather(phase_mask);

mask_for_display = mod(mask1+wavefront+gratingmask,2*pi); %grating to grating mask

mask_for_display = reshape(mask_for_display.', 1, numel(mask_for_display))'; %change
mask_for_display = mask_for_display * 256/2/pi;
ImageOne = libpointer('uint8Ptr', zeros(x_pixels*y_pixels*Bytes,1));
ImageOne.Value = mask_for_display;
% calllib('Blink_C_wrapper', 'Write_image', board_number, ImageOne, width_SLM*height_SLM*depth_SLM/8, wait_For_Trigger, flip_immediate, OutputPulseImageFlip, OutputPulseImageRefresh, timeout_ms);
exposureTimeus = 35; 
% imageBasler_for_feedback;
%figure()
%plot(height_corr)

unifiter900_quarterSLM(index)= 1e2*std(signal_sums)/mean(signal_sums);

end

%%
figure()
histogram((signal_sums)/mean(signal_sums))
xlabel('Signal/Mean');
ylabel('Number of tweezers')

figure()
plot((signal_sums)/mean(signal_sums))
xlabel('Tweeezer Index');
ylabel('Signal/Mean')
title('Uniformity over 7465 tweezers')

%%
figure()
x_axis = 1:iterations;
xlabel('Iteration');
ylabel('WGS Tweezer Amplitude Std / Mean Tweeezer Amplitude ')
title('WGS initialization convergence for increasing system size')
axis([0 100 0 0.02])
hold on
% plot(mean(errorsig100,1));
% plot(mean(errorsig2500,1));
% plot(mean(errorsig6400,1));
% plot(mean(errorsig10000,1));
plot(mean(errorsig14400,1));
plot(mean(errorsig14400_hmupd,1));
legend('old','new')
%legend('100 tweezers, 10x10','2500 tweezers, 50x50','6400 tweezers, 80x80','10000 tweezers, 100x100', '14400 tweezers, 120x120'); %
hold off
%%
figure()
x_axis = 1:length(unifiter4900_wofilt); %loop;
hold on
plot(unifiter4900_2/100)
plot(unifiter4900_sparsespread1/100)
plot(unifiter4900_sparserspread1/100)
xlabel('Iteration');
ylabel(' Tweezer Amplitude Std / Mean Tweeezer Amplitude ')
legend('4900 tweezers','1280 tweezers', '250 tweezers'); %
title( 'Uniformization across same area, different tweezer number' )

hold off
%%
%data stored: unifiter100, unifiter70circ, unifiter200 (400 tw),
%unifiter1000,unifiter2500 
figure()
x_axis = 1:length(unifiter2000_halfSLM); %loop;
hold on
% plot(x_axis,unifiter100/100)
% plot(x_axis,unifiter200/100)
% plot(x_axis,unifiter1000/100)
%plot(x_axis,unifiter2500/100)

plot(x_axis,unifiter2000_quarterSLM/100)
plot(x_axis,unifiter900_quarterSLM/100)
plot(x_axis,unifiter400_quarterSLM/100)
% plot(x_axis,unifiter4900_2/100)
% plot(x_axis,unifiter7465/100)
xlabel('Iteration');
ylabel(' Tweezer Amplitude Std / Mean Tweeezer Amplitude ')
legend('1600 tweezers','900 tweezers','400 tweezers')
%legend('100 tweezers, 10x10','400 tweezers, 20x20','1000 tweezers, 20x50','2500 tweezers, 50x50','4900 tweezers, 70x70','7465 tweezers'); %
title( 'Uniformization achieved using quarter of SLM pixels for varied tweezer number' )
hold off
%%

xx= [100,400,1000,2500,4900,7465];

yy = [mean(unifiter100(7:10)), mean(unifiter200(7:10)), mean(unifiter1000(7:10)), mean(unifiter2500(7:10)), mean(unifiter4900_2(7:10)),mean(unifiter7465(7:10)) ]/100;
figure()
plot(xx,yy)
xlabel('Number of tweezers')
ylabel('Non-uniformity (Standard Deviation/ Mean )')

%% Equalization efficiency
% x_axis = -0.08  + [1:loop]*0.02;
x_axis = 1:loop;
% x_axis = 1:numel(list_pos_atom_1056_x);
% x_axis = 1:length(list_deviation);
label_x = 'Iteration';
% label_x = 'Weight of 4th Zernike polynomial';
wait_um = 1;
figure(6)
hold on
semilogy(x_axis,list_deviation,'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); xlabel(label_x); ylabel('Standard deviation (% of the mean value)');
ylim([0.1,30]);%xlim([1,13]);
grid on;
figure(7)
hold on
semilogy(x_axis,abs(list_dist_x_dev),'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); 
semilogy(x_axis,abs(list_dist_y_dev),'o-r'); pbaspect([(1+sqrt(5))/2 1 1]);
xlabel(label_x); ylabel('Standard deviation (% of the mean value)');
legend('x direction','y direction');
ylim([0.01,2]); 
grid on;
hold off
figure(31)
hold on
plot(x_axis,(list_pos_atom_1056_x),'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); 
plot(x_axis,(list_pos_atom_460_x),'o-r'); pbaspect([(1+sqrt(5))/2 1 1]);
plot(x_axis,(list_pos_atom_1_x),'o-g'); pbaspect([(1+sqrt(5))/2 1 1]);
xlabel(label_x); ylabel('Position along X (number of waists)');
ylim([-0.01,0.01]); 
legend('trap 200','trap 100','trap 1');
grid on;
hold off

figure(30)
hold on
plot(x_axis,(list_pos_atom_1056_y),'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); 
plot(x_axis,(list_pos_atom_460_y),'o-r'); pbaspect([(1+sqrt(5))/2 1 1]);
plot(x_axis,(list_pos_atom_1_y),'o-g'); pbaspect([(1+sqrt(5))/2 1 1]);
xlabel(label_x); ylabel('Position along Y (number of waists)');
ylim([-0.01,0.01]); 
legend('trap 200','trap 100','trap 1');
grid on;
hold off

% plot(1:loop,list_deviation2,'o-r'); pbaspect([(1+sqrt(5))/2 1 1]); xlabel('Iteration'); ylabel('Standard deviation (% of the mean value)'); 
% hold off
% legend({'Images + GS','Only Images'},'Location','northeast')

%%

x_axis = -0.08  + [1:loop]*0.02;
x_axis = 1:loop;
% x_axis = 1:numel()
% x_axis = 1:length(list_deviation);
label_x = 'Iteration';
% label_x = 'Weight of 4th Zernike polynomial';
wait_um = 1;
figure(6)
% hold on
semilogy(x_axis,list_deviation,'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); xlabel(label_x); ylabel('Standard deviation (% of the mean value)');
ylim([0.1,30]);%xlim([1,13]);
grid on;
figure(7)
hold on
semilogy(x_axis,abs(list_dist_x_dev_slm),'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); 
semilogy(x_axis,abs(list_dist_y_dev_slm),'o-r'); pbaspect([(1+sqrt(5))/2 1 1]);
xlabel(label_x); ylabel('Standard deviation (% of the mean value)');
legend('x direction','y direction');
% ylim([0.001,0.1]); 
grid on;
hold off
figure(19)
hold on
plot(x_axis,(list_pos_atom_1056_x_slm-mean(list_pos_atom_1056_x_slm))/mean(mean(waist_um,2)),'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); 
plot(x_axis,(list_pos_atom_460_x_slm-mean(list_pos_atom_460_x_slm))/mean(mean(waist_um,2)),'o-r'); pbaspect([(1+sqrt(5))/2 1 1]);
plot(x_axis,(list_pos_atom_1_x_slm-mean(list_pos_atom_1_x_slm))/mean(mean(waist_um,2)),'o-g'); pbaspect([(1+sqrt(5))/2 1 1]);
xlabel(label_x); ylabel('Position along X (number of waists)');
ylim([-0.1,0.1]); 
legend('trap 200','trap 100','trap 1');
grid on;
hold off

figure(20)
hold on
plot(x_axis,(list_pos_atom_1056_y_slm-mean(list_pos_atom_1056_y_slm))/mean(mean(waist_um,2)),'ko-'); pbaspect([(1+sqrt(5))/2 1 1]); 
plot(x_axis,(list_pos_atom_460_y_slm-mean(list_pos_atom_460_y_slm))/mean(mean(waist_um,2)),'o-r'); pbaspect([(1+sqrt(5))/2 1 1]);
plot(x_axis,(list_pos_atom_1_y_slm-mean(list_pos_atom_1_y_slm))/mean(mean(waist_um,2)),'o-g'); pbaspect([(1+sqrt(5))/2 1 1]);
xlabel(label_x); ylabel('Position along Y (number of waists)');
ylim([-0.1,0.1]); 
legend('trap 200','trap 100','trap 1');
grid on;
hold off



%% Save phase_mask
save_phase = 1;
if save_phase
   save_idx = 2;
   path='C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\phase_masks';
%    imwrite(phase_mask,[path, sprintf('phase%d.bmp',save_idx)]);
   imwrite(phase_mask/(2*pi),'C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\phase_masks\a_try.bmp')
end
