close all
clear all

path = 'E:\Data\human_subjects';
addpath(path);

%%
name = 'Input';
prompt = {'Enter Subject ID:';'Enter Breathing Condition:';'Acceleration Method:';'Lag Computation?';'Add Padding?';'Peak Detection Method:'};
formats = struct('type', {}, 'style', {}, 'items', {}, ...
  'format', {}, 'limits', {}, 'size', {});
formats(1,1).type   = 'list';
formats(1,1).style = 'popupmenu';
formats(1,1).items  = {'P01-221021_01','P02-221026_01', 'P03-221025_01', 'P04-221104_01', 'P05-221104_01', 'P07-221027_01', 'P08-221027_01', 'P09-221027_01', 'P10-221027_01', 'P11-221028_01', 'P12-221028_01', 'P13-221103_01', 'P14-221103_01', 'P15-221103_01'};
formats(2,1).type   = 'list';
formats(2,1).style = 'popupmenu';
formats(2,1).items  = {'EndExhalation','EndInhalation'};
formats(3,1).type   = 'list';
formats(3,1).style = 'popupmenu';
formats(3,1).items  = {'gradient','diff'};
formats(4,1).type   = 'list';
formats(4,1).style = 'popupmenu';
formats(4,1).items  = {'Yes','No'};
formats(5,1).type   = 'list';
formats(5,1).style = 'popupmenu';
formats(5,1).items  = {'No','Yes'};
formats(6,1).type   = 'list';
formats(6,1).style = 'popupmenu';
formats(6,1).items  = {'Windowing'};

[answer, canceled] = inputsdlg(prompt, name, formats);

if canceled
    return;
end

%%
subject_id = string(formats(1,1).items{answer{1}});
breathing = string(formats(2,1).items{answer{2}});
method = string(formats(3,1).items{answer{3}});
lag_computation = string(formats(4,1).items{answer{4}});
padding = string(formats(5,1).items{answer{5}});
peak_detection_method = string(formats(6,1).items{answer{6}});



%% select the files for processing
% Output displacement file
% PPG
ppg_folder_path = fullfile(path,subject_id,'Output',subject_id+'_'+breathing,'PPG');
d = dir(ppg_folder_path);
fn = {d.name};
[indx,~] = listdlg('PromptString',{'Select directory for PPG.',...
    'Only one folder be selected.',''},...
    'ListSize',[400,300],'ListString',fn);
if length(indx) > 1
    error('Only one file can be selected at a time.');
    return;
end

% load ppg file
ppg_filename = 'ppg_'+subject_id+'_'+breathing+'.mat';
load(fullfile(ppg_folder_path,fn{indx(1)},ppg_filename));

if breathing == 'EndExhalation' || breathing == 'EndInhalation'
    % Template track displacement
    displacement_folder_path = fullfile(path,subject_id,'Output',subject_id+'_'+breathing,'Template_tracking');
    d = dir(displacement_folder_path);
    fn = {d.name};
    [indx,~] = listdlg('PromptString',{'Select directory for Template Track.',...
        'Only one folder be selected.',''},...
        'ListSize',[400,300],'ListString',fn);
    if length(indx) > 1
        error('Only one file can be selected at a time.');
        return;
    end
    
    template_displacement_filename = 'qr_signals_'+subject_id+'_'+breathing+'.mat';
    % load Template track displacement signal
    load(fullfile(displacement_folder_path,fn{indx(1)},template_displacement_filename));
end


% Raw video file
video_filename = subject_id+'_'+breathing+'.MOV';
video_path = fullfile(path,subject_id,video_filename);
%end

% ECG file
ecg_filename = subject_id+'_'+breathing+'.mat';
ecg_path = fullfile(path,subject_id,ecg_filename);

% load ecg data
load(ecg_path,'b1');

%% parameters
fs = 5000;      % sampling frequency

if subject_id == 'P01-221021_01'
    fps = 59.975;
elseif subject_id == 'P20-250227_01'
    fps = 59.97;
elseif subject_id == 'P21-250228_01'
    fps = 59.97;
else
    vidObj = VideoReader(video_path);
    fps = vidObj.FrameRate;
end


%%
% syncing iWorx data and video data
% selecting the two taps at the beginning and end of the recording
fig_st = figure("Name","Mark starting and ending");
set(fig_st,'Position',[0 600 1920 300]);
plot(abs(b1(:,4)))
title('Carefully click on the 2 marker peaks!')
[x,~] = ginput(2);
x = round(x);
[~,i1] = max(abs(b1(x(1)-fs/10:x(1)+fs/10,4)));
[~,i2] = max(abs(b1(x(2)-fs/10:x(2)+fs/10,4)));

str_marker = (x(1)-fs/10) + i1 - 1;     % first marker
end_marker = (x(2)-fs/10) + i2 - 1;    % second marker

t = b1(:,1);                            % iWorx time vector
t_trunc = t(str_marker:end_marker);     % truncated time vector (for signal synchronization)


%% processing video data
fs_iphone = 44100;      % sampling freq of the audio of the video
videoAudio = audioread(video_path);

% selecting peak in the video audio to sync the data with the accelerometer
fig_ad = figure("Name","Audio start");
set(fig_ad,'Position',[0 600 1920 300]);
plot(abs(videoAudio))
title('Carefully click on the 1st marker peak!')
[x1,~] = ginput(1);
x1 = round(x1);
[~,i3] = max(abs(videoAudio(x1-fs_iphone/10:x1+fs_iphone/10)));

aud_str_marker = (x1-fs_iphone/10) + i3 - 1;     % first marker

vid_str_marker = round(aud_str_marker*fps/fs_iphone);     % first marker
vid_end_marker = vid_str_marker + round((end_marker-str_marker)*fps/fs);     % second marker


%% resampling factor
[p,q] = rat(fps/fs);

%% ECG read
ecg_raw = b1(str_marker:end_marker,2);

ecg_prep = sig_pre_process(ecg_raw, fs);

ecg = resample(ecg_prep, p, q);

figure
[~,qrs_i_raw,~] = pan_tompkin(ecg,fps,1);

% correcting the R peak by finding the max peak inside a defined window
for i=1:numel(qrs_i_raw)
    window_strt = max(1, qrs_i_raw(i) - round(fps / 10));
    window_end = min(length(ecg), qrs_i_raw(i) + round(fps / 10));
    [~,R_peak_loc] = max(ecg(window_strt:window_end));
    qrs_i_raw(i) = window_strt+R_peak_loc-1;
end


RRI = diff(qrs_i_raw);
EKG_HR = mean(1./(RRI/fps)*60);

HRm = mean(diff(qrs_i_raw));
ps = round(1*HRm/4);
pe = round(3*HRm/4);

count = 0;
if qrs_i_raw(1) - ps < 0
    for k=1:length(qrs_i_raw)
        if qrs_i_raw(k) - ps < 0
            count = count+1;
        end
    end
    qrs_i = qrs_i_raw(count+1:end);        
else
    qrs_i = qrs_i_raw;
end

 
%% Accelerometer SCG read
if breathing == 'EndExhalation' || breathing == 'EndInhalation' 
    % Top
    if subject_id == 'P12-221028_01'
        top_scgx = b1(:,7);
        top_scgy = b1(:,6);
    else
        top_scgx = b1(:,6);
        top_scgy = b1(:,5);
    end    
    
    top_scgx = resample(top_scgx, p, q);
    top_scgy = resample(top_scgy, p, q);
    
    % Middle
    if subject_id == 'P12-221028_01'
        middle_scgx = b1(:,10);
        middle_scgy = b1(:,9);
    else
        middle_scgx = b1(:,9);
        middle_scgy = b1(:,8);
    end
        
    middle_scgx = resample(middle_scgx, p, q);
    middle_scgy = resample(middle_scgy, p, q);
    
    % Bottom
    if subject_id == 'P12-221028_01'
        bottom_scgx = b1(:,13);
        bottom_scgy = b1(:,12);
    else
        bottom_scgx = b1(:,12);
        bottom_scgy = b1(:,11);
    end
        
    bottom_scgx = resample(bottom_scgx, p, q);
    bottom_scgy = resample(bottom_scgy, p, q);
    
    
    
    %% Computing lags between vision and accelerometer
    % Lags are computed on bottom sensor only
    if lag_computation == 'Yes'
        qr = signals(3,:,:);   % bottom sensor
        qr = reshape(qr,size(qr,2),2);
        displacement_t = linspace(0,length(qr)/fps,length(qr));
        [~,scgy_vid,~] = acceleration(qr, displacement_t, method, fps);
    
        % band passing
        fl = 1; % Hz
        fh = fps/2 - 0.001; % Hz
        N = 4;     
        [b, a] = butter(N, [fl fh] / (fps / 2), 'bandpass');
        scgy_vid = filtfilt(b, a, scgy_vid);
    
        % croping the data outside of the markers    
        scgy_vid = scgy_vid(vid_str_marker:vid_end_marker);
        
        % removing average and normalizing the signals    
        scgy_vid = scgy_vid - mean(scgy_vid);
        scgy_vid = scgy_vid/std(scgy_vid);
    
        % lag
        lag = finddelay(scgy_vid, bottom_scgy);
    
    end
    
    %% Template Vision based 
    if size(signals,1) > 3
        error('More that 3 QR code detected');
    end
    
    % band passing parameter
    fl = 1; % Hz
    fh = fps/2 - 0.001; % Hz
    N = 4;    
    [b, a] = butter(N, [fl fh] / (fps / 2), 'bandpass');
    
    %for qr_no=1:size(signals,1)
    for qr_no=3:size(signals,1)
        % reading displacement and then convert it to acceleration
        % Extract displacement signal of qr1
        qr = signals(qr_no,:,:);
        qr = reshape(qr,size(qr,2),2);
        displacement_t = linspace(0,length(qr)/fps,length(qr));
        [scgx_vid,scgy_vid,~] = acceleration(qr, displacement_t, method, fps);       
        
        % band pass
        scgx_vid = filtfilt(b, a, scgx_vid);
        scgy_vid = filtfilt(b, a, scgy_vid);
    
        % croping the data outside of the markers
        scgx_vid = scgx_vid(vid_str_marker:vid_end_marker);
        scgy_vid = scgy_vid(vid_str_marker:vid_end_marker);
    
        % removing average and normalizing the signals
        scgx_vid = scgx_vid - mean(scgx_vid);
        scgy_vid = scgy_vid - mean(scgy_vid);
        
        scgx_vid = scgx_vid/std(scgx_vid);
        scgy_vid = scgy_vid/std(scgy_vid);
    
        % correcting lags
        if lag_computation == 'Yes'
            scgx_vid = circshift(scgx_vid, lag);
            scgy_vid = circshift(scgy_vid, lag);
        end
    end

end % if breathing == 'EndExhalation' || breathing == 'EndExhalation' || breathing == 'AllSignals'

%% bandpass filter parameter
fl = 0.667;
fh = 2.5; 
N = 6;
[b, a] = butter(N, [fl fh] / (fps / 2), 'bandpass');

%% PPG
ppg_rgb = ppg_rgb(vid_str_marker:vid_end_marker, :);

ppg_red = ppg_rgb(:,1);
ppg_green = ppg_rgb(:,2);
ppg_blue = ppg_rgb(:,3);

% bandpass
ppg_red_filt = filtfilt(b, a, ppg_red);
ppg_green_filt = filtfilt(b, a, ppg_green);
ppg_blue_filt = filtfilt(b, a, ppg_blue);

% time
time_fps = linspace(0,(length(ppg_green)-1)/fps,length(ppg_green))';
time_fs = linspace(0,(length(ecg_raw)-1)/fs,length(ecg_raw))';

% correct ecg signal/ecg to match with vision ppg
if length(ecg) > length(ppg_green)
    % Truncate
    ecg = ecg(1:length(ppg_green));
    if ismember(subject_id, new_subject_list)
        ppg_gold = ppg_gold(1:length(ppg_green));
    end
    if qrs_i_raw(end) > length(ppg_green)
        qrs_i_raw(end) = [];
    end
elseif length(ecg) < length(ppg_green)
    % Pad signal with zeros
    ecg(length(ppg_green)) = 0;    
end

figure("Name",subject_id+" "+breathing+": ECG vs PPG")
ax = subplot(4,1,1);
%plot(ax,time_fps,ecg);
plot(ax,time_fs,ecg_raw);
title(ax, subject_id+" "+breathing+": ECG");
ax = subplot(4,1,2);
plot(ax,time_fps,ppg_red);
title(ax, "Red");
ax = subplot(4,1,3);
plot(ax,time_fps,ppg_green);
title(ax, "Green");
ax = subplot(4,1,4);
plot(ax,time_fps,ppg_blue);
title(ax, "Blue");


% for figures in manuscript
figure("Name",subject_id+" "+breathing+": ECG vs PPG")
ax = subplot(4,1,1);
%plot(ax,time_fps,ecg);
plot(ax,time_fs(round(2*fs):end-round(2*fs)),ecg_raw(round(2*fs):end-round(2*fs)));
title(ax, subject_id+" "+breathing+": ECG");
ax = subplot(4,1,2);
plot(ax,time_fps(round(2*fps):end-round(2*fps)),ppg_red(round(2*fps):end-round(2*fps)));
title(ax, "Red");
ax = subplot(4,1,3);
plot(ax,time_fps(round(2*fps):end-round(2*fps)),ppg_green(round(2*fps):end-round(2*fps)));
title(ax, "Green");
ax = subplot(4,1,4);
plot(ax,time_fps(round(2*fps):end-round(2*fps)),ppg_blue(round(2*fps):end-round(2*fps)));
title(ax, "Blue");


% filtered PPG
fig_rgb_filt = figure("Name",subject_id+" "+breathing+": ECG vs PPG filtered");
set(fig_rgb_filt,'Position',[1040 0 640 600]);
ax = subplot(6,1,1);
plot(ax,time_fps,ecg);
title(ax,subject_id+" "+breathing+": ECG");
ax = subplot(6,1,2);
plot(ax,time_fps,ppg_red_filt);
title(ax, "Red");
ax = subplot(6,1,3);
plot(ax,time_fps,ppg_green_filt);
title(ax, "Green");
ax = subplot(6,1,4);
plot(ax,time_fps,ppg_blue_filt);
title(ax, "Blue");


%% Wavelet based
% Define the wavelet and level of decomposition
wavelet = 'db4'; % Daubechies wavelet
level = 4; % Level of decomposition

% Perform wavelet decomposition
[C, L] = wavedec(ppg_green, level, wavelet);
% Reconstruct the signal from the approximation coefficients
ppg_green_wavelet = wrcoef('a', C, L, wavelet, level);
ppg_green_wavelet_filt = filtfilt(b, a, ppg_green_wavelet);

% Plot the original and reconstructed signals
figure("Name","Wavelet-based denoising");
subplot(4, 1, 1);
plot(time_fps,ppg_green);
title('Original Signal Green Channel');
xlabel('Sample');
ylabel('Amplitude');

subplot(4, 1, 2);
plot(time_fps,ppg_green_wavelet);
title('Wavelet Reconstructed Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(4, 1, 3);
plot(time_fps,ppg_green_wavelet_filt);
title('Filtered Reconstructed Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(4, 1, 4);
plot(time_fps,ecg);
title('ECG');

%% frequency analysis
% Compute the frequency axis
n = length(ppg_green); 
f_axis = (0:n-1)*(fps/n); 

% Compute the FFT of the signal
[fft_ppg_red, ppg_df_red] = findDominantFrequency(ppg_red, fps);
[fft_ppg_green, ppg_df_green] = findDominantFrequency(ppg_green, fps);
[fft_ppg_blue, ppg_df_blue] = findDominantFrequency(ppg_blue, fps);


% Display the dominant frequency
disp(['Dominant Frequency']);
disp(['Red: ', num2str(ppg_df_red), ' Hz']);
disp(['Green: ', num2str(ppg_df_green), ' Hz']);
disp(['Blue: ', num2str(ppg_df_blue), ' Hz']);


%% add padding (if necessary)
if padding == "Yes"
    % Number of samples for 3 seconds
    num_samples = round(3 * fps);

    % Padding using the beginning and end of the signal
    start_pad_ppg_red = ppg_red(1:num_samples); 
    end_pad_ppg_red = ppg_red(end-num_samples+1:end); 
    start_pad_ppg_green = ppg_green(1:num_samples); 
    end_pad_ppg_green = ppg_green(end-num_samples+1:end);
    start_pad_ppg_blue = ppg_blue(1:num_samples); 
    end_pad_ppg_blue = ppg_blue(end-num_samples+1:end);

    % Align the end of the start padding with the start of the main signal
    % Red
    [acor_start_red, lag_start_red] = xcorr(start_pad_ppg_red(end-num_samples/2+1:end), ppg_red(1:num_samples/2), 'coeff');
    [~, idx_start_red] = max(acor_start_red);
    lag_start_red = lag_start_red(idx_start_red);
    aligned_start_pad_ppg_red = circshift(start_pad_ppg_red, -lag_start_red);
    % Green
    [acor_start_green, lag_start_green] = xcorr(start_pad_ppg_green(end-num_samples/2+1:end), ppg_green(1:num_samples/2), 'coeff');
    [~, idx_start_green] = max(acor_start_green);
    lag_start_green = lag_start_green(idx_start_green);
    aligned_start_pad_ppg_green = circshift(start_pad_ppg_green, -lag_start_green);
    % blue
    [acor_start_blue, lag_start_blue] = xcorr(start_pad_ppg_blue(end-num_samples/2+1:end), ppg_blue(1:num_samples/2), 'coeff');
    [~, idx_start_blue] = max(acor_start_blue);
    lag_start_blue = lag_start_blue(idx_start_blue);
    aligned_start_pad_ppg_blue = circshift(start_pad_ppg_blue, -lag_start_blue);

    % Align the start of the end padding with the end of the main signal
    % Red
    [acor_end_red, lag_end_red] = xcorr(end_pad_ppg_red(1:num_samples/2), ppg_red(end-num_samples/2+1:end), 'coeff');
    [~, idx_end_red] = max(acor_end_red);
    lag_end_red = lag_end_red(idx_end_red);
    aligned_end_pad_ppg_red = circshift(end_pad_ppg_red, -lag_end_red);
    % Green
    [acor_end_green, lag_end_green] = xcorr(end_pad_ppg_green(1:num_samples/2), ppg_green(end-num_samples/2+1:end), 'coeff');
    [~, idx_end_green] = max(acor_end_green);
    lag_end_green = lag_end_green(idx_end_green);
    aligned_end_pad_ppg_green = circshift(end_pad_ppg_green, -lag_end_green);
    % Blue
    [acor_end_blue, lag_end_blue] = xcorr(end_pad_ppg_blue(1:num_samples/2), ppg_blue(end-num_samples/2+1:end), 'coeff');
    [~, idx_end_blue] = max(acor_end_blue);
    lag_end_blue = lag_end_blue(idx_end_blue);
    aligned_end_pad_ppg_blue = circshift(end_pad_ppg_blue, -lag_end_blue);

    % Concatenate the padded signal
    ppg_red_padded = [aligned_start_pad_ppg_red; ppg_red; aligned_end_pad_ppg_red];
    ppg_green_padded = [aligned_start_pad_ppg_green; ppg_green; aligned_end_pad_ppg_green];
    ppg_blue_padded = [aligned_start_pad_ppg_blue; ppg_blue; aligned_end_pad_ppg_blue];

    RGB = [ppg_red_padded, ppg_green_padded, ppg_blue_padded]; % padded

else
    RGB = [ppg_red, ppg_green, ppg_blue];

end

%%% Pre-processing
% moving average filter
RGB = movmean(RGB, 15, 1);

%% visualization of different method
% Green Channel
green_ppg = RGB(:,2);
green_ppg = detrend_custom(green_ppg, 100);
green_ppg = filtfilt(b, a, green_ppg);

% POS
pos_ppg = POS_ppg_extration(RGB, fps, false);
pos_ppg = detrend_custom(pos_ppg', 100);
pos_ppg = filtfilt(b, a, pos_ppg);

% CHROM
chrom_ppg = CHROM_ppg_extration(RGB, fps, fl, fh, N);

% ICA
ica_ppg = ICA_POH(RGB, fps, fl, fh, N);

% OMIT
omit_ppg = OMIT(RGB, fps, fl, fh, N);

% LGI
lgi_ppg = LGI(RGB, fps, fl, fh, N);

% Remove the padding
if padding == "Yes"
    pos_ppg = pos_ppg(num_samples+1:end-num_samples);
    chrom_ppg = chrom_ppg(num_samples+1:end-num_samples);
    ica_ppg = ica_ppg(num_samples+1:end-num_samples);
    omit_ppg = omit_ppg(num_samples+1:end-num_samples);
    lgi_ppg = lgi_ppg(num_samples+1:end-num_samples);
    
end

% Heart rate
[dom_freq_HR_ppg_green, locs_ppg_green, locs_ppg_green_track, HR_ppg_green, NNi_ppg_green] = heart_rate(green_ppg, green_ppg, fps, "Green", peak_detection_method);
[dom_freq_HR_omit_ppg, locs_omit_ppg, locs_omit_ppg_track, HR_omit_ppg, NNi_omit_ppg] = heart_rate(omit_ppg, green_ppg, fps, "OMIT", peak_detection_method);
[dom_freq_HR_pos_ppg, locs_pos_ppg, locs_pos_ppg_track, HR_pos_ppg, NNi_pos_ppg] = heart_rate(pos_ppg, green_ppg, fps, "POS", peak_detection_method);
[dom_freq_HR_chrom_ppg, locs_chrom_ppg, locs_chrom_ppg_track, HR_chrom_ppg, NNi_chrom_ppg] = heart_rate(chrom_ppg, green_ppg, fps, "CHORM", peak_detection_method);
[dom_freq_HR_ica_ppg, locs_ica_ppg, locs_ica_ppg_track, HR_ica_ppg, NNi_ica_ppg] = heart_rate(ica_ppg, green_ppg, fps, "ICA", peak_detection_method);
[dom_freq_HR_lgi_ppg, locs_lgi_ppg, locs_lgi_ppg_track, HR_lgi_ppg, NNi_lgi_ppg] = heart_rate(lgi_ppg, green_ppg, fps, "LGI", peak_detection_method);
[dom_freq_HR_wavelet_ppg_green, locs_wavelet_ppg_green, locs_wavelet_ppg_green_track, HR_wavelet_ppg_green, NNi_wavelet_ppg_green] = heart_rate(ppg_green_wavelet_filt, green_ppg, fps, "Wavelet", peak_detection_method);

fig_rPPG = figure("Name",subject_id+"\_"+breathing+": "+peak_detection_method);
set(fig_rPPG,'Position',[0 0 960 840]);
ax1 = subplot(8,1,1);
plot(time_fps,ecg);
hold on;
plot(time_fps(qrs_i_raw),ecg(qrs_i_raw), "o")
title("ECG: HR - "+num2str(EKG_HR, '%05.2f'));

ax2 = subplot(8,1,2);
plot(time_fps,green_ppg);
hold on;
plot(time_fps(locs_ppg_green), green_ppg(locs_ppg_green),"o")
if peak_detection_method == "Windowing"
    plot(time_fps(locs_ppg_green(locs_ppg_green_track)), green_ppg(locs_ppg_green(locs_ppg_green_track)),"k*")

end
title("Green: HR - "+num2str(mean(HR_ppg_green), '%05.2f')+ ...
    ";  dom\_freq\_based\_HR:"+num2str(dom_freq_HR_ppg_green, '%05.2f'));

ax3 = subplot(8,1,3);
plot(time_fps,omit_ppg);
hold on;
plot(time_fps(locs_omit_ppg),omit_ppg(locs_omit_ppg),"o")
if peak_detection_method == "Windowing"
    plot(time_fps(locs_omit_ppg(locs_omit_ppg_track)),omit_ppg(locs_omit_ppg(locs_omit_ppg_track)),"k*")

end
title("OMIT ppg: HR - "+num2str(mean(HR_omit_ppg), '%05.2f')+ ...
    ";  dom\_freq\_based\_HR:"+num2str(dom_freq_HR_omit_ppg, '%05.2f'));

ax4 = subplot(8,1,4);
plot(time_fps,pos_ppg);
hold on;
plot(time_fps(locs_pos_ppg),pos_ppg(locs_pos_ppg),"o")
if peak_detection_method == "Windowing"
    plot(time_fps(locs_pos_ppg(locs_pos_ppg_track)),pos_ppg(locs_pos_ppg(locs_pos_ppg_track)),"k*")

end
title("POS ppg: HR - "+num2str(mean(HR_pos_ppg), '%05.2f')+ ...
    ";  dom\_freq\_based\_HR:"+num2str(dom_freq_HR_pos_ppg, '%05.2f'));

ax5 = subplot(8,1,5);
plot(time_fps(1:length(chrom_ppg)),chrom_ppg);
hold on;
plot(time_fps(locs_chrom_ppg),chrom_ppg(locs_chrom_ppg),"o")
if peak_detection_method == "Windowing"
    plot(time_fps(locs_chrom_ppg(locs_chrom_ppg_track)),chrom_ppg(locs_chrom_ppg(locs_chrom_ppg_track)),"k*")

end
title("CHROM ppg: HR - "+num2str(mean(HR_chrom_ppg), '%05.2f')+ ...
    ";  dom\_freq\_based\_HR:"+num2str(dom_freq_HR_chrom_ppg, '%05.2f'));

ax6 = subplot(8,1,6);
plot(time_fps,ica_ppg);
hold on;
plot(time_fps(locs_ica_ppg),ica_ppg(locs_ica_ppg),"o")
if peak_detection_method == "Windowing"
    plot(time_fps(locs_ica_ppg(locs_ica_ppg_track)),ica_ppg(locs_ica_ppg(locs_ica_ppg_track)),"k*")

end
title("ICA ppg: HR - "+num2str(mean(HR_ica_ppg), '%05.2f')+ ...
    ";  dom\_freq\_based\_HR:"+num2str(dom_freq_HR_ica_ppg, '%05.2f'));

ax7 = subplot(8,1,7);
plot(time_fps,lgi_ppg);
hold on;
plot(time_fps(locs_lgi_ppg),lgi_ppg(locs_lgi_ppg),"o")
if peak_detection_method == "Windowing"
    plot(time_fps(locs_lgi_ppg(locs_lgi_ppg_track)),lgi_ppg(locs_lgi_ppg(locs_lgi_ppg_track)),"k*")

end
title("LGI ppg: HR - "+num2str(mean(HR_lgi_ppg), '%05.2f')+ ...
    ";  dom\_freq\_based\_HR:"+num2str(dom_freq_HR_lgi_ppg, '%05.2f'));

ax8 = subplot(8,1,8);
plot(time_fps, ppg_green_wavelet_filt);
hold on;
plot(time_fps(locs_wavelet_ppg_green),ppg_green_wavelet_filt(locs_wavelet_ppg_green),"o")
title("Wavelet Green: HR - "+num2str(mean(HR_wavelet_ppg_green), '%05.2f')+ ...
    ";  dom\_freq\_based\_HR:"+num2str(dom_freq_HR_wavelet_ppg_green, '%05.2f'));


% Link the x-axes of all subplots
linkaxes([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], 'x');


%% Heart rate variability
RRi = RRI/fps*1000; % RRI in miliseconds
nn_intervals_ecg = RRi;

%% outlier detection and romover
if peak_detection_method == "Windowing"
    % Removed the tracked outliers
    NNi_ppg_green(locs_ppg_green_track) = []; 
    NNi_omit_ppg(locs_omit_ppg_track) = [];
    NNi_pos_ppg(locs_pos_ppg_track) = [];
    NNi_chrom_ppg(locs_chrom_ppg_track) = [];
    NNi_ica_ppg(locs_ica_ppg_track) = [];
    NNi_lgi_ppg(locs_lgi_ppg_track) = [];
end

% % ----------temp 
nn_intervals_green = NNi_ppg_green;
nn_intervals_omit = NNi_omit_ppg;
nn_intervals_pos = NNi_pos_ppg;
nn_intervals_chrom = NNi_chrom_ppg;
nn_intervals_ica = NNi_ica_ppg;
nn_intervals_lgi = NNi_lgi_ppg;
% % ---------------------

% Methods and corresponding NN intervals
methods = {'ECG', 'Green PPG', 'OMIT', 'POS', 'CHROM', 'ICA', 'LGI'};
nn_intervals = {nn_intervals_ecg, nn_intervals_green, nn_intervals_omit, ...
                nn_intervals_pos, nn_intervals_chrom, nn_intervals_ica, nn_intervals_lgi};

% Initialize a cell array to store results
hrv_features = {'Method', 'Mean NNI (ms)', 'SDNN (ms)', 'RMSSD (ms)', ...
    'pNN50 (%)', 'Median NNI (ms)', ...
    'Mean HR (bpm)', 'Max HR (bpm)', 'Min HR (bpm)', 'Std HR (bpm)'};

% Loop through each method to calculate HRV features
for i = 1:numel(methods)
    % Extract NN intervals for the method
    nn = nn_intervals{i};
    
    % Call the `get_time_domain_features` function
    features_with_all_decimel_pts = get_time_domain_features(nn);

    % Round each feature to two decimal points
    features = structfun(@(x) round(x, 2), features_with_all_decimel_pts, 'UniformOutput', false);
    
    % Append results to the cell array
    hrv_features = [hrv_features; {methods{i}, features.mean_nni, features.sdnn, ...
        features.rmssd, features.pnni_50, features.median_nni, ...
        features.mean_hr, features.max_hr, ...
        features.min_hr, features.std_hr}];
end

% displaying results
hrv_features

% Convert cell array to table for easy export
hrv_table = cell2table(hrv_features(2:end, :), 'VariableNames', hrv_features(1, :));

% Save the table to an Excel file
output_filename = subject_id+'_'+breathing+'_'+peak_detection_method+'_without_stat_outlier'+'_HRV.xlsx';
output_filepath = fullfile('EMBC_Conference_2025',output_filename);
writetable(hrv_table, output_filepath);

disp(['HRV features saved to ', output_filepath]);




%% Function to Calculate Acceleration Signal from displacement signal
function [ax,ay,ta] = acceleration(qr_disp_xy, t, method, fs)
    t = t(:);  % for converting to column vector
    dt = t(2) - t(1);
    % Displacement signal
    x = qr_disp_xy(:,1);
    y = qr_disp_xy(:,2);

    % high pass to remove dc component
    fc = 0.5;
    N = 4;
    [b, a] = butter(N, fc/(fs/2), 'high');
    % Apply the high-pass filter
    x = filtfilt(b, a, x);
    y = filtfilt(b, a, y);

%     figure
%     plot(x)
%     figure
%     plot(y)

    % Velocity
    if method == 'diff'
        vx = diff(x)./diff(t);           % velocities at times tv; a vector of the length less than t, x by 1
        %vx = diff(x)/dt;
        vy = diff(y)./diff(t);           % times related to v; a vector of the length less than t, x by 1
        %vy = diff(y)/dt;
        tv = (t(1:end-1)+t(2:end))/2;
    elseif method == 'gradient'        
        %vx = gradient(x)/dt;
        %vy = gradient(y)/dt;
        vx = gradient(x, 1/fs);
        vy = gradient(y, 1/fs);
    end

    % Apply the high-pass filter to remove dc component
    vx = filtfilt(b, a, vx);
    vy = filtfilt(b, a, vy);
        
    % Acceleration
    if method == 'diff'
        ax = diff(vx)./diff(tv);         % accelerations at times ta; a vector of the length less than t, x by 2
        %ax = diff(vx)/dt;
        ay = diff(vy)./diff(tv);         % times related to a; a vector of the length less than t, x by 2
        %ay = diff(vy)/dt;
        ta = (tv(1:end-1)+tv(2:end))/2; 
    elseif method == 'gradient'
        %ax = gradient(vx)/dt;
        %ay = gradient(vy)/dt;   
        ax = gradient(vx, 1/fs);
        ay = gradient(vy, 1/fs);
    end

    if method == 'diff'
        % Since ta vector is less than t by 2, we add some dummy value at the
        % begining and end of the vector. Same will be done for  ax, ay
        ax = vertcat(ax(1), ax, ax(end));
        ay = vertcat(ay(1), ay, ay(end));
        ta = vertcat(ta(1), ta, ta(end));
    elseif method == 'gradient' 
        ta = t;
    else
        print("Please select a Method for acceleration calculation.")
        exit();
    end

end

%% dominent frequency
function [Y, dominant_frequency] = findDominantFrequency(signal, Fs)
    % findDominantFrequency - Finds the dominant frequency of a given signal
    % 
    % Syntax: dominant_frequency = findDominantFrequency(signal, Fs)
    %
    % Inputs:
    %    signal - Input signal (vector)
    %    Fs - Sampling frequency (Hz)
    %
    % Outputs:
    %    dominant_frequency - Dominant frequency (Hz)

    % Remove the DC component
    signal = signal(:);
    signal = signal - mean(signal);
    
    % Length of signal
    L = length(signal);
    
    % Compute the FFT of the signal
    Y = fft(signal);
    
    % Compute the two-sided spectrum P2
    P2 = abs(Y/L);
    
    % Compute the single-sided spectrum P1
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    
    % Define the frequency domain f
    f = Fs*(0:(L/2))/L;
    
    % Find the dominant frequency
    [~, idx] = max(P1);
    dominant_frequency = f(idx);   
    
end


%% ECG preprocess
%% Signal pre-process, i.e., removing average, bandpass filter etc.
function [ecg_filt] = sig_pre_process(ecg, fs)
    % removing average
    ecg = ecg - mean(ecg);
    
    % notch filter to remove 120, 240, 480 Hz components
    freqRatio = [120,240,480] * 2/fs;
    notchWidth = 0.1;       % width of the notch
    notchZeros = [exp( sqrt(-1)*pi*freqRatio ), exp( -sqrt(-1)*pi*freqRatio )]; % Compute zeros
    notchPoles = (1-notchWidth) * notchZeros; % Compute poles
    b = poly(notchZeros); %  Get moving average filter coefficients
    a = poly(notchPoles); %  Get autoregressive filter coefficients
    ecg_filt_raw = filter(b,a,ecg);
    
    % moving average
    ecg_filt_raw = movmean(ecg_filt_raw,round(fs/250));

    % normalization
    ecg_filt_raw = ecg_filt_raw / max(ecg_filt_raw(:));
    
%     % band-pass filter
%     Fn = fs/2;
%     fc = [1 30];
%     fcnm = fc/Fn;
%     Ws = [1-0.1 30+1]/Fn;
%     Rp = 1;
%     Rs = 10;
%     [n,~] = buttord(fcnm,Ws,Rp,Rs);
%     b = fir1(n, fcnm, 'bandpass');
%     scg_filt_raw = fftfilt(b, scg_filt_raw);   

    d = designfilt('highpassiir','FilterOrder',8, ...
    'PassbandFrequency',1,'PassbandRipple',0.2, ...
    'SampleRate',fs);
    ecg_filt_raw = filtfilt(d,ecg_filt_raw);
    
    d_lo = designfilt('lowpassiir','FilterOrder',8, ...
        'PassbandFrequency',30,'PassbandRipple',0.2, ...
        'SampleRate',fs);
    ecg_filt_raw = filtfilt(d_lo,ecg_filt_raw);
        
    ecg_filt = ecg_filt_raw;

end


%% HR calculation
function [dom_freq_HR, locs, locs_track, inst_HR, NNi] = heart_rate(ppg_signal, ppg_green, fs, fig_name, peak_detection_method)
    if peak_detection_method == "Envelop"
        [locs, dom_freq] = find_ppg_peaks(ppg_signal, fs, fig_name); 
        locs_track = false;
    elseif peak_detection_method == "Windowing"
        [locs, locs_track, dom_freq] = find_ppg_peaks_windowing(ppg_signal, ppg_green, fs, true, fig_name);    
    end

    NN_interval = diff(locs);  % in sample points
    inst_HR = 1./(NN_interval/fs)*60;  % bpm
    NNi = NN_interval/fs*1000; % in miliseconds

    % HR calculated from dominent frequency
    dom_freq_HR = dom_freq * 60; % bpm
end


%% Heart rate variablility (HRV) calculations
function time_domain_features = get_time_domain_features(nn_intervals, pnni_as_percent)
    % Returns a struct containing time domain features for HRV analysis.
    % Mostly used on long term recordings (24h) but some studies use some of those features on
    % short term recordings, from 1 to 5 minutes window.

    if nargin<2
        pnni_as_percent = true;
    end

    diff_nni = diff(nn_intervals);
    if pnni_as_percent
        length_int = length(nn_intervals) - 1;
    else
        length_int = length(nn_intervals);
    end

    % Basic statistics
    mean_nni = mean(nn_intervals);
    median_nni = median(nn_intervals);
    range_nni = max(nn_intervals) - min(nn_intervals);

    sdsd = std(diff_nni);
    rmssd = sqrt(mean(diff_nni .^ 2));

    nni_50 = sum(abs(diff_nni) > 50);
    pnni_50 = 100 * nni_50 / length_int;
    nni_20 = sum(abs(diff_nni) > 20);
    pnni_20 = 100 * nni_20 / length_int;

    % Feature found on github and not in documentation
    cvsd = rmssd / mean_nni;

    % Features only for long term recordings
    sdnn = std(nn_intervals, 1);  % ddof = 1 : unbiased estimator => divide std by n-1
    cvnni = sdnn / mean_nni;

    % Heart Rate equivalent features
    heart_rate_list = 60000 ./ nn_intervals;   % minutes * miliseconds 
    mean_hr = mean(heart_rate_list);
    min_hr = min(heart_rate_list);
    max_hr = max(heart_rate_list);
    std_hr = std(heart_rate_list);

    time_domain_features = struct('mean_nni', mean_nni, 'sdnn', sdnn, 'sdsd', sdsd, 'nni_50', nni_50, 'pnni_50', pnni_50, 'nni_20', nni_20, 'pnni_20', pnni_20, 'rmssd', rmssd, 'median_nni', median_nni, 'range_nni', range_nni, 'cvsd', cvsd, 'cvnni', cvnni, 'mean_hr', mean_hr, 'max_hr', max_hr, 'min_hr', min_hr, 'std_hr', std_hr);
end



%% HRV outlier : Remove those nni which are 20% different from mean_vision_hr   
function valid_nn_intervals_std = my_hrv_outlier(nni)
    % Filter out intervals outside the physiological range (e.g., 0.3–1.5 seconds for humans corresponding to 30–150 bpm).
    valid_nn_intervals_phy = nni(nni >= 300 & nni <= 1500);

    %  interquartile range (IQR) filtering
    Q1 = prctile(valid_nn_intervals_phy, 25); % 25th percentile
    Q3 = prctile(valid_nn_intervals_phy, 75); % 75th percentile
    IQR = Q3 - Q1;
    lower_bound1 = Q1 - 1.5 * IQR;
    upper_bound1 = Q3 + 1.5 * IQR;
    
    valid_nn_intervals_irq = valid_nn_intervals_phy(valid_nn_intervals_phy >= lower_bound1 & valid_nn_intervals_phy <= upper_bound1);

    % Standard Deviation filtering
    mean_pp = mean(valid_nn_intervals_irq);
    std_pp = std(valid_nn_intervals_irq);
    lower_bound2 = mean_pp - 2 * std_pp;
    upper_bound2 = mean_pp + 2 * std_pp;
    
    valid_nn_intervals_std = valid_nn_intervals_irq(valid_nn_intervals_irq >= lower_bound2 & valid_nn_intervals_irq <= upper_bound2);

%     % Median Filtering
%     median_pp = median(valid_nn_intervals_std);
%     threshold = 0.2 * median_pp; % Allowable deviation (20%)
%     valid_nn_intervals_median = valid_nn_intervals_std(abs(valid_nn_intervals_std - median_pp) <= threshold);

    
end
