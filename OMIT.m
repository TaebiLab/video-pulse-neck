% Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces.
% Álvarez Casado, C., & Bordallo López, M.
% IEEE Journal of Biomedical and Health Informatics.
% (2023).

function bvp = OMIT(ppg_rgb, fs, fl, fh,N) 
    % Pre-processing signal
    Red = ppg_rgb(:, 1);
    Green = ppg_rgb(:, 2);
    Blue = ppg_rgb(:, 3);
    % Band pass filter
    %fl = 0.7; % low cutoff frequency (Hz) - specified as 40 bpm (~0.667 Hz) in reference
    %fh = 2.0; % high cutoff frequency (Hz) - specified as 240 bpm in reference
    [b, a] = butter(N, [fl / fs * 2, fh / fs * 2], 'bandpass');
    Red = filtfilt(b, a, Red);
    Green = filtfilt(b, a, Green);
    Blue = filtfilt(b, a, Blue);

    ppg_rgb = [Red'; Green'; Blue'];   % dimension (3, N)
    
    % OMIT
    % QR decomposition
    [Q, R] = qr(ppg_rgb, 0);
    
    % Reshape and calculate P
    S = Q(:, 1)';
    P = eye(3) - (S' * S);
    
    % Calculate Y and bvp
    Y = P * ppg_rgb;
    bvp = Y(2, :);
    bvp = bvp(:);
end