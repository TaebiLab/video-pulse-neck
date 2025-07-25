function BVP = ICA_POH(RGB, fs, LPF, HPF, Nn)
    % Cut off frequency.
    %LPF = 0.7;
    %HPF = 2.0;    

    NyquistF = 1/2*fs;
    BGRNorm = zeros(size(RGB));
    Lambda = 100;
    for c = 1:3
        BGRDetrend = detrend_custom(RGB(:, c), Lambda);
        BGRNorm(:, c) = (BGRDetrend - mean(BGRDetrend)) / std(BGRDetrend);
    end
    [~, S] = ica(BGRNorm', 3);

    % Select BVP Source
    MaxPx = zeros(1, 3);
    for c = 1:3
        FF = fft(S(c, :));
        %F = (0:length(FF)-1) / length(FF) * fs * 60;
        FF = FF(2:end);
        N = length(FF);
        Px = abs(FF(1:floor(N / 2)));
        Px = Px .^ 2;
        %Fx = (0:N/2-1) / (N/2) * NyquistF;
        Px = Px / sum(Px);
        MaxPx(c) = max(Px);
    end
    [~, MaxComp] = max(MaxPx);
    BVP_I = S(MaxComp, :);
    [B, A] = butter(Nn, [LPF / NyquistF, HPF / NyquistF], 'bandpass');
    BVP_F = filtfilt(B, A, real(BVP_I));

    BVP = BVP_F(:);
end