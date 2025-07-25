function S = CHROM_ppg_extration(RGB, fs, LPF, HPF, N)        
    WinSec=1.6;
    FN = size(RGB, 1);

    %LPF = 0.7; % low cutoff frequency (Hz) - specified as 40 bpm (~0.667 Hz) 
    %HPF = 2.0; % high cutoff frequency (Hz) - specified as 240 bpm (~4.0 Hz) 
    NyquistF = 1/2*fs;
    [B,A] = butter(N,[LPF/NyquistF HPF/NyquistF], 'bandpass');

    % Window parameters - overlap, add with 50% overlap
    WinL = ceil(WinSec*fs);
    if(mod(WinL,2)) % force even window size for overlap, add of hanning windowed signals
        WinL=WinL+1;
    end    
    NWin = floor((FN-WinL/2)/(WinL/2));
    S = zeros(NWin,1);
    WinS = 1;             % Window Start Index
    WinM = WinS+WinL/2;   % Window Middle Index
    WinE = WinS+WinL-1;   % Window End Index

    for i = 1:NWin        
        RGBBase = mean(RGB(WinS:WinE,:));
        RGBNorm = bsxfun(@times,RGB(WinS:WinE,:),1./RGBBase)-1;
        
        % CHROM
        Xs = squeeze(3*RGBNorm(:,1)-2*RGBNorm(:,2)); %3Rn-2Gn
        Ys = squeeze(1.5*RGBNorm(:,1)+RGBNorm(:,2)-1.5*RGBNorm(:,3)); %1.5Rn+Gn-1.5Bn
        
        Xf = filtfilt(B,A,double(Xs));
        Yf = filtfilt(B,A,double(Ys));
        
        Alpha = std(Xf)./std(Yf);
        
        SWin = Xf - Alpha.*Yf;
        
        SWin = hann(WinL).*SWin;
        %overlap, add Hanning windowed signals
        if(i==1)
            S = SWin;
        else
            S(WinS:WinM-1) = S(WinS:WinM-1)+SWin(1:WinL/2);%1st half overlap
            S(WinM:WinE) = SWin(WinL/2+1:end);%2nd half            
        end
        
        WinS = WinM;
        WinM = WinS+WinL/2;
        WinE = WinS+WinL-1;
    end

end