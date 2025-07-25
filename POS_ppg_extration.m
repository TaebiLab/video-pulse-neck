% POS method
% Algorithmic Principles of Remote PPG

function H = POS_ppg_extration(RGB, fs, useFGTransform)
    % Transform from: Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017, May). Color-distortion filtering for remote photoplethysmography. In Automatic Face & Gesture Recognition (FG 2017), 2017 12th IEEE International Conference on (pp. 71-78). IEEE.
    if nargin < 3
        useFGTransform=false;
    end

    if useFGTransform
        % original LPF = 0.7, HPF=2.5
        LPF = 0.7; % low cutoff frequency (Hz) - specified as 40 bpm (~0.667 Hz) in reference
        HPF = 2.0; % high cutoff frequency (Hz) - specified as 240 bpm in reference
        RGBBase = mean(RGB);
        RGBNorm = bsxfun(@times,RGB,1./RGBBase)-1;
        FF = fft(RGBNorm);
        F = (0:size(RGBNorm,1)-1)*fs/size(RGBNorm,1);
        H = FF*[-1/sqrt(6);2/sqrt(6);-1/sqrt(6)];
        W = (H.*conj(H))./sum(FF.*conj(FF),2);
        FMask = (F >= LPF)&(F <= HPF);
        % FMask(length(FMask)/2+1:end)=FMask(length(FMask)/2:-1:1);
        FMask = FMask + fliplr(FMask);
        W=W.*FMask';% rectangular filter in frequency domain - not specified in original paper
        FF = FF.*repmat(W,[1,3]);
        RGBNorm=real(ifft(FF));
        RGBNorm = bsxfun(@times,RGBNorm+1,RGBBase);
        
        RGB=RGBNorm;
    end
    
    
    % POS
    WinSec=1.6;   % (based on refrence's 32 frame window with a 20 fps camera)
    % lines and comments correspond to pseudo code algorithm on reference page 7       
    N = size(RGB,1);     % line 0 - RGB is of length N frames
    H = zeros(1,N);      % line 1 - initialize to zeros of length of video sequence
    l = ceil(WinSec*fs); % line 1 - window length equivalent to reference: 32 samples of 20 fps camera (default 1.6s)
    
    for n = 1:N        % line 2 - loop from first to last frame in video sequence
        % line 3 - spatial averaging was performed when video was read
        m = n - l + 1;   % line 4 condition
        if(m > 0)        % line 4
            Cn = ( RGB(m:n,:) ./ mean(RGB(m:n,:)) )';    %line 5 - temporal normalization
            S = [0, 1, -1; -2, 1, 1] * Cn;    %line 6 - projection
            h = S(1,:) + ((std(S(1,:)) / std(S(2,:))) * S(2,:));    %line 7 - tuning
            H(m:n) = H(m:n) + (h - mean(h));    %line 8 - overlap-adding
        end  % line 9 - end if
    end  % line 10 - end for

end