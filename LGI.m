% LGI ppg
function bvp = LGI(RGB, fs, fl, fh, N)    
    % Singular Value Decomposition (SVD)
    [U, ~, ~] = svd(RGB');
    S = U(:, 1)';
       
    % Calculate P
    P = eye(3)-S'*S; %rank 1
    
    F(1:length(RGB),3)=0;
    for f=1:length(RGB)
      F(f,:)=(P*RGB(f,:)')';   
    end
    
    pulse=double(F(:,2));

    % Band-pass filter
    [b, a] = butter(N, [fl / fs * 2, fh / fs * 2], 'bandpass');
    bvp = filtfilt(b, a, pulse);
end