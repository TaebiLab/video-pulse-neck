function dom_freq = dominant_frequency(signal, sampling_rate, plot_dom_freq, fig_name)
    if nargin < 3
        plot_dom_freq = false;
        fig_name = "None";
    end
    
    % Remove the DC component
    signal = signal(:);
    signal = signal - mean(signal);

    % Number of samples in the signal
    N = length(signal);
    
    % Perform Fast Fourier Transform (FFT)
    Y = fft(signal);
    
    % Compute the frequencies corresponding to the FFT values
    f = (0:N-1)*(sampling_rate/N);

    % Consider only the first half of the spectrum for analysis
    half_N = floor(N/2) + 1;
    f = f(1:half_N);
    Y = abs(Y(1:half_N)); % magnitude spectrum

    % Find indices where frequency is greater than 0.667 Hz or 40 bpm
    valid_indices = f > 0.667;
    
    % Filter the frequencies and FFT values
    f = f(valid_indices);
    Y = Y(valid_indices);

%     % Remove frequencies between 1.95 Hz - 2.05 Hz since it appears in many
%     % subjects as a outliers.
%     valid_indices = f < 1.9 | f > 2.05;
%     % Filter the frequencies and FFT values
%     f = f(valid_indices);
%     Y = Y(valid_indices);

%     %% Harmonic peaks removal
%     % Peak detection to identify candidates
%     [pks, locs] = findpeaks(Y, f, 'MinPeakHeight', max(Y) * 0.3); % Peaks above 30% of max power
%     
%     % Harmonic Suppression Using Ratios
%     threshold = 0.2; % Tolerance for harmonic detection
%     multiple = 2;  % harmonic multiple of f as 2f
%     fundamental_freqs = []; % Store fundamental frequencies
%     fundamental_peaks = []; % Store corresponding peak values
% 
%     for i = 1:length(locs)
%         is_harmonic = false;
%         for j = 1:i-1
%             ratio = locs(i) / locs(j);
%             if abs(ratio - multiple) < threshold
%                 is_harmonic = true; % Current peak is a harmonic of a lower-frequency peak
%                 break;
%             end
%         end
%         if ~is_harmonic
%             fundamental_freqs = [fundamental_freqs, locs(i)];
%             fundamental_peaks = [fundamental_peaks, pks(i)];
%         end
%     end
% 
%     % Select the lowest frequency from the fundamental frequencies
%     if ~isempty(fundamental_freqs)
%         [~, max_idx] = max(fundamental_peaks); % Find the index of the max peak
%         dom_freq = fundamental_freqs(max_idx); % Use the frequency corresponding to the max peak
%     else
%         dom_freq = NaN; % No valid fundamental frequency found
%     end
% 
%     if plot_dom_freq
%         % Plot the magnitude spectrum
%         figure("Name",fig_name);
%         plot(f, Y, 'b-', 'DisplayName', 'Magnitude Spectrum');
%         hold on;
%         
%         % Highlight detected peaks
%         %plot(locs, pks, 'x', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Detected Peaks');
%         if ~isempty(fundamental_freqs)
%             % Plot only the fundamental frequencies
%             plot(fundamental_freqs, fundamental_peaks, 'x', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Fundamental Frequencies');
%         end
% 
%         % Highlight the dominant frequency
%         if ~isnan(dom_freq)
%             plot(dom_freq, Y(f == dom_freq), 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Dominant Frequency');
%         end
%         
%         % Annotate the plot
%         title('Magnitude Spectrum with Harmonic Suppression');
%         xlabel('Frequency (Hz)');
%         ylabel('|FFT|');
%         grid on;
%         legend('Magnitude Spectrum', 'Detected Peaks', 'Dominant Frequency');
%         
%         % Display the dominant frequency on the plot
%         if ~isnan(dom_freq)
%             text(dom_freq, Y(f == dom_freq), sprintf(' %.2f Hz', dom_freq), 'VerticalAlignment', 'bottom');
%         end
%         hold off;
%     end
    
    % Find the index of the maximum FFT value
    [~, idx] = max(abs(Y));
    
    % Return the dominant frequency
    dom_freq = f(idx);    

    if plot_dom_freq
        % Plot the magnitude spectrum
        figure("Name",fig_name);
        plot(f, abs(Y));
        hold on;
        
        % Highlight the dominant frequency
        plot(f(idx), abs(Y(idx)), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
        
        % Annotate the plot
        title('Magnitude Spectrum with Dominant Frequency');
        xlabel('Frequency (Hz)');
        ylabel('|FFT|');
        grid on;
        legend('Magnitude Spectrum', 'Dominant Frequency');
        
        % Display the dominant frequency on the plot
        text(f(idx), abs(Y(idx)), sprintf(' %.2f Hz', dom_freq), 'VerticalAlignment', 'bottom');
        hold off;
    end
end
