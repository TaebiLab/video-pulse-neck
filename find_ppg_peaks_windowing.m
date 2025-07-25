function [refined_locs, refined_locs_track, dom_freq] = find_ppg_peaks_windowing(ppg_signal, ppg_green, fps, is_figure, fig_name)
    if nargin < 4
        is_figure = false;
        fig_name = "None";
    elseif nargin < 5        
        fig_name = "None";
    end
    % Apply Hilbert transform
    % analytic_signal = hilbert(ppg_signal);
    % upper_en = abs(analytic_signal);
    [upper_en, ~] = envelope(ppg_signal, 30, "analytic");
    
    % Compute the Hilbert transform on the upper envelope to get the lower envelope
    [~, lower_en] = envelope(upper_en, 10, 'peak');
    
    indx = find(ppg_signal<lower_en);
    ppg_temp = ppg_signal;
    %ppg_temp(indx) = lower_en(indx);
    ppg_temp(indx) = 0;
    
    % find dominant frequency of the signal
    if is_figure
        dom_freq = dominant_frequency(ppg_green, fps, true, fig_name+"_dom_freq"); % to plot domfreq
    else
        dom_freq = dominant_frequency(ppg_green, fps);
    end
      
    %min_peak_hight = mean(ppg_signal) * 1.5;
    %min_peak_hight = mean(ppg_temp) * 1.2;
    min_peak_dist = round(1/dom_freq * fps * 0.6);    % 
    %[pks, locs] = findpeaks(ppg_temp, 'MinPeakHeight', min_peak_hight, 'MinPeakDistance', min_peak_dist);
    [pks, locs] = findpeaks(ppg_temp, 'MinPeakDistance', min_peak_dist);

    % peak refinining
    expected_distance = round(1/dom_freq * fps);

    % Calculate the allowable range for distances (±25%)
    min_distance = round(0.75 * expected_distance);
    max_distance = round(1.25 * expected_distance);

    % Calculate RR intervals (distances between peaks)
    rr_intervals = diff(locs);

    % Identify valid intervals
    valid_intervals = (rr_intervals >= min_distance) & (rr_intervals <= max_distance);

    % Identify transitions
    starts = find(diff([0; valid_intervals]) == 1); % Start of each group of 1's
    ends = find(diff([valid_intervals; 0]) == -1);  % End of each group of 1's
    
    % Calculate lengths of each group
    lengths = ends - starts + 1;
    
    % Find the largest group
    [~, max_idx] = max(lengths);
    largest_start = starts(max_idx);
    largest_end = ends(max_idx);

    % Find the peaks and locs of the largest group
    largest_locs = [];
    largest_peaks = [];
    for i = largest_start:largest_end
        if ~ismember(locs(i), largest_locs)
            largest_locs(end+1) = locs(i); 
            largest_peaks(end+1) = pks(i);
        end
        % Add the next peak if the distance is within the threshold
        largest_locs(end+1) = locs(i+1); 
        largest_peaks(end+1) = pks(i+1);
    end


    % Initialize filtered peaks and locations
    filtered_locs = []; % Start with the first peak
    filtered_peaks = [];
    
    % Loop through RR intervals and filter peaks
    for i = 1:numel(rr_intervals)
        if rr_intervals(i) >= min_distance && rr_intervals(i) <= max_distance
            % Add the fist peak if not already in the array
            if ~ismember(locs(i), filtered_locs)
                filtered_locs(end+1) = locs(i); 
                filtered_peaks(end+1) = pks(i);
            end
            % Add the next peak if the distance is within the threshold
            filtered_locs(end+1) = locs(i+1); 
            filtered_peaks(end+1) = pks(i+1);
        end
    end
    
    filtered_locs = filtered_locs(:);
    filtered_peaks = filtered_peaks(:);

    if is_figure
%         figure;
%         plot(ppg_signal);
%         hold on;
%         plot(locs, pks, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5); % Original peaks
%         plot(filtered_locs, filtered_peaks, 'go', 'MarkerSize', 8, 'LineWidth', 1.5); % Filtered peaks
%         xlabel('Time (s)');
%         ylabel('Amplitude');
%         legend('iPPG Signal', 'Original Peaks', 'Filtered Peaks');
%         title('Filtered Peaks within ±25% of Expected Distances');
%         grid on;
    
        figure("Name",fig_name);
        plot(ppg_signal);
        hold on;    
        plot(locs, pks, 'go', 'MarkerSize', 8, 'LineWidth', 1.5); % Filtered peaks
        plot(largest_locs, largest_peaks, 'bo', 'MarkerSize', 8, 'LineWidth', 1.5); % Filtered peaks
    end    

    %% Start peak refinement from the largest group
%     % New expected distance based on largest group
%     expected_distance = mean(diff(largest_locs));
%     % New allowable range for distances (±30%)
%     min_distance = round(0.3 * expected_distance);
%     max_distance = round(1.3 * expected_distance);

    % initialize
    refined_locs = largest_locs;

    % track location of those peaks that are not detected and removed
    refined_locs_track_prev = [];

    % From the end of the largest group    
    start_loc = refined_locs(end);

    % Initialize window size which can adaptively change
    percentage_of_window = 0.25;
    window_size = 2 * expected_distance * percentage_of_window;
    window_increase_factor = 0.05; % 5% increment
    max_window_percentage = 0.40;

    peaks_not_in_one_cardiac_dis = false;

    while true
        window_start = start_loc + round(expected_distance-window_size/2);
        window_end = start_loc + round(expected_distance+window_size/2);

        if window_end > numel(ppg_signal)
            break;
        end
        
        [~, locs_window] = findpeaks(ppg_signal(window_start:window_end));   
        
        % Check if there are peaks in the window
        if ~isempty(locs_window)
            next_peak_loc = window_start + locs_window(1) - 1;                
            if peaks_not_in_one_cardiac_dis
                refined_locs_track_prev = [refined_locs_track_prev; refined_locs(end), next_peak_loc];
                if ~is_figure
                    peaks_not_in_one_cardiac_dis = false; % if not move it to figure section
                end
            end            

            if is_figure
                if ~peaks_not_in_one_cardiac_dis
                    % figure
                    xline(window_start, '--r');
                    xline(start_loc+expected_distance, '-g');
                    xline(window_end, '--r');
                    plot(next_peak_loc,ppg_signal(next_peak_loc), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
                    text(window_start+round(window_size/2), ppg_signal(next_peak_loc), ...
                        sprintf('%d%%',int32(percentage_of_window*100)), ...
                        'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'b');
                else
                    %% ToDo: need to correct the plot
                    plot(next_peak_loc,ppg_signal(next_peak_loc), 'r*', 'MarkerSize', 8, 'LineWidth', 1.5);
                    %xline(start_loc+round(2*expected_distance*0.75), '--k');
                    xline(start_loc+round(expected_distance), '--k');
                    peaks_not_in_one_cardiac_dis = false;
                end
            end

            % Append the next peak location to the refined list
            refined_locs = [refined_locs, next_peak_loc];
            start_loc = next_peak_loc;
            percentage_of_window = 0.25; % reset the window_size percentage
            window_size = 2 * expected_distance * percentage_of_window; % reset the window_size
        elseif round(percentage_of_window, 10) < round(max_window_percentage, 10)
            % Increase window size by 5%
            percentage_of_window = percentage_of_window + window_increase_factor;
            window_size = 2 * expected_distance * percentage_of_window;
        else            
            % max window size is reached
            %fprintf(fig_name+": Max window size reached \n \n");
            % skip the current peak and go to the next
            % apply two times of the window to get the next peak
            peaks_not_in_one_cardiac_dis = true;           
            percentage_of_window = 0.25; % reset the window_size percentage
            start_loc = start_loc+round(expected_distance); % skip current location
        end

    end


    % From the beginning of the largest group          
    ppg_signal_flipped = flip(ppg_signal);
    start_loc = length(ppg_signal_flipped) - refined_locs(1) + 1;   
    percentage_of_window = 0.25;
    window_size = 2 * expected_distance * percentage_of_window;

    peaks_not_in_one_cardiac_dis = false;
    
    while true
        window_start = start_loc + round(expected_distance-window_size/2);
        window_end = start_loc + round(expected_distance+window_size/2);

        if window_end > numel(ppg_signal_flipped)
            break;
        end
        
        [~, locs_window] = findpeaks(ppg_signal_flipped(window_start:window_end));
               
        % Check if there are peaks in the window
        if ~isempty(locs_window)            
            next_peak_loc_flipped = window_start + locs_window(1) - 1;                
            if peaks_not_in_one_cardiac_dis
                refined_locs_track_prev = [(length(ppg_signal) - next_peak_loc_flipped + 1), refined_locs(1); refined_locs_track_prev];
                if ~is_figure
                    peaks_not_in_one_cardiac_dis = false;  % if not moved it to figure section
                end
            end

            next_peak_loc = length(ppg_signal) - next_peak_loc_flipped + 1;
        
            if is_figure
                if ~peaks_not_in_one_cardiac_dis
                    % figure
                    xline(length(ppg_signal)-window_start+1, '--r');
                    xline((length(ppg_signal)-start_loc+1)-expected_distance, '-g');
                    xline(length(ppg_signal)-window_end+1, '--r');
                    plot(next_peak_loc,ppg_signal(next_peak_loc), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5)
                    text(length(ppg_signal)-window_end+1+round(window_size/2), ppg_signal(next_peak_loc), ...
                        sprintf('%d%%',int32(percentage_of_window*100)), ...
                        'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'b');
                else
                    %% ToDo: need to correct the plot
                    plot(next_peak_loc,ppg_signal(next_peak_loc), 'k*', 'MarkerSize', 8, 'LineWidth', 1.5)
                    %xline(length(ppg_signal)-(start_loc+round(2*expected_distance*0.75))+1, '--k');
                    xline(length(ppg_signal)-(start_loc+round(expected_distance))+1, '--k');
                    peaks_not_in_one_cardiac_dis = false;
                end

            end
    
            % Append the next peak location to the refined list
            refined_locs = [next_peak_loc, refined_locs];
            start_loc = next_peak_loc_flipped;
            percentage_of_window = 0.25; % reset the window_size percentage
            window_size = 2 * expected_distance * percentage_of_window; % reset the window_size            
        elseif round(percentage_of_window, 10) < round(max_window_percentage, 10)
            % Increase window size by 5%
            percentage_of_window = percentage_of_window + window_increase_factor;
            window_size = 2 * expected_distance * percentage_of_window;
        else
            % max window size is reached
            %fprintf(fig_name+": Max window size reached \n \n");
            % skip the current peak and go to the next
            % apply two times of the window to get the next peak
            peaks_not_in_one_cardiac_dis = true;
            percentage_of_window = 0.25; % reset the window_size percentage
            start_loc = start_loc+round(expected_distance); % skip current location
        end
        
        
    end

    % find the original position of the tracked peaks
    if ~isempty(refined_locs_track_prev)
        refined_locs_track_prev = refined_locs_track_prev(:,1)';
        % Find the locations
        [~, refined_locs_track] = ismember(refined_locs_track_prev, refined_locs);
    else
        refined_locs_track = [];
    end
    
end


