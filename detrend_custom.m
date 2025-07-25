%% signal detrending
function filtered_signal = detrend_custom(input_signal, lambda_value)
    signal_length = length(input_signal);
    % observation matrix
    H = eye(signal_length);
    ones_vec = ones(signal_length, 1);
    minus_twos = -2 * ones(signal_length, 1);
    diags_data = [ones_vec, minus_twos, ones_vec];
    diags_index = [0, 1, 2];
    D = spdiags(diags_data, diags_index, signal_length - 2, signal_length);
    filtered_signal = (H - inv(H + (lambda_value ^ 2) * (D' * D))) * input_signal;
end
