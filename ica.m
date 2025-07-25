function [W, Zhat] = ica(X, Nsources, Wprev)
    if nargin < 3
        Wprev = 0;
    end

    [nRows, nCols] = size(X);
    if nRows > nCols
        error('The number of rows cannot be greater than the number of columns. Please transpose input.');
    end

    if Nsources > min(nRows, nCols)
        Nsources = min(nRows, nCols);
        warning('The number of sources cannot exceed the number of observation channels. The number of sources will be reduced to %d.', Nsources);
    end

    [Winv, Zhat] = jade(X, Nsources, Wprev);
    W = pinv(Winv);
end