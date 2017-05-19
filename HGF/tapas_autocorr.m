function y = tapas_autocorr(x)
% USAGE:
%     Y = tapas_autocorr(X)
%
% INPUT:
%     X - n-by-m matrix of m time series (columns) of length n
%
% OUTPUT:
%     Y - n-by-m matrix with m columns of autocorrelation coefficients for lags n-1 
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Length of time series
n = size(x,1);

% De-mean time series
x = x - ones(size(x))*diag(mean(x));

% Get the autocovariance
f = fft(x);
fsq = f.*conj(f);
y = ifft(fsq)/n;

% Get the autocorrelation (the next line is equivalent to y = y*diag(1./y(1,:));)
y = y*diag(1./var(x,1));

end
