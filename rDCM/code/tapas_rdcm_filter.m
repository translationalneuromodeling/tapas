function [y_fft, idx] = tapas_rdcm_filter(y_fft, u_fft, h_fft, Ny, options)
% [y_fft, idx] = tapas_rdcm_filter(y_fft, u_fft, h_fft, Ny)
% 
% Specifies informative frequencies and filters the Fourier-transformed 
% signal
%
%   Input:
%       y_fft       - intial Fourier-transformed signal
%       u_fft       - Fourier-transformed driving inputs
%       h_fft       - Fourier-transformed HRF
%       Ny          - intial length of signal (necessary for padding detection)
%       options     - estimation options
%
%   Output:
%       y_fft       - filtered Fourier-transformed signal
%       idx         - frequencies to include in regression
%
 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2021 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% dimensionality of the data
[N, nr] = size(y_fft);


% set precision
prec = 10^(-4);

% freq which are not smoothed out by convolution
h_idx = abs(h_fft) > prec;

% freq which are non-zero due to the input structure
if ( options.filtu == 1 )
    u_idx = sum(abs(u_fft),2) > prec;
else
    u_idx = ones(size(u_fft,1),1);
end

% padding detection
if ( N ~= Ny )
    y_idx = [ones(1+round(Ny/2),1); zeros(N-Ny-1,1); ones(Ny/2,1)];
else
    y_idx = ones(N,1);
end

% indices where h, u and y are present
str_idx = h_idx & u_idx & y_idx;


%% filtering by comapring to noise variance

% relative threshold of signal compare to noise
thr = options.filter_str;

% real and imaginary signal
y_real = real(y_fft);
y_imag = imag(y_fft);

% get the standard deviation of the signal
if ( sum(~str_idx) > 1 )
    std_real = repmat(std(y_real(~str_idx,:)), N, 1);
    std_imag = repmat(std(y_imag(~str_idx,:)), N, 1);
else
    std_real = zeros(N,nr);
    std_imag = zeros(N,nr);
end

% find indices where real or imaginary signal is above threshold
idx = abs(y_real) > thr*std_real | abs(y_imag) > thr*std_imag;


%% high-pass filter

% specify the frequency
if ( options.filtu == 1 )
    hpf  = 16;
    freq = round(7*N/(hpf));
else
    hpf  = max(16 + (thr-1)*4,16);
    freq = round(7*N/(hpf));
end

% high-pass filtering
idx_freq = [ones(1+freq, nr); zeros(N - 2*freq - 1, nr); ones(freq, nr)];
idx      = idx & repmat(str_idx, 1, nr) & idx_freq;


%% freq should be present due to fft constraints

% constant frequency
idx(1,:) = 1;

% symmetricity
idx = idx | idx([1 end:-1:2],:);

% filter the data
y_fft(~idx) = 0;


%% freq to include in regression (zeros in informative regions should be also included)

% iterate over regions
for i = 1:nr
    
    % last freq in first half of the spectrum
    first  = find(idx(1:round(N/2)+1,i),1,'last');
    
    % first frequency in second half of the spectrum
    second = find([zeros(round(N/2),1); idx(round(N/2)+1:end,i)], 1, 'first');
    
    % set back to one
    idx(1:first,i)    = 1;
    idx(second:end,i) = 1;
    
end

end