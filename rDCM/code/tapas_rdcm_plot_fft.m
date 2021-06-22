function tapas_rdcm_plot_fft(y, region_id)
% tapas_rdcm_plot_fft(y, region_id)
% 
% Plots signals in both frequency and time domains.
% 
%   Input:
%   	y               - data
%       region_id       - region index
%
%   Output:
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


% select a specific region
if ( nargin > 1 )
    y = y(:,region_id);
end

% dimensionality of data
[N, nr] = size(y);

% inverse, yi should be real
yi = ifft(y);

% signal is almost real
if sum(abs(imag(y(:)))) < 10^(-10)
    yi = y;
    y = fft(yi);
end

% concatenate time domain and frequency domain data
y = y(:);
yi = yi(:);


% plot the figure
figure;

% frequency domain
subplot(1,2,1);
plot([real(y) imag(y)]); hold on;
plot_delimiter([real(y) imag(y)], N, nr);
legend('Real part', 'Imaginary part');
title('Frequency domain');

% time domain
subplot(1,2,2);
if sum(abs(imag(yi(:)))) < 10^(-10) % numerical correction
    plot(real(yi)); hold on;
    plot_delimiter(real(yi), N, nr);
    legend('FFT inverse');
    title('Time domain');
else
    title('Fourier spectrum is corrupted and cannot be inversed');
end

end
