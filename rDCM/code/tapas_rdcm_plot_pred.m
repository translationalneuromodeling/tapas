function tapas_rdcm_plot_pred(Ep, Y, X, region_id)
% tapas_rdcm_plot_pred(Ep, Y, X, region_id)
% 
% Plots Yd_fft vs the linear combination of the 'true' (according to the
% given model) regressors and parameter values for region_id (if it is 
% specified) in frequency (separate real and imaginary) and time domain.
% 
%   Input:
%   	Ep              - parameter values
%       Y               - data
%       X               - design matrix (predictors)
%       region_id       - region number to be plotted
%
%   Output:
%       options         - estimation options
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


% check whether the dimensions are the same
assert(size(Y,1)==size(X,1), 'Dimensions of signals are not consistent');

% number of regions and inputs
[nr, nu] = size(Ep.C);

% get target and predictor variables
tg = Y;
pr = X*[Ep.A zeros(nr,nr*nu) Ep.C]';

% select a specific region
if nargin > 3
    tg = tg(:,region_id);
    pr = pr(:,region_id);
end

% get finite values
idx      = ~isnan(tg);
tg(~idx) = 0;
pr(~idx) = 0;

% 
tgi = ifft(tg);
pri = ifft(pr);
dfi = tgi - pri;
df  = tg - pr;

% 
idx = sum(idx,2) > 0;
df  = df(idx,:);
dfi = dfi(idx,:);

figure;
if any(sum(~isreal(df(:)))) % input is complex
    
    sp = tight_subplot(1,3);
    % real part
    axes(sp(1));
    plot(real([tg(:) pr(:)])); hold on;
    legend('Target', 'Prediction');
    title(['Real part  ' num2str(mean(real(df(:)).^2)) '\newline' num2str(mean(real(df).^2))]);
    
    % imaginary part
    axes(sp(2));
    plot(imag([tg(:) pr(:)])); hold on;
    legend('Target', 'Prediction');
    title(['Imaginary part  ' num2str(mean(imag(df(:)).^2)) '\newline' num2str(mean(imag(df).^2))]);
    
    % time domain
    axes(sp(3));
    if sum(abs(imag(dfi))) < 10^(-10)  % corruptness is neglectable
        plot(real([tgi(:) pri(:)])); hold on;
        legend('Target', 'Prediction');
        title(['Time domain ' num2str(mean(real(dfi(:)).^2)) ' \newline' num2str(mean(real(dfi).^2))]);
    else
        title('Fourier spectrum is corrupted and cannot be inversed');
    end
else
    plot([tg(:) pr(:)]);
    legend('Target', 'Prediction');
    title(['Time domain ' num2str(mean(df(:).^2)) '\newline' num2str(mean(df.^2))]);
end
