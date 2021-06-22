function [X, Y, DCM, args] = tapas_rdcm_create_regressors(DCM, options)
% [X, Y, DCM, args] = tapas_rdcm_create_regressors(DCM, options)
% 
% Transforms intial DCM signal into a set of regressors X (design matrix)
% and Y (data)
% 
%   Input:
%   	DCM         - model structure
%       options     - estimation options
%
%   Output:
%   	X           - design matrix (predictors)
%       Y           - data
%   	DCM         - either model structure or a file name
%       args        - output arguments
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


%% preparations

% circular shift of the stimulus input function
temp    = circshift(DCM.U.u, options.u_shift);
DCM.U.u = temp;

% unwrapping DCM
y        = DCM.Y.y;
u        = full(DCM.U.u);
u_dt   	 = DCM.U.dt;
y_dt     = options.y_dt;
r_dt     = y_dt / u_dt;
[Nu, nu] = size(u);
[Ny, nr] = size(y);

% complex i
ic = sqrt( - 1);


%% create regressors

% Fourier transform of hemodynamic response function (HRF)
h_fft    = fft(options.h);
h_fft_tr = fft(options.h(1:r_dt:end));

% Fourier transform of BOLD signal
y_fft = fft(y);

% convolution of stimulus input and hemodynamic response function
u = ifft(fft(u).* repmat(h_fft, 1, nu));


% if empty, set constant confound (task) or no confound (rest)
if ( ~isfield(DCM.U,'X0') )
    
    % no inputs to filter for resting-state state
    if ( ~strcmp(DCM.U.name{1},'null') )
        DCM.U.X0      = ones(size(DCM.U.u,1),1);
        options.filtu = 1;
    else
        DCM.U.X0      = zeros(size(DCM.U.u,1),0);
        options.filtu = 0;
    end
    
end

% add confounds (e.g, constant, linear trend, sinusoids)
u = [u, DCM.U.X0];


% interpolation (upsampling) of BOLD data (or not)
if options.padding
    y_fft(round(Ny / 2) + 1, : ) = y_fft(round(Ny / 2) + 1, : ) / 2;
    y_fft = r_dt*[y_fft(1 : round(Ny / 2) + 1, : ); zeros(Nu - Ny - 1, nr); y_fft(round(Ny / 2) + 1 : end, : )];
    Ny    = Nu;
    u_fft = fft(u/16);
else
    if r_dt > 1
        u     = u(1 : r_dt : end, : );
        h_fft = fft(options.h(1 : r_dt : end, : ));
    end
    u_fft = fft(u);
end

% filtering -> find informative frequencies
if ( options.filter_str ~= 0 )
    [y_fft, idx] = tapas_rdcm_filter(y_fft, u_fft, h_fft, size(y, 1), options);
else
    idx = ones(size(y_fft));
end

% coefficients to transform fourier of the function to the fourier of the derivative
coef = exp(2 * pi * ic * (0 : Ny - 1)' / Ny) - 1;

% derivative ~ finite difference
yd_fft = repmat(coef, 1, nr).* y_fft / y_dt;

% filtered out frequencies do not go into regression
yd_fft(~idx) = NaN;

% bilinear term (not supported in present version)
yu_fft = zeros(Ny, nr * (nu+size(DCM.U.X0,2)));
if isfield(options, 'bilinear') && options.bilinear
    y_unconv = ifft(y_fft./ repmat(h_fft_tr, 1, nr));
    for i = 1 : nr
        for j = 1 : (nu+size(DCM.U.X0,2))
            yu_fft( : , (j - 1) * nr + i) = fft(y_unconv( : , i).* u( : , j)).* h_fft;
        end
    end
end


%% combine regressors

% create the X matrix
if strcmp(options.type,'r')
    X = [y_fft yu_fft u_fft];
else
    if ( isfield(options,'scale_u') )
        X = [y_fft yu_fft u_fft];
    else
        X = [y_fft yu_fft u_fft/(r_dt)];
    end
end

% create the Y matrix
Y = yd_fft;
Y = tapas_rdcm_reduce_zeros(X, Y);


% compute autocovariance for bilinear terms (not supported in present version)
P = [];


% define output arguments
args.P      = P;
args.r_dt   = r_dt;
args.type   = options.type;
args.evalCp = options.evalCp;

end
