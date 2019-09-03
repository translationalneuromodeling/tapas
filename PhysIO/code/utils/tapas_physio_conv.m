function [w] = tapas_physio_conv(u, v, filter_type, padding)
% Wrapper around `conv()` for convolution
%   Deals with the padding and time offsets
%
%    [w] = tapas_physio_filter_cardiac(u, v, filter_type)
%
% IN
%   u           Data time series [1-D]
%   v           Convolutional filter time series [1-D]
%   filter_type ['causal', 'symmetric'].
%               If 'causal', the filter is taken to be defined for
%               `t >= 0`, with the first element corresponding to `t=0`.
%               If 'symmetric', the filter is taken to be defined for both
%               positive and negative time, with the central element
%               corresponding to `t=0`.
%               N.B. 'symmetric' implies an *odd* filter length.
%   padding     ['mean', 'zero']. Whether to pad with the mean of `u`, or
%               with zeros.
%
% OUT
%   w           Convolved time series [size(u)]

% Author:   Sam Harrison
% Created:  2019-07-11
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

if ~isvector(u) || ~isvector(v)
    error('tapas_physio_conv: Both inputs must be vectors')
end
% Padding: use data mean to reduce transients by default
if nargin < 4
    padding = 'mean';
end

% Pad value
switch lower(padding)
    case 'mean'
        pad_val = mean(u);
    case 'zero'
        pad_val = 0.0;
    otherwise
        error('Unrecognised padding argument (%s)', padding)
end

% Pad shape
switch lower(filter_type)
    case 'causal'
        u_pad = [pad_val * ones(length(v)-1, 1); u(:)];
    case 'symmetric'
        if mod(length(v), 2) == 1
            pad_elem = (length(v) - 1) / 2;
            u_pad = [pad_val * ones(pad_elem, 1); u(:); pad_val * ones(pad_elem, 1)];
        else
            error('Symmetric filter lengths must be odd!')
        end
    otherwise
        error('Unrecognised padding argument (%s)', filter_type)
end

% Apply convolution and select portion of interest
w = conv(u_pad, v, 'valid');
w = reshape(w, size(u));

end