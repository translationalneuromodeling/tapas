function [correlation,x,y] = tapas_physio_corrcoef12(x,y, isZtransformed,precisionThreshold)
% computes correlation coefficient (i.e. entry (1,2) of correlation matrix)
% quickly between two time series
% The speed-up is mainly due to in-line re-implementation of costly
% statistical functions, i.e. mean, std, cov, and usage of a pre-applied
% z-transformation of either of the inputs
%
%   [correlation,x,y] = tapas_physio_corrcoef12(x,y, isZtransformed,precisionThreshold)
%
%
% IN
%   x               [nSamples,1] column vector of samples
%   y               [nSamples,1] column vector of samples
%   isZtransformed [1,2] vector stating whether x,y or both are
%                   z-transformed, i.e. mean-corrected and divided by their
%                   standard deviation
%                   example:
%                   isZtransformed = [1,0] means that x is z-transformed,
%                   by y is not, i.e. (y-mean(y))/std(y) will be computed
%   precisionThreshold [1,1] If x or y are z-transformed and the standard 
%                   deviation is smaller than precisionThreshold, then the
%                   the result will be set to NaN to avoid numeric
%                   instabilities.
%                   
% OUT
%   correlation     correlation(1,2) of correlation = corrcoef(x,y)
%   x               z-transformed x
%   y               z-transformed y
% EXAMPLE
%   tapas_physio_corrcoef12
%
%   See also

% Author: Lars Kasper
% Created: 2014-08-14
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if nargin < 3
    isZtransformed = [0 0];
end
if nargin < 4
    precisionThreshold = eps('single');
end

% This is the old implementation; uncomment for comparison purposes
%doUseSlow = false;
%if doUseSlow
%    correlation = corrcoef(x,y);
%    correlation = correlation(1,2);
%else %fast, using shortcuts and in-line implementationf of mean/std/cov...


%C(i,j)/SQRT(C(i,i)*C(j,j)).

correlation = 0;
if any(x) && any(y) % all-zero vectors result in zero correlation
    
    nSamples = numel(x);
    normFactor = 1/(nSamples-1);
    
    % make column vectors
    if size(x,1) ~= nSamples
        x = x(:);
        y = y(:);
    end
    
    if ~isZtransformed(1) % perform z-transformation
        x = x - sum(x)/nSamples;
        sig = sqrt(x'*x*normFactor);
        if sig < precisionThreshold
            sig = NaN;
        end
        x = x./sig;
        %x = x./sqrt(x'*x*normFactor);
        %x = x./sqrt(x')./sqrt(x)./normFactor;
    end
    if ~isZtransformed(2) % perform z-transformation
        y = y - sum(y)/nSamples;
        sig = sqrt(y'*y*normFactor);
        if sig < precisionThreshold
            sig = NaN;
        end
        y = y./sig;
    end
    
    if numel(x) == numel(y)
        correlation = x'*y*normFactor;
        % otherwise, correlation stays zero
    end
end

%end % else doUseSlow