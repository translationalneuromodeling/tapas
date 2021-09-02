function [y_sum,y_mean] = tapas_logsumexp(x)


%% -------------------------------------------------------------------------------------------
% [y_sum,y_mean] = tapas_logsumexp(x) takes the values in x, exponates
% them, then takes the sum over the column, and finally applies the natural logarithm.
% The calculation uses the "log-sum-exp" trick: See e.g. http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
% The function also returns the log-mean-exp.
%---------------------------------------------------------------------------------------------
% INPUT:
%       x       - A column vector or matrix of values. All computations are
%       made along the direction of a column.
% 
% Optional:
%
%--------------------------------------------------------------------------------------------
% OUTPUT: 
%       y_sum   - The log-sum-exp of all columns of x.
%       y_mean  - The log-mean-exp of all columns of x.
%           
% Author:  Jakob Heinzle, TNU, UZH & ETHZ - April, 2021
%
% REVISION LOG:
%
%      Jakob Heinzle, 2021/04/16: new function 
%
%%
 
sz = size(x);

if numel(sz)~=2
    error('Input x needs to be a matrix of 2 dimensions');
end

max_x = max(x); %compute maximum of each column
y_sum = max_x + log(sum(exp(x-ones(sz(1),1)*max_x)));

if nargout==2
y_mean = y_sum-log(sz(1)); % compute mean if necessary.
end

return;