function [ balancedPurity ] = tapas_huge_bpurity( labels, estimates ) 
% Calculate balanced purity (see Brodersen2014 Eq. 13 and 14) for a set of
% ground truth labels and a set of estimated labels
% 
% INPUTS:
%   labels    - Vector of ground truth class labels.
%   estimates - Clustering result as array of assignment probabilities or
%               vector of cluster indices.
% 
% OUTPUTS:
%   balancedPurity - Balanced purity score according to Brodersen (2014)
% 
% EXAMPLES:
%   bp = TAPAS_HUGE_BPURITY(labels,estimates)
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 


labels = labels(:);
if numel(estimates) == numel(labels)
    estimates = estimates(:);
end
[N, K] = size(estimates);
assert(N == numel(labels), 'TAPAS:UTIL:InputSizeMismatch', ...
    'Number of rows in estimates must match number of elements in labels.');
if K > 1
    [~, estimates] = max(estimates, [], 2);
else
    assert(all(mod(estimates, 1) == 0) && all(estimates > 0), ...
        'TAPAS:UTIL:NonIntegerIndices', ...
        'Cluster indices must be positive integer.');
    K = max(estimates);
end

% number of classes
C = max(max(labels), K);

% degree of imbalance
xi = zeros(C, 1);
for c = 1:C
    xi(c) = nnz(labels == c)/N;
end
xi = max(xi);

% calculate purity (Brodersen2014 Eq. 13)
counts = zeros(C,C);
for k = 1:C
    % class labels of samples grouped into cluster k
    currentLabels = labels(estimates == k);
    for c = 1:C
        % number of samples belonging to class c
        counts(c,k) = nnz(currentLabels == c);
    end
end
purity = sum(max(counts))/N;

% Brodersen2014 Eq. 14 (n->k)
balancedPurity = (1-1/C)*(purity - xi)/(1 - xi) + 1/C;

end

