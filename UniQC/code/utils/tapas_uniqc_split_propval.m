function propvalSplitArray = tapas_uniqc_split_propval(propvalArray, nSplits)
% Splits PropertyName/Value pairs with cell/multi-element inputs into
% several varargins to be used separately
%
%   vararginSplit = tapas_uniqc_split_propval(propvalArray, nSplits)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_uniqc_split_propval
%
%   See also

% Author:   Lars Kasper
% Created:  2016-01-28
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% split varargins...some into cells, some into
% split variable arguments, leave strings as is, for they
% indicate the argument...
if nargin < 2
    % automatically figure it out from length of longest cellfun
    nSplits = max(cell2mat(cellfun(@numel, propvalArray(2:2:end), ...
        'UniformOutput', false)));
end


propvalSplitArray = cell(nSplits,1);
nVarargins = numel(propvalArray);
for d = 1:nSplits
    propvalSplitArray{d} = cell(1,nVarargins);
    for iArg = 1:nVarargins
        
        currentArg = propvalArray{iArg};
        
        % string => take it!
        if ischar(currentArg)
            currentArgDim = currentArg;
            
            % cell => take corresponding element
        elseif iscell(currentArg)
            currentArgDim = currentArg{d};
            
        elseif isnumeric(currentArg)
            % if one element =>  take it!
            if numel(currentArg) == 1
                currentArgDim = currentArg;
            else
                % otherwise copy corresponding element(s) from dimension
                % first dimension that matches nSplits
                iSplitDim =  find(nSplits == size(currentArg), 1);
                nDims = ndims(currentArg);
                dimReorderArray = [iSplitDim, setdiff(1:nDims, iSplitDim)];
                % take (1,:), permute it first
                currentArgDim = permute(currentArg, dimReorderArray);
                currentArgDim = squeeze(currentArgDim(d,:));
            end
        end
        
        propvalSplitArray{d}{iArg} = currentArgDim;
    end
end

