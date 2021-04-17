function [ opts ] = tapas_huge_parse_inputs( opts, varargin )
% Parse name-value pair type arguments into a struct.
% 
% INPUTS:
%   opts - Struct containing all valid names as field names and 
%          corresponding default values as field values.
% 
% OPTIONAL INPUTS:
%   Name-value pair arguments.
% 
% OUTPUTS:
%   opts - Struct containing name-value pair input arguments as fields.
% 
% EXAMPLE:
%   opts = TAPAS_HUGE_PARSE_INPUTS(struct('a',0,'b',1),'a',10)    Specify
%       'a' and 'b' as valid property names and receive non-default value
%       for 'a'. 
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

if isscalar(varargin) && iscell(varargin{1})
    varargin = varargin{1};
end

nIn = numel(varargin);
validNames = fieldnames(opts);

for iIn = 1:2:nIn - 1
    
    name = varargin{iIn};
    value = varargin{iIn + 1};
    
    assert(ischar(name), 'TAPAPS:HUGE:Nvp:NoneChar', ...
        'Property name must be a character array.');
    assert(any(strcmpi(name, validNames)), 'TAPAPS:HUGE:Nvp:InvalidName', ...
        '"%s" is not a valid property name.', name);
    
    opts.(lower(name)) = value;
    
end

end

