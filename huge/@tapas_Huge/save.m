function [ ] = save( filename, obj, varargin )
% Save properties of HUGE object to mat file.
% 
% INPUTS:
%   filename - File name.
%   obj      - A tapas_Huge object.
% OPTIONAL INPUTS:
%   Names of property to be saved. See examples below.
%
% EXAMPLES:
%   SAVE(filename,obj)    Save properties of 'obj' as individual variables
%       file specified in 'filename'.
% 
%   SAVE(filename,obj,'options','posterior','prior')    Only save the
%       'options', 'posterior' and 'prior' properties of 'obj'.
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




K           = obj.K;         %#ok<*NASGU>
L           = obj.L;
M           = obj.M;        
N           = obj.N;        
R           = obj.R;        
idx         = obj.idx;      
dcm         = obj.dcm;      
inputs      = obj.inputs;
data        = obj.data;
labels      = obj.labels;
options     = obj.options;  
prior       = obj.prior; 
posterior   = obj.posterior; 
trace   	= obj.trace; 
aux         = obj.aux;
model       = obj.model;
    
if isempty(varargin)
    varargin = {'K', 'L', 'M', 'N', 'R', 'idx', 'dcm', 'inputs', 'data', ...
        'labels', 'options', 'prior', 'posterior', 'trace', 'aux', 'model'};
end

save(filename, varargin{:}, '-v7.3');

end

