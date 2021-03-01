function bundle_struct(nameStruct, varargin)
%bundles variables into a structure in the calling function workspace
%
%   bundle_struct(nameStruct, varargin)
%
% IN
%   nameStruct  string of structure variable to be created
%   varargin    strings of all variables to be included in structure
% OUT
%
% EXAMPLE
%   n=5;x=10; myBal = 'haha';
%   bundle_struct('opts', 'n', 'x', 'myBal');
%   whos opts
%
%   See also strip_fields propval

% Author: Lars Kasper
% Created: 2013-11-13
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.

for i=1:nargin-1
    evalin('caller', [nameStruct '.' varargin{i} '=' varargin{i} ';']);
end