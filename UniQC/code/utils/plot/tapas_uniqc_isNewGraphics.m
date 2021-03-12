function isNew = tapas_uniqc_isNewGraphics()
% returns true, if Matlab version is R2014b or new (different handling of
% figure handles)
%
%   isNew = tapas_uniqc_isNewGraphics()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_uniqc_isNewGraphics
%
%   See also

% Author: Lars Kasper
% Created: 2014-11-05
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.

try
  v = ver('MATLAB');
    version = str2double(v.Version);
    isNew = version >=8.4;
catch % If you don't understand this command, you must be older
    isNew = false;
end