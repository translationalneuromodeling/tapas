function pathName = tapas_uniqc_simplify_path(pathName)
% Simplifies path name string by replacing all relative paths in it, i.e. ..\, .\
%
%  pathName = tapas_uniqc_simplify_path(pathName)
%
% IN
%   pathName    string with relative path symbols, e.g.
%               'C:\bla\recon\test\..\code\classes\..\..\test\.\'
% OUT
%   pathName    string, simplified path name, e.g.
%               'C:\bla\recon\test\'

% Author: Lars Kasper
% Created: 2014-08-19
%
% EXAMPLE
%   tapas_uniqc_simplify_path
%
%   See also tapas_uniqc_get_path

% Author:   Lars Kasper
% Created:  2014-11-18
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% replace remove bla\..\  or bla/../ as long as they are in path name

while regexp(pathName, '(\\|/)\.\.')
    pathName = regexprep(pathName, '[^\\/\.]+(\\|/)\.\.(\\|/)', '', 'once');
end



% bluntly remove .\ and ./
pathName = regexprep(pathName, '\.\\', '');
pathName = regexprep(pathName, '\./', '');