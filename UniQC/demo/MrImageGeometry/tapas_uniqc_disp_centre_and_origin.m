function disp_centre_and_origin(obj)
% Displays centre and origin of MrImage object. Helper for demo script.
%
%   disp_centre_and_origin(input);
%
% IN    MrImage object
%
% EXAMPLE
%   disp_centre_and_origin(data)
%
%   See also demo_set_geometry

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-11-06
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

disp(['centre: ', num2str(obj.dimInfo.center(1)), 'mm ', ...
    num2str(obj.dimInfo.center(2)), 'mm ', num2str(obj.dimInfo.center(3)), 'mm']);
disp(['origin: ' num2str(obj.geometry.get_origin()')]);
end