function varargout = example_slider4d()
%creates 4D dataset numerically and runs slider4d to show capability of it
%
%    example_slider4d()
%
% IN
%
% OUT
%
% EXAMPLE
%   example_slider4d
%
%   See also

% Author: Lars Kasper
% Created: 2013-05-15
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.


Y = create_shepp_logan_4d();
nSli = 1;
slider4d(Y, @plot_image_diagnostics, nSli);

if nargout > 1
    varargout{1} = Y;
end