function varargout = tapas_uniqc_example_slider4d()
%creates 4D dataset numerically and runs tapas_uniqc_slider4d to show capability of it
%
%    tapas_uniqc_example_slider4d()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_uniqc_example_slider4d
%
%   See also

% Author: Lars Kasper
% Created: 2013-05-15
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.


Y = tapas_uniqc_create_shepp_logan_4d();
nSli = 1;
tapas_uniqc_slider4d(Y, @tapas_uniqc_plot_image_diagnostics, nSli);

if nargout > 1
    varargout{1} = Y;
end