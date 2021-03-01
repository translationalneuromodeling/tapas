function this = plot_design_matrix(this)
% plots the design matrix scaled for display
%
%   Y = MrGlm()
%   Y.plot_design_matrix()
%
% This is a method of class MrGlm.
%
% IN
%
% OUT
%
% EXAMPLE
%   plot_design_matrix
%
%   See also MrGlm

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-08
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


figure;
imagesc(this.designMatrix);
colormap gray;
title('design matrix');

end