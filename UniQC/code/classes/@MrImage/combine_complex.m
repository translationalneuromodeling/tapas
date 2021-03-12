function outputImage = combine_complex(this)
% combines dimensions with complex parts (magnitude/phase or
% real/imaginary) into a complex array with N-1 dimensions
%
%   Y = MrImage()
%   outputImage = Y.combine_complex()
%
% This is a method of class MrImage.
%
% NOTE: The dimensions reflecting the complex parts are supposed to be
%       named 'complex_mp' or 'complex_ri' for magnitude/phase or
%       real/imaginary part representation, respectively;
%
% IN
%
% OUT
%
% EXAMPLE
%   combine_complex
%
%   See also MrImage MrImage.split_complex MrImage.smooth

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-22
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


isMagnitudePhase = ~isempty(this.dimInfo.get_dim_index('complex_mp'));

if isMagnitudePhase
    outputImage = this.select('complex_mp',1).* ...
        exp(this.select('complex_mp',2).*1i);
    outputImage.remove_dims('complex_mp');
else % real/imag
    outputImage = this.select('complex_ri',1) + ...
        this.select('complex_ri',2).*1i;
    outputImage.remove_dims('complex_ri');
end