function this = apply_deformation_field(this, fileNameDeformationField, varargin)
% Applies a previoulsy estimated deformation field
%
%   Y = MrImage()
%   deformation_field = MrImage;
%   Y.apply_deformation_field(deformation_field)
%
% This is a method of class MrImage.
%
% IN
% MrImage object containing the deformation field
%
% OUT
%
% EXAMPLE
%   apply_deformation_field
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-10
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


this.save(this.get_filename('prefix', 'raw'));
matlabbatch = this.get_matlabbatch('apply_transformation_field', ...
    fileNameDeformationField, varargin{:});
save(fullfile(this.parameters.save.path, 'matlabbatch.mat'), ...
            'matlabbatch');
spm_jobman('run', matlabbatch);
% clean up: move/delete processed spm files, load new data into matrix

this.finish_processing_step('apply_transformation_field');