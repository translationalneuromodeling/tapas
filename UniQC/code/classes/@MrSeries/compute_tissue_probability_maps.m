function this = compute_tissue_probability_maps(this)
% Computes tissue probability maps
%
%   Y = MrSeries()
%   Y.compute_tissue_probability_maps(inputs)
%
% This is a method of class MrSeries.
%
% IN
%
% OUT
%
% EXAMPLE
%   compute_tissue_probability_maps
%
%   See also MrSeries MrImage segment spm_preproc

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-22
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

handleInputImage = this.find('MrImage', 'name', ...
    ['^' this.parameters.compute_tissue_probability_maps.nameInputImage '*']);% find input image...
inputImage = handleInputImage{1}.copyobj;
tissueTypes = this.parameters.compute_tissue_probability_maps.tissueTypes;

mapOutputSpace              = 'native';
deformationFieldDirection   = 'both';
applyBiasCorrection         = false;

this.init_processing_step('compute_tissue_probability_maps', inputImage);

[~, this.tissueProbabilityMaps, deformationFields, biasField] = ...
   inputImage.segment('tissueTypes', tissueTypes, 'mapOutputSpace', mapOutputSpace, ...
   'deformationFieldDirection', deformationFieldDirection, ...
       'saveBiasField', applyBiasCorrection);
   
%% check if additional images has deformationFields/biasField already
% if yes, overwrite, if no, create new additional image

namesField = {
 'forwardDeformationField'
 'backwardDeformationField'
 'biasField'
};

createdFields = [deformationFields; biasField];

nImages = numel(namesField);

for iImage = 1:nImages
   nameImage = sprintf('%s (%s)', namesField{iImage}, inputImage.name);
   createdFields{iImage}.name = nameImage;
   handleImage = this.find('MrImage', 'name', nameImage);
   
   if ~isempty(handleImage)
       overwrite = 2; % overwrite everything, including empty values
       handleImage{1}.update_properties_from(createdFields{iImage}, overwrite);
   else
       this.additionalImages{end+1,1} = createdFields{iImage};
   end
end


%% finish processing by deleting obsolete files, depending on save-parameters

this.finish_processing_step('compute_tissue_probability_maps', ...
    createdFields, inputImage);