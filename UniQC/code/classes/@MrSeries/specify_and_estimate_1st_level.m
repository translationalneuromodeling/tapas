function this = specify_and_estimate_1st_level(this)
% Specifies the 1st level design of the MrSeries.
% Uses SPM's specify 1st level function.
%
%   Y = MrSeries()
%   Y.specify_1st_level(inputs)
%
% This is a method of class MrSeries.
%
% IN
%
% OUT
%
% EXAMPLE
%   specify_1st_level
%
%   See also MrSeries

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-07
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% init processing step
this.init_processing_step('specify_and_estimate_1st_level');

% set general glm save path
this.glm.parameters.save.path = this.data.parameters.save.path;
% init glm using MrGlm; check if regressors and/or conditions are specified
[~, hasRegressors, hasConditions] = this.glm.init_glm;
[~, matlabbatch] = this.glm.get_matlabbatch('specify_1st_level');
if ~hasRegressors
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {''};
end
if ~hasConditions
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
end
% add data to the matlabbatch from MrSeries
matlabbatch = this.get_matlabbatch('specify_and_estimate_1st_level', matlabbatch);
% save matlabbatch
save(fullfile(this.data.parameters.save.path, 'matlabbatch.mat'), ...
    'matlabbatch');
% run matlabbatch
spm_jobman('run', matlabbatch);
% clean up/data sort
this.finish_processing_step('specify_and_estimate_1st_level', this.data);