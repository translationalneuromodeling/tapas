function [this, matlabbatch] = get_matlabbatch(this, module, varargin)
%ONE_LINE_DESCRIPTION
%
%   Y = MrGlm()
%   Y.get_matlabbatch(inputs)
%
% This is a method of class MrGlm.
%
% IN
%   module      'specify_1st_level'
%   varargin    limited set of options to be determined for each module
%
% OUT
% matlabbatch   spm matlabbatch that is executed when the module is
%               performed
%               can be scrutinized via spm_jobman('interactive',
%               matlabbatch)
%
%
% EXAMPLE
%   get_matlabbatch('specify_1st_level');
%
%   See also MrGlm

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-04-20
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


pathThis = fileparts(mfilename('fullpath'));
fileMatlabbatch = fullfile(pathThis, 'matlabbatch', ...
    sprintf('mb_%s.m', module));
run(fileMatlabbatch);

switch module
    case 'specify_1st_level'
        
        % set SPM directory
        spmDirectory = fullfile(this.parameters.save.path, ...
            this.parameters.save.spmDirectory);
        matlabbatch{1}.spm.stats.fmri_spec.dir = {spmDirectory};
        
        % set timing from MrGLM
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = ...
            this.timingUnits;
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = ...
            this.repetitionTime;
        
        % add multiple conditions
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi = ...
            cellstr(fullfile(this.parameters.save.path, 'Conditions.mat'));
        
        % add multiple regressors
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = ...
            cellstr(fullfile(this.parameters.save.path, 'Regressors.mat'));
        
        % set hrf derivatives
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = ...
            this.hrfDerivatives;
        
        % set masking threshold
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = this.maskingThreshold;
        
        % set explicit brain mask
        matlabbatch{1}.spm.stats.fmri_spec.mask = {this.explicitMasking};
        
        % set serial correlations model
        matlabbatch{1}.spm.stats.fmri_spec.cvi = this.serialCorrelations;      
end
