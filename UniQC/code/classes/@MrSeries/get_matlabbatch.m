function matlabbatch = get_matlabbatch(this, module, varargin)
% Returns matlabbatch to perform spm-processing with an MrSeries.
% Fills out all necessary file parameters and options for different
% modules, e.g. specification of the first level design
%
% matlabbatch = get_matlabbatch(MrSeries, module, varargin)
%
% This is a method of class MrSeries.
%
% IN
%   module      'specify_and_estimate_1st_level'
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
%


switch module
    case 'specify_and_estimate_1st_level'
        % get matlabbatch
        matlabbatch = varargin{1};
        
        % add scans
        matlabbatch{1}.spm.stats.fmri_spec.sess.scans =  ...
            cellstr(spm_select('ExtFPList', this.data.parameters.save.path, ...
            ['^' this.data.parameters.save.fileName], Inf));
        
        % add estimation step
        % switch depending on mdoel estimation method (classical or
        % Bayesian)
        pathThis = fileparts(mfilename('fullpath'));
        switch this.glm.estimationMethod
            % load batch for classical model estimation
            case 'classical'
                fileMatlabbatch = fullfile(pathThis, 'matlabbatch', ...
                    sprintf('mb_%s.m', module));
                run(fileMatlabbatch);
                % load batch for Bayesian model estimation (1st level)
            case 'Bayesian'
                fileMatlabbatch = fullfile(pathThis, 'matlabbatch', ...
                    sprintf('mb_%s_%s.m', module, this.glm.estimationMethod));
                run(fileMatlabbatch);
                % fill in parameters regarding estimation
                % AR-model order
                matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.ARP = ...
                    this.glm.ARModelOrderBayes;
                % contrast specification
                matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.gcon = ...
                    this.glm.gcon;
                
        end
end
