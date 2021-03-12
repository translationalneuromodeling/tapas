classdef MrGlm < MrCopyData
    % Class providing General Linear Model of fMRI data
    % (for mass-univariate analysis, i.e. per-voxel regression)
    %
    %
    % EXAMPLE
    %   MrGlm
    %
    %   See also
    
    % Author:   Saskia Klein & Lars Kasper
    % Created:  2014-07-08
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

    
    properties
        
        % Multiple regressors, i.e. confounds that will be regressed directly,
        % without convolution with the hemodynamic response function(s).
        % fields:
        %      realign     Realignment parameters
        regressors = struct( ...
            'realign', [], ...
            'physio', [], ...
            'other', [] ...
            );
        
        % Multiple conditions, i.e. behavioral/neuronal regressors that will be
        % convolved with the hemodynamic response function(s).
        conditions = struct( ...
            'names', [], ...
            'onsets', [], ...
            'durations', [] ...
            );
        
        % The final design matrix used for the voxel-wise regression
        designMatrix = [];
        
        % timing parameters
        timingUnits = '';
        repetitionTime = '';
        
        % HRF derivatives
        hrfDerivatives = '';
        
        % SPM Directory
        parameters = struct( ...
            'save', struct( ...
            'path', '', ... % path where the SPM file is stored, e.g. the MrSeries path
            'spmDirectory', '')... % name of the SPM Directory for the SPM file
            );
        
        % masking threshold, defined as proportion of globals
        maskingThreshold = 0.8;
        
        % explicit mask for the analysis, e.g. the segmentation results for a
        % whithin brain mask
        explicitMasking = '';
        
        % serial correlations, AR(1) or FAST
        serialCorrelations = 'AR(1)';
        
        % estimation method (classical or Bayesian)
        estimationMethod = 'classical'
        
        % AR model order (for Bayesian estimation only)
        ARModelOrderBayes = 3;
        
        % contrasts need to be specified before hand as well
        gcon = struct(...
            'name', {'main_positive', 'main negative'}, ...
            'convec', {1, -1});
        
    end % properties
    
    
    methods
        
        % Constructor of class
        function this = MrGlm()
        end
        
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
        
    end % methods
    
end
