function matlabbatch = get_matlabbatch(this, module, varargin)
% Returns matlabbatch to perform spm-processing with an MrImage. Fills out
% all necessary file parameters and options for different modules, e.g.
% realignment, smoothing
%
%   matlabbatch = get_matlabbatch(MrImage, module, varargin)
%
% This is a method of class MrImage.
%
% IN
%   module      different SPM preprocessing routines, e.g., 'realign', 'smooth'
%   varargin    struct or property name/value pairs, set of SPM options to
%               be determined for each module e.g. fwhm for smoothing
% OUT
%   matlabbatch spm matlabbatch that would be executed if module was performed,
%               can be scrutinized via
%               spm_jobman('interactive', matlabbatch)
%
% EXAMPLE
%   get_matlabbatch
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-02
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


pathThis = fileparts(mfilename('fullpath'));
fileMatlabbatch = fullfile(pathThis, 'matlabbatch', ...
    sprintf('mb_%s.m', module));
try
    run(fileMatlabbatch);
catch % sometimes, subfolders of class folders not recognized in path
    pathNow = pwd;
    [fp, fn, ext] = fileparts(fileMatlabbatch);
    cd(fp)
    run([fn ext]);
    cd(pathNow)
end

[pathRaw, fileRaw, ext] = fileparts(this.get_filename('prefix', 'raw'));
fileRaw = [fileRaw ext];

switch module
    
    case 'apply_transformation_field'
        % set the deformation field
        matlabbatch{1}.spm.spatial.normalise.write.subj.def = ...
            cellstr(varargin{1});
        % enter the image to be transformed
        matlabbatch{1}.spm.spatial.normalise.write.subj.resample = ...
            cellstr(spm_select('ExtFPList', pathRaw, ...
            ['^' this.parameters.save.fileName], Inf));
        % add new voxel size if defined (default is 2x2x2)
        if nargin > 3
            matlabbatch{1}.spm.spatial.normalise.write.woptions.vox ...
                = varargin{2};
        end
        
    case 'coregister_to'
        
        args = varargin{1};
        
        % set filenames for this and stationary reference image
        matlabbatch{1}.spm.spatial.coreg.estimate.ref = args.stationaryImage;
        matlabbatch{1}.spm.spatial.coreg.estimate.other = args.otherImages;
        
        matlabbatch{1}.spm.spatial.coreg.estimate.source = ...
            cellstr(spm_select('ExtFPList', pathRaw, ['^' fileRaw]));
        
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = args.objectiveFunction;
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = args.separation;
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = args.tolerances;
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = args.histSmoothingFwhm;
        % not actually used in batch editor, but when calling spm_coreg
        % with eoptions directly;
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.params = args.trafoParameters;
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.graphics = args.doPlot;
    case 'smooth'
        fwhmMillimeter = varargin{1};
        % load and adapt matlabbatch
        matlabbatch{1}.spm.spatial.smooth.fwhm = fwhmMillimeter;
        matlabbatch{1}.spm.spatial.smooth.data = ...
            cellstr(spm_select('ExtFPList', pathRaw, ['^' fileRaw], Inf));
        
    case 'realign'
        args = varargin{1};
        % update matlabbatch with input parameters
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = args.quality;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = args.separation;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = args.smoothingFwhm;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = args.realignToMean;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = args.interpolation;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = args.wrapping;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = args.weighting;
        matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = args.interpolation;
        matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = args.wrapping;
        matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = args.masking;
        
        matlabbatch{1}.spm.spatial.realign.estwrite.data{1} = ...
            cellstr(spm_select('ExtFPList', pathRaw, ['^' fileRaw], Inf));
        
    case 'reslice'
        fnTargetGeometry = varargin{1};
        matlabbatch{1}.spm.spatial.coreg.write.ref = ...
            cellstr(fnTargetGeometry);
        matlabbatch{1}.spm.spatial.coreg.write.source = ...
            cellstr(spm_select('ExtFPList', pathRaw, ['^' fileRaw], Inf));
        
        args = varargin{2};
        
        matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = args.interpolation;
        matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = args.wrapping;
        matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = args.masking;
        
    case 'segment'
        % parse input arguments
        args = varargin{1};
        
        % files / channels
        % includes biasRegularisation, biasFWHM and saveBiasField
        % set data as well
        saveFileNameArray = args.saveFileNameArray;
        nVols = numel(saveFileNameArray);
        if numel(args.biasRegularisation) ~= nVols
            args.biasRegularisation = repmat(args.biasRegularisation, 1, nVols);
        end
        if numel(args.biasFWHM) ~= nVols
            args.biasFWHM = repmat(args.biasFWHM, 1, nVols);
        end
        if numel(args.saveBiasField) ~= nVols
            args.saveBiasField = repmat(args.saveBiasField, 1, nVols);
        end
        
        for nChannel = 1:nVols
            matlabbatch{1}.spm.spatial.preproc.channel(nChannel).vols = ...
                saveFileNameArray(nChannel);
            % bias regularization
            matlabbatch{1}.spm.spatial.preproc.channel(nChannel).biasreg = ...
                args.biasRegularisation(nChannel);
            matlabbatch{1}.spm.spatial.preproc.channel(nChannel).biasfwhm = ...
                args.biasFWHM(nChannel);
            % set to save bias-corrected image and bias field
            matlabbatch{1}.spm.spatial.preproc.channel(nChannel).write(1) = ...
                args.saveBiasField(nChannel);
            matlabbatch{1}.spm.spatial.preproc.channel(nChannel).write(2) = 1;
        end
        % tissues
        % includes tissue types, output space and tissue probability maps
        % set which tissue types shall be written out and in which space
        allTissueTypes = {'GM', 'WM', 'CSF', 'bone', 'fat', 'air'};
        indOutputTissueTypes = find(ismember(lower(allTissueTypes), ...
            lower(args.tissueTypes)));
        for iTissueType = indOutputTissueTypes
            switch lower(args.mapOutputSpace)
                case 'native'
                    matlabbatch{1}.spm.spatial.preproc.tissue(iTissueType).native = [1 0];
                case {'mni', 'standard', 'template', 'warped'}
                    matlabbatch{1}.spm.spatial.preproc.tissue(iTissueType).warped = [1 0];
            end
        end
        
        % set tissue probability maps
        if isempty(args.fileTPM)
            % Take standard TPMs from spm, but update their paths...
            pathSpm = fileparts(which('spm'));
            nTissues = numel(matlabbatch{1}.spm.spatial.preproc.tissue);
            for iTissue = 1:nTissues
                matlabbatch{1}.spm.spatial.preproc.tissue(iTissue).tpm = ...
                    regexprep(matlabbatch{1}.spm.spatial.preproc.tissue(iTissue).tpm, ...
                    '/Users/kasperla/Documents/code/matlab/spm12b', ...
                    regexprep(pathSpm, '\\', '\\\\'));
            end
        else
            fileTPM = args.fileTPM;
            nTissues = 6;
            for iTissue = 1:nTissues
                matlabbatch{1}.spm.spatial.preproc.tissue(iTissue).tpm = ...
                    cellstr([fileTPM,',',int2str(iTissue)]);
            end
        end
        
        % warping parameters
        matlabbatch{1}.spm.spatial.preproc.warp.mrf = args.mrfParameter;
        
        % clean up
        switch args.cleanUp
            case 'none'
                matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 0;
            case 'light'
                matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
            case 'thorough'
                matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 2;
        end
        
        % warping regularization
        matlabbatch{1}.spm.spatial.preproc.warp.reg = args.warpingRegularization;
        
        % affine regularization
        matlabbatch{1}.spm.spatial.preproc.warp.affreg = args.affineRegularisation;
        
        % smoothness
        matlabbatch{1}.spm.spatial.preproc.warp.fwhm = args.smoothnessFwhm;
        
        % sampling distance
        matlabbatch{1}.spm.spatial.preproc.warp.samp = args.samplingDistance;
        
        % set which deformation field shall be written out
        switch args.deformationFieldDirection
            case 'none'
                matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
            case 'forward'
                matlabbatch{1}.spm.spatial.preproc.warp.write = [0 1];
            case {'backward', 'inverse'}
                matlabbatch{1}.spm.spatial.preproc.warp.write = [1 0];
            case {'both', 'all'}
                matlabbatch{1}.spm.spatial.preproc.warp.write = [1 1];
        end
end