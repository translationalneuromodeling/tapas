function this = load(this, inputDataOrFile, varargin)
% Loads Geometry info from affine image header (.nii/.hdr/.img) or Philips
% (par/rec) or a struct of the object save as a .mat file
%
% NOTE: .mat-header files (for 4D niftis) are ignored, since the same voxel
%       position is assumed in each volume for MrImage
%
%   dimInfo = MrDimInfo()
%   dimInfo.load(inputs)
%
% This is a method of class MrDimInfo.
%
% IN
%
% OUT
%
% EXAMPLE
%   dimInfo = MrImageGeometry()
%   dimInfo.load('test.nii')
%   dimInfo.load('test.hdr/img')
%   dimInfo.load('test.par/rec')
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-10-19
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


fileName = inputDataOrFile;
% check whether file exists
if exist(fileName, 'file')
    
    % get geometry parameters from file
    [~, ~, ext] = fileparts(fileName);
    isValidExtension = ismember(ext, {'.hdr', '.nii', '.img', '.par', '.rec', '.mat'});
    hasExplicitSamplingPoints = ismember(ext, {'.mat'});
    if isValidExtension
        switch ext
            case {'.hdr', '.nii', '.img'}
                % extract dimInfo properties from nifti file
                [dimLabels, resolutions, nSamples, units, firstSamplingPoint] = ...
                    this.read_nifti(fileName);
            case {'.par', '.rec'}
                % extract dimInfo properties from par/rec file
                [dimLabels, resolutions, nSamples, units, firstSamplingPoint] = ...
                    this.read_par(fileName);
            case '.mat'
                load@MrCopyData(this,fileName, varargin{:}); % from MrCopyData, a struct is read in
        end
        
        if ~hasExplicitSamplingPoints % derive via 1st point and equidistant sampling
            
            % now set dims
            nDims = numel(nSamples);
            iDimGeom = 1:nDims;
            
            % update existing geom dimensions, add new ones for
            % non-existing
            iValidDimLabels = this.get_dim_index(dimLabels);
            iDimGeomExisting = find(iValidDimLabels);
            iDimGeomAdd = setdiff(iDimGeom, iDimGeomExisting);
            
            % if dimension labels exist, just update values
            this.set_dims(dimLabels(iDimGeomExisting), ...
                'resolutions', resolutions(iDimGeomExisting), ...
                'nSamples', nSamples(iDimGeomExisting), ...
                'firstSamplingPoint', firstSamplingPoint(iDimGeomExisting), ...
                'units', units(iDimGeomExisting), ...
                'samplingWidths', resolutions(iDimGeomExisting));
            
            % if they do not exist, create dims
            this.add_dims(dimLabels(iDimGeomAdd), ...
                'resolutions', resolutions(iDimGeomAdd), ...
                'nSamples', nSamples(iDimGeomAdd), ...
                'firstSamplingPoint', firstSamplingPoint(iDimGeomAdd), ...
                'units', units(iDimGeomAdd), ...
                'samplingWidths', resolutions(iDimGeomAdd));
        end
    else % no valid extension
        warning('Only dimInfo-struct (.mat), Philips (.par/.rec), nifti (.nii) and analyze (.hdr/.img) files are supported');
    end
    
else
    fprintf('Geometry data for MrDimInfo could not be loaded: file %s not found.\n', ...
        fileName);
end


end