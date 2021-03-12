function outputImage = tapas_uniqc_recon2image(recon, imageType, varargin)
% Converts ReconstructionData-fields into MrImages, respecting GeometryData 
% information (FOV, offcenter, slice thickness)
%
%  outputImage = tapas_uniqc_recon2image(recon, imageType, varargin)
%
% NOTE: geometry conversion works only for single slice so far!
%
% IN
%   recon       ReconstructionData object
%   imageType   part of recon object that shall be converted to an MrImage
%               'recon', 'final_recon' (default)  
%                          recon.final_recon will be used
%                          (e.g. for SENSE-recon)
%               'iterations', 'recon_stack', 'stack'
%                           extracts all iterations of the reconstruction,
%                           i.e. recon.ImReconStack
%                           into dynamics (volumes), i.e. 4th dimension of
%                           MrImage                       
%               'coils', 'single_coils', 'final_recon_single_coils'  
%                           single coil images are converted into different
%                           slices of the image
%               'B1', 'sens', 'B1map', 'sense', 'SENSE'
%                           extracts B1-map, i.e.
%                           recon.sens.sensitivity.data
%               'B0', 'conjphase', 'B0map', 'fieldmap'
%                           extracts B0-map, i.e.
%                           recon.conjphase.w_map.data
% OUT
%   outputImage             MrImage with correct MrImageGeometry from
%                           recon.geom and size of data matrix
%
% EXAMPLE
%   recon = ReconstructionData()
%   Y = tapas_uniqc_recon2image(recon, 'final_recon');
%   Y.plot('signalPart', 'abs');
%
%   See also ReconstructionData

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-29
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

if nargin < 2
    imageType = 'final_recon';
end


%% Data Extraction from recon-object

switch imageType
    case {'final_recon', 'recon'};
        data = recon.final_recon;
        name = 'final_recon';
    case {'iterations', 'recon_stack', 'stack'};
        data = cell2mat(permute(recon.ImReconStack, [3 4 1 2]));
        name = 'iterations';
    case { 'coils', 'single_coils', 'final_recon_single_coils'}
        data = recon.final_recon_single_coils;
        name = 'final_recon_single_coils';
    case {'B1', 'sens', 'B1map', 'sense', 'SENSE'}
        data =  recon.sens.sensitivity.data;
        name = 'SENSE-B1-map';
    case {'B0', 'conjphase', 'B0map', 'fieldmap'}
        data = recon.conjphase.w_map.data;
        name = 'B0-map (rad/s)';
    otherwise
        error('tapas:uniqc:InvalidImageType', ...
            '%s is no valid imageType', imageType);
end



%% Geometry data conversion
% NOTE: geometry conversion works only for single slice so far!

geom    = recon.geom;
nSlices = size(data,3);

FOV_mm          = [geom.FOV(1:2), nSlices*geom.slice_thickness]*1e3;
offcenter_mm    = reshape(geom.offcentre_xyz_slice*1e3, 1, []);

outputImage = MrImage(data, 'resolution_mm', [], ...
           'FOV_mm', FOV_mm, ...
           'offcenter_mm', offcenter_mm);
       try
           outputImage.name = sprintf('%s_%s', name, recon.recon_name);
       catch
           outputImage.name = sprintf('%s_%s', name, num2str(recon.dataId));
       end