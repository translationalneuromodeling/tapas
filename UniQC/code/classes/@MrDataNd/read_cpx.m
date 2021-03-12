function this = read_cpx(this, filename, selectedVolumes, selectedCoils, ...
    signalPart)
% loads Philips cpx files (coil-wise image reconstructions) using GyroTools
% code (read_mr_data, read_cpx)
%
%   Y = MrImage()
%   Y.load_cpx(filename, selectedVolumes, selectedCoils)
%
% This is a method of class MrImage.
%
% IN
%   filename    e.g. 'fmri.cpx'
%   selectedVolumes     [1, nVols] vector of volumes to be loaded
%                       Inf = all volumes in file (default)
%   selectedCoils       [1, nCoils] vector of coils to be loaded
%                       0   = Sum of Squares of all coils (default)
%                       Inf = all coils are loaded (TODO: into which
%                             dimension of MrImage?
%   signalPart         'abs'       - absolute value
%                      'phase'     - phase of signal
%
% OUT
%
% EXAMPLE
%   load_cpx
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-04
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if ~exist('read_cpx.m' , 'file')
    msg = [
        'Readin of Philips CPX data requires read_cpx.m from Gyrotools, ' ...
        'which is not part of UniQC. Please retrieve separately and add to ' ...
        'the Matlab path.'
        ];
    error('tapas:uniqc:MrDataNd:CPXReaderNotFound', msg);
end
hasSelectedVolumes = ~isinf(selectedVolumes);
hasSelectedCoils = ~any(isinf(selectedCoils)) && ~any(selectedCoils==0) ;

readParams = create_read_param_struct(filename);

if hasSelectedVolumes
    readParams.dyn = reshape(selectedVolumes, [], 1);
end

if ~hasSelectedCoils
    selectedCoils = readParams.coil;
end

if nargin < 4
    signalPart = 'abs';
end

selectedCoils = reshape(selectedCoils, 1, []);
nCoils = numel(selectedCoils);

border = 0;
flip = 0;
kspace = 0;

switch lower(signalPart)
    
    case 'abs'
        % read multiple coils one by one to not create memory problem
        for iCoil = 1:nCoils
            fprintf('loading coil %d, (%d/%d)\n', selectedCoils(iCoil), iCoil, nCoils);
            readParams.coil = selectedCoils(iCoil);
            tmpData = read_cpx(filename, border, flip, kspace, readParams);
            if iCoil == 1
                this.data = tmpData.*conj(tmpData);
            else
                this.data = this.data + tmpData.*conj(tmpData);
            end
        end
        this.data = double(squeeze(sqrt(this.data)));
        
    case {'phase', 'angle', 'ang'}
        
        
        % read multiple coils one by one to not create memory problem
        for iCoil = 1:nCoils
            
            % weighted sum of phases per coil, weighted by the squared absolute
            % value of signal in each coil channel
            
            fprintf('loading coil %d, (%d/%d)\n', selectedCoils(iCoil), iCoil, nCoils);
            readParams.coil = selectedCoils(iCoil);
            tmpData = read_cpx(filename, border, flip, kspace, readParams);
            phaseData = angle(tmpData);
            absData = tmpData.*conj(tmpData);
            
            if nCoils == 1 % no weighting needed
                this.data = phaseData;
                sumData = 1;
            else % sum of squares weighting with magnitude
                if iCoil == 1
                    this.data = phaseData.*absData;
                    sumData = absData;
                else
                    this.data = this.data + absData.*phaseData;
                    sumData = sumData + absData;
                end
            end
        end
        this.data = double(squeeze(this.data./sumData));
        
end

% put volumes back into 4th dimension
if numel(readParams.loca) == 1
    this.data = permute(this.data, [1 2 4 3]);
end

