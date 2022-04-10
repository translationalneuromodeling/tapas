function [tableVolSliPhase, indVolPerPhaseSlice, imgCardiacPhasesMeanVols, ...
    imgCardiacPhasesFirstVol, fh] = ...
    tapas_physio_sort_images_by_cardiac_phase(ons_secs_or_c_phase, sqpar, nCardiacPhases, ...
    fnTimeSeries, verbose, dirOut)
%performs retrospective cardiac gating of image time series using phys logfile
%
% [tableVolSliPhase, indVolPerPhaseSlice, imgCardiacPhasesMeanVols, ...
%    imgCardiacPhasesFirstVol] = ...
%   tapas_physio_sort_images_by_cardiac_phase(ons_secs_or_c_phase, sqpar, nCardiacPhases, ...
%    fnTimeSeries, verbose);
%
% Given cardiac phases after running tapas_physio_main_create_regressors,
% this function sorts all slices of all volumes according to the cardiac phase
% they were acquired in, giving an averaged (and 1st pass) movie of
% cardiac-cycle related brain movement, similar to a gated cine-imaging
% technique.
%
% IN
%   ons_secs        physio.ons_secs, structure of onsets in seconds,
%                   see also tapas_physio_new
%
%                   needed: .spulse     time (in seconds) of slice scan
%                                       events
%                           .svolpulse  time (in seconds ) of volume scan
%                                       events
%                           .cpulse     time (in seconds) of heartbeat
%                                       R-peaks
%                   If this is given, c_phase will be computed from the
%                   ons_secs data
%                   OR
%   c_phase         [nSlices*nVolumes,1] vector of cardiac phases
%                   ([0...2pi]) for all slices and volumes acquired
%
%   sqpar           physio.sqpar, structure of sequence parameters, see also tapas_physio_new
%                   needed: .Nslices
%                           .Nscans
%   nCardiacPhases  number of cardiac phases (i.e. cine movie frames) to
%                   bin-sort into
%   fnTimeSeries    file name of (nifti-) time series to re-sort according
%                   to bins
%   verbose         if true, some informative plots are created
%
% OUT
%   tableVolSliPhase [nSlices*nVols, 3], table-like matrix holding per
%                   column 1 : volume index
%                   column 2 : slice index
%                   column 3 : cardiac phase
%   indVolPerPhaseSlice {nCardiacPhases, nSlices}
%                   - cell array where entry {iPhase, iSlice} holds
%                   vector of volumes where iSlice was acquired in cardiac
%                   phase iPhase
%
%   cPhaseMeanVols_<fnTimeSeries>.nii  time series of cardiac phase volumes,
%                  with same slices of the same cardiac phase averaged over
%                  all corresponding volumes
%   cPhaseFirstVol_<fnTimeSeries>.nii  time series of cardiac phase volumes,
%                  with first slice-occurence of each phase stacked
%                  together in volumes
%
% EXAMPLE
%   tapas_physio_sort_images_by_cardiac_phase
%
%   See also

% Author: Lars Kasper
% Created: 2014-06-15
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


if nargin < 5
    verbose = false;
end

hasDirOut = nargin >=6;

if ~isstruct(ons_secs_or_c_phase)
    %% cardiac phase already given, move to next step
    ons_secs    = [];
    c_phase     = ons_secs_or_c_phase;
else
    %% Re-Compute cardiac phases from cardiac pulse events and slice/volume scan events
    ons_secs = ons_secs_or_c_phase;
    c_phase = [];
    cpulse = ons_secs.cpulse;
    spulse = ons_secs.spulse;
    svolpulse = ons_secs.svolpulse;
    
    c_phase    = tapas_physio_get_cardiac_phase(cpulse, spulse, 1, svolpulse);
    
    hasPreps = ~isempty(sqpar.Nprep);
    if hasPreps
        c_phase    = c_phase(((sqpar.Ndummies+sqpar.Nprep)*sqpar.Nslices+1):end);
    else
        c_phase    = c_phase((sqpar.Ndummies*sqpar.Nslices+1):end);
    end
end

%% Histogram and bin the phases ...

nSlices = sqpar.Nslices;
nVols = sqpar.Nscans;

% if too short c_phase, count volumes from back
hascroppedCphase = numel(c_phase)/nSlices < nVols;
if hascroppedCphase
    nVols = numel(c_phase)/nSlices;
    iOffset = sqpar.Nscans -nVols + 1;
end

% cardiacPhaseArray = linspace(0,2*pi,nCardiacPhases);
[h, cardiacPhaseArray] = hist(c_phase, nCardiacPhases);

if verbose
    stringTitle = 'Assess: Number of slice/volume occurences of each cardiac phase';
    fh(1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', stringTitle);
    bar(cardiacPhaseArray,h); xlabel('cardiac phase'); ylabel('counts');
    title(stringTitle);
end

widthBin = mean(diff(cardiacPhaseArray));

leftBorderPhase = cardiacPhaseArray - widthBin/2;
rightBorderPhase = cardiacPhaseArray + widthBin/2;


%% Sort phases into bins
iLocPhaseArray = cell(nCardiacPhases,1);
iSliceArray = cell(nCardiacPhases,1);
iVolArray = cell(nCardiacPhases,1);
%iPhaseVolSliceArray = zeros(nVols, nSlices);
iPhaseVolSliceArray = zeros(nSlices, nVols);

% table holding volume, slice and cardiac phase index in three columns
[I1, I2] = meshgrid(1:nVols,1:nSlices);
tableVolSliPhase = [I1(:), I2(:), nan(nSlices*nVols,1)];

for iPhase = 1:nCardiacPhases
    iLocPhaseArray{iPhase} = find(c_phase >= leftBorderPhase(iPhase) & ...
        c_phase <= rightBorderPhase(iPhase));
    iSliceArray{iPhase} = mod(iLocPhaseArray{iPhase},nSlices) + 1;
    iVolArray{iPhase} = floor(iLocPhaseArray{iPhase}/nVols) + 1;
    %    iPhaseVolSliceArray(iVolArray{iPhase}, iSliceArray{iPhase}) = ...
    %        iPhase;
    iPhaseVolSliceArray(iLocPhaseArray{iPhase}) = ...
        iPhase;
    tableVolSliPhase(iLocPhaseArray{iPhase},3) = iPhase;
end


%% Plot cardiac phase per slice and volume
if verbose
    stringTitle = 'Assess: Cardiac phase per slice and volume';
    fh(2) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', stringTitle);
    imagesc(iPhaseVolSliceArray);
    xlabel('Volumes'); ylabel('slice');
    title(stringTitle);
    colorbar;
end


%% load image file

[~, img4D] = spm_img_load(fnTimeSeries);

if hascroppedCphase
    img4D = img4D(:,:,:,iOffset:end);
end

%% Find for all phases and slices corresponding volumes
indVolPerPhaseSlice = cell(nCardiacPhases,nSlices);
nVolPerPhaseSlice = zeros(nCardiacPhases,nSlices);
for iPhase = 1:nCardiacPhases
    for iSlice = 1:nSlices
        indTmp = find(tableVolSliPhase(:,3) == iPhase & ...
            tableVolSliPhase(:,2) == iSlice);
        indVolPerPhaseSlice{iPhase,iSlice} = tableVolSliPhase(indTmp,1);
        nVolPerPhaseSlice(iPhase, iSlice) = numel(indTmp);
    end
end

if verbose
    stringTitle = 'Assess: Count of volumes falling into phase/slice bin';
    fh(3) = tapas_physio_get_default_fig_params();
    set(fh(3), 'Name', stringTitle);
    imagesc(nVolPerPhaseSlice);
    xlabel('cardiac phase');
    ylabel('slice number');
    title(stringTitle);
    colorbar; 
end


%% Re-sort time series according to cardiac phase, take mean and first ..
% occurences of each phase to get movie
nX = size(img4D,1);
nY = size(img4D,2);
imgCardiacPhasesMeanVols = zeros(nX, nY, nSlices, nCardiacPhases);
imgCardiacPhasesFirstVol = zeros(nX, nY, nSlices, nCardiacPhases);

% fill out image with phases
for iPhase = 1:nCardiacPhases
    for iSlice = 1:nSlices
        if isempty(indVolPerPhaseSlice{iPhase, iSlice})
            warning('No volumes with phase %d, slice %d exist', iPhase, ...
                iSlice);
        else
            imgCardiacPhasesMeanVols(:,:,iSlice,iPhase) = ...
                mean(img4D(:,:,iSlice, indVolPerPhaseSlice{iPhase, iSlice}), 4);
            imgCardiacPhasesFirstVol(:,:,iSlice,iPhase) = ...
                img4D(:,:,iSlice, indVolPerPhaseSlice{iPhase, iSlice}(1));
        end
    end
end


%% Save images
iVolArray = 1:nCardiacPhases;
fnIn = fnTimeSeries;

if hasDirOut
    [~,fn,ext] = fileparts(fnIn);
else
    [dirOut,fn,ext] = fileparts(fnIn);
end

fnOut = fullfile(dirOut, ['cPhaseMeanVols_' fn, ext]);
if exist(fnOut, 'file'), delete(fnOut); end
reuse_nifti_hdr(fnIn, fnOut, imgCardiacPhasesMeanVols, iVolArray);

fnOut = fullfile(dirOut, ['cPhaseFirstVol_' fn, ext]);
if exist(fnOut, 'file'), delete(fnOut); end

reuse_nifti_hdr(fnIn, fnOut, imgCardiacPhasesFirstVol, iVolArray);




%% local functions

function [T, Y, V] = spm_img_load(fn, verbose)
% reads and plots analyze/nifti-img (via spm_read_vols and imtool)
%
%   output = spm_img_load(input)
%
% IN
%   fn          filename
%   verbose     if yes, call imtool for plot
% OUT
%   T           column vector of all voxels
%   Y           read in image values;
%   V           spm_vol information (header)
% EXAMPLE
%   spm_img_load
%
%   See also

% Author: Lars Kasper
% Created: 2013-08-01
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
% $Id: spm_img_load.m 449 2014-03-26 11:32:56Z kasperla $
if nargin < 2
    verbose = false;
end

V = spm_vol(fn);
try
    Y = spm_read_vols(V);
    % maybe only header misalignment of volumes is the problem for nifti
    %...rename temporarily for loading
catch err
    fnHdr = regexprep(fn, '\.nii','\.mat');
    fnTmp = regexprep(fn, '\.nii','\.tmp');
    if exist(fnHdr, 'file')
        movefile(fnHdr, fnTmp);
        warning('Headers of volumes not aligned, ignoring them...');
        V = spm_vol(fn);
        Y = spm_read_vols(V);
        movefile(fnTmp, fnHdr);
    else % nothing we can do, throw error
        throw(err);
    end
end
T = Y(:);
if verbose && ~(isdeployed || ismcc)
    Nslices = size(Y,3);
    for s = 1:Nslices
        imtool(Y(:,:,s))
    end
end


function V = reuse_nifti_hdr(fnIn, fnOut, Y, iVolArray)
%reuse nifti hdr to create other file with it...sets scaling to identical,
%and float64 save option...works for 3D and 4D data!
%
%   V = reuse_nifti_hdr(fnIn, fnOut, Y, iVolArray)
%
% IN
%   fnIn    file name (nifti/analyse) to create new vol from (4D also)
%   fnOut   file to write to
%   Y       (optional) matrix that is writen to fnOut, if given
%   iVolArray   index vector of which volumes shall
%               be manipulated, e.g. [1] default: all
%
% OUT
%   V       volume header to be used with spm_write_vol
%
% EXAMPLE
%   reuse_nifti_hdr
%
%   See also

% Author: Lars Kasper
% Created: 2013-11-12
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
% $Id: reuse_nifti_hdr.m 382 2013-12-11 18:53:04Z kasperla $
W = spm_vol(fnIn);
nVols = length(W);
if nargin < 4
    iVolArray = 1:nVols;
end
nVols = length(iVolArray);

for iV = 1:nVols
    iVol = iVolArray(iV);
    V(iV) = W(iVol);
    V(iV).pinfo = [1 0 0]';
    V(iV).dt = [64 0];
    V(iV).fname = fnOut;
    V(iV).n = [iV, 1];
    if nargin > 2
        if ~exist(fileparts(fnOut),'dir'), mkdir(fileparts(fnOut)); end
        spm_write_vol(V(iV), Y(:,:,:,iV));
    end
end
