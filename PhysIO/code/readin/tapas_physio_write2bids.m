function [] = tapas_physio_write2bids(ons_secs, write_bids, log_files)
% Converts trigger, cardiac and respiratory data from physio structure into
% a .tsv file according to BIDS format. Meta data is written in a corresponding .json file.
%
% Parameters:
% IN:
%     ons_secs:                 ons_secs (onsets in seconds) structure
%                               that contains PhysIO traces and
%                               preprocessing stages
%     write_bids.bids_step      integer describing the stage of the 
%                               processing pipeline at which the files are 
%                               being written.
%                               Options: 
%                               0 - No BIDS files written
%                               1 - Write raw vendor file data to BIDS
%                               2 - Write padded and normalized time series 
%                                   to BIDS
%                               3 - Write padded/normalized data to BIDS, 
%                                   including extracted scan triggers
%                               4 - Write preprocessed data to BIDS (filtering,
%                                   cropping to acquisition window, cardiac 
%                                   pulse detection)
%     write_bids.bids_dir:     where the file should be written to
% OUT: 
%       tsv.gz file             zipped tab-separated value text file, colums
%                               with cardiac, respiratory, [scan trigger], 
%                               [cardiac pulse trigger]
%       json file with meta data
%
% REFERENCES:
% https://bids-specification.readthedocs.io/en/stable/04-modality-specific
% -files/06-physiological-and-other-continuous-recordings.html

% Author: Johanna Bayer
% Created: 2022
% Copyright (C) 2011-2024 TNU, Institute for Biomedical Engineering, 
%               University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public Licence (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

bids_step = write_bids.bids_step;
bids_dir = write_bids.bids_dir;
bids_prefix = write_bids.bids_prefix;

cardiac = ons_secs.c;
respiratory = ons_secs.r;


if bids_step > 0

% after step1
switch bids_step
    case 1
        tag = "raw";
        desc = "raw, after vendor file read-in";
        columnsStrings = ["cardiac", "respiratory"];
        mat=[cardiac respiratory];
 
    case 2
        tag = "norm";
        desc = "processed: normalized amplitudes, padded for scan duration";
        columnsStrings = ["cardiac", "respiratory"];
        mat=[cardiac respiratory];

    case 3 % triggers available, save as well!
        tag = "sync";
        desc = "processed: normalized amplitudes, padded for scan duration, scan trigger extracted";

        % triggers have to be replaced into 1 (trigger) 0 (no trigger)

        trigger_binary = zeros(numel(ons_secs.t),1);

        for iVolume = 1:numel(ons_secs.svolpulse)
            row_number = ons_secs.svolpulse(iVolume)==ons_secs.t;
            trigger_binary(row_number)=1;
        end


        columnsStrings =  ["cardiac", "respiratory", "trigger"];

        mat=[cardiac respiratory trigger_binary];

    case 4 % preprocessed time series with detected cardiac pulses

        tag = "preproc";
        desc = "processed: normalized amplitudes, padded for scan duration, scan trigger extracted, filtered respiratory data";

        respiratory = ons_secs.r;

        % triggers have to be replaced into 1 (trigger) 0 (no trigger)
        trigger_binary = zeros(numel(ons_secs.t),1);

        for iVolume = 1:numel(ons_secs.svolpulse)
            row_number = ons_secs.svolpulse(iVolume)==ons_secs.t;
            trigger_binary(row_number)=1;
        end

        % triggers have to be replaced into 1 (trigger) 0 (no trigger)
        cpulse_binary = zeros(numel(ons_secs.t),1);

        for iVolume = 1:numel(ons_secs.cpulse)
            row_number = ons_secs.cpulse(iVolume)==ons_secs.t;
            cpulse_binary(row_number)=1;
        end

        columnsStrings =  ["cardiac", "respiratory", "trigger", "cardiac_pulse"];

        mat=[cardiac respiratory trigger_binary, cpulse_binary];

end


%% Prepare structure to write into BIDS
cardiacStruct = struct("Description", ...
    sprintf("continuous pulse measurement (%s)", desc), ...
    "Units", "a.u.");
respiratoryStruct = struct("Description", ...
    sprintf("continuous amplitude measurements by respiration belt (%s)", desc), ...
    "Units", "a.u.");
triggerStruct = struct("Description", "continuous binary indicator variable of scan volume trigger signal detected by PhysIO", ...
    "Units", "a.u.");
cardiacPulseStruct = struct("Description", "continuous binary indicator variable of cardiac pulses (peaks) detected by PhysIO", ...
    "Units", "a.u.");

switch bids_step
    case {1,2}
    s = struct("StartTime", log_files.relative_start_acquisition , ...
        "SamplingFrequency", 1./ons_secs.dt, "Columns", columnsStrings, ...
        "Manufacturer", log_files.vendor, ...
        "SoftwareVersions", ...
        sprintf("BIDS Conversion by TAPAS PhysIO Toolbox (%s)", tapas_physio_version()), ...
        "cardiac", cardiacStruct, ...
        "respiratory", respiratoryStruct);
    case 3
    s = struct("StartTime", log_files.relative_start_acquisition , ...
        "SamplingFrequency", 1./ons_secs.dt, "Columns", columnsStrings, ...
        "Manufacturer", log_files.vendor, ...
        "SoftwareVersions", ...
        sprintf("BIDS Conversion by TAPAS PhysIO Toolbox (%s)", tapas_physio_version()), ...
        "cardiac", cardiacStruct, ...
        "respiratory", respiratoryStruct, ...
        "trigger", triggerStruct);
    case 4
        s = struct("StartTime", log_files.relative_start_acquisition , ...
        "SamplingFrequency", 1./ons_secs.dt, "Columns", columnsStrings, ...
        "Manufacturer", log_files.vendor, ...
        "SoftwareVersions", ...
        sprintf("BIDS Conversion by TAPAS PhysIO Toolbox (%s)", tapas_physio_version()), ...
        "cardiac", cardiacStruct, ...
        "respiratory", respiratoryStruct, ...
        "trigger", triggerStruct, ...
        "cardiac_pulse", cardiacPulseStruct);
end

% create JSON file
JSONFILE_name = sprintf('%2$s_desc-%1$s_physio.json',tag, bids_prefix);
fid = fopen(fullfile(bids_dir,JSONFILE_name),'w');
if verLessThan('matlab', '9.10')
    encodedJSON = jsonencode(s);
else
    encodedJSON = jsonencode(s, PrettyPrint=true);
end
% write output
fprintf(fid, encodedJSON);

save_file = fullfile(bids_dir,sprintf('%2$s_desc-%1$s_physio.tsv',tag, bids_prefix));
save(save_file,"mat",'-ascii');

gzip(save_file);

% delete uncompressed .tsv file after zipping
if isfile([save_file, '.gz'])
    delete(save_file)
end

end