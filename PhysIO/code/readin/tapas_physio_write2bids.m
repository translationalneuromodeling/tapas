function [] = tapas_physio_write2bids(ons_secs, write_bids, log_files)
% Converts trigger, cardiac and respiratory data from physio structure into
% a .tsv file according to BIDS format. Meta data is written in a corresponding .json file.
%
% Parameters:
% IN:
%     ons_secs:                 onsecs structure
%     write_bids.bids_step      integer describing the stage of the processing pipeline at which the files are being written.
%                               Options: 1, 2, 3

%     write_bods. bids_dir:     where the file should be written to


% OUT: tsv file(s) cardiac, respiratory, trigger
%      json file with meta data

% REFERENCES:
% https://bids-specification.readthedocs.io/en/stable/04-modality-specific
% -files/06-physiological-and-other-continuous-recordings.html

% Author: Johanna Bayer 2022

bids_step = write_bids.bids_step;
bids_dir = write_bids.bids_dir{1};
bids_prefix = write_bids.bids_prefix;

cardiac = ons_secs.c;
respiratory = ons_secs.r;



% after step1
switch bids_step
    case 1
        tag = "raw";
        columnsStrings = ["cardiac", "respiratory"];
        mat=[cardiac respiratory];

    case 2
        tag = "norm";
        columnsStrings = ["cardiac", "respiratory"];
        mat=[cardiac respiratory];

    case 3 % triggers available, save as well!
        tag = "sync";
        cardiac = ons_secs.c;
        respiratory = ons_secs.r;

        % triggers have to be replaced into 1 (trigger) 0 (no trigger)
        trigger_binary = zeros(numel(ons_secs.t),1);

        for iVolume = 1:numel(ons_secs.svolpulse)
            row_number = ons_secs.svolpulse(iVolume)==ons_secs.t;
            trigger_binary(row_number)=1;
        end


        columnsStrings =  ["cardiac", "respiratory", "trigger"];

        mat=[cardiac respiratory trigger_binary];

end


cardiacStruct = struct("Description", "continuous pulse measurement", ...
    "Units", "a.u.");
respiratoryStruct = struct("Description", "continuous amplitude measurements by respiration belt", ...
    "Units", "a.u.");
triggerStruct = struct("Description", "continuous binary indicator variable of scanner trigger signal detected by PhysIO", ...
    "Units", "a.u.");

% prepare structure to write into BIDS

if bids_step < 3 % before synchronization
    s = struct("StartTime", log_files.relative_start_acquisition , ...
        "SamplingFrequency",log_files.sampling_interval, "Columns", columnsStrings, ...
        "Manufacturer", log_files.vendor, ...
        "SoftwareVersions", "BIDS Conversion by TAPAS PhysIO Toolbox", ...
        "cardiac", cardiacStruct, ...
        "respiratory", respiratoryStruct);
else
    s = struct("StartTime", log_files.relative_start_acquisition , ...
        "SamplingFrequency",log_files.sampling_interval, "Columns", columnsStrings, ...
        "Manufacturer", logfiles.vendor, ...
        "SoftwareVersions", "BIDS Conversion by TAPAS PhysIO Toolbox", ...
        "cardiac", cardiacStruct, ...
        "respiratory", respiratoryStruct, ...
        "trigger", triggerStruct);
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