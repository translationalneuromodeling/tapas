function []= tapa_physio_write2bids(ons_secs, write_bids, log_files)
% Converts trigger, cardiac and respiratory data from physio structure into
% a .tsv file according to BIDS format, with meta data in a corresponding json file.


% IN: ons_secs:     onsecs structure
%     write_bids:   parameter describing the stage of the processing pipeline
%                   at which the files are being written
%    save_dir: where the file should be written to
        

% OUT: tsv file(s) caridac, respiratory, trigger
%      json file with meta data

% REFERENCES:
% https://bids-specification.readthedocs.io/en/stable/04-modality-specific
% -files/06-physiological-and-other-continuous-recordings.html

% Author: Johanna Bayer 2022

which_bids=write_bids.which_bids;
bids_dir=write_bids.bids_dir{1};

% after step1
switch which_bids
    case 1 
        tag = "loc1";
        cardiac = ons_secs.c;
        respiratory = ons_secs.r;

        s = struct("StartTime", log_files.relative_start_acquisition , ...
        "SamplingFrequency",log_files.sampling_interval, "Columns", ["cardiac", "respiratory"]); 

        % create JSON file
        JSONFILE_name= sprintf('%s_JSON.json',tag); 
        fid = fopen(fullfile(bids_dir,JSONFILE_name),'w'); 
        encodedJSON = jsonencode(s); 
        % write output
        fprintf(fid, encodedJSON); 
        
        %write output
        %writematrix(cardiac,fullfile(bids_dir,sprintf('%s_cardiac.txt',tag)),'Delimiter','tab')
        %writematrix(respiratory,fullfile(bids_dir,sprintf('%s_respiratory.txt',tag)),'Delimiter','tab')
    
        mat=[cardiac respiratory];
        save_file= fullfile(bids_dir,sprintf('%s_bids.tsv',tag));
        save(save_file,"mat",'-ascii');
  
 
    case 2
        tag = "normalized";
        cardiac = ons_secs.c;
        respiratory = ons_secs.r;

        s = struct("StartTime", log_files.relative_start_acquisition , ...
        "SamplingFrequency",log_files.sampling_interval, "Columns", ["cardiac", "respiratory"]); 

        % create JSON file
        JSONFILE_name= sprintf('%s_JSON.json',tag); 
        fid = fopen(fullfile(bids_dir,JSONFILE_name),'w'); 
        encodedJSON = jsonencode(s); 
        % write output
        fprintf(fid, encodedJSON);

        % write output
        %writematrix(cardiac,fullfile(bids_dir,sprintf('%s_cardiac.txt',tag)),'Delimiter','tab')
        %writematrix(respiratory,fullfile(bids_dir,sprintf('%s_respiratory.txt',tag)),'Delimiter','tab')
   

       mat=[cardiac respiratory];
       writematrix(mat, fullfile(bids_dir,sprintf('%s_bids.txt',tag)),'Delimiter','\t')

    case 3
        tag = "trigger";
        % triggerafter step 2
        cardiac = ons_secs.c;
        respiratory = ons_secs.r;
    
        trigger= ons_secs.svolpulse;
    
        % triggers have to be replaced into 1 (trigger) 0 (no trigger)
    
        trigger_binary=zeros(numel(ons_secs.t),1);

        for iVolume = 1:numel(ons_secs.svolpulse)
            row_number = ons_secs.svolpulse(iVolume)==ons_secs.t;
            trigger_binary(row_number)=1;
        end
        
        % prepare structure to write into BIDS
        s = struct("StartTime", log_files.relative_start_acquisition , ...
            "SamplingFrequency",log_files.sampling_interval, "Columns", ["cardiac", "respiratory", "trigger"]); 
        
        % create JSON file
        JSONFILE_name= sprintf('%s_JSON.json',tag); 
        fid=fopen(fullfile(bids_dir,JSONFILE_name),'w'); 
        encodedJSON = jsonencode(s); 
        % write output
        fprintf(fid, encodedJSON); 
        
       % write output
       % writematrix(cardiac,fullfile(bids_dir,sprintf('%s_cardiac.txt',tag)),'Delimiter','tab')
       % writematrix(respiratory,fullfile(bids_dir,sprintf('%s_respiratory.txt',tag)),'Delimiter','tab')
       % writematrix(trigger_binary,fullfile(bids_dir,sprintf('%s_trigger_binary.txt',tag)),'Delimiter','\t')
       
       mat=[cardiac respiratory trigger_binary];
       writematrix(mat, fullfile(bids_dir,sprintf('%s_bids.txt',tag)),'Delimiter','\t')
end