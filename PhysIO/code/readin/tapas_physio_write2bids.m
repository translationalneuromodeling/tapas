function []= tapas_physio_write2bids(ons_secs, write_bids, log_files)
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

bids_step=write_bids.bids_step;
bids_dir=write_bids.bids_dir{1};
bids_prefix = write_bids.bids_prefix{1};

% after step1
switch bids_step
    case 1 
        tag = "raw";
        cardiac = ons_secs.c;
        respiratory = ons_secs.r;

        s = struct("StartTime", log_files.relative_start_acquisition , ...
        "SamplingFrequency",log_files.sampling_interval, "Columns", ["cardiac", "respiratory"]); 

        % create JSON file
        JSONFILE_name= sprintf('%2$s_desc_physio_%1$s.json',tag, bids_prefix); 
        fid = fopen(fullfile(bids_dir,JSONFILE_name),'w'); 
        if verLessThan('matlab', '9.10')
            encodedJSON = jsonencode(s); 
        else
            encodedJSON = jsonencode(s, PrettyPrint=true); 
        end
        % write output
        fprintf(fid, encodedJSON); 
        
        %write output
        %writematrix(cardiac,fullfile(bids_dir,sprintf('%s_cardiac.txt',tag)),'Delimiter','tab')
        %writematrix(respiratory,fullfile(bids_dir,sprintf('%s_respiratory.txt',tag)),'Delimiter','tab')
    
        mat=[cardiac respiratory];
        save_file= fullfile(bids_dir,sprintf('%2$s_desc_physio_%1$s.tsv',tag, bids_prefix));
        save(save_file,"mat",'-ascii');
        gzip(save_file);
 
    case 2
        tag = "norm";
        cardiac = ons_secs.c;
        respiratory = ons_secs.r;

        s = struct("StartTime", log_files.relative_start_acquisition , ...
        "SamplingFrequency",log_files.sampling_interval, "Columns", ["cardiac", "respiratory"]); 

        % create JSON file
        JSONFILE_name= sprintf('%2$s_desc_physio_%1$s.json',tag, bids_prefix);
        fid = fopen(fullfile(bids_dir,JSONFILE_name),'w'); 
        if verLessThan('matlab', '9.10')
            encodedJSON = jsonencode(s); 
        else
            encodedJSON = jsonencode(s, PrettyPrint=true); 
        end
        % write output
        fprintf(fid, encodedJSON);

        % write output
        %writematrix(cardiac,fullfile(bids_dir,sprintf('%s_cardiac.txt',tag)),'Delimiter','tab')
        %writematrix(respiratory,fullfile(bids_dir,sprintf('%s_respiratory.txt',tag)),'Delimiter','tab')
   

       mat=[cardiac respiratory];
       save_file= fullfile(bids_dir,sprintf('%2$s_desc_physio_%1$s.tsv',tag, bids_prefix));
       save(save_file,"mat",'-ascii');
       gzip(save_file);

    case 3
        tag = "sync";
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
        JSONFILE_name= sprintf('%2$s_desc_physio_%1$s.json',tag, bids_prefix);
        fid=fopen(fullfile(bids_dir,JSONFILE_name),'w'); 
        if verLessThan('matlab', '9.10')
            encodedJSON = jsonencode(s); 
        else
            encodedJSON = jsonencode(s, PrettyPrint=true); 
        end
        % write output
        fprintf(fid, encodedJSON); 
        
       % write output
       % writematrix(cardiac,fullfile(bids_dir,sprintf('%s_cardiac.txt',tag)),'Delimiter','tab')
       % writematrix(respiratory,fullfile(bids_dir,sprintf('%s_respiratory.txt',tag)),'Delimiter','tab')
       % writematrix(trigger_binary,fullfile(bids_dir,sprintf('%s_trigger_binary.txt',tag)),'Delimiter','\t')
       
       mat=[cardiac respiratory trigger_binary];
       save_file= fullfile(bids_dir,sprintf('%2$s_desc_physio_%1$s.tsv',tag, bids_prefix));
       save(save_file,"mat",'-ascii');
       gzip(save_file);
end