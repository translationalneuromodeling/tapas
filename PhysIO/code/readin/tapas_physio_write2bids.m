function []= tapa_physio_write2bids(ons_secs,which_bids )
% Converts trigger cardiac and respiratory data from physio structure into
% a tsv file according to BIDS format, with meta data 
% in json file

% IN: physio structure
%     

% OUT: tsv file(s) with columns caridac, respiratory, trigger
%    json file with meta data

% after step1
switch which_bids
    case 1 | 2
        cardiac = ons_secs.c;
        respiratory = ons_secs.r;

        s = struct("StartTime", physio.log_files.relative_start_acquisition , ...
        "SamplingFrequency",physio.log_files.sampling_interval, "Columns", ["cardiac", "respiratory"]); 

        % create JSON file
        JSONFILE_name= sprintf('%s_%s_JSON.json',subj, session); 
        fid = fopen(fullfile(save_dir,JSONFILE_name),'w'); 
        encodedJSON = jsonencode(s); 
        % write output
        fprintf(fid, encodedJSON); 

    case 3
    % triggerafter step 2
    cardiac = ons_secs.c;
    respiratory = ons_secs.r;

    trigger= physio.ons_secs.svolpulse;

    % triggers have to be replaced into 1 (trigger) 0 (no trigger)

    trigger_binary=zeros(numel(physio.ons_secs.t),1);

    for iVolume = 1:numel(physio.ons_secs.svolpulse)
        row_number = physio.ons_secs.svolpulse(iVolume)==physio.ons_secs.t;
        trigger_binary(row_number)=1;
    end


    %TODO write to file

    % TODO this needs to be updated to change by subjects - can this info be
    % taken from the pysio structure?
    subj='sub-01';
    session= 'ses-01';
    save_dir=physio.save_dir

    % prepare structure to write into BIDS
    s = struct("StartTime", physio.log_files.relative_start_acquisition , ...
        "SamplingFrequency",physio.log_files.sampling_interval, "Columns", ["cardiac", "respiratory", "trigger"]); 


    % create JSON file
    JSONFILE_name= sprintf('%s_%s_JSON.json',subj, session); 
    fid=fopen(fullfile(save_dir,JSONFILE_name),'w'); 
    encodedJSON = jsonencode(s); 
   % write output
    fprintf(fid, encodedJSON); 

    % write output
    writematrix(cardiac,fullfile(save_dir,'cardiac.txt'),'Delimiter','tab')
    writematrix(respiratory,fullfile(save_dir,'respiratory.txt'),'Delimiter','tab')
    writematrix(trigger_binary,fullfile(save_dir,'trigger_binary.txt'),'Delimiter','tab')

end