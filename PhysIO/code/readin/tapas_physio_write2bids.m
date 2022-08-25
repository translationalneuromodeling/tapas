function [t,c,r,svolpulse]= tapa_physio_write2bids(physio)
% Converts trigger cardiac and respiratory data from physio structure into
% a tsv file according to BIDS format, with meta data 
% in json file

% IN: physio structure
%     

% OUT: tsv file(s) with columns caridac, respiratory, trigger
%    json file with meta data

cardiac = physio.ons_secs.c;
respiratory = physio.ons_secs.r;
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
subj='subject1';
session= 'session1';

% prepare structure to write into BIDS
 s = struct("StartTime", physio.log_files.relative_start_acquisition , ...
        "SamplingFrequency",physio.log_files.sampling_interval, "Columns", ["cardiac", "respiratory", "trigger"]); 
  



JSONFILE_name= sprintf('%s_%s_JSON.json',subj, session); 
    fid=fopen(JSONFILE_name,'w'); 
    encodedJSON = jsonencode(s); 
    % TODO: think about and add output folder
    fprintf(fid, encodedJSON); 

end