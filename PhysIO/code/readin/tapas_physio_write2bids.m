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
%TODO write to file

% TODO this needs to be updated to change by subjects - can this info be
% taken from the pysio structure?
subj='subject1'
session= 'session1'

% prepare structure to write into BIDS
 s = struct("StartTime", physio.log_files.relative_start_acquisition , ...
        "SamplingFrequency",physio.log_files.sampling_interval, "Columns", ["cardiac", "respiratory"]); 
   

JSONFILE_name= sprintf('%s_JSON%d.json',subj, session); 
    fid=fopen(JSONFILE_name,'w') 
    encodedJSON = jsonencode(s); 
    % TODO: think about and add output folder
    fprintf(fid, encodedJSON); 

end