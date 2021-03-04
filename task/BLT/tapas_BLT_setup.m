function [  ] = tapas_BLT_setup(  )
% setup for Breathing Learning Task (BLT)
% 
% author: Yu Yao, Translational Neuromodeling Unit (TNU), University Zurich
%         and ETH Zurich 

% get path to tapas
dTool = mfilename('fullpath');
dTool = fileparts(dTool);
dTapas = fileparts(dTool);
dTapas = fileparts(dTapas);
% add downloader to path
addpath(fullfile(dTapas,'misc'));
addpath(fullfile(dTapas,'external'));
try
    % download example data
    [exdir] = tapas_download_example_data();
catch mx
    if strcmp(mx.identifier,'tapas:example_data:target_exists')
        exdir = mx.message;
        exdir(1:10) = [];
        exdir(end-15:end) = [];
    else
        rethrow(mx);
    end
end

% copy to task folder
copyfile(fullfile(exdir,'BLT','images'),fullfile(dTool,'images'));
disp('done')

end

