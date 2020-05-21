function [version, hash] = tapas_version(verbose)
%% Print and returns the release version of the code.
%
%   Input
%       verbose     -- If true prints the instructions to cite different 
%                      parts of the toolbox. Defaults to false.
%       
%   Output
%       version     -- Current version of tapas
%       hash        -- If possible return the hash of the git repository.
%

% aponteeduardo@gmail.com
% copyright (C) 2017
%

version = tapas_get_current_version();
version = strsplit(version, '.');
hash = ''; % In a future implementation

if nargin < 1
    verbose = 0;
end

if verbose
    tapas_print_logo();
    fprintf(1, '\n\nVersion %s.%s.%s\n', version{:});
    fprintf(1, 'In your citation please include the current version.\n');
    fprintf(1, 'Please cite the corresponding publications according to the toolboxes used:\n');
    fprintf(1, 'PhysIO: https://www.ncbi.nlm.nih.gov/pubmed/27832957\n');
    fprintf(1, 'HGF:    https://www.ncbi.nlm.nih.gov/pubmed/21629826\n');
    fprintf(1, '        https://www.ncbi.nlm.nih.gov/pubmed/25477800\n');
    fprintf(1, 'MPDCM:  https://www.ncbi.nlm.nih.gov/pubmed/26384541\n');
    fprintf(1, 'SERIA:  https://www.ncbi.nlm.nih.gov/pubmed/28767650\n');
    fprintf(1, 'HUGE:   https://www.ncbi.nlm.nih.gov/pubmed/29964187\n');
    fprintf(1, 'rDCM:   https://www.ncbi.nlm.nih.gov/pubmed/29807151\n');
    fprintf(1, '        https://www.ncbi.nlm.nih.gov/pubmed/28259780\n');
end

end

