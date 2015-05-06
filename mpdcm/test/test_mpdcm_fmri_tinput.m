function test_mpdcm_fmri_tinput(fp)
%% Test 
%
% fp -- Pointer to a file for the test output, defaults to 1
%
% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte
%
% Revision log:
%
%

if nargin < 1
    fp = 1;
end

fname = mfilename();
fname = regexprep(fname, 'test_', '');


fprintf(fp, '================\n Test %s\n================\n', fname);


end
