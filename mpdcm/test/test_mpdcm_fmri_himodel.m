function test_mpdcm_fmri_himodel(fp)
%% Test 
%
% fp -- Pointer to a file for the test output, defaults to 1
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 1
    fp = 1;
end

fname = mfilename();
fname = regexprep(fname, 'test_', '');


fprintf(fp, '================\n Test %s\n================\n', fname);


dcm = cell(10, 1);
tdcm = test_mpdcm_fmri_load_td();
dcm(:) = tdcm(1);


% Test whether there is any clear bug
try
    mpdcm_fmri_himodel(dcm)
    fprintf(fp, '       Passed\n');
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
    fprintf(fp, getReport(err, 'extended'));
end


end

