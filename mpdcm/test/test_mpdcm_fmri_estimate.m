function test_mpdcm_fmri_estimate()
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

d = test_mpdcm_fmri_load_td();

try
profile off
profile clear
profile on
    pars = struct();
    pars.T = linspace(1e-1, 1, 30).^5;
    pars.nburnin = 200;
    pars.niter = 200;

    dcm = mpdcm_fmri_estimate(d{1}, pars);

    display('    Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end
profile off
profile viewer
end
