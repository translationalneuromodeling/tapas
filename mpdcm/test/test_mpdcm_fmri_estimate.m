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
    pars.T = linspace(1e-1, 1, 40).^5;
    pars.nburnin = 300;
    pars.niter = 700;

    dcm = mpdcm_fmri_estimate(d{1}, pars);

    fprintf('Fe: %0.5f', dcm.F);
    display('    Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
    disp(getReport(err, 'extended'));
end
profile off
profile viewer
end
