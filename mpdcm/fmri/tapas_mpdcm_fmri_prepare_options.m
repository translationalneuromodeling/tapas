function [dcm] = tapas_mpdcm_fmri_prepare_options(dcm)
%% Sets the default options
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~isfield(dcm, 'options')
    dcm.options = struct();
end

if ~isfield(dcm.options, 'rescale_y')
    dcm.options.rescale_y = 0;
end

if ~isfield(dcm.options, 'detrend_y')
    dcm.options.detrend_y = 0;
end

if ~isfield(dcm.options, 'centre')
    dcm.options.detrend_u = 0;
end

if ~isfield(dcm.options, 'smooth_u')
    dcm.options.smooth_u = 0;
end

end

