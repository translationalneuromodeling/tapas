function [status] = tapas_mpdcm_fmri_destroy_container(container)
%% Destroy a container clearing the memory.
%
% Input
%       container       -- A container object.
% Output
%       status          -- Value representing the status.
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%

[status] = c_mpdcm_destroy_container(container);

end

