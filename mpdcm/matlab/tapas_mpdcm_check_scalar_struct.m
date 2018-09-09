function [] = tapas_mpdcm_check_scalar_struct(astruct, fields)
%% Checks that all the fields in the matrix exist and that they are scalars.
%
% Input
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%


for field = fields
    assert(isfield(astruct, field), ...
        sprintf('mpdcm:check_input:%s', field), ...
        sprintf('Field %s doesn''t exists', field));
    tapas_mpdcm_check_input_matrix(getfield(astruct, field), [1, 1]);
end



end % tapas_mpdcm_check_scalar_struct 

