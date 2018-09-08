function tapas_mpdcm_check_input_ptheta(ptheta)
%% Checks ptheta
%
% Input
%
% Output
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%

assert(isstruct(ptheta), ...
    'mpdcm:int:input:ptheta:not_struct', ...         
    'ptheta should be a struct')

check_ptheta_scalar(ptheta, 'dt');

dt = getfield(ptheta, 'dt');

assert(0 < dt && dt <= 1, ...
    sprintf('mpdcm:int:input:theta:cell:%s:val', 'dt'), ...
    'ptheta.%s should not be < 0 and > 1', 'dt');

check_ptheta_scalar(ptheta, 'dyu');
dyu = getfield(ptheta, 'dyu');

assert(0 < dyu && dyu <= 1, ...
    sprintf('mpdcm:int:input:theta:cell:%s:val', 'dyu'), ...
    'ptheta.%s should not be < 0 and > 1', 'dyu');


check_ptheta_scalar(ptheta, 'udt');

end

function check_ptheta_scalar(ptheta, field)
%% Checks for scalar values in ptheta

assert(isfield(ptheta, field), ...
    sprintf('mpdcm:int:input:ptheta:%s:missing', field), ...
    'ptheta should have field %s', field);

ascalar = getfield(ptheta, field);

assert(isscalar(ascalar), ...
    sprintf('mpdcm:int:input:ptheta:%s:not_scalar', field), ...
    'ptheta.%s should be scalar', field);

assert(isnumeric(ascalar), ...
    sprintf('mpdcm:int:input:ptheta:%s:not_numeric', field), ...
    'ptheta.%s should be numeric', field);
assert(isreal(ascalar), ...
    sprintf('mpdcm:int:input:ptheta:%s:not_real', field), ...
    'ptheta.%s should be real', field);
assert(~issparse(ascalar), ...
    sprintf('mpdcm:int:input:theta:cell:%s:sparse', field), ...
    'ptheta.%s should not be sparse', field);

end
