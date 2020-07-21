function tapas_h2gf_validate_data(data, model, inference)
%% Validates the correctness of the data
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%

tapas_validate_data(data);

sdata = size(data);

if numel(sdata) ~= 2
    error('tapas:h2gf:data', 'Data should have two dimension');
end

if sdata(2) ~= 1
    error('tapas:h2gf:data', 'Second dimension should be 1 instead %d', ...
        sdata(2));
end

% Check also if the irr and ign trials are not there

if ~isfield(data, 'irr')
    error('tapas:h2gf:validate:data:irr', ...
        'irr field must be specified')
end
if ~isfield(data, 'ign')
    error('tapas:h2gf:validate:data:ign', ...
        'ign field must be specified')
end


ns = size(model.graph{1}.htheta.T, 1);

if ns ~= sdata(1)
    error('tapas:h2gf:data', ...
        'First dimension of T is %d, number of subjects is %d', ns, sdata(1));
end

end

