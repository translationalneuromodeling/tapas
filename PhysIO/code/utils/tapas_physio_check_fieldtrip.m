function verbose = tapas_physio_check_feldtrip(verbose)
% Will remove found FieldTrip installations
% from matlab path
% FieldTrip is conficting with PhysIO, see
% https://github.com/translationalneuromodeling/tapas/issues/166
% https://github.com/translationalneuromodeling/tapas/issues/180

ft_paths = which('ft_defaults', '-all');

if isempty(ft_paths)
    return;
end

w_msg = ['Removing FieldTrip paths (',...
         'https://github.com/translationalneuromodeling/',...
         'tapas/issues/166',...
         ')\n'];

warning('off', 'MATLAB:rmpath:DirNotFound');
for p = 1:size(ft_paths, 1)
    ft_paths{p} = fileparts(ft_paths{p});
    w_msg = [w_msg, '\t', ft_paths{p}, '\n']; %#ok<AGROW>
    rmpath(genpath(ft_paths{p}));
end
warning('on', 'MATLAB:rmpath:DirNotFound');

if verbose.level >= 3
  msg = sprintf(['You can restore FieldTrip path using:\n\t',...
                 'addpath(''%s'');\n\tft_defaults;'],...
                ft_paths{1});
  w_msg = [w_msg, '\n', msg];
end
verbose = tapas_physio_log(w_msg, verbose, 1);

end
