function [config] = tapas_get_config(tpath)
%% Loads the configuration of the project.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

fname = fullfile(tpath, 'config.static.ini');

config = ini2struct(fname);

fname = fullfile(tpath, 'config.ini');
tc = ini2struct(fname);
tf = fields(tc);

for i = 1:numel(tf)
    config = setfield(config, tf{i}, getfield(tc, tf{i}));
end

end

function Struct = ini2struct(FileName)
% Parses .ini file
% Returns a structure with section names and keys as fields.
%
% Based on init2struct.m by Andriy Nych
% 2014/02/01

f = fopen(FileName,'r');                    % open file
while ~feof(f)                              % and read until it ends
    s = strtrim(fgetl(f));                  % remove leading/trailing spaces
    if isempty(s) || s(1)==';' || s(1)=='#' % skip empty & comments lines
        continue
    end
    if s(1)=='['                            % section header
        Section = genvarname(strtok(s(2:end), ']'));
        Struct.(Section) = [];              % create field
        continue
    end
    
    [Key,Val] = strtok(s, '=');             % Key = Value ; comment
    Val = strtrim(Val(2:end));              % remove spaces after =
    
    if isempty(Val) || Val(1)==';' || Val(1)=='#' % empty entry
        Val = [];
    elseif Val(1)=='"'                      % double-quoted string
        Val = strtok(Val, '"');
    elseif Val(1)==''''                     % single-quoted string
        Val = strtok(Val, '''');
    else
        Val = strtok(Val, ';');             % remove inline comment
        Val = strtok(Val, '#');             % remove inline comment
        Val = strtrim(Val);                 % remove spaces before comment
        
        [val, status] = str2num(Val);       %#ok<ST2NM>
        if status, Val = val; end           % convert string to number(s)
    end
    
    if ~exist('Section', 'var')             % No section found before
        Struct.(genvarname(Key)) = Val;
    else                                    % Section found before, fill it
        Struct.(Section).(genvarname(Key)) = Val;
    end

end
fclose(f);
end
