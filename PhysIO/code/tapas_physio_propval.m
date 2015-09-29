function [merged unused] = tapas_physio_propval(propvals, defaults, varargin)

% Create a structure combining property-value pairs with default values.
%
% [MERGED UNUSED] = PROPVAL(PROPVALS, DEFAULTS, ...)
%
% Given a cell array or structure of property-value pairs
% (i.e. from VARARGIN or a structure of parameters), PROPVAL will
% merge the user specified values with those specified in the
% DEFAULTS structure and return the result in the structure
% MERGED.  Any user specified values that were not listed in
% DEFAULTS are output as property-value arguments in the cell array
% UNUSED.  STRICT is disabled in this mode.
%
% ALTERNATIVE USAGE:
% 
% [ ARGS ] = PROPVAL(PROPVALS, DEFAULTS, ...)
%
% In this case, propval will assume that no user specified
% properties are meant to be "picked up" and STRICT mode will be enforced.
% 
% ARGUMENTS:
%
% PROPVALS - Either a cell array of property-value pairs
%   (i.e. {'Property', Value, ...}) or a structure of equivalent form
%   (i.e. struct.Property = Value), to be merged with the values in
%   DEFAULTS.
%
% DEFAULTS - A structure where field names correspond to the
%   default value for any properties in PROPVALS.
%
% OPTIONAL ARGUMENTS:
% 
% STRICT (default = true) - Use strict guidelines when processing
%   the property value pairs.  This will warn the user if an empty
%   DEFAULTS structure is passed in or if there are properties in
%   PROPVALS for which no default was provided.
%
% EXAMPLES:
%
% Simple function with two optional numerical parameters:
% 
% function [result] = myfunc(data, varargin)
% 
%   defaults.X = 5;
%   defaults.Y = 10;
%
%   args = propvals(varargin, defaults)
%
%   data = data * Y / X;
% 
% >> myfunc(data)
%    This will run myfunc with X=5, Y=10 on the variable 'data'.
%
% >> myfunc(data, 'X', 0)
%    This will run myfunc with X=0, Y=10 (thus giving a
%    divide-by-zero error)
%
% >> myfunc(data, 'foo', 'bar') will run myfunc with X=5, Y=10, and
%    PROPVAL will give a warning that 'foo' has no default value,
%    since STRICT is true by default.
%

% License:
%=====================================================================
%
% This is part of the Princeton MVPA toolbox, released under
% the GPL. See http://www.csbmb.princeton.edu/mvpa for more
% information.
% 
% The Princeton MVPA toolbox is available free and
% unsupported to those who might find it useful. We do not
% take any responsibility whatsoever for any problems that
% you have related to the use of the MVPA toolbox.
%
% ======================================================================

% Backwards compatibility
pvdef.ignore_missing_default = false;
pvdef.ignore_empty_defaults = false;

% check for the number of outputs
if nargout == 2
  pvdef.strict = false;
else
  pvdef.strict = true;
end

pvargs = pvdef;

% Recursively process the propval optional arguments (possible
% because we only recurse if optional parameters are given)
if ~isempty(varargin) 
  pvargs = propval(varargin, pvdef);
end

% NOTE: Backwards compatibility with previous version of propval
if pvargs.ignore_missing_default | pvargs.ignore_empty_defaults
  pvargs.strict = false;
end

% check for a single cell argument; assume propvals is that argument
if iscell(propvals) && numel(propvals) == 1 
  propvals = propvals{1};
end

% check for valid inputs
if ~iscell(propvals) & ~isstruct(propvals) & ~isempty(propvals)
  error('Property-value pairs must be a cell array or a structure.');
end

if ~isstruct(defaults) & ~isempty(defaults)
  error('Defaults struct must be a structure.');
end

% check for empty defaults structure
if isempty(defaults)
  if pvargs.strict & ~pvargs.ignore_missing_default
   error('Empty defaults structure passed to propval.');
  end
  defaults = struct();
end

defaultnames = fieldnames(defaults);
defaultvalues = struct2cell(defaults);

% prepare the defaults structure, but also prepare casechecking
% structure with all case stripped
defaults = struct();
casecheck = struct();

for i = 1:numel(defaultnames)
  defaults.(defaultnames{i}) = defaultvalues{i};
  casecheck.(lower(defaultnames{i})) = defaultvalues{i};
end

% merged starts with the default values
merged = defaults;
unused = {};
used = struct();

properties = [];
values = [];

% If propvals is empty, quit
if isempty(propvals)
    return;
end

% To extract property value pairs, we use different methods
% depending on how they were passed in
if isstruct(propvals)   
  properties = fieldnames(propvals);
  values = struct2cell(propvals);
else
  properties = { propvals{1:2:end} };
  values = { propvals{2:2:end} };
end

if numel(properties) ~= numel(values)
  error(sprintf('Found %g properties but only %g values.', numel(properties), ...
                numel(values)));
end

% merge new properties with defaults
for i = 1:numel(properties)

  if ~ischar(properties{i})
    error(sprintf('Property %g is not a string.', i));
  end

  % convert property names to lower case
  properties{i} = properties{i};

  % check for multiple usage
  if isfield(used, properties{i})
    error(sprintf('Property %s is defined more than once.\n', ...
                  properties{i}));
  end
  
  % Check for case errors
  if isfield(casecheck, lower(properties{i})) & ...
    ~isfield(merged, properties{i}) 
    error(['Property ''%s'' is equal to a default property except ' ...
           'for case.'], properties{i});
  end  
    
  % Merge with defaults  
  if isfield(merged, properties{i})
    merged.(properties{i}) = values{i};
  else
    % add to unused property value pairs
    unused{end+1} = properties{i};
    unused{end+1} = values{i};    

    % add to defaults, just in case, if the user isn't picking up "unused"
    if (nargout == 1 & ~pvargs.strict)
      merged.(properties{i}) = values{i};
    end

    if pvargs.strict
      error('Property ''%s'' has no default value.', properties{i});
    end
    
  end

  % mark as used
  used.(properties{i}) = true;
end

