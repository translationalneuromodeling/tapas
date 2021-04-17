function [ obj ] = import( obj, listDcms, listConfounds, omit, bAppend )
% Import fMRI time series data into HUGE object.
%
%   WARNING: Importing data into a HUGE object will delete any data and
%   results which are already stored in that object.
%   
% INPUTS:
%   obj  - A tapas_Huge object.
%   dcms - A cell array containing DCM structs in SPM's DCM format.
% 
% OPTIONAL INPUTS:
%   confounds - Group-level confounds (e.g., age, sex, etc). 'confounds'
%               must be empty or an array with as many rows as there are
%               elements in 'dcm'.
%   omit      - specifies DCM parameters which should be omitted from
%               clustering. Parameters omitted from clustering will still
%               be estimated, but under a static Gaussian prior. 'omit'
%               should be a struct with fields a, b, c, and/or d which are
%               bool arrays with sizes matching the corresponding fields of
%               the DCMs. If omit is an array, it is interpreted as the
%               field a. If omit is 1, it is expanded to an identity matrix
%               of suitable size.
%   append    - bool, if true keep current fMRI time series and append new
%               data in 'dcms'. Note: estimation results will still be
%               deleted.
% 
% OUTPUTS:
%   obj - A tapas_Huge object containing the imported data.
% 
% EXAMPLES:
%   [obj] = IMPORT(obj,dcms)    Import the fMRI time series and DCM
%       network structure stored in dcms into obj.
% 
%   [obj] = IMPORT(obj,dcms,confounds)    Import group-level confounds
%       (like age or sex) in addition to fMRI data.
% 
%   [obj] = IMPORT(obj,dcms,[],1)    Import fMRI data and network
%       structure. Exclude self-connections from clustering. 
%
% See also tapas_Huge.REMOVE, tapas_Huge.EXPORT
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 


%% process inputs
if nargin < 5
    bAppend = false;
end

if nargin < 4
    omit = [];
end
if ~isempty(omit)
    assert(~isempty(listDcms), 'TAPAS:HUGE:import:missingDcm', ...
        'Omitting parameters can only be done while importing DCMs.');
end

if nargin < 3
    listConfounds = [];
end
if ~isempty(listConfounds)
    assert(~isempty(listDcms), 'TAPAS:HUGE:import:missingDcm', ...
        'Confounds can only be imported in combination with DCMs.');
end

if isempty(listDcms)
    return
end

if ~bAppend
    % remove current data
    obj = obj.remove( );
end


%% convert DCM list
if isvector(listDcms)&&isstruct(listDcms)
    try
        listDcms = {listDcms(:).DCM}';
    catch
        listDcms = num2cell(listDcms);
    end
else
    assert(iscell(listDcms),'TAPAS:HUGE:inputFormat',...
        'listDcms must be cell array of DCMs in SPM format');
end


%% check confounds list
if ~isempty(listConfounds)
    
    assert( size(listConfounds,1) == numel(listDcms), ...
        'TAPAS:HUGE:confSize', ['Number of rows in listConfounds must' ...
        ' match number of entries in listDCMs.']);
     
    assert(obj.M == 0 || size(listConfounds,2) == obj.M, ...
        'TAPAS:HUGE:confSize', ['Number of colums in listConfounds does' ...
        ' not match dimension of confounds already present.']);
    
    obj.M = size(listConfounds,2);
     
end


%% add DCMs subject by subject
for n = 1:numel(listDcms)
    
    dcm = listDcms{n};    
    try
        % check DCM structure
        check_dcm(obj, dcm);
    
        % update N
        obj.N = obj.N + 1;

        % init object's DCM structure
        if isempty(obj.dcm)
            obj = init_dcm(obj, dcm);
        end

        % bold signal
        assert(size(dcm.Y.y, 2) == obj.R, ...
            'TAPAS:HUGE:import:sizeMismatch', ...
            'Number of BOLD time series does not match number of regions');
        % center data
        dcm.Y.y = bsxfun(@minus, dcm.Y.y, mean(dcm.Y.y));
        % echo time
        if ~isfield(dcm,'TE')
            dcm.TE = 0.04; % default value for echo time: 40ms
            fprintf(['Missing echo time for subject %u. ' ...
                     'Using %.3f seconds instead.\n'], n, dcm.TE)
        end
        % 1st-level confounds
        if isfield(dcm.Y, 'X0')
            assert(size(dcm.Y.X0, 1) == size(dcm.Y.y, 1), ...
                'TAPAS:HUGE:import:sizeMismatch', ...
                'Size of 1st-level confounds does not match BOLD');
        else
            dcm.Y.X0 = [];
        end
        if isempty(dcm.Y.X0)
            dcm.res = [];
        else
        % calculate residual forming matrix and remove 1st-lvl confounds
            dcm.res = eye(size(dcm.Y.X0,1)) - ...
                dcm.Y.X0*((dcm.Y.X0'*dcm.Y.X0 + ...
                eye(size(dcm.Y.X0,2))*1e-10)\dcm.Y.X0');
            dcm.Y.y = dcm.res*dcm.Y.y;
        end
        % group-level confounds
        if isempty(listConfounds)
            confounds = zeros(obj.M, 1);
        else
            confounds = listConfounds(n,:)';
        end
        % add Ep field as initial value
        if ~isfield(dcm, 'Ep')
            dcm.Ep = [];
        end

        % assemble data
        obj.data(obj.N, 1) = struct( ...
            'bold',      dcm.Y.y, ... % BOLD time series
            'te',        dcm.TE, ... % echo time TE
            'tr',        dcm.Y.dt, ... % repetition time TR
            'X0',        dcm.Y.X0, ... % subject-level confounds
            'res',       dcm.res, ... % residual forming matrix
            'confounds', confounds, ... % group-level confounds
            'spm',       dcm.Ep);     % posterior from SPM

        % inputs
        obj.inputs(obj.N, 1) = struct('u', dcm.U.u, 'dt', dcm.U.dt);
    
    catch err
        fprintf('Import: Error adding subject %u.\n', n)
        rethrow(err)
    end

end

% reset prior and posterior
obj.prior = [];
obj.posterior = [];
obj.trace = [];

% calculate indices and number of parameters
obj.idx = calculate_index( obj.dcm, obj.R, omit );


end



function [ obj ] = init_dcm( obj, dcm )
% initialize DCM structure and labels

obj.dcm = dcm;
% regions
obj.R = dcm.n;
if isfield(dcm.Y, 'name')
    obj.labels.regions = dcm.Y.name;
else
    s = repmat({'region '}, 1, obj.R);
    r = cellfun(@num2str, num2cell(1:obj.R), 'UniformOutput', false);
    obj.labels.regions = strcat(s, r);
end
% inputs
obj.L = size(dcm.U.u, 2);
if isfield(dcm.U, 'name')
    obj.labels.inputs = dcm.U.name;
else
    s = repmat({'input '}, 1, obj.L);
    l = cellfun(@num2str, num2cell(1:obj.L), 'UniformOutput', false);
    obj.labels.inputs = strcat(s, l);
end

end



function [ idx ] = calculate_index( dcm, R, omit )
%CALCULATE_INDEX calculate indices and number of parameters for current DCM
%stucture

if isempty(omit)
    omit = struct();
elseif isnumeric(omit)
    omit = struct('a', omit);
else
    assert(isstruct(omit), 'TAPAS:HUGE:import:inputType', ...
        'omit must be a struct with fields a, b, c or d');
end
% expand scalar 'a' field to identity matrix
if isfield(omit,'a') && isscalar(omit.a) && omit.a == 1
    omit.a = eye(R);
end

idx = struct( );
% indices of parameters to cluster
clustering = dcm_and(dcm, omit, true);
indicators = [clustering.a(:); clustering.b(:); clustering.c(:); ...
    clustering.d(:)];
idx.clustering = find(indicators);
assert(~isempty(idx.clustering), 'TAPAS:HUGE:import:omit', ...
    ['Cannot exclude all DCM parameters from clustering. If you do' ...
     ' not wish to perform clustering, set K=1 (empirical Bayes).']);
% indices of parameters not to cluster
homogenous = dcm_and(dcm, omit, false);
indicators = [homogenous.a(:); homogenous.b(:); homogenous.c(:); ...
    homogenous.d(:)];
idx.homogenous = [find(indicators); length(indicators) + (1:R*2+1)'];

% number of clustering and homogenous parameters
idx.P_c = numel(idx.clustering);
idx.P_h = numel(idx.homogenous);
% number of parameters of fully connected model
idx.P_f = numel(indicators) + R*2 + 1;

end


function [ dcm1 ] = dcm_and( dcm1, dcm2, bInv )

if nargin < 3
    bInv = false;
end

% if a field does not exist in dcm2, treat it as a matrix of false with
% same size as the corresponding field in dcm1
if isfield(dcm2, 'a')
    dcm1.a = dcm1.a & ((dcm2.a & ~bInv) | (~dcm2.a & bInv));
else
    dcm1.a(:) = dcm1.a(:) & bInv;
end

if isfield(dcm2, 'b')
    dcm1.b = dcm1.b & ((dcm2.b && ~bInv) | (~dcm2.b & bInv));
else
    dcm1.b(:) = dcm1.b(:) & bInv;
end

if isfield(dcm2, 'c')
    dcm1.c = dcm1.c & ((dcm2.c & ~bInv) | (~dcm2.c & bInv));
else
    dcm1.c(:) = dcm1.c(:) & bInv;
end

if isfield(dcm2, 'd')
    dcm1.d = dcm1.d & ((dcm2.d & ~bInv) | (~dcm2.d & bInv));
else
    dcm1.d(:) = dcm1.d(:) & bInv;
end

end
%%
% 
% 
% 
function [ ] = check_dcm( obj, dcm )
% CHECK_DCM compare dcm structures

% check DCM structure (SPM format)
assert(isfield(dcm,'n'),...
    'TAPAS:HUGE:Import:MissingR','Missing number of regions.')

assert(isfield(dcm,'a'),'TAPAS:HUGE:Import:MissingA','Missing A matrix.')

assert(isfield(dcm,'b'),'TAPAS:HUGE:Import:MissingB','Missing B matrix.')

assert(isfield(dcm,'c'),'TAPAS:HUGE:Import:MissingC','Missing C matrix.')

assert(isfield(dcm,'d'),'TAPAS:HUGE:Import:MissingD','Missing D matrix.')

% check data field
assert(isfield(dcm,'Y') && isfield(dcm.Y,'y'),...
    'TAPAS:HUGE:Import:MissingBOLD','Missing BOLD time series.')

assert(isfield(dcm,'Y') && isfield(dcm.Y,'dt'),...
    'TAPAS:HUGE:Import:MissingTR','Repetition time is missing.')

% check input field
assert(isfield(dcm,'U') && isfield(dcm.U,'u'),...
    'TAPAS:HUGE:Import:MissingInput','Missing exprimental input.')

assert(isfield(dcm,'U') && isfield(dcm.U,'dt'),...
    'TAPAS:HUGE:Import:MissingFs',...
    'Sampling interval of experimental input is missing.')

rSmp = dcm.Y.dt/dcm.U.dt;
assert(rem(rSmp,1)==0,'TAPAS:HUGE:Import:MismatchFs',...
    'Ratio between TR and input sampling interval must be integer.');

% check array sizes
assert(mod(numel(dcm.Y.y), dcm.n) == 0,...
    'TAPAS:HUGE:Import:MismatchSize',...
    'Number of BOLD samples inconsistent with number of regions.')
assert(length(dcm.U.u(rSmp:rSmp:end,:)) == length(dcm.Y.y),...
    'TAPAS:HUGE:MismatchSize',...
    'Number of BOLD samples inconsistent with number of input samples.');


% check for consistency with current DCM structure
if isempty(obj.dcm)
    return
end

assert(obj.R == dcm.n,'TAPAS:HUGE:Import:MismatchR',...
    'Number of regions does not match.')

assert(obj.L == size(dcm.U.u,2), 'TAPAS:HUGE:MismatchU',...
    'Number of experimental inputs does not match.');

assert(all(obj.dcm.a(:) == dcm.a(:)),...
    'TAPAS:HUGE:Import:MismatchA','A matrix does not match.')

assert(all(obj.dcm.b(:) == dcm.b(:)),...
    'TAPAS:HUGE:Import:MismatchB','B matrix does not match.')

assert(all(obj.dcm.c(:) == dcm.c(:)),...
    'TAPAS:HUGE:Import:MismatchC','C matrix does not match.')

assert(all(obj.dcm.d(:) == dcm.d(:)),...
    'TAPAS:HUGE:Import:MismatchD','D matrix does not match.')

end


