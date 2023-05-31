function tapas_physio_interactive_noiseROI_mask( mask_path )
%TAPAS_PHYSIO_INTERACTIVE_NOISEROI_MASK function will help visualizing the
%effect of 'threshold' and 'erosion' in tapas_physio_create_noise_rois_regressors()
%
% SYNTAX
%   tapas_physio_interactive_noiseROI_mask() % a popup window will be used to select a mask
%   tapas_physio_interactive_noiseROI_mask('path/to/my_mask.nii')
%
% NOTES
%   The noiseROI will be created on the fly and kept in memory, it will not
%   be written on disk.
%   Selected values can be saved as global variables for latter usage.
%

default_threshold = 0.95; % from 0 to 1
default_erosion   = 1;    % 0, 1, 2, ...


%% Mask selection

if nargin < 1
    mask_path = spm_select(1,'image','Select a mask');
    if isempty(mask_path)
        return
    end
end


%% Load mask & display it

Vmask = spm_vol(mask_path);
Ymask = spm_read_vols(Vmask);

spm_check_registration(Vmask);
spm_orthviews('context_menu','interpolation',3); % disable interpolation // 3->NN , 2->Trilin , 1->Sinc


%% Vroi
% the noiseROI will be computed and displayed on the fly without writting to the disk

% copy header
Vroi = Vmask;
% change some settings so the volume remains "virtual"
% thanks to @gllmflndn for helping on this part
Vroi.dt = [spm_type('float64') 0];
Vroi.pinfo = [1 ; 0 ; 0];

[Yroi , descrip ] = threshold_erode_mask(Ymask, default_threshold, default_erosion);
Vroi.fname = descrip; % only useful for the display in the menu

Vroi.dat = Yroi; % this is not a file object, just a standard matrix
print_stats( Yroi )


%% UserData (useful for the GUI)

UserData = struct;
UserData.Vmask = Vmask;
UserData.Ymask = Ymask;
UserData.Vroi  = Vroi;


%% GUI
% add a panel so the user can enter the values in the SPM figure

F = spm_figure('FindWin','Graphics');

% this code might not work in old version of Matlab, since it use the "object" syntax
% old versions might have to use get() & set() functions
panel = uipanel(F,...
    'Title','Threshold & Erosion',...
    'Units', 'Normalized',...
    'Position',[...
    F.Children(2).Position(1)...
    F.Children(4).Position(2)...
    F.Children(2).Position(3)...
    F.Children(4).Position(4)...
    ],...
    'UserData', UserData);

% from botton to top, since matlab figure origin (0,0) is in the bottom left corner

uicontrol(panel,...
    'Tag', 'pushbutton_save',...
    'Style','pushbutton',...
    'Units','normalized',...
    'Position',[0.1 0.1 0.8 0.2],...
    'String', 'Save values & close',...
    'Tooltip','Saved as global variables',...
    'Callback',@save_values_Callback);

uicontrol(panel,...
    'Tag','text_erosion',...
    'Style','Text',...
    'Units', 'Normalized',...
    'Position',[0.1 0.4 0.3 0.25],...
    'String', '3D erosion');

uicontrol(panel,...
    'Tag','edit_erosion',...
    'Style','Edit',...
    'Units', 'Normalized',...
    'Position',[0.4 0.4 0.4 0.25],...
    'String', num2str(default_erosion),...
    'Callback', @update_noiseROI_Callback,...
    'Tooltip','integer values, like 0 1 2 3 ...');

uicontrol(panel,...
    'Tag','text_threshold',...
    'Style','Text',...
    'Units', 'Normalized',...
    'Position',[0.1 0.7 0.3 0.25],...
    'String', 'Threshold');

uicontrol(panel,...
    'Tag','edit_threshold',...
    'Style','Edit',...
    'Units', 'Normalized',...
    'Position',[0.4 0.7 0.4 0.25],...
    'String', num2str(default_threshold),...
    'Callback', @update_noiseROI_Callback,...
    'Tooltip', 'from 0 to 1');


%% Add the overlay
% code from spm_orthviews:add_c_image

global st % global variable used by using spm_orthviews()

r = 1;

spm_orthviews('addcolouredimage',r,Vroi ,[1 0 0]);
hlabel = sprintf('%s (%s)',Vroi.fname ,'Red');
c_handle    = findobj(findobj(st.vols{r}.ax{1}.cm,'label','Overlay'),'Label','Remove coloured blobs');
ch_c_handle = get(c_handle,'Children');
set(c_handle,'Visible','on');
uimenu(ch_c_handle(2),'Label',hlabel,'ForegroundColor',[1 0 0],...
    'Callback','c = get(gcbo,''UserData'');spm_orthviews(''context_menu'',''remove_c_blobs'',2,c);',...
    'Tag','noiseROI_menu');
spm_orthviews('redraw')


end % function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ Yroi , descrip ] = threshold_erode_mask(Ymask, threshold, erosion)

Yroi = Ymask/max(abs(Ymask(:))); % normamalize input range to [0..1]
Yroi(Yroi< threshold) = 0;
Yroi(Yroi>=threshold) = 1;

for i = 1 : erosion
    Yroi = spm_erode(Yroi);
end

descrip = sprintf('threshold = %4.2f // erosion = %d', threshold, erosion);

end % function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function print_stats( Yroi )

fprintf('[%s]: n_Voxels_in_ROI = %6d \n', mfilename, sum(Yroi(:)));

end % function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_noiseROI_Callback(hObject,~)

global st % to overlay the final ROIs, using spm_orthviews

% Retrieve UserData
panel = get(hObject,'Parent');
UserData = get(panel,'UserData');


% Check values
threshold = get( findobj('Tag', 'edit_threshold'), 'String');
threshold = str2double(threshold);
if threshold<0 || threshold>1 || isnan(threshold)
    threshold = 0.95;
    set( findobj('Tag', 'edit_threshold'), 'String', num2str(threshold));
end

erosion = get( findobj('Tag', 'edit_erosion'), 'String');
erosion = str2double(erosion);
erosion = round(erosion);
if erosion<0 || erosion>999 || isnan(erosion)
    erosion = 1;
    set( findobj('Tag', 'edit_erosion'), 'String', num2str(erosion));
end


% Apply new values on mask
[ Yroi , descrip ] = threshold_erode_mask(UserData.Ymask, threshold, erosion);
UserData.Vroi.dat = Yroi;
UserData.Vroi.name = descrip;
print_stats( Yroi )


% Redraw : update overlay
st.vols{1}.blobs{1}.vol = UserData.Vroi;
handles = findobj('Tag', 'noiseROI_menu');
handles.Text = sprintf('%s (Red)', descrip);
spm_orthviews('redraw')


end % function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function save_values_Callback(~,~)

F = spm_figure('FindWin','Graphics');

threshold = get( findobj('Tag', 'edit_threshold'), 'String');
threshold = str2double(threshold);

erosion = get( findobj('Tag', 'edit_erosion'), 'String');
erosion = str2double(erosion);
erosion = round(erosion);

global tapas_physio_interactive_noiseROI_mask_threshold tapas_physio_interactive_noiseROI_mask_erosion
tapas_physio_interactive_noiseROI_mask_threshold = threshold;
tapas_physio_interactive_noiseROI_mask_erosion   = erosion;

fprintf('[%s]: Values saved in global variables : \n', mfilename);
fprintf('tapas_physio_interactive_noiseROI_mask_threshold = %4.2f \n', threshold);
fprintf('tapas_physio_interactive_noiseROI_mask_erosion   = %d \n', erosion);
fprintf('Retrieve them using ''global tapas_physio_interactive_noiseROI_mask_threshold tapas_physio_interactive_noiseROI_mask_erosion'' \n')

delete(F)

end % function
