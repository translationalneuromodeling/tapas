function data = read_cpx(file, border, flip_img, kspace, read_params, compression_parameter)

%--------------------------------------------------------------------------
% data = read_cpx(file, border, flip_img)
%
% read_cpx: Function for reading the whole cpx-file and writes the data in
%           a predefined 9 dimensional array
%
% Input:		file		cpx-file must be given in ''. e.g. 'file.cpx'
%               border      0 returns the original sized image
%                           1 makes a square image
%               flip_img    0 no flip of the image
%                           1 flips the image (like the scanner does)
%               read_params optional input. Specifies the images to be
%                           read. read_params is a struct created
%                           by the function create_read_param_struct
%
% Output:       data        A 9 dimensional array which consists all the
%                           data in the cpx file. The array has the
%                           following structure:
%                           Column: 1:resolution y, 2:resolution x 3:Stack,
%                                   4:Slice, 5:Coil, 6:Heart phase, 7:Echo,
%                                   8:Dynamics, 9:Segments, 10:Segments2
%
% Notice! For scans with body coil and surface coil in different stacks,
% e.g. SENSE reference scan, the body coil is numbered as coil no. 1 in
% stack no. 2, the surface coils are numbered as coils 2,3,4... in stack no. 1!
%------------------------------------------------------------------------


switch nargin
    case 5
        compression_parameter = [];
    case 4
        read_params = create_read_param_struct(file);
        compression_parameter = [];
    case 3
        read_params = create_read_param_struct(file);
        kspace = 0;
        compression_parameter = [];
    case 2
        flip_img = 0;
        kspace = 0;
        read_params = create_read_param_struct(file);
        compression_parameter = [];
    case 1
        flip_img = 0;
        border = 0;
        kspace = 0;
        read_params = create_read_param_struct(file);
        compression_parameter = [];
end

%Reads the header of the file
header = read_cpx_header(file,'no');
[rows,columns] = size(header);

% Calculates the number of slices, coils, etc...
stacks   = length(read_params.loca);
slices   = length(read_params.slice);
coils    = length(read_params.coil);
hps      = length(read_params.phase);
echos    = length(read_params.echo);
dynamics = length(read_params.dyn);
segments = length(read_params.seg);
segments2= length(read_params.seg2);


res_x       = header(1,9);
res_y       = header(1,10);
compression = header(1,11);
flip        = header(1,12);

offset_table_cpx=create_offset_table(header);

% defines the array for the output
if border
    res = max([res_x, res_y]);
    res_x = res;
    res_y = res;
end

if ~isempty(compression_parameter)
    data = zeros(res_x, res_y, stacks, slices, compression_parameter{1}, hps, echos, dynamics, segments, segments2,'single');
    data3 = zeros(res_x, res_y,coils);
else
    data = zeros(res_x, res_y, stacks, slices, coils, hps, echos, dynamics, segments, segments2,'single');
    data3 = zeros(res_x, res_y,coils);
end

% Define a waitbar
h = waitbar(0, 'Loading file...');
set(h,'Units','pixels')
scnsize = get(0,'ScreenSize');
set(h,'Position',[floor(scnsize(3)/2)-160,floor(scnsize(4)/2)-30,360,75]);

fid = fopen(file);

% Runs through all images in the file, reads them and writes it in the
% correct position in the output array "data"
i = 1;
total_loops = 1;
for loop = 1:2
    for st = 1:stacks
        for sl = 1:slices
            for se2 = 1:segments2
                for ph = 1:hps
                    for ec = 1:echos
                        for dy = 1:dynamics
                            for se = 1:segments
                                for co = 1:coils
                                    offset = offset_table_cpx(read_params.loca(st),read_params.slice(sl),read_params.coil(co),read_params.phase(ph),read_params.echo(ec),read_params.dyn(dy),read_params.seg(se),read_params.seg2(se2));
                                    if offset >=0
                                        if loop == 2
                                            image = read_cpx_image(file, offset, border, flip_img);
                                            if kspace
                                                image = fftshift(fft2(fftshift(image)));
                                            end
                                            data3(:,:,co) = image;                                            
                                            waitbar(i/total_loops,h)
                                            i = i+1;
                                        else
                                            total_loops = total_loops +1;
                                        end
                                    end
                                end
                                if ~isempty(compression_parameter)
%                                     data_temp= squeeze(combine_data(reshape(data3,size(data3,1),size(data3,2), 1,size(data3,3)),compression_parameter{2}));
                                    data(:,:,st,sl,:,ph,ec,dy,se, se2) = reshape(combine_data_gui(reshape(data3,size(data3,1),size(data3,2), 1,size(data3,3)),compression_parameter{2}),size(data3,1),size(data3,2),1,1,compression_parameter{1});
                                else
                                    data(:,:,st,sl,:,ph,ec,dy,se, se2) = reshape(data3,size(data3,1),size(data3,2),1,1,size(data3,3));
                                end
                                
                            end
                        end
                    end
                end
            end
        end
    end
end

close(h);
fclose all;