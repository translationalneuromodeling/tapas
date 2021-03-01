function data = read_cpx_image(file, offset, border, flip_img)

%--------------------------------------------------------------------------
% data = read_cpx_image(file, offset, border, flip_img)
%
% read_cpx_image : Function for reading one image, defined through the
%                  input parameters
% 
% Input:		file		cpx-file must be given in ''. e.g. 'file.cpx'
%               offset      offset to the image from header (element (.,8))
%               border      0 returns the original sized image
%                           1 makes a square image
%               flip_img    0 no flip of the image
%                           1 flips the image (like the scanner does)
%
%               Notice! All numbers starts from 1. E.g.the first slice is 
%               numbered with 1
%
% Output:       data        The requested image in a 2 dimensional array
%
%
% Notice! For scans with body coil and surface coil in different stacks, 
% e.g. SENSE reference scan, the body coil is numbered as coil no. 1 in
% stack no. 2, the surface coils are numbered as coils 2,3,4... in stack no. 1!
%------------------------------------------------------------------------


%Reads the header of the requested image
fid = fopen(file);

fseek(fid, offset-512,'bof');
h1 = fread(fid, 15, 'long');
factor = fread(fid,2,'float');
h2 = fread(fid, 10,'long');

res_x = h1(11);
res_y = h1(12);
compression = h1(14);

%Reads the requested image
fseek(fid, offset,'bof');
switch (compression)
    case 1
        data = zeros(res_x*res_y*2,1,'single');
        data=fread(fid, res_x*res_y*2, 'float'); 
    case 2
        data = zeros(res_x*res_y*2,1,'single');
        data(:)=fread(fid, res_x*res_y*2,'short');
        data=factor(2)+factor(1).*data;
    case 4
        data = zeros(res_x*res_y*2,1,'single');
        data=fread(fid, res_x*res_y*2, 'int8');
        data=factor(2)+factor(1).*data;
end
data = complex(data(1:2:end),data(2:2:end));
data = reshape(data,res_x,res_y);

%Adds the border if requested
if border & (res_x ~= res_y)
    res = max([res_x, res_y]);
    data_temp = zeros(res, res);
    if res_x > res_y
        data_temp(:,floor((res - res_y)/2): res - ceil((res - res_y)/2+0.1)) = data;
    else
        data_temp(floor((res - res_x)/2): res - ceil((res - res_x)/2+0.1),:) = data;
    end
    data = data_temp;
    clear data_temp;           
end

%Flips the image if requested
if flip_img
    s = size(data);
    data = data(end:-1:1,:);
%     data=data';
%     data = data(:,s(1)+1-(1:s(1)));
%     data = data(s(2)+1-(1:s(2)),:);
end

fclose(fid);