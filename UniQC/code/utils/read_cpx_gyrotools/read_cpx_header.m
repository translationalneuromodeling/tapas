function header = read_cpx_header(file, output)

%--------------------------------------------------------------------------
% header = read_cpx_header(file)
%
% read_cpx_header: Function for reading the header of a cpx file
%
% Input:		file		cpx-file must be given in ''. e.g. 'file.cpx'
%               output      can be 'yes' or 'no'. Default is 'yes'
%                           Specifies if the information about the cpx file is written
%                           to the command line
%
% Output:       header      gives out the header of the cpx file as an
%                           array. The structure of this array is the
%                           following:
%                           Column: 1:Stack, 2:Slice, 3:Coil, 4:Heart phase, 
%                                   5:Echo, 6:Dynamics, 7:Segments, 
%                                   8:data offset, 9:Scaling factor1, 
%                                   10:Scaling factor2, 11:Compression,
%                                   12:Flip, 13: Scaling factor1, 14:
%                                   scaling factor2, 15: Mix, 16: Prep Dir,
%                                   17: Sequence Number 18: Segment2, 
%                                   19: Syncho Nr.
%                
%------------------------------------------------------------------------

if nargin == 1
    output = 'yes';
end

% Calculate the size of the cpx-file
fid = fopen(file);
fseek(fid, 0, 'eof');
filesize = ftell(fid);
fseek(fid,0,'bof');

% Read in the first header for 
h1 = fread(fid, 15, 'long');
factor = fread(fid,2,'float');
h2 = fread(fid, 111,'long');

res_x = h1(11);
res_y = h1(12);
compression = h1(14);
if ~h2(26)
    offset = h1(10);
else
    offset = h2(26);
end

matrix_data_blocks = h1(13);

% Calculates the number of images in the cpx-file 
image_exist = 1; i=0;
while image_exist
%     header_offset = (res_x * res_y * 8 /compression + offset)*i;
    header_offset = (matrix_data_blocks * 512 + offset)*i;
    fseek(fid, header_offset, 'bof');
    h1 = fread(fid, 15, 'long');
    image_exist = h1(9);    
    i = i+1;
end
images = i-1;

% Defines the header:
% header Columns : 1:Stack, 2:Slice, 3:Coil, 4:Heart phase, 5:Echo, 6:Dynamics, 
%                  7:Segments, 8:data offset, 9:Resolution x, 10:Resolution y, 
%                  11: Compression, 12: Flip, 13:Scaling factor1, 14:Scaling factor2
%                  15: Mix, 16: Prep Dir, 17: Sequence Nr.
%                  18: Segment2, 19: Syncho Number
header = zeros(images, 19); 

% Runs over all images in the file and writes out its header
for i = 0: images-1
    header_offset = (matrix_data_blocks * 512 + offset)*i;
    fseek(fid, header_offset, 'bof');
    h1 = fread(fid, 15, 'long');
    factor = fread(fid,2,'float');
    h2 = fread(fid, 111,'long');
    header(i+1,1) = h1(2);                  % Stack                         
    header(i+1,2) = h1(3);                  % Slice         
    header(i+1,3) = h2(2);                  % Coil
    header(i+1,4) = h1(6);                  % Heart phase
    header(i+1,5) = h1(5);                  % Echo
    header(i+1,6) = h1(7);                  % Dynamics    
    header(i+1,7) = h1(8);                  % Segments
    if ~h2(26)
        header(i+1,8) = h1(10);                 % Data offset
    else
        header(i+1,8) = h2(26);                 % Data offset
    end
    header(i+1,9) = h1(11);                 % Resolution x
    header(i+1,10) = h1(12);                % Resolution y
    header(i+1,11) = h1(14);                % Compression
    header(i+1,12) = h2(111);               % Flip
    header(i+1,13) = factor(1);             % Scaling factor 1
    header(i+1,14) = factor(2);             % Scaling factor 2
    header(i+1,15) = h1(1);                 % mix
    header(i+1,16) = h1(4);                 % Prep Dir
    header(i+1,17) = h1(15);                % Sequence Number
    header(i+1,18) = h2(1);                 % Segment2
    header(i+1,19) = h2(3);                 % Syncho number
    
    if h1(9) == 0
       'Header Problem!! Too many images calculated'
       break
   end
end

% Reads in the last header and checks the parameter "Complex Matrix
% Existence" if it hasn't the value 0, the file is corrupt 
last_header_offset = (matrix_data_blocks * 512 + offset)*images;
fseek(fid, last_header_offset, 'bof');
h1 = fread(fid, 15, 'long');
factor = fread(fid,2,'float');
h2 = fread(fid, 10,'long');
if h1(9) ~= 0 
    'Header Problem'
    return
end

% Prints the parameters on the screen
if strcmp(output,'yes')
    s1=sprintf('\nResolution in x-direction: %d \nResolution in y-direction: %d \nNumber of stacks: %d \nNumber of slices: %d \nNumber of coils: %d \nNumber of heart phases: %d \nNumber of echos: %d \nNumber of dynamics: %d \nNumber of segments: %d \nNumber of segments2: %d',header(1,9),header(1,10),max(header(:,1))+1,max(header(:,2))+1,max(header(:,3))+1, max(header(:,4))+1,max(header(:,5))+1,max(header(:,6))+1,max(header(:,7))+1,max(header(:,18))+1);
    disp(s1);    
end
