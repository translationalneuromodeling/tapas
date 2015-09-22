function tapas_findreplace(file,otext,ntext,varargin)

%FINDREPLACE finds and replaces strings in a text file
%
% SYNTAX:
%
% findreplace(file,otext,ntext)
% findreplace(file,otext,ntext,match)
%
% findreplace  : This function finds and replaces strings in a text file
%
%       file:           text file name (with or without path)
%       otext:          text to be replaced (old text)
%       ntext:          replacing text (new text)
%       match:          either (1) for match case or (0) to ignore case
%                       default value is (1)
%
% Example:
%   findreplace('sample.txt','Moller','Moler');
%   findreplace('sample.txt','jake','Jack',0);
%   findreplace('sample.txt','continue it is','continue its',0);
%
%   Copyright 2005 Fahad Al Mahmood
%   Version: 1.0    $Date: 24-Dec-2005

% Obtaining the file full path
[fpath,fname,fext] = fileparts(file);
if isempty(fpath)
    out_path = pwd;
elseif fpath(1)=='.'
    out_path = [pwd filesep fpath];
else
    out_path = fpath;
end

% Reading the file contents
k=1;
all=0;
opt=[];
first_time=1;
change_counter=0;
fid = fopen([out_path filesep fname fext],'r');
while 1
    line{k} = fgetl(fid);
    if ~ischar(line{k})
        break;
    end
    k=k+1;
end
fclose(fid);
old_lines = line;

%Number of lines
nlines = length(line)-1;

for i=1:nlines
    if nargin==3, match=1;
    else match=varargin{1}; end

    if match==1, loc = regexp(line{i},otext);
    elseif match==0, loc = regexpi(line{i},otext);
    end

    if ~isempty(loc)
        nloc = 1;
        for j=1:length(loc)
            if all==0
                % Displaying keyboard instructions
                if first_time
                    disp(' ');
                    disp('(y) change (n) skip (a) change all (s) stop');
                    disp(' ');
                    first_time=0;
                end
                disp(line{i});
                opt = input(underline(loc(j),length(otext)),'s');
                if opt=='a'
                    line{i} = regexprep(line{i},otext,ntext, 'preservecase',nloc);
                    change_counter = change_counter + 1;
                    all=1;
                    if length(loc)>j
                        loc(j:end) = loc(j:end) + (length(ntext)-length(otext));
                    end
                elseif opt=='y'
                    line{i} = regexprep(line{i},otext,ntext, 'preservecase',nloc);
                    change_counter = change_counter + 1;
                    if length(loc)>j
                        loc(j:end) = loc(j:end) + (length(ntext)-length(otext));
                    end
                elseif opt=='s';
                    break;
                else
                    nloc = nloc + 1;
                end
            else
                line{i} = regexprep(line{i},otext,ntext, 'preservecase',nloc);
                change_counter = change_counter + 1;
                if length(loc)>j
                    loc(j:end) = loc(j:end) + (length(ntext)-length(otext));
                end
            end
        end
    end
    if opt=='s';
        break
    end
end

line = line(1:end-1);

disp(' ');
if change_counter~=0
    % Writing to file
    fid2 = fopen([out_path filesep fname fext],'w');

    for i=1:nlines
        fprintf(fid2,[line{i} '\n']);
    end
    fclose(fid2);
    disp([num2str(change_counter) ' Changes Made & Saved Successfully']);
    disp(' ');
else
    disp('No Match Found / No Change Applied');
    disp(' ');
end

function uline = underline(loc,length)
s=' ';
l='-';
uline=[];
if loc==1
    for i=1:length
        uline=[uline l];
    end
else
    for i=1:loc-1
        uline = [uline s];
    end
    for i=1:length
        uline=[uline l];
    end
end
uline = [uline s];

