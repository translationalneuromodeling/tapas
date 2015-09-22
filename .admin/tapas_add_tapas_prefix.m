function tapas_add_tapas_prefix()

% directory where the renaming is to be done
dir1 = 'C:\Sudhir\Tapas\admin\testfolder\testtest\';

% prefix to be added
prefix = 'tapas_toolbox_';

% collect all filenames from current directory

filelist= dir(strcat(dir1,'*.m'));

% go through each file and replace filename
for i=1:length(filelist)
    for j=1:length(filelist)
        fin = fopen(strcat(dir1,filelist(i).name));
        fout = fopen(strcat(dir1,prefix,filelist(i).name),'w');
        while ~feof(fin)
            s = fgetl(fin);
            [fpathstr,fname,fext] = fileparts(strcat(dir1,filelist(j).name));
            s1 = strrep(s, fname, strcat(prefix,fname));
            fprintf(fout,'%s\n',s1);
        end
        fclose(fin);
        fclose(fout);        
       % rename file
       movefile(strcat(dir1,prefix,filelist(i).name),strcat(dir1,filelist(i).name));       
     end
end

% 
% 
% % rename all files
for i=1:length(filelist)
    movefile(strcat(dir1,filelist(i).name),strcat(dir1,prefix,filelist(i).name))
end
