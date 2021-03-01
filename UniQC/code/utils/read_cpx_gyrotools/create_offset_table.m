function offset_table_cpx=create_offset_table(header)

%--------------------------------------------------------------------------
% offset_table_cpx=create_offset_table(header)
%
% offset_table_cpx: Creates an array with all image offsets in the cpx-file
%                   The offset of a certain image can be obtained by:
%                   offset_table_cpx(stack, slice, coil, heart_phase, echo, dynamic, segment, segment2)
%
% Input:		header              The header of the cpx-file. Can be obtained
%                                   with the function "read_cpx_header"
%
% Output:       offset_table_cpx    An array with the image offsets in the
%                                   cpx-file
%                           
%                
%------------------------------------------------------------------------


[rows, columns] = size(header);

for i = 1: rows
    if header(i,8) == 0;
        offset_table_cpx(header(i,1)+1, header(i,2)+1, header(i,3)+1, header(i,4)+1, header(i,5)+1, header(i,6)+1, header(i,7)+1, header(i,18)+1) = -100;
    else
        offset_table_cpx(header(i,1)+1, header(i,2)+1, header(i,3)+1, header(i,4)+1, header(i,5)+1, header(i,6)+1, header(i,7)+1, header(i,18)+1) = header(i,8);
    end
end
offset_table_cpx(find(offset_table_cpx==0)) = -1;
offset_table_cpx(find(offset_table_cpx==-100)) = 0;