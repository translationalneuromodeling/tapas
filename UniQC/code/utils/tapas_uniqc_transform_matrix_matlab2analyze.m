% performs transformation from geometry of
% recon-code to output of spm_read_vols
%
%
%  INPUT
%       Y  - congruent with FullImReconStackData & imagesc-orientation
%
% OUTPUT
%       newY - from spm_read_vols

function newY = tapas_uniqc_transform_matrix_matlab2analyze(Y)

nSlices = size(Y,3);
nVolumes = size(Y,4);
newY = zeros([size(Y,2), size(Y,1), nSlices, nVolumes]);

for vol = 1:nVolumes
    for sl = 1:nSlices
        newY(:,:,sl, vol) = flipud(fliplr(Y(:,:,sl, vol).'));
    end
end

end