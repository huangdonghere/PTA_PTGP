%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                  %
% This is a demo for the PTA and PTGP algorithms. If you find the  %
% code useful for your research,please cite the paper below.       %
%                                                                  %
% Dong Huang, Jian-Huang Lai, and Chang-Dong Wang. Robust ensemble %
% clustering using probability trajectories, IEEE Transactions on  %
% Knowledge and Data Engineering, 2016, 28(5), pp.1312-1326.       %
%                                                                  %
% The code has been tested in Matlab R2014a and Matlab R2015a on a %
% workstation with Windows Server 2008 R2 64-bit.                  %
%                                                                  %
% https://www.researchgate.net/publication/284259332               %
%                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function fullResults = mapMicroclustersBackToObjects(results, mcLabels)

[~, I] = sort(mcLabels(:,1));
mcLabels2 = mcLabels(I,:);

N = size(mcLabels,1);
cntRes = size(results, 2);
fullResults = zeros(N, cntRes);

for i = 1:cntRes
    fullResults(:,i) = results(mcLabels2(:,2),i);
end