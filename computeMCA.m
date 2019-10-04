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

function S = computeMCA(baseCls)
%% Compute the microcluster based co-association matrix.

[n, nBC] = size(baseCls);
cntCol = max(baseCls);

S = zeros(n,n);
for k = 1:nBC
    for idx = 1:cntCol(k)
        tmp1 = baseCls(:,k)==idx;
        tmp2 = 1:n;
        tmp = tmp2(tmp1); clear tmp1 tmp2
        
        for idxTmp = 1:numel(tmp)
            S(tmp(idxTmp), tmp) = S(tmp(idxTmp), tmp) + 1;
        end
    end
end
S = S*1.0/nBC;