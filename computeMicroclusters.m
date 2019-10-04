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

function [newBaseCls, mClsLabels] = computeMicroclusters(baseCls)
%% Obtain the set of microclusters w.r.t. the clustering ensemble.

%% Get micro-clusters
[sortBaseCls, I] = sortrows(baseCls);
[uniqueBaseCls, uI] = unique(baseCls,'rows');
mClsLabels = zeros(numel(I),2); % micro-cluster labels
mClsLabels(:,1) = I;
for i = 1:numel(uI)
    mClsLabels(I==uI(i), 2) = i;
end 

flag_r2014a = 0;
for i = 1:size(mClsLabels,1)-1
    if mClsLabels(i,2)~=0 && mClsLabels(i+1,2)==0
        test = sortBaseCls(i,:)-sortBaseCls(i+1,:);
        if sum(abs(test(:))) == 0
            flag_r2014a = 1;
        end
    end
end

if flag_r2014a>0
    tmp = mClsLabels(1,2); 
    for i = 2:numel(I) 
        if mClsLabels(i,2) ~= 0
            tmp = mClsLabels(i,2);
        else
            mClsLabels(i,2) = tmp;
        end
    end
else
    tmp = mClsLabels(end,2);
    for i = (numel(I)-1):(-1):1
        if mClsLabels(i,2) ~= 0
            tmp = mClsLabels(i,2);
        else
            mClsLabels(i,2) = tmp;
        end
    end
end

%% Get the newBaseCls (represented by micro-clusters)
%%
newBaseCls = baseCls(uI,:);