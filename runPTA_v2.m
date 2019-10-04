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

function [results_al, results_cl, results_sl] = runPTA_v2(S, ks)
% Input: the co-association matrix
%        and the numbers of clusters.
% Output: clustering results by PTA-AL, PTA-CL, and PTA-SL, respectively.

N = size(S,1);

d = stod2(S); clear S %convert similarity matrix to distance vector
% single linkage
Zsl = linkage(d,'single');
% complete linkage
Zcl = linkage(d,'complete');
% average linkage 
Zal = linkage(d,'average'); clear d

results_sl = ones(N, numel(ks));
results_cl = ones(N, numel(ks));
results_al = ones(N, numel(ks));
disp('.');
 for i = 1:numel(ks)
    K = ks(i);
    disp(['Obtain ',num2str(K),' clusters by PTA-AL\CL\SL.']); tic;
    results_sl(:,i) = cluster(Zsl,'maxclust',K);
    results_cl(:,i) = cluster(Zcl,'maxclust',K);
    results_al(:,i) = cluster(Zal,'maxclust',K);toc;
 end
disp('.');
 
 function d = stod2(S)
%==========================================================================
% FUNCTION: d = stod(S)
% DESCRIPTION: This function converts similarity values to distance values
%              and change matrix's format from square to vector (input
%              format for linkage function)
%
% INPUTS:   S = N-by-N similarity matrix
%
% OUTPUT:   d = a distance vector
%==========================================================================
% copyright (c) 2010 Iam-on & Garrett
%==========================================================================

N = size(S,1);
s = zeros(1,N*(N-1)/2);
nextIdx = 1;
for a = 1:N-1 %change matrix's format to be input of linkage fn
    s(nextIdx:nextIdx+(N-a-1)) = S(a,[a+1:end]);
    nextIdx = nextIdx + N - a;
end
d = 1 - s; %compute distance (d = 1-sim)