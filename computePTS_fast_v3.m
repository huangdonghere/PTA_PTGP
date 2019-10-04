%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                  %
% This is a demo for the PTA and PTGP algorithms. If you find this %
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

function Sim = computePTS_fast_v3(S,mcLabels,para)
%% Compute the probability trajectory based similarity matrix.

%% Get the probability transition matrix
N = size(S,1);

mcSizes = zeros(N,1);
for iS = 1:numel(mcSizes)
    mcSizes(iS) = sum(sum(mcLabels(:,2)==iS));
end
mcSizes = mcSizes(:);

thresPos = N-floor(para.K)+1;
if thresPos>N, thresPos=N; end
if thresPos<1, thresPos=1; end

for i = 1:size(S,1), S(i,i)=0; end
sortedS = sort(S,2);
thresholds = sortedS(:, thresPos);clear sortedS;
for ii = 1:N % for each row 
    S(ii,S(ii,:)<thresholds(ii)) = 0;
end
S = max(S, S');
S = bsxfun(@times, S, mcSizes');

rowSum = sum(S,2);
isoletedIdx = find(rowSum==0);
rowSum(isoletedIdx)=-1; % If a point is isoleted, which means it has no connections to any other points, we label it by a negative value.
% P = S./repmat(rowSum, 1, N); % transition probability.
P = bsxfun(@rdivide, S, rowSum);
clear S
%% Compute PTS
%% Old implementation %%
% sumP = zeros(N, N*para.T);
% tmpP = P;
% sumP(:,1:N) = P; 
% for ii = 1:para.T-1
%     tmpP = tmpP*P;
%     sumP(:,N*ii+1:N*(ii+1)) = tmpP;
% end
% 
% inProdP = sumP*sumP';   % inner product
% inProdPii = repmat(diag(inProdP), 1, N);
% inProdPjj = inProdPii';
% Sim = inProdP./sqrt(inProdPii.*inProdPjj);


%% Facilitate the Computation of PTS %%
tmpP = P;
P = sparse(P); % Use P as a sparse matrix
inProdP = tmpP*P';
for ii = 1:(para.T-1)
    tmpP = tmpP*P;
    inProdP = inProdP + tmpP*tmpP';
end
clear tmpP P
% inProdPii = repmat(diag(inProdP), 1, N);
% inProdPjj = inProdPii';
% Sim = inProdP./sqrt(inProdPii.*inProdPjj);
inProdPii = repmat(diag(inProdP), 1, N);
Sim = inProdP./sqrt(inProdPii.*inProdPii');
Sim(isoletedIdx,:) = 1e-20;
Sim(:,isoletedIdx) = 1e-20; % remove the isolated point.
