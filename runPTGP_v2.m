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

function results = runPTGP_v2(baseCls, PTS_sim, range)


[N,M] = size(baseCls);

disp('Build the MCBG graph.');
tic;
maxCls = max(baseCls);
for i = 1:numel(maxCls)-1
    maxCls(i+1) = maxCls(i+1)+maxCls(i);
end
cntCls = maxCls(end);
baseCls(:,2:end) = baseCls(:,2:end) + repmat(maxCls(1:end-1),N,1); clear maxCls

% Build the bipartite graph with 0-1 weights. If an object i belongs to a
% cluster j, the B(i,j)=1; otherwise, B(i,j)=0.
B=sparse(repmat([1:N]',1,M),baseCls(:),1,N,cntCls); clear baseCls

% Weight the edges in B according to the PTS similarity.
Da = diag(1./sum(B));
B = PTS_sim*B*Da; clear Da
toc;

results = zeros(N, numel(range));
disp('.');
for idx = 1:numel(range)
    try 
        disp(['Obtain ',num2str(range(idx)),' clusters by PTGP.']); tic;
        results(:, idx) = bipartiteGraphPartitioning(B,range(idx));toc;
    catch
        results(:, idx) = results(:, idx) + 1;
    end
end
disp('.');


function labels = bipartiteGraphPartitioning(B,Nseg)

% B - |X|-by-|Y|, cross-affinity-matrix

[Nx,Ny] = size(B);
if Ny < Nseg
    error('The cluster number is too large!');
end

dx = sum(B,2);
dx(dx==0) = 1e-10; 
Dx = sparse(1:Nx,1:Nx,1./dx); clear dx
Wy = B'*Dx*B;

%%% compute Ncut eigenvectors
% normalized affinity matrix
d = sum(Wy,2);
D = sparse(1:Ny,1:Ny,1./sqrt(d)); clear d
nWy = D*Wy*D; clear Wy
nWy = (nWy+nWy')/2;

% computer eigenvectors
[evec,eval] = eig(full(nWy)); clear nWy % use eigs for large superpixel graphs  
[~,idx] = sort(diag(eval),'descend');
Ncut_evec = D*evec(:,idx(1:Nseg)); clear D

%%% compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
evec = Dx * B * Ncut_evec; clear B Dx Ncut_evec

% normalize each row to unit norm
evec = bsxfun( @rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10 );

% k-means
labels = kmeans(evec,Nseg,'MaxIter',100,'Replicates',3);
