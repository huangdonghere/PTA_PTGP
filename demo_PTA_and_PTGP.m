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

function demo_PTA_and_PTGP()
%% A demo for the PTA and PTGP algorithms.

clear all;
close all;
clc;


%% Load the base clustering pool.
% We have generated a pool of 200 candidate base clusterings for each dataset. 
% Please uncomment the dataset that you want to use and comment the other ones.

% If you don't want to use the pre-generated base clusterings, you may as
% well generate your own base clusterings by using k-means or any other
% clustering algorithms. 

% dataName = 'MF';
% dataName = 'IS';
% dataName = 'MNIST';
dataName = 'ODR';
% dataName = 'LS';
% dataName = 'PD';
% dataName = 'USPS';
% dataName = 'FC';
% dataName = 'KDD99_10P';
% dataName = 'KDD99';

members = [];
gt = [];
load(['bc_pool_',dataName,'.mat'],'members','gt');

[N, poolSize] = size(members);
trueK = numel(unique(gt));

%% Settings
% Ensemble size M
M = 10;
% How many times the PTA and PTGP algorithms will be run.
cntTimes = 20; 
% You can set cntTimes to a greater (or smaller) integer if you want to run
% the algorithms more (or less) times.

% For each run, M base clusterings will be randomly drawn from the pool.
% Each row in bcIdx corresponds to an ensemble of M base clusterings.
bcIdx = zeros(cntTimes, M);
for i = 1:cntTimes
    tmp = randperm(poolSize);
    bcIdx(i,:) = tmp(1:M);
end

%% Run PTA and PTGP repeatedly.
% The numbers of clusters.
clsNums = [2:20, 25:5:50];
clsNums = unique([clsNums,trueK]);

% Scores
nmiScoresBestK_PTA = zeros(cntTimes, 3);
nmiScoresTrueK_PTA = zeros(cntTimes, 3);
nmiScoresBestK_PTGP = zeros(cntTimes, 1);
nmiScoresTrueK_PTGP = zeros(cntTimes, 1);
for runIdx = 1:cntTimes
    disp('**************************************************************');
    disp(['Run ', num2str(runIdx),':']);
    disp('**************************************************************');
    
    %% Construct the ensemble of M base clusterings
    % baseCls is an N x M matrix, each row being a base clustering.
    baseCls = members(:,bcIdx(runIdx,:));
    
    %% Produce microclusters
    disp('Produce microclusters ... ');
    tic; [mcBaseCls, mcLabels] = computeMicroclusters(baseCls); toc;
    tilde_N = size(mcBaseCls,1);
    disp('--------------------------------------------------------------');
    
    %% Compute the microcluster based co-association matrix.
    disp('Compute the MCA matrix ... ');
    tic; MCA = computeMCA(mcBaseCls); toc;
    disp('--------------------------------------------------------------');
    
    %% Set parameters K and T.
    para.K = floor(sqrt(tilde_N)/2);
    para.T = floor(sqrt(tilde_N)/2);
    if para.K>20, para.K=20; end
    if para.T>20, para.T=20; end
    
    %% Compute PTS
    disp('Compute PTS ... ');
    tic; PTS = computePTS_fast_v3(MCA,mcLabels,para); toc;
    disp('--------------------------------------------------------------');
    
    %% Perform PTA
    disp('Run the PTA algorithm ... '); 
    [mcResultsAL,mcResultsCL,mcResultsSL] = runPTA_v2(PTS, clsNums);
    % The i-th column in results_al\results_cl\results_sl represents the
    % consensus clustering with clsNums(i) clusters by PTA-AL\CL\SL.
    disp('--------------------------------------------------------------');
    
    %% Perform PTGP 
    disp('Run the PTGP algorithm ... '); 
    mcResultsPTGP = runPTGP_v2(mcBaseCls, PTS, clsNums);     
    disp('--------------------------------------------------------------'); 

    %% Display the clustering results.
    disp('Map microclusters back to objects ... '); tic;
    resultsAL = mapMicroclustersBackToObjects(mcResultsAL, mcLabels);
    resultsCL = mapMicroclustersBackToObjects(mcResultsCL, mcLabels);
    resultsSL = mapMicroclustersBackToObjects(mcResultsSL, mcLabels);
    resultsPTGP = mapMicroclustersBackToObjects(mcResultsPTGP, mcLabels);toc;
    disp('--------------------------------------------------------------');
    
    disp('##############################################################'); 
    scoresAL = computeNMI(resultsAL,gt);
    scoresCL = computeNMI(resultsCL,gt);
    scoresSL = computeNMI(resultsSL,gt);
    scoresPTGP = computeNMI(resultsPTGP,gt);
    
    trueKidx = find(clsNums==trueK);
    nmiScoresBestK_PTA(runIdx,:) = [max(scoresAL),max(scoresCL),max(scoresSL)];
    nmiScoresTrueK_PTA(runIdx,:) = [scoresAL(trueKidx),scoresCL(trueKidx),scoresSL(trueKidx)];
    
    nmiScoresBestK_PTGP(runIdx) = max(scoresPTGP);
    nmiScoresTrueK_PTGP(runIdx) = scoresPTGP(trueKidx);
    
    disp(['The Scores at Run ',num2str(runIdx)]);
    disp('    ---------- The NMI scores w.r.t. best-k: ----------    ');
    disp(['PTA-AL : ',num2str(nmiScoresBestK_PTA(runIdx,1))]);
    disp(['PTA-CL : ',num2str(nmiScoresBestK_PTA(runIdx,2))]);
    disp(['PTA-SL : ',num2str(nmiScoresBestK_PTA(runIdx,3))]);
    disp(['PTGP   : ',num2str(nmiScoresBestK_PTGP(runIdx))]);
    disp('    ---------- The NMI scores w.r.t. true-k: ----------    ');
    disp(['PTA-AL : ',num2str(nmiScoresTrueK_PTA(runIdx,1))]);
    disp(['PTA-CL : ',num2str(nmiScoresTrueK_PTA(runIdx,2))]);
    disp(['PTA-SL : ',num2str(nmiScoresTrueK_PTA(runIdx,3))]);
    disp(['PTGP   : ',num2str(nmiScoresTrueK_PTGP(runIdx))]);
    disp('##############################################################'); 

    %% Save results
    save(['results_',dataName,'.mat'],'bcIdx','nmiScoresBestK_PTA','nmiScoresTrueK_PTA','nmiScoresBestK_PTGP','nmiScoresTrueK_PTGP');  
end

disp('**************************************************************');
disp(['   ** Average Performance over ',num2str(cntTimes),' runs on the ',dataName,' dataset **']);
disp(['Data size:     ', num2str(N)]);
disp(['Ensemble size: ', num2str(M)]);
disp('   ---------- Average NMI scores w.r.t. best-k: ----------   ');
disp(['PTA-AL : ',num2str(mean(nmiScoresBestK_PTA(:,1)))]);
disp(['PTA-CL : ',num2str(mean(nmiScoresBestK_PTA(:,2)))]);
disp(['PTA-SL : ',num2str(mean(nmiScoresBestK_PTA(:,3)))]);
disp(['PTGP   : ',num2str(mean(nmiScoresBestK_PTGP))]);
disp('   ---------- Average NMI scores w.r.t. true-k: ----------   ');
disp(['PTA-AL : ',num2str(mean(nmiScoresTrueK_PTA(:,1)))]);
disp(['PTA-CL : ',num2str(mean(nmiScoresTrueK_PTA(:,2)))]);
disp(['PTA-SL : ',num2str(mean(nmiScoresTrueK_PTA(:,3)))]);
disp(['PTGP   : ',num2str(mean(nmiScoresTrueK_PTGP))]);
disp('**************************************************************');
disp('**************************************************************');
