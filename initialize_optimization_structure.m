%   Initialize optimization structure
%%
function [] = initialize_optimization_structure()
    global Op;
    global isLinkSizeInclusive;
    global isFixedUturn;
    global nbobs;
    global nParams_v;  
    global nParams_mu;
    Op.Optim_Method = OptimizeConstant.TRUST_REGION_METHOD;
    Op.ETA1 = 0.05;
    Op.ETA2 = 0.75;
    Op.maxIter = 150;
    Op.k = 0;
    Op.n = nParams_v;
    if isLinkSizeInclusive == true
        Op.n = Op.n + 1;
    end
    if isFixedUturn == true
        Op.n = Op.n - 1;
    end
    Op.m = Op.n;
    Op.n = Op.n + nParams_mu;
    %if isFixedMu == false                
    %end
    Op.x = -ones(Op.n,1) * 1.5;
    Op.x(Op.m+1:Op.n) = 0;
    Op.tol = 1e-6;
    Op.radius = 0.125;
    Op.delta = 0.125;
    Op.Ak = eye(Op.n);
    Op.H = eye(Op.n);
end