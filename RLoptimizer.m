....
%%  
%   Recursive Nested logit estimator
%   Relaxing the scales for random terms
%   MAI ANH TIEN - DIRO
%   14.4.2014
%   MAIN PROGRAM
%   ---------------------------------------------------
%% Credits
Credits;
globalVar;
global resultsTXT; 
% Initialize email notification
notifyMail('set','amyeuphich@gmail.com','sntal2908');

%% Data 

file_linkIncidence = './Input/linkIncidence.txt';
file_AttEstimatedtime = './Input/ATTRIBUTEestimatedtime.txt';
file_turnAngles = './Input/ATTRIBUTEturnangles.txt';
file_observations = './Input/observationsForEstimBAI.txt';
%file_observations = './simulatedData/ObservationsAll_NoLS.txt';

%% Set estimation type
global isFixedMu;
isFixedMu = 0;

%% Initialize the optimizer structure
 %Op_old = Op;
 isLinkSizeInclusive = true;
 isFixedUturn = false;
 loadData;
 Op = Op_structure;
 initialize_optimization_structure();

 %Op.x = Op_old.x(1:Op.n);
 %Op.delta = 0.05;
 %Op.H = getHessian();
 %Op.x = [-2.4,-1,-0.4,-4.4, -0.2, 0.0, 0.0]';
 Op.x = [-3.060,-1.057,-0.353,-4.431, -0.227, 0.1, 0.1, 0.1]';
 Op.x = [-2.139;-0.748;-0.224;-3.301;-0.155;0.341;-0.581;-0.092];
 %Op.x = [-2,-1,-1,-4, -0.2, -0.1, -1]';
 %Op.x = [-2.494;-0.933;-0.411;-4.459;0;0;0];
 %Op.x = [-1.854;-0.679;-0.258;-3.340;0.515;-0.674;-0.086];
 %mu = Scale(:,1)+0.2;
 %Op.x = [-1;-1;-1;-1];
 % Op.x = [-1.340446e+00,-7.776641e-01,-8.111989e-01,-5.534458e+00,-2.35543e+00]';

%% Relax Att
for i = Op.m+1: Op.n
    u = sparse(zeros(size(incidenceFull)));
    Atts(i) = Matrix2D(u);
end

Op.Optim_Method = OptimizeConstant.TRUST_REGION_METHOD;
Op.Hessian_approx = OptimizeConstant.BHHH;
Gradient = zeros(nbobs,Op.n);

% Generate Observations
%   createSimulatedObs;
%   file_observations = './simulatedData/ObservationsAll.txt';
%  loadData;

%% Starting optimization
tic ;
disp('Start Optimizing ....');
[Op.value, Op.grad ] = LL(Op.x);
PrintOut(Op);
header = [sprintf('%s \n',file_observations) Op.Optim_Method];
header = [header sprintf('\nNumber of observations = %d \n', nbobs)];
header = [header sprintf('Hessian approx methods = %s \n', OptimizeConstant.getHessianApprox(Op.Hessian_approx))];
resultsTXT = header;

%% Loop
while (true)    
  Op.k = Op.k + 1;
  if strcmp(Op.Optim_Method,OptimizeConstant.LINE_SEARCH_METHOD);
    ok = line_search_iterate();
    if ok == true
        PrintOut(Op);
    else
        disp(' Unsuccessful process ...')
        break;
    end
  else
    ok = btr_interate();
    PrintOut(Op);
  end
  [isStop, Stoppingtype, isSuccess] = CheckStopping(Op);  
  %----------------------------------------
  % Check stopping criteria
  if(isStop == true)
      isSuccess
      fprintf('The algorithm stops, due to %s \n', Stoppingtype);
      resultsTXT = [resultsTXT sprintf('The algorithm stops, due to %s \n', Stoppingtype)];
      break;
  end
end
%% Compute variance - Covariance matrix
%PrintOut(Op);
%disp(' Calculating VAR-COV ...');
getCov;
%Finishing ...
ElapsedTtime = toc
resultsTXT = [resultsTXT sprintf('\n Number of function evaluation %d \n', Op.nFev)];
resultsTXT = [resultsTXT sprintf('\n Estimated time %d \n', ElapsedTtime)];
%% Send email nitification 
try
   notifyMail('send', resultsTXT);
catch exection
   fprintf('\n Can not send email notification !!! \n');
end

