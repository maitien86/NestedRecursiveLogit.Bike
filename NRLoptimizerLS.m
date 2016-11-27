%%  
%   Recursive Nested logit estimator
%   MAI ANH TIEN - DIRO
%   07.June.2015
%   MAIN PROGRAM
%   ---------------------------------------------------
%% Credits
Credits;
globalVar;
global resultsTXT; 
global nParams_v; % Number of parameters in link utilities 
global nParams_mu; % Number of parameters in scales
global isFixedMu;
global AttsNames;
nParams_v = 14;
nParams_mu = 1;
isFixedMu = 0;
%%

%file_linkIncidence = './simulatedData/linkIncidence.txt';
%file_AttEstimatedtime = './simulatedData/ATTRIBUTEestimatedtime.txt';
%file_turnAngles = './simulatedData/ATTRIBUTEturnangles.txt';
%file_observations = './simulatedData/ObservationsAll.txt';

%% Set estimation type
SampleObs  = [];
%% Initialize the optimizer structure
%Op_old = Op;
isLinkSizeInclusive = true;
isFixedUturn = false;
loadDataNRL;
[nbobs, ~] = size(Obs); 
Op = Op_structure;
initialize_optimization_structure();
%Op.x(1:5) = [-2.4774; -0.7716;-0.3377;-1.7823;-1.0210];%Z==0 is empty but
%negative elements...
%Op.x(1:5)=[-5.2570;-4.6702;-5.0949;-1.5374;-5.1486]';%Z==0 not empty (1500 links)
%Op.x(1:5) = [-2.4774; -0.7716;-0.3377;1.0;1.0];%Z==0 is empty but negative
%elements...
%Op.x(1:5) = [-4.4774; -1.7716;-1.3377;1.0;1.0];% Z == 0 is empty and no neg elem!

%starting point of 11/11-13/11 
%Op.x(1:5) = [-1.940;-3.807;-1.601;0.337;1.492];

%This is the starting point of Rl with LS:
%Op.x(1:14) = [-2.41; -1.073;-1.013;-4.702;-1.596;0.42;1.86;0.87;-7.34;6.39;-0.2;0.35;-0.29;1.42];
%------------------------------------
%This is the starting point of RL:
%Op.x(1:14)=[-1.94;0;0;-3.807;-1.601;0.337;1.492;0;0;0;0;0;0;0];
Op.x(1:15)=[-1.94;0;0;-3.807;-1.601;0.337;1.492;0;0;0;0;0;0;0;0];
%------------------------------------
%Op.x(1:13) = [-1.940;-0.934;-0.858;-3.807;-1.601;0.337;1.492;0.701;-5.107;4.073;-0.224;0.348;-0.319];
% if isLinkSizeInclusive ==  true
%     Op.x = [-2.139;-0.748;-0.224;-3.301;-0.155;0.341;-0.581;-0.092];
% else
%     Op.x = [-2.139;-0.748;-0.224;-3.301;0.341;-0.581;-0.092];
% end

%% Relax Att
%getAtt();
for i = Op.m+1: Op.n
%        u = sparse((size(incidenceFull)));
        Atts(i).value = incidenceFull * 0;
end

Op.Optim_Method = OptimizeConstant.TRUST_REGION_METHOD;
Op.Hessian_approx = OptimizeConstant.BHHH;

Gradient = zeros(nbobs,Op.n);

%% Optimizing ... 
fprintf('\nEstimating ... \n');
tic;
%note: i added maxiter = 130 condition (maelle)
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','GradObj','on','MaxIter',130,'MaxFunEvals',130);
[x,fval,exitflag,output,grad] = fminunc(@LL,Op.x,options);
Op.value = fval;
Op.x = x;
Op.grad = grad;

%Save to file
formatout1='yymmdd';
formatout2='hhMM';
date1=datestr(now,formatout1);
date2=datestr(now,formatout2);
FolderString=horzcat('./Results/',date1);
if ~exist(FolderString, 'dir')
  % The folder does not exist.
  % Create that folder.
  mkdir(FolderString);
end
FileString=horzcat('./Results/',date1,'/',date2,'.txt');
FileID=fopen(FileString,'w');
[isStop, Stoppingtype, isSuccess] = CheckStopping(Op);
fprintf(FileID,'The algorithm stops, due to %s \n', Stoppingtype);
fprintf(FileID,'The attributes are \n');
for cellitem = 1:length(AttsNames)
    fprintf(FileID,'%s \n',AttsNames{cellitem});
end           
fprintf(FileID,'[Iteration]: %d\n', Op.k);
fprintf(FileID,'     LL = %f\n', Op.value);
fprintf(FileID,'     x = \n');
fprintf(FileID,'         %i\n', Op.x');
fprintf(FileID,'     norm of step = %f\n', norm(Op.step));
fprintf(FileID,'     radius = %f\n', Op.delta);  
fprintf(FileID,'     Norm of grad = %f\n', norm(Op.grad));
relatice_grad = relative_gradient(Op.value, Op.x, Op.grad, 1.0);
fprintf(FileID,'     Norm of relative gradient = %f\n', relatice_grad);
fprintf(FileID,'     Number of function evaluation = %f\n', Op.nFev);

global StandarError;
StandarError = zeros(1,Op.n);
disp(' Calculating VAR-COV ...');
getCov;

%Finishing ...
ElapsedTtime = toc
resultsTXT = [resultsTXT sprintf('\n Number of function evaluation %d \n', Op.nFev)];
resultsTXT = [resultsTXT sprintf('\n Estimated time %d \n', ElapsedTtime)];

%Continue saving to File:
fprintf(FileID,'\n Number of function evaluation %d \n', Op.nFev);
fprintf(FileID,'\n Estimated time %d \n', ElapsedTtime);
fprintf(FileID,'Estimated : %5.8f \n',Op.x);
fprintf(FileID,'Standard deviation : %5.8f \n',StandarError);
fprintf(FileID,resultsTXT);
fclose(FileID);

%% Send email notification 
try
   notifyMail('send', resultsTXT);
catch exection
   fprintf('\n Can not send email notification !!! \n');
end

