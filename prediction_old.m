%% Compute Prediction

function results = prediction_old(isLS, isObs)

    global Op;
    %global file_AttEstimatedtime;
    %global file_turnAngles;
    global isLinkSizeInclusive;
    global isFixedUturn;
    global TXT; 
    global SampleObs;

    notifyMail('set','amyeuphich@gmail.com','sntal2908');
    %% Data 

    %file_AttEstimatedtime = './Input/ATTRIBUTEestimatedtime.txt';
    %file_turnAngles = './Input/ATTRIBUTEturnangles.txt';

    isLinkSizeInclusive = isLS;
    isFixedUturn = false;
    loadDataNRL;
    Op = Op_structure;
    initialize_optimization_structure();
    
    if isObs ==  true
        PredSample = spconvert(load('../PredSampleObs.txt'));
        note = 'OBS';
    else
        PredSample = spconvert(load('../PredSampleODs.txt'));
        note = 'ODS';
    end
    TXT = ['Nested prediction:',note,':'];
    TXT = [TXT sprintf('\n Link size = %d \n', isLinkSizeInclusive)];
    nTest = round(size(PredSample,1) / 2);
    
    %% Estimation for training samples
    for i = 1: nTest
        train = PredSample(i*2-1,:);
        test  = PredSample(i*2,:);
        SampleObs = train; 
        NRLestimator(train, isLS);
        LL = getPLL_nested(test);
        TXT = [TXT sprintf('%d : %f \n', i, LL)];
    end    
    try
          notifyMail('send', TXT);
    catch exection
            fprintf('\n Can not send email notification !!! \n');
    end
end