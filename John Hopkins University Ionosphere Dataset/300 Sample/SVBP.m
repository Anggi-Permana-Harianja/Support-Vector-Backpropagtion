%% SUPPORT VECTOR BACKPROPAGATION function
%==========================================================================
function Classified = SVBP(newSample, IonosphereGroups, Backpropdata, Theta1, Theta2)
%% inisialisasi Backpropagation output
IonosphereOutput = zeros(size(newSample, 1), 2); %DiabetesOutput(newSamplex2)

%% simulate the sample
for i = 1:size(newSample, 1)
    IonosphereOutput(i,:) = simulate(newSample(i,:), Theta1, Theta2);
end

%% creating svm Struct
svmStruct = svmtrain(Backpropdata, IonosphereGroups, 'Kernel_Function', 'rbf', 'boxconstraint', Inf, 'showplot', 'true');

%% classify
Classified = svmclassify(svmStruct, IonosphereOutput, 'showplot', true);

end