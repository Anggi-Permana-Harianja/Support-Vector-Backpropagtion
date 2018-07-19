%% SUPPORT VECTOR BACKPROPAGATION segment
%==========================================================================
function Classified = SVBP(newSample, BreastGroups, Backpropdata, Theta1, Theta2)
%% inisialisasi Backpropagation output
BreastOutput = zeros(size(newSample, 1), 2); %BreastOutput(newSamplex2)

%% simulate the sample
for i = 1:size(newSample, 1)
    BreastOutput(i,:) = simulate(newSample(i,:), Theta1, Theta2);
end

%% creating SVM struct
svmStruct = svmtrain(Backpropdata, BreastGroups, 'Kernel_Function', 'rbf', 'boxconstraint', Inf, 'showplot', 'true');

%% classify it
Classified = svmclassify(svmStruct, BreastOutput, 'showplot', 'true');

end