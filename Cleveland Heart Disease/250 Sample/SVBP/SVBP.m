%% SUPPORT VECTOR BACKPROPAGATION function
%==========================================================================
function Classified = SVBP(newSample, HeartGroups, Backpropdata, Theta1, Theta2)
%% inisialisasi HeartOutput
HeartOutput = zeros(size(newSample, 1), 2); %HeartOutput(newSamplex2)

%% simulate the sample
for i = 1:size(newSample, 1)
    HeartOutput(i,:) = simulate(newSample(i,:), Theta1, Theta2);
end

%% creating svm struct
svmStruct = svmtrain(Backpropdata, HeartGroups, 'Kernel_Function', 'rbf', 'boxconstraint', Inf, 'showplot', true);

%% classify
Classified = svmclassify(svmStruct, HeartOutput, 'showplot', true);

end
