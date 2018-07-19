%% SUPPORT VECTOR MACHINE segment
%==========================================================================
function Classified = SVBP(newSample, IrisGroups, Backpropdata, Theta1, Theta2)
%% insialisasi backpropagation output
IrisOutput = zeros(size(newSample, 1), 2); %IrisOutput(newSamplex2)

%% simulate the sample
for i = 1:size(newSample, 1)
    IrisOutput(i,:) = simulate(newSample(i,:), Theta1, Theta2);
end

%% creating SVMStruct
svmStruct = svmtrain(Backpropdata, IrisGroups, 'Kernel_Function', 'rbf', 'boxconstraint', Inf, 'showplot', 'true');

%% classify it
Classified = svmclassify(svmStruct, IrisOutput, 'showplot', 'true');

end