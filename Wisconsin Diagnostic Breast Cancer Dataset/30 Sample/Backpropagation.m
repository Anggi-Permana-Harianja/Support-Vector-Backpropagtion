%% BACKPROPAGATION classfier segment
%==========================================================================
function [J Theta1 Theta2 Backpropdata] = Backpropagation(InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    BreastInputs, BreastTargets, BreastGroups, lambda)
%% vectorize Theta
nn_params = [Theta1(:);Theta2(:)]; %nn_params(50x1)

%% feedforward cost function
J = nnCostFunction(nn_params, InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    BreastInputs, BreastTargets, BreastGroups, lambda);

%% random initialize weights
initial_nn_params = [Theta1(:); Theta2(:)]; %initial_nn_params(50x1)

%% training neural network
options = optimset('Display', 'iter', 'TolFun', 1e-8);
costFunction = @(p)nnCostFunction(p, InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    BreastInputs, BreastTargets, BreastGroups, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%% reshape Theta
Theta1 = reshape(nn_params(1:40), size(Theta1)); %Theta1(4x10)
Theta2 = reshape(nn_params(41:end), size(Theta2)); %Theta2(2x5)

%% Backpropagation data
Backpropdata = zeros(size(BreastInputs, 1), OutputLayer); %Backpropdata(100x2)
for i = 1:size(BreastInputs, 1)
    Backpropdata(i,:) = simulate(BreastInputs(i,:), Theta1, Theta2);
end

%% plot in scatter
scatter(Backpropdata(:,1), Backpropdata(:,2), 'DisplayName', 'Backpropdata(:,1), Backpropdata(:,2)'); figure(gcf);
pause;

end