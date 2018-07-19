%% BACKPROPAGATION CLASSIFIER function
%==========================================================================
function [J Jgrad Theta1 Theta2 Backpropdata] = Backpropagation(InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    HeartInputs, HeartGroups, lambda)
%% vectorize Theta
nn_params = [Theta1(:); Theta2(:)]; %nn_params(114x1)

%% feedforward propagation
J = nnCostFunction(nn_params, InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    HeartInputs, HeartGroups, lambda);

%% random initialize weights
initial_nn_params = [Theta1(:); Theta2(:)]; %initial_nn_params(114x1)

%% training neural network
options = optimset('Display', 'iter', 'TolFun', 1e-8);
costFunction = @(p)nnCostFunction(p, InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    HeartInputs, HeartGroups, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%% reshape Theta
Theta1 = reshape(nn_params(1:98), size(Theta1)); %Theta1(7x14)
Theta2 = reshape(nn_params(99:end), size(Theta2)); %Theta2(2x8)

%% gradient cost function
Jgrad = gradnnCostFunction(HeartInputs, HeartGroups, Theta1, Theta2,lambda);

%% Backpropagation data
Backpropdata = zeros(size(HeartInputs, 1), OutputLayer); %Backpropdata(140x2)
for i = 1:size(HeartInputs, 1)
    Backpropdata(i,:) = simulate(HeartInputs(i,:), Theta1, Theta2);
end

%% plot in scatter
scatter(Backpropdata(:,1), Backpropdata(:,2), 'DisplayName', 'Backpropdata(:,1), Backpropdata(:,2)'); figure(gcf);
pause;

end