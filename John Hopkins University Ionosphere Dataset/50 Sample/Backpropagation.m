%% BACKPROPAGATION classifier segment
%==========================================================================
function [J Theta1 Theta2 Backpropdata] = Backpropagation(InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    IonosphereInputs, IonosphereTargets, IonosphereGroups, lambda)
%% vectorize Theta
nn_params = [Theta1(:); Theta2(:)]; %nn_params(613x1) Theta1(17x35) Theta2(2x18)

%% feedforward propagation
J = nnCostFunction(nn_params, InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    IonosphereInputs, IonosphereTargets, IonosphereGroups, lambda);

%% random initialize weights
initial_nn_params = [Theta1(:); Theta2(:)]; %initial_nn_params(613x1)

%% training neural network
options = optimset('Display', 'iter', 'TolFun', 1e-18);
costFunction = @(p)nnCostFunction(p, InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    IonosphereInputs, IonosphereTargets, IonosphereGroups, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%% reshape Theta
Theta1 = reshape(nn_params(1:595), size(Theta1)); %Theta1(17x35)
Theta2 = reshape(nn_params(596:end), size(Theta2)); %Theta2(2x18)

%% Backpropagation data
Backpropdata = zeros(size(IonosphereInputs, 1), OutputLayer); %Backpropdata(50x2)
for i = 1:size(IonosphereInputs, 1)
    Backpropdata(i,:) = simulate(IonosphereInputs(i,:), Theta1, Theta2);
end

%% plot in scatter
scatter(Backpropdata(:,1), Backpropdata(:,2), 'DisplayName', 'Backpropdata(:,1), Backpropdata(:,2)'); figure(gcf);
pause;

end