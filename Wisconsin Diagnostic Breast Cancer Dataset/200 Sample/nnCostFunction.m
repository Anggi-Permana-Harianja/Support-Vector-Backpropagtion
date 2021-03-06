%% neural network cost function
%==========================================================================
function [J grad] = nnCostFunction(nn_params, InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    BreastInputs, BreastTargets, BreastGroups, lambda)
%% reshape Theta
Theta1 = reshape(nn_params(1:40), size(Theta1)); %Theta1(4x10)
Theta2 = reshape(nn_params(41:end), size(Theta2)); %Theta2(2x5)

%% inisialisasi m
m = length(BreastInputs);

%% inisialisasi cost function
J = 0;

%% inisialisasi Theta_grad
Theta1_grad = zeros(size(Theta1)); %Theta1_grad(4x10)
Theta2_grad = zeros(size(Theta2)); %Theta2_grad(2x5)


%% target for each class
yk = zeros(OutputLayer, size(BreastInputs, 1)); %yk(2x100)
for i = 1:size(BreastInputs, 1)
    yk(BreastTargets(i), i) = 1;
end

%% feedforward cost function
BreastInputs = [ones(size(BreastInputs, 1), 1) BreastInputs]; %BreastInputs(100x10)
for i = 1:size(BreastInputs, 1)
    a1 = BreastInputs(i,:); %a1(1x10)
    z2 = Theta1 * a1'; %z2(4x1) Theta1(4x10) a1'(10x1)
    a2 = sigmoid(z2); %a2(4x1)
    a2 = [1;a2]; %a2(5x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x5) a2(5x1)
    a3 = sigmoid(z3); %a3(2x1)
    
    J = J + -yk(:,i)' * log(a3) - (1-yk(:,i)') * log(1-a3);
end
J = J/m;
J = J + (lambda/(2*m)) * sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));

%% implement backpropagation
for i = 1:size(BreastInputs, 1)
    a1 = BreastInputs(i,:); %a1(1x10)
    z2 = Theta1 * a1'; %z2(4x1) Theta1(4x10) a1'(10x1)
    a2 = sigmoid(z2); %a2(4x1)
    a2 = [1;a2]; %a2(5x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x5) a2(5x1)
    a3 = sigmoid(z3); %a3(2x1)
    
    delta3 = a3 - yk(:,i); %delta3(2x1) a3(2x1) yk(2x1)
    delta2 = (Theta2' * delta3) .* [1;sigmoidGradients(z2)]; %delta2() Theta2'(5x2) delta3(2x1) [1;sigmoidGradients(z2)](5x1)
    delta2 = delta2(2:end); %delta2(4x1)
    
    Theta1_grad = Theta1_grad + delta2 * a1; %Theta1_grad(4x10) delta2(4x1) a1(1x10)
    Theta2_grad = Theta2_grad + delta3 * a2'; %Theta2_grad(2x5) delta3(2x1) a2'(1x5)
end
Theta1_grad = (1/m) * Theta1_grad + lambda * Theta1 .* [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; %Theta1_grad(4x10)
Theta2_grad = (1/m) * Theta2_grad + lambda * Theta2 .* [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; %Theta2_grad(2x5)
grad = [Theta1_grad(:);Theta2_grad(:)]; %grad(50x1)

end