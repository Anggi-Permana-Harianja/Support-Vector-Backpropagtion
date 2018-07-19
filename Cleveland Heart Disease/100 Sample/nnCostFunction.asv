%% neural network cost function
%%=========================================================================
function [J grad] = nnCostFunction(nn_params, InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    HeartInputs, HeartGroups, lambda)
%% reshape Theta
Theta1 = reshape(nn_params(1:98), size(Theta1)); %Theta1(7x14)
Theta2 = reshape(nn_params(99:end), size(Theta2)); %Theta2(2x8)

%% inisialisasi m
m = length(HeartInputs); 

%% inisialisasi cost function
J = 0;

%% inisialisasi Theta_grad
Theta1_grad = zeros(size(Theta1)); %Theta1(7x14)
Theta2_grad = zeros(size(Theta2)); %Theta2(2x8)

%% target for each class
yk = zeros(OutputLayer, size(HeartInputs, 1)); %yk(2x100)
for i = 1:size(HeartInputs)
    yk(HeartGroups(i), i) = 1;
end

%% feedforward cost function
HeartInputs = [ones(size(HeartInputs, 1), 1) HeartInputs]; %HeartInputs(100x14)
for i = 1:size(HeartInputs, 1)
    a1 = HeartInputs(i,:); %HeartInputs(1x14)
    z2 = Theta1 * a1'; %z2(7x1) Theta1(7x14) a1'(14x1)
    a2 = sigmoid(z2); %a2(7x1)
    a2 = [1;a2]; %a2(8x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x8) a2(8x1)
    a3 = sigmoid(z3); %a3(2x1)
    
    J = J + -yk(:,i)' * log(a3) - (1-yk(:,i)') * log(1-a3);
end
J = J/m;
J = J + (lambda/(2*m)) * sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));

%% implement backpropagation
for i = 1:size(HeartInputs, 1)
    a1 = HeartInputs(i,:); %HeartInputs(1x14)
    z2 = Theta1 * a1'; %z2(7x1) Theta1(7x14) a1'(14x1)
    a2 = sigmoid(z2); %a2(7x1)
    a2 = [1;a2]; %a2(8x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x8) a2(8x1)
    a3 = sigmoid(z3); %a3(2x1)
    
    delta3 = a3 - yk(:,i); %delta3(2x1) a3(2x1) yk(2x1)
    delta2 = (Theta2' * delta3) .* [1;sigmoidGradients(z2)]; %delta2(8x1) Theta2'(8x2) delta3(2x1) [1;sigmoidGrdients(z2)](8x1)
    delta2 = delta2(2:end); %delta2(7x1)
    
    Theta1_grad = Theta1_grad + delta2 * a1; %Theta1_grad(7x14) delta2(7x1) a1(1x14)
    Theta2_grad = Theta2_grad + delta3 * a2'; %Theta2_grad(2x8) delta3(2x1) a2'(1x8)
end
Theta1_grad = (1/m) * Theta1_grad + lambda * Theta1 .* [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; %Theta1_grad(7x14)
Theta2_grad = (1/m) * Theta2_grad + lambda * Theta2 .* [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; %Theta2_grad(2x8)
grad = [Theta1_grad(:); Theta2_grad(:)]; %grad(114x1)

end