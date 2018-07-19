%% neural network cost function
%==========================================================================
function [J grad] = nnCostFunction(nn_params, Inputlayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    IonosphereInputs, IonosphereTargets, IonosphereGroups, lambda)
%% reshape Theta
Theta1 = reshape(nn_params(1:595), size(Theta1)); %Theta1(17x35)
Theta2 = reshape(nn_params(596:end), size(Theta2)); %Theta2(2x18)

%% inisialisasi m
m = length(IonosphereInputs);

%% inisialisasi cost function
J = 0;

%% inisialisasi Theta_grad
Theta1_grad = zeros(size(Theta1)); %Theta1_grad(17x35)
Theta2_grad = zeros(size(Theta2)); %Theta2_grad(2x18)

%% target for each class
yk = zeros(OutputLayer, size(IonosphereInputs, 1)); %yk(2x300)
for i = 1:size(IonosphereInputs, 1)
    yk(IonosphereTargets(i), i) = 1;
end

%% feedforward cost function
IonosphereInputs = [ones(size(IonosphereInputs, 1), 1) IonosphereInputs]; %IonosphereInputs(300x35)
for i = 1:size(IonosphereInputs)
    a1 = IonosphereInputs(i,:); %a1(1x35)
    z2 = Theta1 * a1'; %z2(17x1) Theta1(17x35) a1'(35x1)
    a2 = sigmoid(z2); %a2(17x1)
    a2 = [1;a2]; %a2(18x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x18) a2(18x1)
    a3 = sigmoid(z3); %a3(2x1)
    
    J = J + -yk(:,i)' * log(a3) - (1-yk(:,i)') * log(1-a3);
end
J = J/m;
J = J + (lambda/(2*m)) * sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));

%% implemet backpropagation
for i = 1:size(IonosphereInputs)
    a1 = IonosphereInputs(i,:); %a1(1x35)
    z2 = Theta1 * a1'; %z2(17x1) Theta1(17x35) a1'(35x1)
    a2 = sigmoid(z2); %a2(17x1)
    a2 = [1;a2]; %a2(18x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x18) a2(18x1)
    a3 = sigmoid(z3); %a3(2x1)
    
    delta3 = a3 - yk(:,i); %delta3(2x1) a3(2x1) yk(2x1)
    delta2 = (Theta2' * delta3) .* [1;sigmoidGradients(z2)]; %delta2(18x1) Theta2'(18x2) delta3(2x1) [1;sigmoidGradients(z2)](18x1)
    delta2 = delta2(2:end); %delta2(17x1)
    
    Theta1_grad = Theta1_grad + delta2 * a1; %Theta1_grad(17x35) delta2(17x1) a1(1x35)
    Theta2_grad = Theta2_grad + delta3 * a2'; %Theta2_grad(2x18) delta3(2x1) a2'(1x18)
end
Theta1_grad = (1/m) * Theta1_grad + lambda * Theta1 .* [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; %Theta1_grad(17x35)
Theta2_grad = (1/m) * Theta2_grad + lambda * Theta2 .* [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; %Theta2_grad(2x18)
grad = [Theta1_grad(:); Theta2_grad(:)]; %grad(613x1)

end