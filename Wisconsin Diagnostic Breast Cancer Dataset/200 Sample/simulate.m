%% simulate function
%==========================================================================
function OutputBackprop = simulate(Inputs, Theta1, Theta2)
Inputs = [ones(size(Inputs, 1), 1) Inputs]; %Inputs(Inputsx10)
for i = 1:size(Inputs, 1)
    a1 = Inputs(i,:); %a1(1x10)
    z2 = Theta1 * a1'; %z2(4x1) Theta1(4x10) a1'(10x1)
    a2 = sigmoid(z2); %z2(4x1) 
    a2 = [1;a2]; %a2(5x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x5) a2(5x1)
    a3 = sigmoid(z3); %a3(2x1)
    OutputBackprop = a3'; %OutputBackprop(1x2)
end
end