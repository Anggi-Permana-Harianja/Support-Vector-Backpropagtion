%% simulate function
%==========================================================================
function OutputBackprop = simulate(Inputs, Theta1, Theta2)
Inputs = [ones(size(Inputs, 1), 1) Inputs]; %Inputs(Inputsx35)
for i = 1:size(Inputs, 1)
    a1 = Inputs(i,:); %Inputs(1x35)
    z2 = Theta1 * a1'; %z2(17x1) Theta1(17x35) a1'(35x1)
    a2 = sigmoid(z2); %a2(17x1)
    a2 = [1;a2]; %a2(18x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x18) a2(18x1)
    a3 = sigmoid(z3); %a3(2x1)
    OutputBackprop = a3'; %OutputBackprop(1x2)
end
end