%% simulate function
%==========================================================================
function OutputBackprop = simulate(Inputs, Theta1, Theta2)
Inputs = [ones(size(Inputs, 1), 1) Inputs]; %Inputs(Inputsx14)
for i = 1:size(Inputs, 1)
    a1 = Inputs(i,:); %a1(1x14)
    z2 = Theta1 * a1'; %z2(7x1) Theta1(7x14) a1'(14x1)
    a2 = sigmoid(z2); %a2(7x1)
    a2 = [1;a2]; %a2(8x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x8) a2(8x1)
    a3 = sigmoid(z3); %a3(2x1)
    OutputBackprop = a3'; %OuputBackprop(1x2)
end

end 