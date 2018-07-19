%% gradient cost function
%==========================================================================
function Jgrad = gradnnCostFunction(HeartInputs, HeartGroups, Theta1, Theta2, lambda)
%% inisialisasi Jgrad
Jgrad = 0;

%% inisialisasi m
m = length(HeartInputs);

%% target for each class
yk = zeros(2, size(HeartInputs, 1)); %yk(2x50)
for i = 1:size(HeartInputs)
    yk(HeartGroups(i), i) = 1;
end

%% implement gradient cost function
HeartInputs = [ones(size(HeartInputs, 1), 1) HeartInputs]; %HeartInputs(50x14)
for i = 1:size(HeartInputs, 1)
    a1 = HeartInputs(i,:); %a1(1x14)
    z2 = Theta1 * a1'; %z2(7x1) Theta1(7x14) a1'(14x1)
    a2 = sigmoidGradients(z2); %a2(7x1)
    a2 = [1;a2]; %a2(8x1)
    z3 = Theta2 * a2; %z3(2x1) Theta2(2x8) a2(8x1)
    a3 = sigmoid(z3); %a3(2x1)
    
    Jgrad = Jgrad + -yk(:,i) - (1-yk(:,i)')*log(1-a3);
end
Jgrad = Jgrad /m;
Jgrad = Jgrad + (lambda/(2*m)) * sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));

end