%% random initialize weights function
%==========================================================================
function W = randInitWeights(L_in, L_out)
W = zeros(size(L_out, 1+L_in));
epsilon_init = 0.02;
W = rand(L_out, 1+L_in) * (2*epsilon_init) - epsilon_init;
end