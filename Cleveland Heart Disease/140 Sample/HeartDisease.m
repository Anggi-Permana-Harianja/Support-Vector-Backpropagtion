%% SUPPORT VECTOR BACKPROPAGATION FOR HEART DISEASE CLASSIFIER
%==========================================================================
%% start up
clear all; close all; clc

%% loading data
load('HeartInputs.mat'); %HeartInputs(200x13)
load('HeartTargets.mat'); %HeartTargets(200x1)
HeartGroups = ismember(HeartTargets, 1); %HeartGroups(200x1)

%% inisialisasi layer neural network
InputLayer = 13;
HiddenLayer = 7;
OutputLayer = 2;

%% loading bobot
load('weights.mat'); %Theta1(7x14) Theta2(2x8)

%% loading learning rate
lambda = 0.02;

%% BACKPROPAGATION CLASSIFIER segment
[J Jgrad Theta1 Theta2 Backpropdata] = Backpropagation(InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    HeartInputs, HeartGroups, lambda);
% display result
fprintf('Minimum Square Error: %f\n', J)
Theta1
Theta2
fprintf('Gradients Minimum Square Error: %f\n', Jgrad)
pause;

%% SUPPORT VECTOR MACHINE segment
loop = -1;
while loop < 0
    newSample = input('Masukkan SAMPLE baru untuk diklasifikasi: \n');
    newTargets = input('Masukkan TARGETS untuk masing-masing sample: \n');
    if isempty(newSample)
        continue;
    else
        %% heart disease classifier
        Classified = SVBP(newSample, HeartGroups, Backpropdata, Theta1, Theta2)
        akurasi = 0;
        for i = 1:size(Classified, 1)
            if Classified(i) == newTargets(i)
                akurasi = akurasi + 1;
            else
                akurasi  = akurasi + 0;
            end
        end
        akurasi = (akurasi/size(newSample, 1)) * 100;
        fprintf('Tingkat akurasi = %f percent\n', akurasi)
    end
end
