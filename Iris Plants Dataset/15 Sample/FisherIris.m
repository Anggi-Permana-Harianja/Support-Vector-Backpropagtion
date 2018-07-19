%% SUPPORT VECTOR BACKPROPAGATION FOR IRIS CLASSIFIER
%==========================================================================
%% start up
clear all; close all; clc

%% loading data
load('IrisInputs.mat'); %IrisInputs(30x4)
load('IrisTargets.mat'); %IrisTargets(30x1)
IrisGroups = ismember(IrisTargets, 1); %IrisGroups(30x1)

%% inisialisasi neural network
InputLayer = 4;
HiddenLayer = 4;
OutputLayer = 2;

%% loading bobot
load('weights.mat'); %Theta1(4x5) Theta2(2x5)

%% loading learning rate
lambda = 0.02;

%% BACKPROPAGATION classifier segment
[J Theta1 Theta2 Backpropdata] = Backpropagation(InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    IrisInputs, IrisTargets, IrisGroups, lambda);

%% SUPPORT VECTOR MACHINE segment
loop = -1;
while loop < 0
    %% new sample and targets
    newSample = input('Masukkan sample untuk di klasifikasi:\n');
    newTargets = input('Masukkan masing-masing target: \n');
    if isempty(newSample)
        continue;
    else
        %% Classify it!
        Classified = SVBP(newSample, IrisGroups, Backpropdata, Theta1, Theta2)
        
        %% tingkat akurasi
        akurasi = 0;
        for i = 1:size(newSample, 1)
            if Classified(i) == newTargets(i)
                akurasi = akurasi + 1;
            else
                akurasi = akurasi + 0;
            end
        end
        fprintf('%d dari %d sample berhasil di-klasifikasi dengan benar\n', akurasi, size(newSample, 1));
    end
end
