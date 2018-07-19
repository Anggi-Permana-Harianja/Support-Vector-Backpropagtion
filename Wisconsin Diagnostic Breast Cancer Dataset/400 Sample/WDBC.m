%% SUPPORT VECTOR BACKPROPAGATION FOR BREAST CANCER CLASSIFIER
%==========================================================================
%% start up
clear all; close all; clc

%% loading data
load('BreastInputs.mat'); %BreastInputs(100x9)
load('BreastTargets.mat'); %BreastTargets(100x1)
BreastGroups = ismember(BreastTargets, 1); %BreastGroups(100x1)

%% inisialisasi neural network
InputLayer = 9;
HiddenLayer = 4;
OutputLayer = 2;

%% loading bobot
load('weights.mat'); %Theta1(10x4) Theta2(2x5)

%% loading learning rate
lambda = 0.02;

%% BACKPROPAGATION classifier segment
[J Theta1 Theta2 Backpropadata] = Backpropagation(InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    BreastInputs, BreastTargets, BreastGroups, lambda)

%% SUPPORT VECTOR MACHINE segment
loop = -1;
while loop < 0
    %% new sample and target
    newSample = input('Masukkan sample baru untuk diklasifikasi: \n');
    newTargets = input('Masukkan target pada masing-masing sample: \n');
    if isempty(newSample)
        continue;
    else
        %% Classify It
        Classified = SVBP(newSample, BreastGroups, Backpropadata, Theta1, Theta2)
        
        %% tingkat akurasi
        akurasi = 0;
        for i = 1:size(newSample, 1)
            if Classified(i) == newTargets(i)
                akurasi = akurasi + 1;
            else
                akurasi = akurasi + 0;
            end
        end
        fprintf('%d dari %d sample berhasil diklasifikasi dengan benar\n', akurasi, size(newSample, 1));
    end
end
