%% SUPPORT VECTOR BACKPROPAGATION FOR IONOSPHERE CLASSIFICATION
%==========================================================================
%% start up
clear all; close all; clc

%% loading data
load('IonosphereInputs.mat'); %IonosphereInputs(200x35)
load('IonosphereTargets.mat'); %IonosphereTargets(200x1)
IonosphereGroups = ismember(IonosphereTargets, 1); %IonosphereGroups(200x1)

%% inisialisasi neural network
InputLayer = 34;
HiddenLayer = 17;
OutputLayer = 2;

%% loading bobot
load('weights.mat'); %Theta1(17x35) Theta2(2x18)

%% loading learning rate
lambda = 0.02;

%% BACKPROPAGATION classifier segment
[J Theta1 Theta2 Backpropdata] = Backpropagation(InputLayer, HiddenLayer, OutputLayer, ...
    Theta1, Theta2, ...
    IonosphereInputs, IonosphereTargets, IonosphereGroups, lambda);

%% SUPPORT VECTOR MACHINE segment
loop = -1;
while loop < 0
    %% new sample and target
    newSample = input('Masukkan sample untuk diklasifikasi: \n');
    newTargets = input('Masukkan target untuk masing-masing sample: \n');
    if isempty(newSample)
        continue;
    else
        %% classify it
        Classified = SVBP(newSample, IonosphereGroups, Backpropdata, Theta1, Theta2)
        
        %% tingkat akurasi
        akurasi = 0;
        for i = 1:size(newSample, 1)
            if Classified(i) == newTargets(i)
                akurasi = akurasi + 1;
            else
                akurasi = akurasi + 0;
            end
        end
        fprintf('%d dari %d sampel berhasil diklasifikasi dengan benar\n', akurasi, size(newSample, 1))
    end
end
