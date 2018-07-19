%% Support Vector Machine for Heart Disease
%==========================================================================
clear all; close all; clc

%% load data
load('HeartInputs.mat'); %HeartInpurs(50x13)
load('HeartTargets.mat'); %HeartTargets(50x1)
HeartGroups = ismember(HeartTargets, 1); %HeartGroups(50x1)

%% creating svm Struct
svmStruct = svmtrain(HeartInputs, HeartGroups, 'Kernel_Function', 'rbf', 'boxconstraint', Inf);

%% interactive SVM
loop = -1;
while loop < 0
    %% new sample and targets
    newSample = input('Masukkan sample baru untuk di klasifikasi: \n');
    newTargets = input('Masukkan target pada masing-masing sample: \n');

    %% classify it
    Classified = svmclassify(svmStruct, newSample)

    %% nilai akurasi
    akurasi = 0;
    for i = 1:size(newSample, 1)
        if Classified(i) == newTargets(i)
            akurasi = akurasi + 1;
        else
            akurasi = akurasi + 0;
        end
    end
    akurasi = (akurasi/size(newSample, 1)) * 100;
    fprintf('Nilai akurasi: %f percent\n', akurasi)
end
