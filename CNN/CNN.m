clear;
clc;
close all;

TH_PROBAB = 0.6;

%load ConcatenatedImagesAndLabels.mat;
%load ConcatenatedImagesAndLabelsv2.mat;
%load ConcatenatedImagesAndLabelsContrastFilter.mat;
%load ConcatenatedImagesAndLabelsFilterContrast.mat;
load ConcatenatedImagesAndLabelsOpen.mat;

size(trainingImages)

numImageCategories = 26;
categories(trainingLabels)

figure
thumbnails = trainingImages(:,:,:,1:100);
montage(thumbnails)

%%
% Create the image input layer
[height, width, numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize)

%%
% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

middleLayers = [

% The first convolutional layer has a bank of 32 5x5x3 filters. A
% symmetric padding of 2 pixels is added to ensure that image borders
% are included in the processing. This is important to avoid
% information at the borders being washed away too early in the
% network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)

% Note that the third dimension of the filter can be omitted because it
% is automatically deduced based on the connectivity of the network. In
% this case because this layer follows the image layer, the third
% dimension must be 3 to match the number of channels in the input
% image.

% Next add the ReLU layer:
reluLayer()

% Follow it with a max pooling layer that has a 3x3 spatial pooling area
% and a stride of 2 pixels. This down-samples the data dimensions from
% 32x32 to 15x15.
maxPooling2dLayer(3, 'Stride', 2)

% Repeat the 3 core layers to complete the middle of the network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

]

%%
finalLayers = [

% Add a fully connected layer with 64 output neurons. The output size of
% this layer will be an array with a length of 64.
fullyConnectedLayer(64)

% Add an ReLU non-linearity.
reluLayer

% Add the last fully connected layer. At this point, the network must
% produce 10 signals that can be used to measure whether the input image
% belongs to one category or another. This measurement is made using the
% subsequent loss layers.
fullyConnectedLayer(numImageCategories)

% Add the softmax loss layer and classification layer. The final layers use
% the output of the fully connected layer to compute the categorical
% probability distribution over the image classes. During the training
% process, all the network weights are tuned to minimize the loss over this
% categorical distribution.
softmaxLayer
classificationLayer
]

%%
layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

%%
layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);

%%
% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.01, ... %% Changed
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ... %% Changed
    'L2Regularization', 0.004, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);

%%
% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network.
alphabetNet = trainNetwork(trainingImages, trainingLabels, layers, opts);

%%

% Run the network on the training set.
YTrain = classify(alphabetNet, trainingImages);

% Calculate the accuracy.
accuracyTrain = sum(YTrain == trainingLabels)/numel(trainingLabels)

% Run the network on the test set.
YTest = classify(alphabetNet, testingImages);

% Calculate the accuracy.
accuracyTest = sum(YTest == testingLabels)/numel(testingLabels)

%%

N = 36;

index_images = round(rand(1, N) * 2372);

for line = 1 : sqrt(N)
    for column = 1 : sqrt(N)
        index = (line - 1) * sqrt(N) + column;
        
        img = testingImages(:,:,:,index_images(index));
        probabilities = predict(alphabetNet, img);
        [maxProbab, indexMaxProbab] = max(probabilities);
        
        expectedLabel = testingLabels(index_images(index));
        obtainedLabel = YTest(index_images(index));
        
        subplot(sqrt(N), sqrt(N), index);
        imshow(img);       
        
        if expectedLabel == obtainedLabel
            title(sprintf('%s (%s)', obtainedLabel, expectedLabel), 'Color', 'g');
        else
            title(sprintf('%s (%s)', obtainedLabel, expectedLabel), 'Color', 'r');
        end
         
    end
end

suptitle({'Classification results: Obtained label (Expected Label)', 'Green - correct, Red - wrong'});