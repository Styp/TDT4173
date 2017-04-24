clear;
clc;
close all;

%%

trainingImages = uint8([]);
trainingLabels = [];
indexTrainingImages = 0;

testingImages = uint8([]);
testingLabels = [];
indexTestingImages = 0;

for ASCIIcode = 97 : 122
    
    folderName = char(ASCIIcode)
    
    folderTrainingPath = ['/home/shomec/a/alexanc/Desktop/TDT4173-FinalProject/chars74k-lite_split/' folderName '/TrainingSet'];
    folderTestingPath = ['/home/shomec/a/alexanc/Desktop/TDT4173-FinalProject/chars74k-lite_split/' folderName '/TestingSet'];
    
    noTrainingImages = length(dir(folderTrainingPath)) - 2;
    noTestingImages = length(dir(folderTestingPath)) - 2;
    
    for index = 0 : noTrainingImages - 1
        indexTrainingImages = indexTrainingImages + 1;
        
        imageName = [folderName '_' int2str(index) '.jpg'];
        
        img = imread([folderTrainingPath '/' imageName]);
        
        trainingImages(:, :, :, indexTrainingImages) = img;
               
        trainingLabels = [trainingLabels; folderName];
    end
    
    for index = noTrainingImages : noTrainingImages + noTestingImages - 1
        indexTestingImages = indexTestingImages + 1;
        
        imageName = [folderName '_' int2str(index) '.jpg'];
        
        img = imread([folderTestingPath '/' imageName]);
        
        testingImages(:, :, :, indexTestingImages) = img;
        
        testingLabels = [testingLabels; folderName];
    end
    
end

trainingLabels = categorical(cellstr(trainingLabels));
testingLabels = categorical(cellstr(testingLabels));