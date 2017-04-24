clear;
clc;
close all;



for ASCIIcode = 98 : 122
    
    folderName = char(ASCIIcode)
    folderPath = ['/home/shomec/a/alexanc/Documents/TTK4157/chars74k-lite/' folderName];

    noImages = length(dir(folderName)) - 2;
    
    noTrainingImages = round(2 * noImages / 3);
    noTestingImages = noImages - noTrainingImages;
    
    mkdir (folderPath, 'TrainingSet');
    mkdir (folderPath, 'TestingSet');
    
    for index = 0 : noTrainingImages - 1
        imageName = [folderName '/' folderName '_' int2str(index) '.jpg'];
        copyfile (imageName, [folderPath '/TrainingSet']);
        delete (imageName);
    end

    for index = noTrainingImages : noImages - 1
        imageName = [folderName '/' folderName '_' int2str(index) '.jpg'];
        copyfile (imageName, [folderPath '/TestingSet']);
        delete (imageName);
    end
    
end