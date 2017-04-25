clear;
clc;
close all;

tic

%%
STEP = 3;
WINDOW = 25;
CNN_WINDOW = 20;
THRESHOLD = 0.6;
WHITE_TH = 235;

%%
load ../CNN/alphabetCNNnet.mat;

image1 = imread('../detection-images/detection-1.jpg');
image2 = imread('../detection-images/detection-2.jpg');
image3 = imread('../detection-images/detection-3.jpg');

%%
image = image1;

sizeImage = size(image);

figure;
imshow(image);

for line = 1 : STEP : sizeImage(1) - WINDOW
    for column = 1 : STEP : sizeImage(2) - WINDOW
        % (line, column) = coordinate of top-left pixel of sliding window
        
        window = image(line : line + WINDOW - 1, column : column + WINDOW - 1);
        
        window = imresize(window, [CNN_WINDOW CNN_WINDOW]);
        
        whiteBorder = 1;
        for index = 1 : CNN_WINDOW
            if window(1, index) < WHITE_TH || window(index, 1) < WHITE_TH || window(CNN_WINDOW, index) < WHITE_TH || window(index, CNN_WINDOW) < WHITE_TH
                whiteBorder = 0;
                break;
            end
        end
        
        if whiteBorder == 1
            %figure;
            %imshow(window);
        
            probabilities = predict(alphabetNet, window);
         
            [maxProbab, indexMaxProbab] = max(probabilities);
         
           if (maxProbab > THRESHOLD)
                 %figure;
                 %imshow(window);
                 %title(sprintf('%c - %d', char(indexMaxProbab + 96), maxProbab));
                 
                 hold on;
                 rectangle('Position', [column, line, WINDOW, WINDOW], 'EdgeColor', 'b');
                 text(column + WINDOW / 2, line - 5, char(indexMaxProbab + 96), 'Color','blue','FontSize',14)
                 
                 %pause(0.1);
           end
        end
      
        %pause(0.5);
        
    end
end

toc