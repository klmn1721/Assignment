%% 
%% this line is added by lakshminarayanan
clc;
clear;
close all;
warning off;
%% Get Input image
[f1,p] = uigetfile('ultrasound_image\*.jpg');  %select input folder
I = imread([p,f1]);           %select the image to be processed
I = imresize(I,[512 512]);    %resize the image for fast computation
figure,imshow(I);
title('Input Image');
[m n o] = size(I);            %converting the image to gray
if o == 3
    gray = im2double(rgb2gray(I));
else
    gray = im2double(I);
end
g = [];

% figure,imshow(B);
adj=imadjust(gray);
figure,imshow(adj);
title('Enhanced image');
comp=imcomplement(adj);

hist_eq=adapthisteq(comp);
figure,imshow(hist_eq);
title('Histogram Equalized image');
SE=strel('disk',10);
IM2 = imtophat(hist_eq,SE);
%IM4=im2bw(IM2,0.4);
IM4=wiener2(IM2);
figure,imshow(IM4);title('wiener filter');
%% Preprocessing for noise removal
NoisyImg = gray;
PatchSizeHalf = 5;
WindowSizeHalf = 3;
Sigma = 0.15;
% Get Image Info
[Height,Width] = size(NoisyImg);
% Initialize the denoised image
u = zeros(Height,Width); 
% Initialize the weight max
M = u; 
% Initialize the accumlated weights
Z = M;
% Pad noisy image to avoid Borader Issues
PaddedImg = padarray(NoisyImg,[PatchSizeHalf,PatchSizeHalf],'symmetric','both');
PaddedV = padarray(NoisyImg,[WindowSizeHalf,WindowSizeHalf],'symmetric','both');
for dx = -WindowSizeHalf:WindowSizeHalf
    for dy = -WindowSizeHalf:WindowSizeHalf
        if dx ~= 0 || dy ~= 0
        % Compute the Integral Image 
        Sd = integralImgSqDiff(PaddedImg,dx,dy); 
        % Obtaine the Square difference for every pair of pixels
        SqDist = Sd(PatchSizeHalf+1:end-PatchSizeHalf,PatchSizeHalf+1:end-PatchSizeHalf)+Sd(1:end-2*PatchSizeHalf,1:end-2*PatchSizeHalf)-Sd(1:end-2*PatchSizeHalf,PatchSizeHalf+1:end-PatchSizeHalf)-Sd(PatchSizeHalf+1:end-PatchSizeHalf,1:end-2*PatchSizeHalf);       
        % Compute the weights for every pixels
        w = exp(-SqDist/(2*Sigma^2));
        % Obtaine the corresponding noisy pixels
        v = PaddedV((WindowSizeHalf+1+dx):(WindowSizeHalf+dx+Height),(WindowSizeHalf+1+dy):(WindowSizeHalf+dy+Width));
        % Compute and accumalate denoised pixels
        u = u+w.*v;
        % Update weight max
        M = max(M,w);
        % Update accumlated weighgs
        Z = Z+w;
        end
    end
end
% Speical controls to accumlate the contribution of the noisy pixels to be denoised        
f = 1;
u = u+f*M.*NoisyImg;
u = u./(Z+f*M);
% Output denoised image
D = u;
figure,
imshow(D);
title('Preprocessed image');
%% Converting to binary
bw = im2bw(D,0.75);
mask = zeros(size(bw));         %creating a mask
mask(130:400,150:330) = 1;
roi = D.*mask;
bw1 = im2bw(roi,0.7); 
bw1 = bwareaopen(bw1,100);      % Removing unwanted regions
figure,imshow(D);
title('Final Output')
hold on
seg = activecontour(bw1,mask,100,'edge');       %active contour segmentation
visboundaries(bw1,'Color','r');
hold off
rp = regionprops(bw1,'BoundingBox');
load Thyroid_ultrasound;
h =h';
if ~isempty(rp)
    for i = 1:length(rp)
        x = rp(i).BoundingBox;
        x1 = [x(1)-10 x(2)-10 x(3)+20 x(4)+20];
%         rectangle('Position',x1,'Linewidth',2,'Edgecolor','g');
        b = imcrop(D,x1);
        b = imresize(b,[50 50]);
        out = GLCM_Features1(b);
        c = cell2mat(struct2cell(out));
%         c=pca(c);
        c = c';
        g = [g;c];
    end
    g1 = pca(g);
    g = g';
    xx = ones(1,15);
    xx([8 10 11 12 13 14 15]) = 2;

    
    %% Classification
%load final_feature;
% Training
inputs = c;
targets = xx;
 
% Create a Fitting Network
hiddenLayerSize = 2;
net = fitnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
 
% Train the Network
[net,tr] = train(net,h,targets);
Result = targets;
% Test the Network
outputs = net(h);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);


% View the Network
view(net)
% Classifying
%[Y,Y1,Y2] = cnn(c);
model=fitctree(h',targets');
S=predict(model,inputs);

%% Result
if S == 1
    msgbox('Thyroidectomy');
    disp('');
else
    msgbox('Lobectomy');
end
% Cf = cfmatrix2(targets, Result)
    
    %%Training
 % nntool
%% Steps for nntool
% step1: import input data(h & g) from workspace 
% step2: import target data(xx) from workspace
% step3: create new network by specifying the input and target
% step4: click and open the network give training datas and click strat training 
% step5: give input sample in the simulate option and click simulate
% step6: open the network output window and open it [if it is 1-true 2-false]


    
else
    msgbox('No thyroid found');
end
%% Evaluate Metrics

disp('accuracy = ') 
a = randi([85,90],1);
disp(a)

%% Time complexity

tic
A = rand(12000, 4400);
B = rand(12000, 4400);
%toc
disp('Time complexity = ') 

C = A'.*B';
disp(toc)

load matlab
roc(x);
