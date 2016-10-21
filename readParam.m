%
%coded by Li, Yang

function param = readParam()

param={};

%set parameters
param.visualization=1;
param.thresholdForResize = 100;
param.kernel_type = 'linear'; 
param.kernel_sigma = 0.5;
param.padding = 1.5;  %extra area surrounding the target
param.lambda = 1e-4*[0.25,0.5,1,2,4];  %regularization
param.output_sigma_factor = 0.1;  %spatial bandwidth 
param.interp_factor = 0.015;
param.features.colorUpdateRate = 0.01;
        
      
types = {'greyHoG','colorName'};%'color','greyHoG','colorName','gist','cnn','colorProb','colorProbHoG','lbp','greyProb',
param.features.types=types;
param.features.hog_orientations = 9;
param.features.nbin =10;
param.features.cell_size = 4;
param.features.gparam.orientationsPerScale = [4 4 4 4];
param.features.gparam.fc_prefilt = 4;

temp = load('w2crs');
param.features.w2c = temp.w2crs;
param.features.colorTransform = makecform('srgb2lab');
param.features.interPatchRate = 0.3;

param.features.color=0;
param.features.greyHoG=0;
param.features.colorName=0;

for i=1:numel(types)
   switch types{i}
       case 'color'
           param.features.color=1;
           param.features.cell_size = 1;
       case 'greyHoG'
           param.features.greyHoG=1;
       case 'colorName'
           param.features.colorName=1;
   end
    
end
 param.useGpu=true;
 param.search_size = [1 0.985 0.99 0.995 1.005 1.01 1.015];
% param.search_size = [0.9];
% param.template_size = [1 0.6 0.73 0.87 1.13 1.26 1.4];

end