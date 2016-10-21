function data=prepareData(patch, features)
    
    data.patch = patch;
    data.cImg = double(patch)/255;

    if features.color || features.greyHoG || features.greyProb
        if size(patch,3)>1
            data.gImg = double(rgb2gray(patch))/255;
            data.grey = rgb2gray(patch);
        else
            data.gImg = double(patch)/255;
            data.grey = patch;
        end
    end
end