function data = calculateFeatures(rawdata, features,cos_window)
    sz = size(cos_window);

    if features.color 
       x= rawdata.cImg;
       if ~(size(x,1)==sz(1)&&size(x,2)==sz(2))
           x = imresize(x,sz);
       end
       for n = 1:3
           x_ch = x(:,:,n);
           x(:,:,n) = x(:,:,n) - mean(x_ch(:));
       end
       x= x - mean(x(:));
%        data{5} =x;
       data = x;
%       id =id +1;
 
    end

    if features.colorName
        x= rawdata.patch;       
        tmp = size(x);
        r = prod(tmp(1:2)==sz(1:2));
        if ~r
           x = imresize(x,sz);
        end
       out_pca = get_feature_map(x, 'cn', features.w2c);
       out_npca = get_feature_map(x,'gray',features.w2c);
       x = cat(3,out_pca,out_npca);
       
       if ~(exist('data','var'))
            data=x;
       else
            data =cat(3,data,x);
       end
    end
        
    if features.greyHoG
        x = double(fhog(single(rawdata.cImg), features.cell_size, features.hog_orientations));
        x(:,:,end) = [];
        if ~(exist('data','var'))
            data=x;
        else
            data =cat(3,data,x);
 %        data{id} = x;%
 %        id =id +1;
        end
    end
    
    data= double(bsxfun(@times, data, cos_window));

end


