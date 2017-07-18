function patch = getPatch(img,pos,tmp_sz, window_sz)
    param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
                tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
    param0 = affparam2mat(param0); 
    patch = uint8(warpimg(double(img), param0, window_sz));
%       patch_tmp = img(pos(1)-floor(tmp_sz(1)/2):pos(1)+floor(tmp_sz(1)/2), ...
%           pos(2)-floor(tmp_sz(2)/2):pos(2)+floor(tmp_sz(2)/2), :);
%       patch = imresize(patch_tmp,window_sz,'bilinear');
end

