depthImage = imread("0.png");
rgb = imread("rgb_0.png");
mask = imread("mask_0.png");

mask = (mask>10);

rgb_color = reshape(rgb, [], 3);

depth = double(depthImage(:,:,1))/255;
depth = depth.* mask(:,:,1);

far = 5;
near = 0.3;

height = 480;
width = 640;

z = 1 - (far-near)*depth + near;

K = [-572.4124, 0, 320; 0, -573.5692, 240; 0, 0, 1];

K_inv = inv(K);

pts = zeros(height*width, 3);
color = uint8(zeros(height*width, 3));

t = 1;bn  
for v = 1:height
    for u = 1:width
        point = [u ; v; 1];
        rep = z(v,u) .* K_inv * point;
        pts(t, :) = [rep(1) rep(2) z(v,u)];
        color(t, :) = [rgb(v, u, 1), rgb(v,u,2), rgb(v,u,3)];
        t=t+1;
    end
end

[row, col] = find(pts(:,3) > 0);

pts(row, :) = [];
color(row, :) = [];
%%

pcshow(pts, color);

ptCloud = pointCloud(pts, 'Color', color);
pcwrite(ptCloud, 'testPointCloud.pcd', 'Encoding', 'ascii');