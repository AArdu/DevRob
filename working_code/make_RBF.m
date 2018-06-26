%% create RBF with cartesian data
% clear all 

raw_data0 = jsondecode(fileread('./all_data/data.json'));
raw_data1 = jsondecode(fileread('./all_data/data1.json'));

centres0 = [raw_data0.circles(:,:,1), raw_data0.circles(:,:,2)];
centres1 = [raw_data1.circles(:,:,1), raw_data1.circles(:,:,2)];

arm_cart0 = raw_data0.arm_pos(:, 4, :);
head_cart0 = raw_data0.head_pos(:, 4, :);

arm_cart1 = raw_data1.arm_pos(:, 4, :);
head_cart1 = raw_data1.head_pos(:, 4, :);

% append coords coming from different files
arm_cart_o = squeeze([arm_cart0; arm_cart1]);
head_cart_o = squeeze([head_cart0; head_cart1]);
centres_o = [centres0; centres1];

neg_centres = [centres_o(:, 1) * -1, centres_o(:, 2)];
neg_head_cart = [head_cart_o(:, 1:5), head_cart_o(:, end) * -1];
neg_arm_cart = [arm_cart_o(:, 1), arm_cart_o(:, 2) * -1, arm_cart_o(:, 3:end)];

centres = [centres_o; centres_o; neg_centres; neg_centres; centres_o; centres_o; neg_centres; neg_centres];
arm_cart = [arm_cart_o; arm_cart_o; neg_arm_cart; neg_arm_cart; arm_cart_o; arm_cart_o; neg_arm_cart; neg_arm_cart];
head_cart = [head_cart_o; head_cart_o; neg_head_cart; neg_head_cart; head_cart_o; head_cart_o; neg_head_cart; neg_head_cart];

% permute data randomly
new_sorting = randperm(length(centres));
centres = centres(new_sorting,:);
arm_cart = arm_cart(new_sorting,:);
head_cart = head_cart(new_sorting,:);

%% train new RBF
net = newrb([centres(30:end, :), head_cart(30:end, :)].', arm_cart(30:end, :).', 0, 2, 300, 10);
view(net)

%% test
id = [1:30];
% id = randperm(length(centres));
% id = id(1:30);
y_pred_cart = sim(net, [centres(id, :), head_cart(id, :)].');
y_true_cart = arm_cart(id,:);

mean_err = mean(mean(y_pred_cart - y_true_cart'));
max_err = max(max(y_pred_cart - y_true_cart'));
sprintf("Error in %i trials: \n Mean = %f \n Max = %f ",length(id), mean_err, max_err)

%% save rbf and get weights
save("./all_data/new_rbf.mat")

biases = net.b();
biases1 = biases(1);
biases1 = biases1{1,1};
biases2 = biases(2);
biases2 = biases2{1,1};


weight1 = net.IW();
weight2 = net.LW();
weight1 = weight1(1);
weight2 = weight2(2,1);
weigth1 = weight1{1,1};
weigth2 = weight2{1,1};

save("./all_data/rbfweights.mat", 'biases1', 'biases2', 'weigth1', 'weigth2')