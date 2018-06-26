%% create RBF with radians
clear all 

raw_data0 = jsondecode(fileread('./all_data/data.json'));
raw_data1 = jsondecode(fileread('./all_data/data1.json'));

centres0 = [raw_data0.circles(:,:,1), raw_data0.circles(:,:,2)];
centres1 = [raw_data1.circles(:,:,1), raw_data1.circles(:,:,2)];

joints0 = zeros(length(raw_data0.joints), 4);
joints1 = zeros(length(raw_data1.joints), 4);

for i = 1:length(raw_data0.joints)
    for j = 1:4
        joints0(i, j) = cell2mat(raw_data0.joints{i}{j}(2));
    end
end

for i = 1:length(raw_data1.joints)
    for j = 1:4
        joints1(i, j) = cell2mat(raw_data1.joints{i}{j}(2));
    end
end

% append coords coming from different files
joints_o = [joints0; joints1];

centres_o = [centres0; centres1];
centres_o = [centres_o(:, 1) - 320, centres_o(:, 2) - 240];
centres_o = [-(centres_o(:, 1)/640*60.97*pi/180), centres_o(:, 2)/480*47.64*pi/180];
centres_o = [centres_o(:, 1) + joints_o(:, 1), centres_o(:, 2) + joints_o(:, 2)];

neg_centres = [centres_o(:, 1) * -1, centres_o(:, 2)];
neg_joints = [joints_o(:, 1) * -1, joints_o(:, 2), joints_o(:, 1) * -1, joints_o(:, 4)];

centres = [centres_o; centres_o; neg_centres; neg_centres; centres_o; centres_o; neg_centres; neg_centres];
joints = [joints_o; joints_o; neg_joints; neg_joints; joints_o; joints_o; neg_joints; neg_joints];

% permute data randomly
new_sorting = randperm(length(centres));
centres = centres(new_sorting,:);
joints = joints(new_sorting,:);


%% train new RBF
net = newrb(centres(30:end, :).', joints(30:end, 3:4).', 0, 2, 300, 10);
view(net)

%% test
id = [1:30];
y_pred_cart = sim(net, centres(id, :).');
y_true_cart = [joints(id, 3:4)];

mean_err = mean(mean(y_pred_cart - y_true_cart'));
max_err = max(max(y_pred_cart - y_true_cart'));
sprintf("Error in %i trials: \n Mean = %f \n Max = %f ",length(id), mean_err, max_err)

%% save rbf and get weights
save("./all_data/new_rbf_angles.mat")

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

save("./all_data/rbfweights_angles.mat", 'biases1', 'biases2', 'weigth1', 'weigth2')