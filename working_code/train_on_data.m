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

%% train on jointvalues

centres = [centres0; centres1];
joints = [joints0; joints1];

net = newrb(joints(2:end,:).', centres(2:end,:).', 0, 2.0943951024, 300, 10);
view(net)

% test 
y_pred_rad = sim(net, joints(1,:)')
y_true_rad = centres(1,:)

%% get cartesian coordinates of joints
arm_cart0 = raw_data0.arm_pos(:, 4, :);
head_cart0 = raw_data0.head_pos(:, 4, :);

arm_cart1 = raw_data1.arm_pos(:, 4, :);
head_cart1 = raw_data1.head_pos(:, 4, :);

arm_cart = [arm_cart0; arm_cart1];
head_cart = [head_cart0; head_cart1];

joints_cart = [head_cart, arm_cart];
joints_cart_neg = joints_cart * -1;
joints_cart = [joints_cart; joints_cart_neg];
joints_cart = reshape(joints_cart, [length(joints_cart), 12]);

centres = [centres0; centres1];
centres_neg = centres * -1;
centres = [centres; centres_neg];

% double training data with same samples, I know it's a bit like cheating
centres = [centres; centres];
joints_cart = [joints_cart; joints_cart];

%% train on cartesian coordinates
close all
clear net

new_sorting = randperm(length(centres));
centres = centres(new_sorting,:);
joints_cart = joints_cart(new_sorting,:);

net = newrb(joints_cart(30:end, :, :).', centres(30:end,:).', 0, 2, 300, 10);
view(net)

%% test
id = [1:30];
% id = randperm(length(centres));
% id = id(1:30);
y_pred_cart = sim(net, joints_cart(id,:)');
y_true_cart = centres(id,:);

mean_err = mean(mean(y_pred_cart - y_true_cart'));
max_err = max(max(y_pred_cart - y_true_cart'));
sprintf("Error in %i trials: \n Mean = %f \n Max = %f ",length(id), mean_err, max_err)
