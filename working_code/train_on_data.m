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

centres = [centres0; centres1];
joints = [joints0; joints1];

%centres = centres0;
%joints = joints0;

net = newrb(joints(2:end,:).', centres(2:end,:).', 0, 2.0943951024, 300, 10);
view(net)

sim(net, joints(1,:)')
centres(1,:)