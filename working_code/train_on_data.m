clear all 
raw_data = jsondecode(fileread('data.json'));

centres = [raw_data.circles(:,:,1), raw_data.circles(:,:,2)];
joints = zeros(length(raw_data.joints), 4);
for i = 1:length(raw_data.joints)
    for j = 1:4
        joints(i, j) = cell2mat(raw_data.joints{i}{j}(2));
    end
end

net = newrb(centres.', joints.');
view(net)