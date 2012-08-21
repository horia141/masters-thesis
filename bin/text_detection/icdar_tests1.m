train_data = cat(4,train_letters{4},train_nonletters{4});
train_data_ci = classifier_info({'letter' 'nonletter'},[1*ones(1,dataset.count(train_data) / 2) 2*ones(1,dataset.count(train_data) / 2)]);
train_data_r = dataset.flatten_image(train_data);
test_data = cat(4,test_letters{4},test_nonletters{4});
test_data_r = dataset.flatten_image(test_data);
test_data_ci = classifier_info({'letter' 'nonletter'},[1*ones(1,dataset.count(test_data) / 2) 2*ones(1,dataset.count(test_data) / 2)]);

%% Nothing

clear train_data_r_*
clear test_data_r_*
clear t_*

train_data_r_use = train_data_r;
test_data_r_use = test_data_r;

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% DC

clear train_data_r_*
clear test_data_r_*
clear t_*

t_dc = transforms.record.dc_offset(train_data_r);

train_data_r_use = t_dc.code(train_data_r);
test_data_r_use = t_dc.code(test_data_r);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse	(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% STD

clear train_data_r_*
clear test_data_r_*
clear t_*

t_std = transforms.record.standardize(train_data_r);

train_data_r_use = t_std.code(train_data_r);
test_data_r_use = t_std.code(test_data_r);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% ZCA

clear train_data_r_*
clear test_data_r_*
clear t_*

t_zca = transforms.record.zca(train_data_r);

train_data_r_use = t_zca.code(train_data_r);
test_data_r_use = t_zca.code(test_data_r);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% NORM

clear train_data_r_*
clear test_data_r_*
clear t_*

t_norm = transforms.record.normalize(train_data_r);

train_data_r_use = t_norm.code(train_data_r);
test_data_r_use = t_norm.code(test_data_r);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% DC+STD

clear train_data_r_*
clear test_data_r_*
clear t_*

t_dc = transforms.record.dc_offset(train_data_r);

train_data_r_use_1 = t_dc.code(train_data_r);
test_data_r_use_1 = t_dc.code(test_data_r);

t_std = transforms.record.standardize(train_data_r_use_1);

train_data_r_use = t_std.code(train_data_r_use_1);
test_data_r_use = t_std.code(test_data_r_use_1);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse	(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% DC+ZCA

clear train_data_r_*
clear test_data_r_*
clear t_*

t_dc = transforms.record.dc_offset(train_data_r);

train_data_r_use_1 = t_dc.code(train_data_r);
test_data_r_use_1 = t_dc.code(test_data_r);

t_zca = transforms.record.zca(train_data_r_use_1);

train_data_r_use = t_zca.code(train_data_r_use_1);
test_data_r_use = t_zca.code(test_data_r_use_1);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse	(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% DC+NORM

clear train_data_r_*
clear test_data_r_*
clear t_*

t_dc = transforms.record.dc_offset(train_data_r);

train_data_r_use_1 = t_dc.code(train_data_r);
test_data_r_use_1 = t_dc.code(test_data_r);

t_norm = transforms.record.normalize(train_data_r_use_1);

train_data_r_use = t_norm.code(train_data_r_use_1);
test_data_r_use = t_norm.code(test_data_r_use_1);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse	(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% DC+STD+NORM

clear train_data_r_*
clear test_data_r_*
clear t_*

t_dc = transforms.record.dc_offset(train_data_r);

train_data_r_use_1 = t_dc.code(train_data_r);
test_data_r_use_1 = t_dc.code(test_data_r);

t_std = transforms.record.standardize(train_data_r_use_1);

train_data_r_use_2 = t_std.code(train_data_r_use_1);
test_data_r_use_2 = t_std.code(test_data_r_use_1);

t_norm = transforms.record.normalize(train_data_r_use_2);

train_data_r_use = t_norm.code(train_data_r_use_2);
test_data_r_use = t_norm.code(test_data_r_use_2);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse	(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% DC+ZCA+NORM

clear train_data_r_*
clear test_data_r_*
clear t_*

t_dc = transforms.record.dc_offset(train_data_r);

train_data_r_use_1 = t_dc.code(train_data_r);
test_data_r_use_1 = t_dc.code(test_data_r);

t_zca = transforms.record.zca(train_data_r_use_1);

train_data_r_use_2 = t_zca.code(train_data_r_use_1);
test_data_r_use_2 = t_zca.code(test_data_r_use_1);

t_norm = transforms.record.normalize(train_data_r_use_2);

train_data_r_use = t_norm.code(train_data_r_use_2);
test_data_r_use = t_norm.code(test_data_r_use_2);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse	(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% STD+NORM

clear train_data_r_*
clear test_data_r_*
clear t_*

t_std = transforms.record.standardize(train_data_r);

train_data_r_use_1 = t_std.code(train_data_r);
test_data_r_use_1 = t_std.code(test_data_r);

t_norm = transforms.record.normalize(train_data_r_use_1);

train_data_r_use = t_norm.code(train_data_r_use_1);
test_data_r_use = t_norm.code(test_data_r_use_1);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse	(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

%% ZCA+NORM

clear train_data_r_*
clear test_data_r_*
clear t_*

t_zca = transforms.record.zca(train_data_r);

train_data_r_use_1 = t_zca.code(train_data_r);
test_data_r_use_1 = t_zca.code(test_data_r);

t_norm = transforms.record.normalize(train_data_r_use_1);

train_data_r_use = t_norm.code(train_data_r_use_1);
test_data_r_use = t_norm.code(test_data_r_use_1);

assert(dataset.count(train_data_r_use) ~= dataset.count(test_data_r_use));

for c = CSA
	cl = classifiers.linear.svm(sparse	(train_data_r_use),train_data_ci,'Primal','L2','L2',c,'1v1',[45 48]);
	[~,~,score,~,~] = cl.classify(sparse(test_data_r_use),test_data_ci);
	fprintf('For C=%.3f => %.2f%%\n',c,score);
end

