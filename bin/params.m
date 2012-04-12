classdef params
    methods (Static,Access=public)
        function [params_list] = gen_all(params_desc,varargin)
            assert(tc.scalar(params_desc));
            assert(tc.struct(params_desc));
            assert(tc.checkf(@(c)params.allowed_field(c) || ...
                                 (tc.scalar(c) && tc.struct(c,'type','field_name','dependency_fn') && ...
                                  tc.scalar(c.type) && tc.string(c.type) && strcmp(c.type,'depend') && ...
                                  tc.scalar(c.field_name) && tc.string(c.field_name) && ...
                                  tc.scalar(c.dependency_fn) && tc.function_h(c.dependency_fn)) || ...
                                 (tc.scalar(c) && tc.struct(c,'type','field_name','keys','values') && ...
                                  tc.scalar(c.type) && tc.string(c.type) && strcmp(c.type,'condition') && ...
                                  tc.scalar(c.field_name) && tc.string(c.field_name) && ...
                                  tc.vector(c.keys) && tc.cell(c.keys) && ...
                                  tc.checkf(@params.allowed_value,c.keys) && ...
                                  tc.vector(c.values) && tc.cell(c.values) && ...
                                  tc.checkf(@params.allowed_field,c.values)),...
                                     struct2cell(params_desc)));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.function_h,varargin));

            field_names = fieldnames(params_desc);
            field_values = struct2cell(params_desc);
            
            assert(tc.checkf(@(ii)~isfield(field_values{ii},'type') || ...
                                   find(cellfun(@(c)strcmp(field_values{ii}.field_name,c),field_names(1:ii-1))),1:length(field_values)));

            params_list = params.do_gen_all(1,length(field_values),struct(),field_names,field_values);
            
            for ii = 1:length(varargin)
                params_list = params_list(arrayfun(@(p)varargin{ii}(p),params_list));
            end
        end
        
        function [action_struct] = depend(field_name,dependency_fn)
            assert(tc.scalar(field_name));
            assert(tc.string(field_name));
            assert(tc.scalar(dependency_fn));
            assert(tc.function_h(dependency_fn));

            action_struct.type = 'depend';
            action_struct.field_name = field_name;
            action_struct.dependency_fn = dependency_fn;
        end

        function [action_struct] = condition(field_name,varargin)
            assert(tc.scalar(field_name));
            assert(tc.string(field_name));
            assert(tc.vector(varargin));
            assert(mod(length(varargin),2) == 0);
            assert(tc.cell(varargin));
            assert(tc.checkf(@params.allowed_value,varargin(1:2:end)));
            assert(tc.checkf(@params.allowed_field,varargin(2:2:end)));

            action_struct.type = 'condition';
            action_struct.field_name = field_name;
            action_struct.keys = varargin(1:2:end);
            action_struct.values = varargin(2:2:end);
        end
        
        function [s] = to_string(param)
            assert(tc.scalar(param));
            assert(tc.struct(param));
            assert(length(fieldnames(param)) >= 1);
            assert(tc.checkf(@params.allowed_value,struct2cell(param)));
            
            field_names = fieldnames(param);
            
            s = sprintf('%s: %s',field_names{1},utils.value_to_string(param.(field_names{1})));

            for ii = 2:length(field_names)
                if tc.empty(param.(field_names{ii}))
                    s = sprintf('%s\n%s: ',s,field_names{ii});
                else
                    s = sprintf('%s\n%s: %s',s,field_names{ii},utils.value_to_string(param.(field_names{ii})));
                end
            end
        end
        
        function [s] = to_table_string(params_list)
            assert(tc.vector(params_list));
            assert(tc.struct(params_list));
            assert(tc.checkf(@(s)tc.checkf(@params.allowed_value,struct2cell(s)),params_list));
            
            field_names = fieldnames(params_list);
            contents = cell(length(params_list) + 1,length(field_names) + 1);
            contents(1,1) = {'#'};
            contents(1,2:end) = field_names;
            contents(2:end,1) = arrayfun(@utils.value_to_string,1:length(params_list),'UniformOutput',false);
            
            for ii = 1:length(params_list)
                for jj = 1:length(field_names)
                    if tc.empty(params_list(ii).(field_names{jj}))
                        contents{ii + 1,jj + 1} = '';
                    else
                        contents{ii + 1,jj + 1} = utils.value_to_string(params_list(ii).(field_names{jj}));
                    end
                end
            end

            max_sizes_per_column = zeros(size(contents,2),1);
            
            for ii = 1:size(contents,2)
                max_sizes_per_column(ii) = max(cellfun('length',contents(:,ii)));
            end
            
            s = '';
            
            for ii = 1:size(contents,2)
                s = sprintf('%s + %s',s,repmat('-',1,max_sizes_per_column(ii)));
            end
            
            s = sprintf('%s +\n',s);
            
            for ii = 1:size(contents,2)
                s = sprintf('%s | %s%s',s,contents{1,ii},repmat(' ',1,max_sizes_per_column(ii) - length(contents{1,ii})));
            end
            
            s = sprintf('%s |\n',s);
            
            for ii = 1:size(contents,2)
                s = sprintf('%s + %s',s,repmat('-',1,max_sizes_per_column(ii)));
            end
            
            s = sprintf('%s +\n',s);

            for ii = 2:size(contents,1)
                for jj = 1:size(contents,2)
                    s = sprintf('%s | %s%s',s,contents{ii,jj},repmat(' ',1,max_sizes_per_column(jj) - length(contents{ii,jj})));
                end

                s = sprintf('%s |\n',s);
            end
            
            for ii = 1:size(contents,2)
                s = sprintf('%s + %s',s,repmat('-',1,max_sizes_per_column(ii)));
            end
            
            s = sprintf('%s +\n',s);
        end
    end
        
    methods (Static,Access=private)
        function [params_list] = do_gen_all(level,max_level,c_param,field_names,field_values)
            if level == max_level + 1
                params_list = c_param;
            else
                if tc.empty(field_values{level})
                    c_param.(field_names{level}) = field_values{level};
                    params_list = params.do_gen_all(level+1,max_level,c_param,field_names,field_values);
                elseif tc.scalar(field_values{level}) && (tc.value(field_values{level}) || tc.function_h(field_values{level}))
                    c_param.(field_names{level}) = field_values{level};
                    params_list = params.do_gen_all(level+1,max_level,c_param,field_names,field_values);
                elseif tc.vector(field_values{level}) && (tc.logical(field_values{level}) || tc.number(field_values{level}))
                    params_list = [];

                    for ii = 1:length(field_values{level})
                        c_param.(field_names{level}) = field_values{level}(ii);
                        params_list = [params_list; params.do_gen_all(level+1,max_level,c_param,field_names,field_values)];
                    end
                elseif tc.vector(field_values{level}) && tc.cell(field_values{level})
                    params_list = [];

                    for ii = 1:length(field_values{level})
                        c_param.(field_names{level}) = field_values{level}{ii};
                        params_list = [params_list; params.do_gen_all(level+1,max_level,c_param,field_names,field_values)];
                    end
                elseif isfield(field_values{level},'type') && strcmp(field_values{level}.type,'depend')
                    c_param.(field_names{level}) = field_values{level}.dependency_fn(c_param.(field_values{level}.field_name));
                    params_list = params.do_gen_all(level+1,max_level,c_param,field_names,field_values);
                elseif isfield(field_values{level},'type') && strcmp(field_values{level}.type,'condition')
                    for ii = 1:length(field_values{level}.keys)
                        c_key = field_values{level}.keys{ii};
                        comp_value = c_param.(field_values{level}.field_name);

                        if tc.empty(c_key) && tc.empty(comp_value)
                            selected_value = field_values{level}.values{ii};
                            break;
                        elseif tc.logical(c_key) && tc.scalar(comp_value) && tc.logical(comp_value) && (c_key == comp_value)
                            selected_value = field_values{level}.values{ii};
                            break;
                        elseif tc.number(c_key) && tc.scalar(comp_value) && tc.number(comp_value) && (c_key == comp_value)
                            selected_value = field_values{level}.values{ii};
                            break;
                        elseif tc.string(c_key) && tc.scalar(comp_value) && tc.string(comp_value) && strcmp(c_key,comp_value)
                            selected_value = field_values{level}.values{ii};
                            break;
                        elseif tc.function_h(c_key) && tc.scalar(comp_value) && tc.function_h(comp_value) && ...
                                strcmp(func2str(c_key),func2str(comp_value))
                            selected_value = field_values{level}.values{ii};
                            break;
                        end
                    end
                    
                    if ~exist('selected_value','var')
                        assert(false);
                    end

                    if tc.empty(selected_value)
                        c_param.(field_names{level}) = selected_value;
                        params_list = params.do_gen_all(level+1,max_level,c_param,field_names,field_values);
                    elseif tc.scalar(selected_value) && (tc.value(selected_value) || tc.function_h(selected_value))
                        c_param.(field_names{level}) = selected_value;
                        params_list = params.do_gen_all(level+1,max_level,c_param,field_names,field_values);
                    elseif tc.vector(selected_value) && (tc.logical(selected_value) || tc.number(selected_value))
                        params_list = [];

                        for ii = 1:length(selected_value)
                            c_param.(field_names{level}) = selected_value(ii);
                            params_list = [params_list; params.do_gen_all(level+1,max_level,c_param,field_names,field_values)];
                        end
                    elseif tc.vector(selected_value) && tc.cell(selected_value)
                        params_list = [];

                        for ii = 1:length(selected_value)
                            c_param.(field_names{level}) = selected_value{ii};
                            params_list = [params_list; params.do_gen_all(level+1,max_level,c_param,field_names,field_values)];
                        end
                    else
                        assert(false);
                    end
                else
                    assert(false);
                end
            end
        end

        function [o] = allowed_value(obj)
            o = tc.empty(obj) || (tc.scalar(obj) && tc.value(obj));
        end

        function [o] = allowed_field(obj)
            o = tc.empty(obj) || ...
                (tc.scalar(obj) && tc.value(obj)) || ...
                (tc.vector(obj) && (tc.logical(obj) || tc.number(obj))) || ...
                (tc.vector(obj) && tc.cell(obj));
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "params".\n');
            
            fprintf('  Functions "gen_all", "depend" and "condition".\n');
            
            param_desc.filters_count = [192 256];
            param_desc.filter_row_count = [3 5];
            param_desc.filter_col_count = params.depend('filter_row_count',@(v)v);
            param_desc.reduce_function = {@transforms.image.random_corr.max @transforms.image.random_corr.sqr};
            param_desc.reduce_spread = params.condition('reduce_function',@transforms.image.random_corr.max,[3 4],...
                                                                          @transforms.image.random_corr.sqr,[3 6]);
            param_desc.kernel_type = {'linear' 'rbf'};
            param_desc.kernel_param = params.condition('kernel_type','linear',0,'rbf',[0.1 1]);
            
            assert(tc.struct(param_desc.filter_col_count,'type','field_name','dependency_fn'));
            assert(tc.same(param_desc.filter_col_count.type,'depend'));
            assert(tc.same(param_desc.filter_col_count.field_name,'filter_row_count'));
            assert(tc.same(param_desc.filter_col_count.dependency_fn,@(v)v));
            assert(tc.struct(param_desc.reduce_spread,'type','field_name','keys','values'));
            assert(tc.same(param_desc.reduce_spread.type,'condition'));
            assert(tc.same(param_desc.reduce_spread.field_name,'reduce_function'));
            assert(tc.same(param_desc.reduce_spread.keys,{@transforms.image.random_corr.max @transforms.image.random_corr.sqr}));
            assert(tc.same(param_desc.reduce_spread.values,{[3 4] [3 6]}));
            assert(tc.struct(param_desc.kernel_param,'type','field_name','keys','values'));
            assert(tc.same(param_desc.kernel_param.type,'condition'));
            assert(tc.same(param_desc.kernel_param.field_name,'kernel_type'));
            assert(tc.same(param_desc.kernel_param.keys,{'linear' 'rbf'}));
            assert(tc.same(param_desc.kernel_param.values,{0 [0.1 1]}));
            
            params_list = params.gen_all(param_desc);
            
            assert(tc.same(params_list(1),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(2),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(3),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(4),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(5),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(6),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(7),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(8),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(9),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(10),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(11),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(12),struct('filters_count',192,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(13),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(14),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(15),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(16),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(17),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(18),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(19),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(20),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(21),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(22),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(23),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(24),struct('filters_count',192,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(25),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(26),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(27),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(28),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(29),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(30),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(31),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(32),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(33),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(34),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(35),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(36),struct('filters_count',256,'filter_row_count',3,'filter_col_count',3,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(37),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(38),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(39),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(40),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(41),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(42),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.max,'reduce_spread',4,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(43),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(44),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(45),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',3,...
                                                 'kernel_type','rbf','kernel_param',1)));
            assert(tc.same(params_list(46),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','linear','kernel_param',0)));
            assert(tc.same(params_list(47),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','rbf','kernel_param',0.1)));
            assert(tc.same(params_list(48),struct('filters_count',256,'filter_row_count',5,'filter_col_count',5,...
                                                 'reduce_function',@transforms.image.random_corr.sqr,'reduce_spread',6,...
                                                 'kernel_type','rbf','kernel_param',1)));
            
            
            clearvars -except display;
            
            fprintf('  Function "to_string".\n');
            
            s.a = true;
            s.b = 10.3;
            s.c = -10;
            s.d = 10;
            s.e = 0.3;
            s.f = 'hello';
            s.g = @params.to_string;
            
            assert(tc.same(params.to_string(s),sprintf('a: true\nb: 10.300000\nc: -10\nd: 10\ne: 0.300000\nf: hello\ng: params.to_string')));
            
            clearvars -except display;
            
            fprintf('  Function "to_table_string".\n');
            
            s.a = 1:2;
            s.b = {'hello' 'world'};
            
            sl = params.gen_all(s);
            
            assert(tc.same(params.to_table_string(sl),sprintf(strcat(' + - + - + ----- +\n',...
                                                                     ' | # | a | b     |\n',...
                                                                     ' + - + - + ----- +\n',...
                                                                     ' | 1 | 1 | hello |\n',...
                                                                     ' | 2 | 1 | world |\n',...
                                                                     ' | 3 | 2 | hello |\n',...
                                                                     ' | 4 | 2 | world |\n',...
                                                                     ' + - + - + ----- +\n'))));
            
            clearvars -except display;
        end
    end
end
