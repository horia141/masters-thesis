classdef utilstest
    methods (Static,Access=public)
        function [s,ci] = classifier_data_3()
            s = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            ci = classification_info({'1' '2' '3'},[1*ones(100,1);2*ones(100,1);3*ones(100,1)]);
        end

        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_clear_data_3()
            s = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            ci = classification_info({'1' '2' '3'},[1*ones(100,1);2*ones(100,1);3*ones(100,1)]);
            [tr_i,ts_i] = ci.partition('holdout',0.2);
            s_tr = dataset.subsample(s,tr_i);
            s_ts = dataset.subsample(s,ts_i);
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_mostly_clear_data_3()
            s_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([3 3],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)];
            s_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],18);
                    3 3;
                    1 3;
                    mvnrnd([3 3],[0.01 0; 0 0.01],18);
                    3 1;
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],18)
                    3 1;
                    3 3];
            ci_tr = classification_info({'1' '2' '3'},[1*ones(80,1);2*ones(80,1);3*ones(80,1)]);
            ci_ts = classification_info({'1' '2' '3'},[1*ones(20,1);2*ones(20,1);3*ones(20,1)]);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_unclear_data_3()
            s = [mvnrnd([3 1],[0.5 0; 0 0.5],100);
                 mvnrnd([3 3],[0.5 0; 0 0.5],100);
                 mvnrnd([1 3],[0.5 0; 0 0.5],100)];
            ci = classification_info({'1' '2' '3'},[1*ones(100,1);2*ones(100,1);3*ones(100,1)]);
            [tr_i,ts_i] = ci.partition('holdout',0.2);            
            s_tr = dataset.subsample(s,tr_i);
            s_ts = dataset.subsample(s,ts_i);
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
        
        function [s,ci] = classifier_data_2()
            s = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            ci = classification_info({'1' '2'},[1*ones(100,1);2*ones(100,1)]);
        end

        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_clear_data_2()
            s = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            ci = classification_info({'1' '2'},[1*ones(100,1);2*ones(100,1)]);
            [tr_i,ts_i] = ci.partition('holdout',0.2);
            s_tr = dataset.subsample(s,tr_i);
            s_ts = dataset.subsample(s,ts_i);
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_mostly_clear_data_2()
            s_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)];
            s_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],19);
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],19)
                    3 1];
            ci_tr = classification_info({'1' '2'},[1*ones(80,1);2*ones(80,1)]);
            ci_ts = classification_info({'1' '2'},[1*ones(20,1);2*ones(20,1)]);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_unclear_data_2()
            s = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            ci = classification_info({'1' '2'},[1*ones(100,1);2*ones(100,1)]);
            [tr_i,ts_i] = ci.partition('holdout',0.2);            
            s_tr = dataset.subsample(s,tr_i);
            s_ts = dataset.subsample(s,ts_i);
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end

        function [] = show_classification_border(cl,sample_tr,sample_ts,ci_tr,ci_ts,range)
            assert(tc.scalar(cl));
            assert(tc.classifier(cl));
            assert(tc.dataset_record(sample_tr));
            assert(tc.same(dataset.geometry(sample_tr),2));
            assert(tc.dataset_record(sample_ts));
            assert(tc.same(dataset.geometry(sample_ts),2));
            assert(tc.scalar(ci_tr));
            assert(tc.classification_info(ci_tr));
            assert(tc.scalar(ci_ts));
            assert(tc.classification_info(ci_ts));
            assert(tc.vector(range));
            assert(length(range) == 4);
            assert(tc.number(range));
            assert(range(1) < range(2));
            assert(range(3) < range(4));
            assert(ci_tr.compatible(sample_tr));
            assert(ci_ts.compatible(sample_ts));
            
            hnd = logging.handlers.zero(logging.level.All);
            log = logging.logger({hnd});

            figure();
            hold on;
            gscatter(sample_tr(:,1),sample_tr(:,2),ci_tr.labels_idx,'rgb','o',6);
            gscatter(sample_ts(:,1),sample_ts(:,2),ci_ts.labels_idx,'rgb','o',6);
            ptmp = allcomb(-1:0.05:5,-1:0.05:5);
            ptmp2 = allcomb(-1:0.2:5,-1:0.2:5);
            l = cl.classify(ptmp,-1,log);
            [~,cfd] = cl.classify(ptmp2,-1,log);
            subplot(2,2,1);
            gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
            axis(range);
            subplot(2,2,2);
            mesh(-1:0.2:5,-1:0.2:5,reshape(cfd(:,1),31,31),cat(3,reshape(cfd(:,1),31,31),zeros(31,31),zeros(31,31)));
            axis([range 0 1]);
            subplot(2,2,3);
            mesh(-1:0.2:5,-1:0.2:5,reshape(cfd(:,2),31,31),cat(3,zeros(31,31),reshape(cfd(:,2),31,31),zeros(31,31)));
            axis([range 0 1]);
            subplot(2,2,4);
            if size(cfd,2) > 2
                mesh(-1:0.2:5,-1:0.2:5,reshape(cfd(:,3),31,31),cat(3,zeros(31,31),zeros(31,31),reshape(cfd(:,3),31,31)));
                axis([range 0 1]);
            end
            pause(3);
            close(gcf());
        end
    end

    methods (Static,Access=public)
        function test(~)
        end
    end
end
