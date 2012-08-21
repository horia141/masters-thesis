if exist('DEBUG','var') && (DEBUG == true)
    IMAGES_DIR_PATH = '../data/icdar/sample';
    SEGMENTATION_PATH = '../data/icdar/sample/segmentation.xml';
    DESIRED_SIZES = [20 40];
else
    IMAGES_DIR_PATH = '../data/icdar/SceneTrialTrain';
    SEGMENTATION_PATH = '../data/icdar/SceneTrialTrain/segmentation.xml';
    DESIRED_SIZES = [8 12 16 20 24 28 32 36 40];
end

%% Load images according to segmentation file and resize them according to DESIRED_SIZES.

document = xmlread(SEGMENTATION_PATH);
root = document.getDocumentElement;

letters = {};
letters_class = '';
current_letter = 1;

nonletters = {};
current_nonletter = 1;

for ii = 0:(root.getLength - 1)
    if ~root.item(ii).hasChildNodes
        continue;
    end
    
    image = root.item(ii);
        
    image_path = char(image.item(1).getTextContent);
    tagged_rectangles = image.item(5);
    
    fprintf('Processing image "%s".\n',image_path);
    
    current_image = imread(fullfile(IMAGES_DIR_PATH,image_path));
        
    for jj = 0:(tagged_rectangles.getLength - 1)
        if ~tagged_rectangles.item(jj).hasChildNodes
            continue;
        end
        
        tagged_rectangle = tagged_rectangles.item(jj);
        tag = char(tagged_rectangle.item(1).getTextContent);
        segmentation = tagged_rectangle.item(3);
        
        row_start = sscanf(char(tagged_rectangle.getAttribute('y')),'%f') + 1;
        row_start = max(min(size(current_image,1) - 1,row_start),1);
        row_stop = row_start + sscanf(char(tagged_rectangle.getAttribute('height')),'%f');
        row_stop = max(min(size(current_image,1),row_stop),2);
        col_start = sscanf(char(tagged_rectangle.getAttribute('x')),'%f') + 1;
        col_start = max(min(size(current_image,2) - 1,col_start),1);
        col_stop = col_start + sscanf(char(tagged_rectangle.getAttribute('width')),'%f');
        col_stop = max(min(size(current_image,2),col_stop),2);
        columns = [];
        last_col = col_start;
        
        for kk = 0:(segmentation.getLength - 1)
            if strcmp(char(segmentation.item(kk).getNodeName),'#text')
                continue;
            end
            
            new_col = col_start + sscanf(char(segmentation.item(kk).getTextContent),'%f');
            new_col = max(min(size(current_image,2),new_col),2);
            columns = [columns; last_col new_col];
            last_col = new_col;
        end
        
        columns = [columns; last_col col_stop];

        for kk = 1:length(tag)
            if row_start >= row_stop || columns(kk,1) >= columns(kk,2)
                continue;
            end
            
            for ll = 1:length(DESIRED_SIZES)
                resized_letter_t = double(current_image(row_start:row_stop,columns(kk,1):columns(kk,2),:)) / 255;
                letters{current_letter,ll} = imresize(rgb2gray(resized_letter_t),[DESIRED_SIZES(ll) DESIRED_SIZES(ll)]);
            end
            letters_class(current_letter) = lower(tag(kk));
            current_letter = current_letter + 1;
        end
        
        for kk = 1:length(tag)
            for ll = 1:length(DESIRED_SIZES)
                resized_nonletter_row = randi(size(current_image,1) - DESIRED_SIZES(ll) - 1);
                resized_nonletter_col = randi(size(current_image,2) - DESIRED_SIZES(ll) - 1);
                nonletters{current_nonletter,ll} = rgb2gray(double(current_image(resized_nonletter_row:(resized_nonletter_row + DESIRED_SIZES(ll) - 1),...
                                                                                 resized_nonletter_col:(resized_nonletter_col + DESIRED_SIZES(ll) - 1),:)) / 255);
            end
            current_nonletter = current_nonletter + 1;
        end
    end
end

allowed_length = min(size(letters,1),size(nonletters,1));

train_letters = cell(1,9);

for ii = 1:9
    train_letters{ii} = zeros(DESIRED_SIZES(ii),DESIRED_SIZES(ii),1,6185);
    for jj = 1:allowed_length
        train_letters{ii}(:,:,1,jj) = letters{jj,ii};                         
    end
end

train_letters_class = letters_class;

train_nonletters = cell(1,9);                                            

for ii = 1:9
    train_nonletters{ii} = zeros(DESIRED_SIZES(ii),DESIRED_SIZES(ii),1,6185);
    for jj = 1:allowed_length
        train_nonletters{ii}(:,:,1,jj) = nonletters{jj,ii};                      
    end
end

%% Select only "proper" letters and nonletters.

% proper_letters = {};
% proper_letters_class = '';
% current_proper_letter = 1;
% 
% proper_nonletters = {};
% current_proper_nonletter = 1;
% 
% fprintf('Selecting Letters:\n');
% 
% for ii = 1:current_letter
%     for jj = 1:length(DESIRED_SIZES)
%         subplot(1,length(DESIRED_SIZES),jj);
%         imshow(letters{ii,jj});
%     end
%     keep = input('  Keep? [Y/n]','s');
%     
%     if check.empty(keep) || check.same(keep,'Y')
%         proper_letters(current_proper_letter,:) = letters(ii,:);
%         proper_letters_class(current_proper_letter) = letters_class(ii);
%         current_proper_letter = current_proper_letter + 1;
%     end
% end
% 
% fprintf('Selecting NonLetters:\n');
% 
% for ii = 1:current_nonletter
%     for jj = 1:length(DESIRED_SIZES)
%         subplot(1,length(DESIRED_SIZES),jj);
%         imshow(nonletters{ii,jj});
%     end
%     keep = input('  Keep? [Y/n]','s');
%     
%     if check.same(keep,'Y') || check.empty(keep)
%         proper_nonletters(current_proper_nonletter,:) = nonletters(ii,:);
%         current_proper_nonletter = current_proper_nonletter + 1;
%     end
% end
