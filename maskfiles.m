
load('./nyu_depth_v2_labeled.mat','instances');
load('./nyu_depth_v2_labeled.mat','labels');
load('./nyu_depth_v2_labeled.mat','images');

outp= '/mnt/c/Users/HP/Desktop/KMI/NYUdepth/masked_imgs/bins';

%disp(length(labels))
%disp(length(instances))
disp(size(labels))

for i=1:length(instances)
   
   image= images(:,:,:,i);
   %imshow(image); 
   
   %For each image, retrieve associated masks 
   [masks, lab] = get_instance_masks(labels(:,:,i), instances(:,:,i));   
   
   %Find indices related to ID. e.g.,  no. 5  = 'chair'
   indices =  find(lab==307);
   
   m=[];
   
   if ~isempty(indices)
       
       m = masks(:,:, indices);
       
       
   end 
   
   s=size(m);
   
   if s(:,2)~=0
                      
         try
             idx=s(:,3); 
         catch
             idx=1;
         end 
        
         for k=1:idx
              
             cmask=repmat(m(:,:,k),[1,1,3]); 
             imagecopy= image;
             imagecopy(~cmask)=0;
             
             %if i==7
             %  disp(cmask);
             %end         
             
             fname = sprintf(fullfile(outp,'img__%d__%d.png'), i, k); 
             %imshow(imagecopy);
             imwrite(imagecopy, fname);
         end
   else
         
         fprintf('Skip image %d', i);
         
   end
   
   
   %disp('Next instance')
   %fname = sprintf('chairs/file__%d.mat', i);
   %save(fname,'m'); 
   
end

 