 %Given a dataset of RGB images
 %And one mat file for each set of binary masks belonging to the same image
 %returns the masked RGB

 %Uncomment to load for the first time
 load('./nyu_depth_v2_labeled.mat','images');
 outp= '/mnt/c/Users/HP/Desktop/KMI/NYUdepth/masked_imgs/';

 chairfiles= dir('chairs/*.mat');
 n=1; 
 
 %disp(chairfiles.name)
 %disp(class(chairfiles.name))
 
 for file=chairfiles'
     
     name = file.name;
     load(fullfile(file.folder, file.name), 'm'); 
     disp(size(m));
     s=size(m); 
     
     image= images(:,:,:,n);
     %imshow(image);
     
     if s(:,2)~=0
                      
         try
             idx=s(:,3); 
         catch
             idx=1;
         end 
        
         for i=1:idx
              
             cmask=repmat(m(:,:,i),[1,1,3]); 
             imagecopy= image;
             imagecopy(~cmask)=0;
             if i==7
                disp(cmask);
             end
             fname = sprintf(fullfile(outp,'img__%d__%d.png'), n, i); 
             %imshow(imagecopy);
             imwrite(imagecopy, fname);
         end
     else
         
         fprintf('Skip image %s', file.name);
         
     end
     
     
     n=n+1;
 end