load('./nyu_depth_v2_labeled.mat','instances');
load('./nyu_depth_v2_labeled.mat','labels');
%load('./nyu_depth_v2_labeled.mat','namesToIds');
%load('./nyu_depth_v2_labeled.mat','images');

classes = zeros(1,894);


for i=1:length(instances)
   
    instance = instances(:,:,i);
    label = labels(:,:, i);
    %disp(instance)
    
    [masks, lab] = get_instance_masks(label, instance);
    
    %disp(lab)
    
    for n=1:length(lab)
        
       current_lab = lab(n); 
       
       classes(1,current_lab) = classes(1, current_lab) + 1 ; 
        
    end
    %{
    
    [rows,cols] = size(instance);
    
    seen = uint8.empty; %zeros(894, 'uint16');
    
    for r=1:rows
        
       
       for c=1:cols
           
          
           current = instance(r,c);
           current_lab= label(r,c);
           
           
           if current_lab==0 
               
               continue
               
           end 
           
           
           %curr_cell = {current_lab};
           %disp(class(curr_cell));
           %disp(class(seen));
          
           
           if ismember(current, seen)
                 continue
             
           end
           
           
           classes(1,current_lab) = classes(1, current_lab) + 1 ;
           
           %keylab = keys(map.namesToIds);
           %class_name = keylab{current};
           
           seen = [seen, current];
       
           
       end
        
        
    end
       
    
    %}
    
    
end


csvwrite('/mnt/c/Users/HP/Desktop/KMI/NYUdepth/descriptives.csv', classes)