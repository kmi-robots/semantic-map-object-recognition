load('./nyu_depth_v2_labeled.mat','instances');
load('./nyu_depth_v2_labeled.mat','labels');
load('./nyu_depth_v2_labeled.mat','namesToIds');


classes = zeros(894);
seen = uint16.empty; %zeros(894, 'uint16');


for i=1:length(instances)
   
    instance = instances(:,:,i);
    label = labels(:,:, i);
    
    [rows,cols] = size(instance);
    
    
    for r=1:rows
        
       
       for c=1:cols
           
          
           current = instance(r,c);
           current_lab= label(r,c);
           
           
           if current_lab==0
               
               continue
               
           end 
           
           
           curr_cell = {current_lab};
           %disp(class(curr_cell));
           %disp(class(seen));
          
           
           if ismember(current_lab, seen)
                 continue
             
           end
           
           
           classes(current_lab) = classes(current_lab) + 1 ;
           
           %keylab = keys(map.namesToIds);
           %class_name = keylab{current};
           
           seen = [seen, current_lab];
       
           
       end
        
        
    end
       
    
    
    
    
end


csvwrite('/mnt/c/Users/HP/Desktop/KMI/NYUdepth/descriptives.csv', classes)