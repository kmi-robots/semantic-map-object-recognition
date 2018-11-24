import keras 
import sys 
from keras import backend as K 
from keras.layers import Conv2D, MaxPooling2D, Dense,Input, Flatten 
from keras.models import Model, Sequential 
from keras.engine import InputSpec, Layer 
from keras import regularizers 
from keras.optimizers import SGD, Adam 
from keras.utils.conv_utils import conv_output_length 
from keras import activations 
import numpy as np


#### A bunch of ugly hardcoded object IDs    ##############################

#Micro, i.e., same object different views
chair1=['9e14d77634cf619f174b6156db666192-0.png', '9e14d77634cf619f174b6156db666192-2.png', '9e14d77634cf619f174b6156db666192-5.png', '9e14d77634cf619f174b6156db666192-7.png', '9e14d77634cf619f174b6156db666192-10.png', '9e14d77634cf619f174b6156db666192-12.png', 'c.png']
chair2=['49918114029ce6a63db5e7f805103dd-0.png', '49918114029ce6a63db5e7f805103dd-1.png' ,'49918114029ce6a63db5e7f805103dd-5.png', '49918114029ce6a63db5e7f805103dd-6.png', '49918114029ce6a63db5e7f805103dd-9.png', '49918114029ce6a63db5e7f805103dd-11.png', '49918114029ce6a63db5e7f805103dd-13.png']

plant1=['4d637018815139ab97d540195229f372-1.png', '4d637018815139ab97d540195229f372-3.png', '4d637018815139ab97d540195229f372-7.png', '4d637018815139ab97d540195229f372-8.png', '4d637018815139ab97d540195229f372-11.png', '4d637018815139ab97d540195229f372-12.png'] 
bin1=['7bde818d2cbd21f3bac465483662a51d-0.png', '7bde818d2cbd21f3bac465483662a51d-3.png', '7bde818d2cbd21f3bac465483662a51d-10.png', '7bde818d2cbd21f3bac465483662a51d-12.png']
bin2=['8ab06d642437f3a77d8663c09e4f524d-0.png', '8ab06d642437f3a77d8663c09e4f524d-3.png', '8ab06d642437f3a77d8663c09e4f524d-5.png', '8ab06d642437f3a77d8663c09e4f524d-8.png', '8ab06d642437f3a77d8663c09e4f524d-9.png', '8ab06d642437f3a77d8663c09e4f524d-13.png']
display1=['2d5d4d79cd464298566636e42679cc7f-0.png', '2d5d4d79cd464298566636e42679cc7f-1.png', '2d5d4d79cd464298566636e42679cc7f-2.png', '2d5d4d79cd464298566636e42679cc7f-5.png', '2d5d4d79cd464298566636e42679cc7f-6.png', '2d5d4d79cd464298566636e42679cc7f-7.png', '2d5d4d79cd464298566636e42679cc7f-9.png', '2d5d4d79cd464298566636e42679cc7f-11.png', '2d5d4d79cd464298566636e42679cc7f-13.png']
display2=['17226b72d812ce47272b806070e7941c-1.png', '17226b72d812ce47272b806070e7941c-3.png', '17226b72d812ce47272b806070e7941c-4.png', '17226b72d812ce47272b806070e7941c-5.png', '17226b72d812ce47272b806070e7941c-6.png', '17226b72d812ce47272b806070e7941c-8.png', '17226b72d812ce47272b806070e7941c-9.png', '17226b72d812ce47272b806070e7941c-13.png']
printer1= ['7c1ac983a6bf981e8ff3763a6b02b3bb-0.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-1.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-4.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-5.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-8.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-10.png', '7c1ac983a6bf981e8ff3763a6b02b3bb-13.png']
printer2= ['2135295ad1c580e7ffbff4948728b4f5-0.png', '2135295ad1c580e7ffbff4948728b4f5-1.png', '2135295ad1c580e7ffbff4948728b4f5-2.png', '2135295ad1c580e7ffbff4948728b4f5-3.png', '2135295ad1c580e7ffbff4948728b4f5-6.png', '2135295ad1c580e7ffbff4948728b4f5-7.png', '2135295ad1c580e7ffbff4948728b4f5-8.png', '2135295ad1c580e7ffbff4948728b4f5-9.png', '2135295ad1c580e7ffbff4948728b4f5-13.png']
bottle1= ['9eccbc942fc8c0011ee059e8e1a2ee9-6.png', '9eccbc942fc8c0011ee059e8e1a2ee9-7.png', '9eccbc942fc8c0011ee059e8e1a2ee9-10.png', '9eccbc942fc8c0011ee059e8e1a2ee9-11.png']
bottle2= ['62451f0ab130709ef7480cb1ee830fb9-0.png', '62451f0ab130709ef7480cb1ee830fb9-1.png', '62451f0ab130709ef7480cb1ee830fb9-6.png', '62451f0ab130709ef7480cb1ee830fb9-8.png']

#bottle3= ['d851cbc873de1c4d3b6eb309177a6753_0.png', 'd851cbc873de1c4d3b6eb309177a6753_6.png', 'd851cbc873de1c4d3b6eb309177a6753_11.png', 'd851cbc873de1c4d3b6eb309177a6753_12.png']
paper1=['3dw.b463019e5f3b9a9b5c6bfbdfe6a8f99.png', '3dw.b463019e5f3b9a9b5c6bfbdfe6a8f99-2.png', '3dw.b463019e5f3b9a9b5c6bfbdfe6a8f99-3.png', '3dw.b463019e5f3b9a9b5c6bfbdfe6a8f99-4.png']
paper2=['3dw.14b4294e99cb10215f606243e56be258.png', '3dw.14b4294e99cb10215f606243e56be258-2.png', '3dw.14b4294e99cb10215f606243e56be258-3.png', '3dw.14b4294e99cb10215f606243e56be258-4.png']
book1=['3dw.13d22e3b3657e229ce6cd687d82659e9.png', '3dw.13d22e3b3657e229ce6cd687d82659e9-2.png', '3dw.13d22e3b3657e229ce6cd687d82659e9-3.png', '3dw.13d22e3b3657e229ce6cd687d82659e9-4.png']
book2= ['3dw.1d493a57a21833f2d92c7cdc3939488b.png', '3dw.1d493a57a21833f2d92c7cdc3939488b-2.png', '3dw.1d493a57a21833f2d92c7cdc3939488b-3.png', '3dw.1d493a57a21833f2d92c7cdc3939488b-4.png']
table1=['f9f9d2fda27c310b266b42a2f1bdd7cf-4.png', 'f9f9d2fda27c310b266b42a2f1bdd7cf-10.png', 'f9f9d2fda27c310b266b42a2f1bdd7cf-11.png', 'f9f9d2fda27c310b266b42a2f1bdd7cf-13.png']
table2=['7807caccf26f7845e5cf802ea0702182-1.png', '7807caccf26f7845e5cf802ea0702182-6.png', '7807caccf26f7845e5cf802ea0702182-11.png', '7807caccf26f7845e5cf802ea0702182-12.png']
box1=['3dw.f0f1419ffe0e4475242df0a63deb633.png', '3dw.f0f1419ffe0e4475242df0a63deb633-2.png', '3dw.f0f1419ffe0e4475242df0a63deb633-3.png', '3dw.f0f1419ffe0e4475242df0a63deb633-4.png']
box2=['3dw.405e820d9717e1724cb8ef90d0735cb6.png', '3dw.405e820d9717e1724cb8ef90d0735cb6-2.png', '3dw.405e820d9717e1724cb8ef90d0735cb6-3.png', '3dw.405e820d9717e1724cb8ef90d0735cb6-4.png']
window1=['3dw.2f322060f3201f71caf432acbbd622b.png', '3dw.2f322060f3201f71caf432acbbd622b-2.png']
window2=['3dw.e884d2ee658acf6faa0e334660b67084.png', '3dw.e884d2ee658acf6faa0e334660b67084-2.png', '3dw.e884d2ee658acf6faa0e334660b67084-3.png', '3dw.e884d2ee658acf6faa0e334660b67084-4.png' ]
door1=['3dw.f40d5808bf78f8003f7c9f4b711809d.png', '3dw.f40d5808bf78f8003f7c9f4b711809d-2.png']
door2=['3dw.c67fa55ac55e0ca0a3ef625a8daeb343.png', '3dw.c67fa55ac55e0ca0a3ef625a8daeb343-2.png']
sofa1=['4820b629990b6a20860f0fe00407fa79-0.png', '4820b629990b6a20860f0fe00407fa79-7.png', '4820b629990b6a20860f0fe00407fa79-9.png', '4820b629990b6a20860f0fe00407fa79-13.png']
sofa2=['87f103e24f91af8d4343db7d677fae7b-0.png', '87f103e24f91af8d4343db7d677fae7b-6.png', '87f103e24f91af8d4343db7d677fae7b-7.png', '87f103e24f91af8d4343db7d677fae7b-12.png']
lamp1=['3dw.3c5db21345130f0290f1eb8f29abcea8.png', '3dw.3c5db21345130f0290f1eb8f29abcea8-2.png']
lamp2=['6770adca6c298f68fc3f90c1b551a0f7-4.png', '6770adca6c298f68fc3f90c1b551a0f7-6.png', '6770adca6c298f68fc3f90c1b551a0f7-8.png', '6770adca6c298f68fc3f90c1b551a0f7-12.png']



#Macro, i.e., same object class
chairs=  list(set().union(chair1,chair2))
plants=  plant1
bins=  list(set().union(bin1,bin2))
displays =  list(set().union(display1,display2))
printers = list(set().union(printer1, printer2))

bottles = list(set().union(bottle1, bottle2)) #bottle3))
papers = list(set().union(paper1, paper2)) #bottle3))
books = list(set().union(book1, book2)) #bottle3))
tables = list(set().union(table1, table2)) #bottle3))
boxes = list(set().union(box1, box2)) #bottle3))
windows = list(set().union(window1, window2)) #bottle3))
doors = list(set().union(door1, door2)) #bottle3))
sofas = list(set().union(sofa1, sofa2)) #bottle3))
lamps = list(set().union(lamp1, lamp2)) #bottle3))

#all_ids = list(set().union(chairs,plants,bins)) # displays, printers))
all_ids = list(set().union(chairs, bottles, papers, books, tables, boxes, windows, doors, sofas, lamps))

object_list = [chairs, bottles, papers, books, tables, boxes, windows, doors, sofas, lamps]
flags = ["chairs", 'bottles', 'papers', 'books', 'tables', 'boxes', 'windows', 'doors', 'sofas', 'lamps']


class Normalized_Correlation_Layer(Layer):
    
    
    #This layer does Normalized Correlation.
    
    #It needs to take two inputs(layers),
    #currently, it only supports the border_mode = 'valid',
    #if you need to output the same shape as input, 
    #do padding before giving the layer.
    


    def __init__(self, patch_size=(5,5),
                 dim_ordering='tf',
                 border_mode='same',
                 stride=(1, 1),
                 activation=None,
                 **kwargs):

        if border_mode != 'same':
            raise ValueError('Invalid border mode for Correlation Layer '
                             '(only "same" is supported as of now):', border_mode)
        self.kernel_size = patch_size
        self.subsample = stride
        self.dim_ordering = dim_ordering
        self.border_mode = border_mode
        self.activation = activations.get(activation)
        super(Normalized_Correlation_Layer, self).__init__(**kwargs)


    def compute_output_shape(self, input_shape):
        
        if self.dim_ordering == 'tf':
            inp_rows = input_shape[0][1]
            inp_cols = input_shape[0][2]
        else:
            raise ValueError('Only support tensorflow.')
        
        if self.border_mode != "same":
            rows = conv_output_length(inp_rows, self.kernel_size[0],
                                       self.border_mode, 1)
            cols = conv_output_length(inp_cols, self.kernel_size[1],
                                       self.border_mode, 1)
        else:
            rows = inp_rows
            cols = inp_cols
        
        return (input_shape[0][0], rows, cols,self.kernel_size[0]*cols*input_shape[0][-1])
    

    def call(self, x, mask=None):
        
        input_1, input_2 = x
        stride_row, stride_col = self.subsample
        inp_shape = input_1._keras_shape
        
        '''
        print(input_1._keras_shape)
        print('input1 shape is %s' % str(inp_shape))
        print('input1 shape is %s' % str(inp_shape[-1]))
        '''
        
        #print(input_1.shape)
        #print(input_2.shape)
        
        output_shape = self.compute_output_shape([inp_shape, inp_shape])
        
        #Add padding to both feature maps
        padding_row = (int(self.kernel_size[0] / 2),int(self.kernel_size[0]/2))
        padding_col = (int(self.kernel_size[1] / 2),int(self.kernel_size[1]/2))
        input_1 = K.spatial_2d_padding(input_1, padding =(padding_row,padding_col))
        input_2 = K.spatial_2d_padding(input_2, padding = ((padding_row[0]*2, padding_row[1]*2),padding_col))
        
        #should equal 2
        #print(input_1.shape)
        #print(input_2.shape)
        #print(padding_col)
        
        #sys.exit(0)
        output_row = output_shape[1] 
        output_col = output_shape[2]
        #print('output shape row is %s' % str(output_row))
        #print('output shape col is %s' % str(output_col))
        
        output = []
        #sys.exit(0)
        
        # range is (0, 25), i.e., loop for all the depths
        for k in range(inp_shape[-1]):
            xc_1 = []
            xc_2 = []
            
            
            #extract patches of 5 from both feature maps
            
            #first two rows, to take care of padding top = 2
            for i in range(padding_row[0]):
                
                #range is (0,12) in this case
                for j in range(output_col):
                    xc_2.append(K.reshape(input_2[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1], k],
                                          (-1, 1,self.kernel_size[0]*self.kernel_size[1])))
                    
                    #This should be (-1, 1, 25)
                    #print(K.reshape(input_2[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1], k],
                    #                      (-1, 1,self.kernel_size[0]*self.kernel_size[1])).shape)
                    
            #print('xc1_2 length 1 is % i' % int(len(xc_2))) #24 = 12*2
            
            #from 0 to 37
            #rint(padding_row)
            #sys.exit(0)
            #
            for i in range(output_row):
                slice_row = slice(i, i + self.kernel_size[0])
                slice_row2 = slice(i+padding_row[0], i +self.kernel_size[0]+padding_row[0])
                #from 0 to 12
                for j in range(output_col):
                    slice_col = slice(j, j + self.kernel_size[1])
                    
                    
                    xc_2.append(K.reshape(input_2[:, slice_row2, slice_col, k],
                                          (-1, 1,self.kernel_size[0]*self.kernel_size[1])))
                    
                    #This should be (-1, 1, 25)
                    #print(K.reshape(input_2[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1], k],
                    #                      (-1, 1,self.kernel_size[0]*self.kernel_size[1])).shape)
                    
                    if i % stride_row == 0 and j % stride_col == 0:
                        
                        xc_1.append(K.reshape(input_1[:, slice_row, slice_col, k],
                                              (-1, 1,self.kernel_size[0]*self.kernel_size[1])))
                    
                    #print(K.reshape(input_2[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1], k],
                    #                      (-1, 1,self.kernel_size[0]*self.kernel_size[1])).shape)
                    
                #print('xc1_1 length is % i' % int(len(xc_1)))
                #sys.exit(0)
            #print('xc1_1 length is % i' % int(len(xc_1)))      #444
            #print('xc1_2 length 2 is % i' % int(len(xc_2)))    #468  
            
            
            #extra last two rows of padding = 2 added on the bottom
            for i in range(output_row, output_row+padding_row[0]):
                for j in range(output_col):
                    xc_2.append(K.reshape(input_2[:, i:i+ self.kernel_size[0], j:j+self.kernel_size[1], k],
                                          (-1, 1,self.kernel_size[0]*self.kernel_size[1])))
            
            
            #print('xc1_2 length 3 is % i' % int(len(xc_2)))   #492 = 468 + 24
            #sys.exit(0)
            
            xc_1_aggregate = K.concatenate(xc_1, axis=1) # batch_size x w'h' x (k**2*d), w': w/subsample-1
            #This should be (-1, 60, 25)?    
            
            #print('Imp prints start here:')
            #print(xc_1_aggregate.shape)
            
            #compute the normalized correlation
            xc_1_mean = K.mean(xc_1_aggregate, axis=-1, keepdims=True)
            xc_1_std = K.std(xc_1_aggregate, axis=-1, keepdims=True)
            xc_1_aggregate = (xc_1_aggregate - xc_1_mean) / xc_1_std
    
            
            xc_2_aggregate = K.concatenate(xc_2, axis=1) # batch_size x wh x (k**2*d), w: output_row
            #This should be (-1, 60, 25)
            #print(xc_1_aggregate.shape)
          
            #similarly to compute the normalized version of feature map 2
            xc_2_mean = K.mean(xc_2_aggregate, axis=-1, keepdims=True)
            xc_2_std = K.std(xc_2_aggregate, axis=-1, keepdims=True)
            xc_2_aggregate = (xc_2_aggregate - xc_2_mean) / xc_2_std
            
            xc_1_aggregate = K.permute_dimensions(xc_1_aggregate, (0, 2, 1))
            
            
            block = []
            len_xc_1= len(xc_1)
            #print(len_xc_1) # 444 where it should be 60
            #print(str(inp_shape[1])) #37
            
            for i in range(len_xc_1):
              
                sl1 = slice(int(i/inp_shape[1])*inp_shape[1],
                        int(i/inp_shape[1])*inp_shape[1]+inp_shape[1]*self.kernel_size[0])
                
                #the dot product between the two normalized units is stored in block
                block.append(K.reshape(K.batch_dot(xc_2_aggregate[:,sl1,:],
                                      xc_1_aggregate[:,:,i]),(-1,1,1,inp_shape[1]*self.kernel_size[0])))

            block = K.concatenate(block, axis=1)
            block = K.reshape(block,(-1,output_row,output_col,inp_shape[2]*self.kernel_size[0]))
            #print(block.shape) #should be (37, 12, 60)
            #sys.exit(0)
            output.append(block)
        
        #print(len(output))
        output = K.concatenate(output, axis=-1)
        
        #print(output.shape)
        
        
        output = self.activation(output)
        #print(output.shape)
        return output

    def get_config(self):
        config = {'patch_size': self.kernel_size,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'stride': self.subsample,
                  'dim_ordering': self.dim_ordering}
        base_config = super(Correlation_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def normalized_X_corr_model():
    
    
    a = Input((160,60,3))
    b = Input((160,60,3))
    
    model = Sequential()
    
    model.add(Conv2D(kernel_size = (5,5), filters = 20,input_shape = (160,60,3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(kernel_size = (5,5), filters =  25, activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    
    model1 = model(b)
    model2 = model(a)
    
    #print(model1.shape)
    #print(model2.shape)
    
    normalized_layer = Normalized_Correlation_Layer(stride = (1,1), patch_size = (5, 5))([model1, model2])
    
    
    final_layer = Conv2D(kernel_size=(1,1), filters=25, activation='relu')(normalized_layer)
    final_layer = Conv2D(kernel_size=(3,3), filters=25, activation = None)(final_layer)
    final_layer = MaxPooling2D((2,2))(final_layer)
    final_layer = Dense(500)(final_layer)
    
    #print(final_layer.shape)
    
    final_layer =Flatten()(final_layer)
    
    #print(final_layer.shape)
    
    #WARNING! Contrarily to Keras' documentation,this does not flatten the input to binary as expected
    #we had to flatten above
    final_layer = Dense(2, activation = "softmax")(final_layer)
    
    #Finally, a new model is created with inputs as the images to be passed as a list, which gives a binary output.
    x_corr_mod = Model(inputs=[a,b], outputs = final_layer)
    
    try:
        x_corr_mod.summary()
    except:
        pass
    print(x_corr_mod.output._keras_shape)
    return x_corr_mod

#def norm_model(input_size = (8,8,2)):
#    a = Input(input_size) 
#    b = Input(input_size)
#    output = Normalized_Correlation_Layer(stride = (1,1), patch_size = (5,5))([a,b])
#    m = Model(inputs=[a,b], outputs= output)
#    return m
 


if __name__ == "__main__":
    
    
    import sys, os
    
    
    #Initializing empty model first
    print("Initializing Siamese Net")
    
    #Uncomment later
    siamese= normalized_X_corr_model()
    #sys.exit(0)
    
    #Reads images passed through command line as arrays
    #argv 1 = path to train set 
    #arv2 = path to test set
    
    import cv2
        
    train_examples =[os.path.join(sys.argv[1], img) for img in os.listdir(sys.argv[1])]
    #print(len(train_examples))
    isfirst= True
    first= True
    
    print("Starting image vec reading")
    
    '''
    pairs=[]
    for i, path in enumerate(train_examples):
      
        
        print("Iter no. **** %i" % i)
        #Read img as RGB, resize and fix formats
        im1  = cv2.imread(path)
        x1 = cv2.resize(im1, (60,160))
        x1 = np.asarray(x1)
        x1 = x1.astype('float32')
        x1 = np.expand_dims(x1, axis= 0)
        
        #print(x1.shape)
        
        
        
        for path2 in train_examples[i+1:]:
                     
                im2  = cv2.imread(path2)
                x2 = cv2.resize(im2, (60,160))
                x2 = np.asarray(x2)
                x2 = x2.astype('float32')
                x2 = np.expand_dims(x2, axis= 0)
                
                pairs.append((path, path2))
                
                if isfirst:
                    
                    a = x1
                    b = x2
                    
                else:
                  
                    a = np.vstack((a,x1))
                    b = np.vstack((b,x2))
                    
                isfirst= False
                
        #Pair up all list elements except the current one
        #rint(a.shape)
        #print(b.shape)
        
      
        
    
    
    #Label pairs as positive or negative examples
    for path1, path2 in pairs:
    #for i, path1 in enumerate(train_examples):
      
            
           filename1= path1.split("/")[len(path1.split("/"))-1]
           

           filename2= path2.split("/")[len(path2.split("/"))-1]
          
           label= np.array([0,1]) #np.zeros((2))
          
           for l in object_list:
           
               if filename1 in l and filename2 in l:
                  
                    #both in same category
                    label = label= np.array([1,0]) #np.ones((2))
                    break
                  
                  
           if first:
               labels = label
               first= False
                  
           else:
                  
               labels = np.vstack((labels,label))
              
    print("Training set (images and labels) ready!")
    
    print(labels.shape)
    print(a.shape)
    print(b.shape)
    #np.save('imgset_left.npy', a)
    #np.save('imgset_right.npy', b)
    np.save('gt_labels.npy', labels)
    
    
    sys.exit(0) 
    
    '''
    print('Loading training sets...')
    a= np.load('imgset_left.npy')
    b =np.load('imgset_right.npy')
    labels = np.load('gt_labels.npy')
    
    
    print("Compiling the Keras model")    
    #The Keras model initialized above is compiled here
    siamese.compile(loss = 'categorical_crossentropy',  optimizer = Adam(lr = 0.0001, decay = 1e-6))
        
    #And trained here for a given number of epochs
    #one line per epoch should be drawn
    
    print("Starting training")
    
    output = siamese.fit([a,b], labels, batch_size=1, validation_split=0.2, shuffle = True, verbose = 2, epochs = 100)
    
    '''
    TODO: create
    [a_test, b_test] and label_test
    
    
    test_mod.evaluate([a_test, b_test], label_test, batch_size=1, )
    
    '''
    
    
    np.save("output", output)
