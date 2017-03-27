# Convnet
def create_ConvNet():#row=32,col=32,channels=3,out_dim=6):    
    global channels,row,col
    global in_dim,out_dim

    # input�� label Dim ����
    input = cntk.blocks.input_variable((channels,row,col))
    label = cntk.blocks.input_variable(out_dim)
    scaled_input = cntk.ops.element_times(input,(1/256))
    
    # moedl ����
    with cntk.layers.default_options(activation=cntk.ops.relu):
        model = cntk.layers.Sequential([
            cntk.layers.For(range(2),lambda : [
                cntk.layers.Convolution2D((3,3), 64),
                cntk.layers.Convolution2D((3,3), 64),
                cntk.layers.MaxPooling((3,3), strides = 2)
            ]),
            cntk.layers.For(range(2),lambda i : [ 
                cntk.layers.Dense([256,128][i]),
                cntk.layers.Dropout(0.5)
            ]),
            cntk.layers.Dense(out_dim,activation=None)
        ])

    return model
