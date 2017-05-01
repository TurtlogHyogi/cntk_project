import cntk

# Convnet
def create_ConvNet(train_args):
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
            cntk.layers.Dense(train_args.out_dim,activation=None)
        ])

    return model
