from __future__ import print_function
import cntk
import cntk.io.transforms as xforms
import _cntk_py
import os
import Dataset
import logging
import re
import argparse
from importlib import import_module

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train and create an model')
    parser.add_argument('--out', default='/', help='path to folder output dataset.')

    tgroup = parser.add_argument_group('Options for Train')
    tgroup.add_argument('--resize', type=int, default=50, help='resize the shorter edge of image to the newsize, original images will be packed by default.')
    tgroup.add_argument('--channels', type=int, default=3 ,help='read Dataset channels')
    tgroup.add_argument('--epoch-size', type=int, default=0, help='number of images at one epoch')
    tgroup.add_argument('--out-dim', type=int, default=0, help='number of output classes')
    args = parser.parse_args()

    return args

# Reader
def create_reader(Dataset_result,train,train_args):
    for file in Dataset_result:
        if file.find('map.txt') != -1:
            map_file = file
        if file.find('mean.xml') != -1:
            mean_file = file
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        return print (r'map.txt or mean.xml file is not found')
    transforms = []
    if train:
        transforms +=[
            xforms.crop(crop_type='randomside',side_ratio=0.8,jitter_type='uniratio')
        ]
    transforms += [
        xforms.scale(width=train_args.resize,height=train_args.resize,channels=train_args.channels,interpolations='linear'),
        xforms.mean(mean_file)
    ]
    return cntk.io.MinibatchSource(cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
        features = cntk.io.StreamDef(field='image', transforms=transforms),
        labels = cntk.io.StreamDef(field='label', shape=train_args.out_dim))),
        randomize=train)

# Read_total_num
def read_num(file):
    num = 0
    with open(file,'r') as file:   
         while True:
            line = file.readline()
            if not line:
                break
            num += 1
    return num

# Read_size
def read_size(file):
    with open(file) as mean:
        p = re.compile('<[a-zA-Z]{3}>[0-9]+</[a-zA-Z]{3}>')
        while True:
            line = mean.readline()
            size = p.search(line)
            if size:
                return int((size.group())[5:-6])


def Train_create(dataset_dir, framework, out_model_dir, max_epochs, mb_size, network_name):
    if not os.path.exists(dataset_dir):
        return print("Dataset directory is wrong")

    if not os.path.exists(out_model_dir):
        os.makedirs(out_model_dir)
    cntk.device.set_default_device(cntk.device.gpu(cntk.device.best().type()))

    if framework == 3: # if cntk
    
        logger = logging.getLogger('Train')
        logger.setLevel(logging.DEBUG)
        filehandler = logging.FileHandler(os.path.join(w_out_dataset_dir,'create_train_db.log'),'w')
        streamhandler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s:%(asctime)s], %(message)s')
        filehandler.setFormatter(formatter)
        streamhandler.setFormatter(formatter)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)

        train_args = parse_args()
        train_args.resize = read_size([xml_file for xml_file in Dataset.Dataset_result(w_out_dataset_dir) if 'mean.xml' in xml_file][0])
        train_args.out_dim = read_num([map_file for map_file in Dataset.Dataset_result(w_out_dataset_dir) if 'labels.txt' in map_file][0])
        train_args.epoch_size = read_num([label_file for label_file in Dataset.Dataset_result(w_out_dataset_dir) if 'map.txt' in label_file][0])

        
        train_reader = create_reader(Dataset_result=Dataset.Dataset_result(dataset_dir),train_args=train_args,train=True)
    
        input = cntk.blocks.input_variable((train_args.channels,train_args.resize,train_args.resize))
        scaled_input = cntk.ops.element_times(input,(1/256))
        label = cntk.blocks.input_variable(train_args.out_dim)

        if network_name == 'convnet':
 
            z = import_module('network.'+network_name).get_network(train_args)(scaled_input)
            
            ce = cntk.ops.cross_entropy_with_softmax(z,label)
            pe = cntk.ops.classification_error(z,label)
            
            lr_per_sample = [0.0015625]*20+[0.00046875]*20+[0.00015625]*20+[0.000046875]*10+[0.000015625]
            lr_schedule = cntk.learning_rate_schedule(lr_per_sample, unit=cntk.learner.UnitType.sample, epoch_size=train_args.epoch_size)
            mm_time_constant = [0]*20+[600]*20+[1200]
            mm_schedule = cntk.learner.momentum_as_time_constant_schedule(mm_time_constant,epoch_size=train_args.epoch_size)
            learner = cntk.learner.momentum_sgd(z.parameters, lr=lr_schedule, momentum=mm_schedule, l2_regularization_weight=0.002)

            trainer = cntk.Trainer(z,(ce,pe),learner)
    
        input_map = {
            input : train_reader.streams.features,
            label : train_reader.streams.labels
        }
        
        progress_printer = cntk.utils.ProgressPrinter(tag="Training",num_epochs=max_epochs)
        
        sample, loss, metric = 0, 0, 0
        for epoch in range(max_epochs):
            sample_count = 0
            while sample_count<train_args.epoch_size:
                mb = train_reader.next_minibatch(min(mb_size,train_args.epoch_size-sample_count),input_map=input_map)
                trainer.train_minibatch(mb)
                sample_count += trainer.previous_minibatch_sample_count
                progress_printer.update_with_trainer(trainer, with_metric=True)
            
            new_sample, new_loss, new_metric = progress_printer.samples_since_start, progress_printer.loss_since_start, progress_printer.metric_since_start
            message = 'epoch = {} of {},'.format(1+epoch,max_epochs)+'loss = %f, metric = %f%%'%(((new_loss-loss)/(new_sample-sample)),100*(new_metric-metric)/(new_sample-sample))
            logger.info(message)
            sample, loss, metric = new_sample, new_loss, new_metric
            
            z.save(os.path.join(out_model_dir, "ConvNet_{}.dnn".format(epoch)))

        return print('Train_create finish')

def Train_result(model_dir):
    if not os.path.exists(model_dir):
        return print('model_dir is not found.')

    return os.listdir(model_dir)

w_my_dataset_dir = Dataset.w_my_dataset_dir
l_my_dataset_dir = Dataset.l_my_dataset_dir
w_out_dataset_dir = Dataset.w_out_dataset_dir
l_out_dataset_dir = Dataset.l_out_dataset_dir
w_out_model_dir = Dataset.w_out_dataset_dir + '/model'
l_out_model_dir = Dataset.l_out_dataset_dir + '/model'

if __name__ == '__main__':
    print('epoch_size={},out_dim={}'.format(read_num(Dataset.Dataset_result(w_out_dataset_dir)[1]),read_num(Dataset.Dataset_result(w_out_dataset_dir)[0])))
    Train_create(dataset_dir = w_out_dataset_dir,
                 framework = 3, 
                 out_model_dir = w_out_model_dir, 
                 max_epochs = 3, 
                 mb_size = 50, 
                 network_name = 'convnet')
    print(Train_result(w_out_model_dir))

