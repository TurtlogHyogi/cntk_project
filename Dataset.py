from __future__ import print_function
import os
import numpy as np
import random
import threading,time
import logging
from PIL import Image
from math import log10
import xml.etree.ElementTree as et
import xml.dom.minidom
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Create dataset(resized_img,mean_xml,map.txt,label,txt)')
    parser.add_argument('--root', default='/', help='path to folder containing images.')
    parser.add_argument('--out', default='/', help='path to folder output dataset.')

    dgroup = parser.add_argument_group('Options for creating Dataset')
    dgroup.add_argument('--resize', type=int, default=50, help='resize the shorter edge of image to the newsize, original images will be packed by default.')
    dgroup.add_argument('--channel', type=int, default=3, help='channels of image. RGB -> 3, Gray -> 1')
    dgroup.add_argument('--total-img-num', type=int, default=0, help='logging data, total number of dataset')
    dgroup.add_argument('--current-img-num', type=int, default=0, help='logging data, number of dataset made')
    dgroup.add_argument('--check', type=bool, default=False, help='if check=True -> running print_log thread, else -> stop print_log thread')
    args = parser.parse_args()

    return args

def get_foldernames(in_dataset_dir):
    filenames = os.listdir(in_dataset_dir)    
    foldernames = []

    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == '':
            foldernames.append(filename)

    return foldernames

def get_imgnames(foldername):
    imgnames = []
    imgnames = os.listdir(foldername)    

    return imgnames

def resize_to_PNGimg(in_filename,out_filename,resize):
    raw_img = Image.open(in_filename)
    resized_img = raw_img.resize((resize,resize))
    resized_img.save(out_filename)               

def savemean(fname,data,dataset_args):
    root = et.Element('opencv_storage')
    et.SubElement(root,'Channel').text = str(dataset_args.channel)
    et.SubElement(root,'Row').text = str(dataset_args.resize)
    et.SubElement(root,'Col').text = str(dataset_args.resize)
    meanImg = et.SubElement(root,'MeanImg', type_id = 'opencv-matrix')
    et.SubElement(meanImg,'rows').text = '1'
    et.SubElement(meanImg,'cols').text = str(dataset_args.channel*dataset_args.resize*dataset_args.resize)
    et.SubElement(meanImg,'dt').text = 'f'

    et.SubElement(meanImg,'data').text = ' '.join('%e' % data[i] for i in range(dataset_args.channel*dataset_args.resize*dataset_args.resize))

    tree = et.ElementTree(root)
    tree.write(fname) # make fname but it's line is 1
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = ' '))

def resizing(in_dataset_dir,out_dataset_dir,resize,dataset_args):
    foldernames = get_foldernames(in_dataset_dir)
    label=0

    img_mean = np.zeros((resize,resize,3)) 

    with open(out_dataset_dir+'/train_map.txt','w') as map:
        with open(out_dataset_dir+'/labels.txt','w') as labels:
            dataset_args.check = True
            for foldername in foldernames: 
                labels.write(foldername+'\n')
                abs_in_foldername = os.path.join(in_dataset_dir,foldername)
                abs_out_foldername = os.path.join(out_dataset_dir,'train_db')
                if not os.path.exists(abs_out_foldername):
                        os.makedirs(abs_out_foldername)
                imgnames = get_imgnames(abs_in_foldername)
                
                for in_imgname in imgnames: 
                    pixindex=0
                    abs_in_imgname = os.path.join(abs_in_foldername,in_imgname) 
                    extension = ['.jpg','.png','.jpeg','.bmp']
                    if os.path.splitext(abs_in_imgname)[1] in extension:
                        out_imgname = '{:0{}d}.png'.format(dataset_args.current_img_num,int(log10(dataset_args.total_img_num)+1))    
                        abs_out_imgname = os.path.join(abs_out_foldername,out_imgname)
                        resize_to_PNGimg(abs_in_imgname,abs_out_imgname,resize)

                        img = Image.open(abs_out_imgname)
                        if img.mode == 'RGB':
                            arr = np.array(img,dtype=np.float32)
                            img_mean += arr/dataset_args.total_img_num
                            dataset_args.current_img_num += 1
                            map.write(abs_out_imgname+'\t'+str(label)+'\n')
                label+=1

    img_mean = np.ascontiguousarray(np.transpose(img_mean,(2,0,1)))
    img_mean = img_mean.reshape(3*resize*resize)
    savemean(out_dataset_dir + './train_mean.xml',img_mean, dataset_args)
    dataset_args.check = False # stop print_log thread
    
def create_dataset(in_dataset_dir, out_dataset_dir, resize, framework, dataset_args):
    if not os.path.exists(in_dataset_dir):
        return print('Dataset directory is Wrong.')
    
    if not os.listdir(in_dataset_dir):
        return print('Dataset is not found.')
    
    if not os.path.exists(out_dataset_dir):
        os.makedirs(out_dataset_dir)
    resizing(in_dataset_dir,out_dataset_dir,resize,dataset_args)
    return True

def print_dataset_log(in_dataset_dir, out_dataset_dir, resize, framework, dataset_args):
    while not dataset_args.check:
        pass

    logger = logging.getLogger('Dataset')
    logger.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler(os.path.join(w_out_dataset_dir,'creat_val_db.log'),'w')
    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s:%(asctime)s], %(message)s')
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    
    while dataset_args.check == True:
        message = 'Total={0}, Current={1}, Progress={2:0.4f}'.format(dataset_args.total_img_num,dataset_args.current_img_num,(dataset_args.current_img_num*100/dataset_args.total_img_num))
        logger.info(message)
        time.sleep(1)
    if dataset_args.check == False:
        message = 'Total={0}, Current={1}, Progress={2:0.4f}'.format(dataset_args.total_img_num,dataset_args.current_img_num,(dataset_args.current_img_num*100/dataset_args.total_img_num))
        logger.info(message)
        

def Dataset_result(out_dataset_dir):
    found_dir = os.path.exists(out_dataset_dir)
    found_files = os.listdir(out_dataset_dir)
    found_dataset = []

    if not found_dir:
        return print('Directory is not found')    

    if not found_files:
        return print('Dataset is not found')
    
    for found_file in found_files:
        if os.path.splitext(found_file)[1] in ('.txt','.xml'):
            found_file = os.path.join(out_dataset_dir,found_file)
            found_dataset.append(found_file)

    found_dataset.sort()

    return found_dataset
    
def Dataset_create(in_dataset_dir, out_dataset_dir, resize, framework):
    if framework == 3: # CNTK
        dataset_args = parse_args()
        dataset_args.root = in_dataset_dir
        dataset_args.out = out_dataset_dir
        dataset_args.resize = resize
    
        foldernames = get_foldernames(in_dataset_dir)

        for foldername in foldernames:
            abs_in_foldername = os.path.join(in_dataset_dir,foldername)
            imgnames = get_imgnames(abs_in_foldername)
        
            extension = ['.jpg','.png','.jpeg','.bmp']
            for imgname in imgnames:
                if os.path.splitext(imgname)[1] in extension:
                    dataset_args.total_img_num += 1

        functions = [create_dataset,print_dataset_log]
        threads = []
        for function in functions:
            func_args = (in_dataset_dir,out_dataset_dir,resize,framework,dataset_args)
            th = threading.Thread(target=function, args=func_args, name = function)
            th.start()
            threads.append(th)
        for thread in threads:
            thread.join()
        print('all ths finished')
    return True

w_my_dataset_dir = r'D:\Github\cntk_dataset\img'
w_out_dataset_dir = r'D:\Github\cntk_dataset\outdataset'
l_my_dataset_dir = r'/root/git/cntk_dataset/img'
l_out_dataset_dir = r'/root/git/cntk_dataset/out_dataset'

if __name__ == '__main__':
    Dataset_create(in_dataset_dir = w_my_dataset_dir,
                   out_dataset_dir = w_out_dataset_dir,
                   resize = 32,
                   framework = 3)
    print(Dataset_result(w_out_dataset_dir))

