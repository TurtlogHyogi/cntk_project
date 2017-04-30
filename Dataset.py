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

def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)

def list_image(in_dataset_dir, out_dataset_dir):
    i = 0

    cat = {}
    for path, dirs, files in os.walk(in_dataset_dir, followlinks=True):
        dirs.sort()
        files.sort()
        for fname in files:
            fpath = os.path.join(path, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in ['.bmp','.png','.jpeg','.jpg']):
                if path not in cat:
                    cat[path] = len(cat)
                yield (i, os.path.relpath(fpath, in_dataset_dir), cat[path])
                i += 1
    with open(out_dataset_dir+'/labels.txt','w') as labels:
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            labels.write(os.path.basename(os.path.relpath(k, in_dataset_dir))+'\n')

def make_list(in_dataset_dir, out_dataset_dir):
    image_list = list_image(in_dataset_dir,out_dataset_dir)
    image_list = list(image_list)
    #random.seed(100)
    #random.shuffle(image_list)
    N = len(image_list)
    chunk_size = N
    for i in range(1):
        chunk = image_list[i * int(chunk_size):(i + 1) * int(chunk_size)]
        str_chunk = ''
        sep = chunk_size * 1
        write_list(out_dataset_dir+'/train_map.txt', chunk)
        
def resize_to_PNGimg(in_filename,out_filename,resize):

    raw_img = Image.open(in_filename)
    resized_img = raw_img.resize((resize,resize))
    resized_img.save(out_filename)               

def savemean(fname,data,resize):
    root = et.Element('opencv_storage')
    et.SubElement(root,'Channel').text = '3'
    et.SubElement(root,'Row').text = str(resize)
    et.SubElement(root,'Col').text = str(resize)
    meanImg = et.SubElement(root,'MeanImg', type_id = 'opencv-matrix')
    et.SubElement(meanImg,'rows').text = '1'
    et.SubElement(meanImg,'cols').text = str(resize*resize*3)
    et.SubElement(meanImg,'dt').text = 'f'

    et.SubElement(meanImg,'data').text = ' '.join('%e' % data[i] for i in range(3072))

    tree = et.ElementTree(root)
    tree.write(fname) # make fname but it's line is 1
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = ' '))

def resizing(in_dataset_dir,out_dataset_dir,resize):

    global total_img_num,current_num_img,labelnum,check
    foldernames = [] 
    imgnames = []
    img_foldernames = []
    
    foldernames = get_foldernames(in_dataset_dir)
    label=0

    for foldername in foldernames:
        file = 0
        abs_in_foldername = os.path.join(in_dataset_dir,foldername)
        imgnames = get_imgnames(abs_in_foldername)
        
        extension = ['.jpg','.png','.jpeg','.bmp']
        for imgname in imgnames:
            if os.path.splitext(imgname)[1] in extension:
                total_img_num += 1
                file += 1

        if file != 0:
            img_foldernames.append(foldername)
                    
    labelnum = len(img_foldernames)
    
    check = 1   # start print_log thread

    img_mean = np.zeros((resize,resize,3)) 

    with open(out_dataset_dir+'/train_map.txt','w') as map:
        with open(out_dataset_dir+'/labels.txt','w') as labels:
            for foldername in img_foldernames: 
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
                        out_imgname = '{:0{}d}.png'.format(current_num_img,int(log10(total_img_num)+1))    
                        abs_out_imgname = os.path.join(abs_out_foldername,out_imgname)
                        resize_to_PNGimg(abs_in_imgname,abs_out_imgname,resize)

                        img = Image.open(abs_out_imgname)
                        if img.mode == 'RGB':
                            arr = np.array(img,dtype=np.float32)
                            img_mean += arr/total_img_num
                            current_num_img += 1
                            map.write(abs_out_imgname+'\t'+str(label)+'\n')

            label+=1
    img_mean = np.ascontiguousarray(np.transpose(img_mean,(2,0,1)))
    img_mean = img_mean.reshape(3*resize*resize)
    savemean(out_dataset_dir + './train_mean.xml',img_mean,resize)
    check=0 # stop print_log thread
    
def create_dataset(in_dataset_dir, out_dataset_dir, resize, framework):
    if not os.path.exists(in_dataset_dir):
        return print('Dataset directory is Wrong.')
    
    if not os.listdir(in_dataset_dir):
        return print('Dataset is not found.')
    
    if not os.path.exists(out_dataset_dir):
        os.makedirs(out_dataset_dir)
    
    resizing(in_dataset_dir,out_dataset_dir,resize)
    return True

def print_dataset_log():

    global total_img_num
    global current_num_img
    global check
    
    logger = logging.getLogger('Dataset')
    logger.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler(os.path.join(w_out_dataset_dir,'creat_val_db.log'),'w')
    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s:%(asctime)s], %(message)s')
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    
    while check == 1:
        message = 'Total=%d, Current=%d, Progress=%0.4f%%'%(total_img_num,current_num_img,(current_num_img*100/total_img_num))
        logger.info(message)
        time.sleep(1)
    if check == 0:
        message = 'Total=%d, Current=%d, Progress=%0.4f%%'%(total_img_num,current_num_img,(current_num_img*100/total_img_num))
        print('Dataset creating finished')
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
    global check
    if framework == 3: # CNTK
        log = threading.Thread(target = print_dataset_log)
        dataset = threading.Thread(target = create_dataset, args=(in_dataset_dir,out_dataset_dir,resize,framework))

        dataset.start()    
        time.sleep(0.1)
        log.start()
        while check:
            time.sleep(1)
            continue
    return True


total_img_num = 0 # -> epoch_size in Train.py
current_num_img = 0
check = 0
labelnum = 0
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

