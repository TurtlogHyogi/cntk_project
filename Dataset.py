from __future__ import print_function
import os
import cntk
import numpy as np
import threading,time
import logging
from PIL import Image
from math import log10

total_img_num = 0 # -> epoch_size in Train.py
current_num_img = 0
check = 0
pixels = []
labelnum = 0

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

def resize_to_PNGimg(in_filename,out_filename,row,col):

    raw_img = Image.open(in_filename)
    resized_img = raw_img.resize((row,col))
    resized_img.save(out_filename)               

def resizing(in_dataset_dir,out_dataset_dir,row,col):

    global total_img_num,current_num_img,labelnum
    foldernames = []
    imgnames = []
    img_foldernames = []
    
    foldernames = get_foldernames(in_dataset_dir)
    label=0

    for foldername in foldernames:
        file = 0
        abs_in_foldername = os.path.join(in_dataset_dir,foldername)
        imgnames = get_imgnames(abs_in_foldername)
        
        extension = ['.jpg','.png','.jpen','.bmp']
        for imgname in imgnames:
            if os.path.splitext(imgname)[1] in extension:
                total_img_num += 1
                file += 1

        if file != 0:
            img_foldernames.append(foldername)
                    
    labelnum = len(img_foldernames)
    with open(out_dataset_dir + './train_mean.xml','w') as mean:
        mean.write('''<?xml version="1.0" ?>
<opencv_storage>
  <Channel>{}</Channel>
  <Row>{}</Row>
  <Col>{}</Col>
  <MeanImg type_id="opencv-matrix">
    <rows>1</rows>
    <cols>3072</cols>
    <dt>f</dt>
    <data>'''.format(3,row,col))

        with open(out_dataset_dir+'./train_data.txt','w') as text:
            with open(out_dataset_dir+'./train_map.txt','w') as map:
                with open(out_dataset_dir+'./labels.txt','w') as labels:
                     for foldername in img_foldernames: # ������ ���� resize and save data,map,mean by folder
                        labels.write(foldername+'\n')
                        abs_in_foldername = os.path.join(in_dataset_dir,foldername)
    #                    abs_out_foldername = os.path.join(out_dataset_dir,foldername)
                        abs_out_foldername = os.path.join(out_dataset_dir,'train_db')
   
                        if not os.path.exists(abs_out_foldername):
                            os.makedirs(abs_out_foldername)

                        imgnames = get_imgnames(abs_in_foldername)
        
                        print ('Saving raw_images to resized_images in {}'.format(foldername))
                        for in_imgname in imgnames: # ���� ���� �ִ� �̹����� ����
                            pixindex=0
                            abs_in_imgname = os.path.join(abs_in_foldername,in_imgname) # ��ȯ�� �̹��� ������

                            extension = ['.jpg','.png','.jpen','.bmp']
                            if os.path.splitext(abs_in_imgname)[1] in extension:
                                out_imgname = '{:0{}d}.png'.format(current_num_img,int(log10(total_img_num)+1))                                                            # abs_out_image�� �ٲ������
                                abs_out_imgname = os.path.join(abs_out_foldername,out_imgname) # ��ȯ�� �̹��� ������
    #                            print(abs_out_imgname)
                                resize_to_PNGimg(abs_in_imgname,abs_out_imgname,row,col) # �̹��� resize�� .png�� ����
                            
                                text.write('|labels ') # data.txt �������
                                for num in range(labelnum):
                                    if num == label:
                                        text.write('1 ')
                                    else:
                                        text.write('0 ')

    #                            print(abs_in_imgname)
                                text.write('|features ')
                                img = Image.open(abs_out_imgname)
    #                            print(img.size)
                                pix = img.load()
                                # rgb�� �ƴ϶� grayscale�̸� img.convert()���� pixel�� �ٸ��� �޾ƿ����
                                for rgb in range(3): # rgb * row*col
                                    for y in range(col): # y * col
                                        for x in range(row): # x
    #                                        print(pix[x,y][rgb])
                                            text.write(str(pix[x,y][rgb])+' ')
                                            pixel = (pix[x,y][rgb])/total_img_num

                                            if len(pixels) < 3*row*col:
                                                pixels.append(pixel) # pixels�� ������� ������ �ȼ��� ����
                                            else:
                                                pixels[pixindex] += (pix[x,y][rgb])/total_img_num

                                            pixindex = pixindex + 1

                                text.write('\n')
                                current_num_img += 1
                                map.write(abs_out_imgname+'\t'+str(label)+'\n')

                        label+=1
                        print('Done\n')

        for i in range(3*row*col):
#            pixels[i] = '%e'%pixels[i]
            if (i+1) == 3*row*col:
                mean.write('%e'%pixels[i])
            else:       
                mean.write('%e'%pixels[i]+' ')
        mean.write('''</data>
  </MeanImg>
</opencv_storage>
''')

def create_dataset(in_dataset_dir, out_dataset_dir, row, col, framework):
    global check
    # in_dataset_dir�� ����x ����
    if not os.path.exists(in_dataset_dir):
        return print('Dataset directory is Wrong.')
    
    # in_dataset_dir�� ������ ���� x
    if not os.listdir(in_dataset_dir):
        return print('Dataset is not found.')
    
    if not os.path.exists(out_dataset_dir):
        print("out dataset directory is not found. We make it")
        os.makedirs(out_dataset_dir)
    
    resizing(in_dataset_dir,out_dataset_dir,row,col)
    check = 1
    return True

def print_dataset_log():

    global total_img_num
    global current_num_img
    global check
    
    logger = logging.getLogger('Dataset')
    logger.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler(os.path.join(out_dataset_dir,'creat_val_db.log'),'w')
    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s], %(message)s')
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    message = 'Log :: Total=%d, Current=%d, Progress=%0.4f%%'%(total_img_num,current_num_img,(current_num_img*100/total_img_num))
    
    while check == 0:
        #print(message)
        logger.info(message)
        time.sleep(1)

########################################################################################################################################################################################################
##############################################################################      Dataset.py       ###################################################################################################
########################################################################################################################################################################################################

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

    return found_dataset
   
def Dataset_create(in_dataset_dir, out_dataset_dir, row, col, framework):
    if framework == 3: # CNTK
        log = threading.Thread(target = print_dataset_log)
        dataset = threading.Thread(target = create_dataset, args=(in_dataset_dir,out_dataset_dir,row,col,framework))

        dataset.start()    
        time.sleep(0.2)
        log.start()
    return print('Dataset creating finished')


my_dataset_dir = r'D:\Github\dataset\img\mydataset'
CIFAR_10 = r'D:\Github\dataset\img\cifar10'

out_dataset_dir = r'D:\Github\dataset\img\mydataset\out_dataset'
    
if __name__ == '__main__':
    #Dataset_create(CIFAR_10,out_dataset_dir,32,32,3)
    Dataset_create(my_dataset_dir,out_dataset_dir,32,32,3)
    result=Dataset_result(out_dataset_dir)
