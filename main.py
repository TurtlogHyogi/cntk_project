import Dataset
import Train

w_my_dataset_dir = Dataset.w_my_dataset_dir
l_my_dataset_dir = Dataset.l_my_dataset_dir
w_out_dataset_dir = Dataset.w_out_dataset_dir
l_out_dataset_dir = Dataset.l_out_dataset_dir
w_out_model_dir = Train.w_out_model_dir
l_out_model_dir = Train.l_out_model_dir

if __name__ == '__main__':
    Dataset.Dataset_create(in_dataset_dir = l_my_dataset_dir,
                           out_dataset_dir = l_out_dataset_dir,
                           resize = 32,
                           framework=3)
    print(Dataset.Dataset_result(l_out_dataset_dir))
    Train.Train_create(dataset_dir = l_out_dataset_dir,
                       framework = 3, 
                       out_model_dir = l_out_model_dir, 
                       max_epochs = 5, 
                       mb_size = 64, 
                       network_name = 'convnet')
    print(Train.Train_result(l_out_model_dir))

    
