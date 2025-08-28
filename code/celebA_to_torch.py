import time
import torch
from dataloader_func import load_CelebA_dataset, prep_celeba
from quality_metrics_func import remove_repeats_block


def main():
    start_time_total = time.time()
    dir_path = '../datasets/'
    train_folder_path = dir_path + 'train/'
    test_folder_path = dir_path + 'test/'
    train_list_path = dir_path + 'train_filenames.txt'
    test_list_path = dir_path + 'test_filenames.txt'

    all_ims = load_CelebA_dataset(
        train_folder_path, test_folder_path, s=.25, 
        train_list_path=train_list_path, test_list_path=test_list_path)
    train_set, test_set = prep_celeba(all_ims)
        
    print('train: ', train_set.shape)
    print('test: ', test_set.shape)

    # remove repeated images 
    data_cleaned = remove_repeats_block(train_set, threshold=.95)
    print('shape after removed repeats:' , data_cleaned.shape)
    
    torch.save(data_cleaned, dir_path + '/train40x40_no_repeats.pt')
    torch.save(test_set, dir_path + '/test40x40.pt')
        
    print("--- %s seconds ---" % (time.time() - start_time_total))
    

if __name__ == "__main__" :
    main()    
    
