import os

def test_file_count():
  BASE_DATA_PATH = '/content/gdrive/MyDrive/fourthbrain-capstone/data/oct/OCT2017/'
  TRAIN_BASE_PATH = os.path.join(BASE_DATA_PATH, 'train/')
  TEST_BASE_PATH = os.path.join(BASE_DATA_PATH, 'test/')

  file_count_dict = {'train/CNV':37205, 'train/DME': 11348, 'train/DRUSEN':8616, 'train/NORMAL': 26315, 
                     'test/CNV': 250, 'test/DME': 250, 'test/DRUSEN': 250, 'test/NORMAL': 250}


  parents = ['train', 'test']
  children = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

  # create a base dataframe from the directory items

  for p in parents:
      for c in children:
          files = os.listdir(os.path.join(BASE_DATA_PATH, p, c))
          instance_type = [ p for i in range(len(files)) ]
          path = [ f'/OCT2017/{p}/{c}' for i in range(len(files))]
          assert file_count_dict[p+'/'+c]==len(files), '{} should have {} files but it has {} files'.format(p+'/'+c, file_count_dict[p+'/'+c], len(files))
  print('file counts are correct!')

