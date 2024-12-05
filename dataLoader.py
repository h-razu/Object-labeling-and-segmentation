import shutil
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import prettyprinter as pp
from tqdm.auto import tqdm



class DataLoader:
    def __init__(self, label_path, images_path, object_classes_path):
        self.label_path = Path(label_path)
        self.images_path = Path(images_path)
        self.object_classess_path = Path(object_classes_path)


    def make_validation_set(self):
        #make pair of (train_image, train_label)
        train_imgages = sorted(list(self.images_path.glob('*')))
        train_labels = sorted(list(self.label_path.glob('*')))
        train_data = list(zip(train_imgages, train_labels))

        #split the train dataset into train and val set
        print("Spliting the train dataset into train and val set")
        train, val = train_test_split(train_data,test_size=0.2,shuffle=True)
        print(f'Train data size: {len(train)}')
        print(f'Val data size: {len(val)}')
        print()

        #make new data directory for processed data
        self.train_path = Path('dataset/modified/train').resolve()
        self.train_path.mkdir(exist_ok=True)
        self.valid_path = Path('dataset/modified/valid').resolve()
        self.valid_path.mkdir(exist_ok=True)

        #copy train image to processed data directory
        pp.pprint('Copying train image to processed data directory')
        for t_image, t_label in tqdm(train):
            image_path = self.train_path / t_image.name
            label_path = self.train_path / t_label.name
            shutil.copy(t_image,image_path)
            shutil.copy(t_label,label_path)

        #copy val image to processed data directory
        pp.pprint('Copying validate image to processed data directory')
        for v_image, v_label in tqdm(val):
            image_path = self.valid_path / v_image.name
            label_path = self.valid_path / v_label.name
            shutil.copy(v_image,image_path)
            shutil.copy(v_label,label_path)



    def make_yaml_file(self):
        with open(self.object_classess_path,'r') as f:
            classes = json.load(f)

        yaml_file = 'names:\n'
        yaml_file += '\n'.join(f'- {c}' for c in classes)
        yaml_file += f'\nnc: {len(classes)}'
        yaml_file += f'\ntrain: {str(self.train_path)}\nval: {str(self.valid_path)}'
        with open('dataset/data.yaml','w') as f:
            f.write(yaml_file)
