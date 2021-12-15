import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from dataPreprocChest import Vocabulary, JsonReader
import numpy as np
from torchvision import transforms
import pickle


class ChestXrayDataSet(Dataset):
    def __init__(self,
                 image_dir,
                 caption_json,
                 file_list,
                 vocabulary,
                 transforms=None):
        self.image_dir = image_dir
        self.caption = JsonReader(caption_json)
        self.file_names, self.labels = self.__load_label_list(file_list)
        self.vocab = vocabulary
        self.transform = transforms


    def __load_label_list(self, file_list):
        labels = []
        filename_list = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[:2]
                label = items[2:]
                label = [int(i) for i in label]
                for i in range(len(image_name)):
                    image_name[i] = '{}.png'.format(image_name[i])
                    # # mimic-cxr
                    # image_name[i] = '{}.jpg'.format(image_name[i])
                filename_list.append(image_name)
                labels.append(label)
        return filename_list, labels

    def __getitem__(self, index):
        image_name = self.file_names[index]
        img_front = Image.open(os.path.join(self.image_dir, image_name[0]).replace('\\','/')).convert('RGB')
        img_later = Image.open(os.path.join(self.image_dir, image_name[1]).replace('\\','/')).convert('RGB')


        label = self.labels[index]
        if self.transform is not None:
            img_front = self.transform(img_front)
            img_later = self.transform(img_later)

        try:
            text = self.caption[image_name[0]]
        except Exception as err:
            text = 'normal. '
        try:
            text = self.caption[image_name[1]]
        except Exception as err:
            text = 'normal. '

        target = list()

        return img_front, img_later, image_name, label, target

    def __len__(self):
        return len(self.file_names)


class GastroDataSet(Dataset):
    def __init__(self,
                 image_dir_Gastro,
                 caption_json,
                 file_list,
                 vocabulary,
                 transforms=None):
        self.image_dir = image_dir_Gastro
        self.caption = JsonReader(caption_json)
        self.file_names, self.labels = self.__load_label_list(file_list)
        self.vocab = vocabulary
        self.transform = transforms


    def __load_label_list(self, file_list):
        labels = []
        filename_list = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[:2]
                label = items[2:]
                label = [int(i) for i in label]
                for i in range(len(image_name)):
                    image_name[i] = '{}.jpg'.format(image_name[i])
                filename_list.append(image_name)
                labels.append(label)
        return filename_list, labels

    def __getitem__(self, index):
        image_name = self.file_names[index]
        img_first = Image.open(os.path.join(self.image_dir, image_name[0]).replace('\\','/')).convert('RGB')
        img_second = Image.open(os.path.join(self.image_dir, image_name[1]).replace('\\','/')).convert('RGB')
        img_third = Image.open(os.path.join(self.image_dir, image_name[2]).replace('\\','/')).convert('RGB')
        img_fourth = Image.open(os.path.join(self.image_dir, image_name[3]).replace('\\','/')).convert('RGB')
        img_fifth = Image.open(os.path.join(self.image_dir, image_name[4]).replace('\\','/')).convert('RGB')



        label = self.labels[index]
        if self.transform is not None:
            img_first = self.transform(img_first)
            img_second = self.transform(img_second)
            img_third = self.transform(img_third)
            img_fourth = self.transform(img_fourth)
            img_fifth = self.transform(img_fifth)

        # try:
        #     text = self.caption[image_name[0]]
        # except Exception as err:
        #     text = 'normal. '
        # try:
        #     text = self.caption[image_name[1]]
        # except Exception as err:
        #     text = 'normal. '

        return img_first, img_second, img_third, img_fourth, img_fifth, image_name, label, target

    def __len__(self):
        return len(self.file_names)


def collate_fn(data):
    img_front, img_later, image_id, label, captions, sentence_num, max_word_num = zip(*data)

    img_front = torch.stack(img_front, 0)
    img_later = torch.stack(img_later, 0)
    # images = torch.stack(images, 0)

    max_sentence_num = max(sentence_num)
    max_word_num = max(max_word_num)

    targets = np.zeros((len(captions), max_sentence_num + 1, max_word_num))
    prob = np.zeros((len(captions), max_sentence_num + 1))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0

    return img_front, img_later, image_id, torch.Tensor(label), targets, prob


def get_loader(image_dir,
               caption_json,
               file_list,
               vocabulary,
               transform,
               batch_size,
               s_max=10,
               n_max=50,
               shuffle=False):
    dataset = ChestXrayDataSet(image_dir=image_dir,
                               caption_json=caption_json,
                               file_list=file_list,
                               vocabulary=vocabulary,
                               transforms=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    vocab_path = '../data/vocab.pkl'
    image_dir = '../data/images'
    image_dir_Gastro = '../data/image_Gastro'
    caption_json = '../data/debugging_captions.json'
    file_list = '../data/debugging.txt'
    batch_size = 6
    resize = 256
    crop_size = 224

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(image_dir=image_dir,
                             caption_json=caption_json,
                             file_list=file_list,
                             vocabulary=vocab,
                             transform=transform,
                             batch_size=batch_size,
                             shuffle=False)

    for i, (image, image_id, label, target, prob) in enumerate(data_loader):
        print(image.shape)
        print(image_id)
        print(label)
        print(target)
        print(prob)
        break
