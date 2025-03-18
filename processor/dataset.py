import random
import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import logging
from collections import Counter


logger = logging.getLogger(__name__)



class MSDProcessor(object):
    def __init__(self, data_path, bert_name, clip_processor):
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.clip_processor = clip_processor

    def load_from_file(self, mode="train"):

        logger.info("Loading data from {}".format(self.data_path[mode]))

        with open(self.data_path[mode], "r", encoding="utf-8") as f:
            dataset = json.load(f)  # 加载整个数据集

            raw_texts, raw_captions, raw_labels, imgs = [], [], [], []

            for index in range(0, len(dataset)):  # 一条一条地读取数据
                sample = dataset[index]

                img_id = sample[0] + '.jpg'
                text = sample[1]
                # for val and test dataset, the sample[2] is hashtag label
                if mode == "train":
                    label = sample[2]
                    # text = sample[3]
                else:
                    # label =sample[2] hashtag label
                    label = sample[3]
                    # text = sample[4]
                caption = sample[-3]
                # 将所有数据分别放到对应的列表中
                raw_texts.append(text)
                raw_labels.append(label)
                raw_captions.append(caption)
                imgs.append(img_id)

        assert len(raw_texts) == len(raw_labels) == len(raw_captions) == len(imgs), "{}, {}, {}, {}".format(
            len(raw_texts), len(raw_labels), len(raw_captions), len(imgs))

        return {"texts": raw_texts, "labels": raw_labels, "captions": raw_captions, "imgs": imgs}


class MSDDataset(Dataset):
    def __init__(self, processor, img_path, max_seq=128, mode="train"):
        self.processor = processor
        self.img_path = img_path
        # 分词器
        self.tokenizer = self.processor.tokenizer
        self.data_dict = self.processor.load_from_file(mode)
        self.clip_processor = self.processor.clip_processor
        self.max_seq = max_seq


    def __len__(self):
        return len(self.data_dict['texts'])

    def __getitem__(self, idx):
        text, label, caption, img = self.data_dict['texts'][idx], self.data_dict['labels'][idx], \
                                    self.data_dict['captions'][idx], self.data_dict['imgs'][idx]

        tokens_text = self.tokenizer.tokenize(text)
        tokens_caption = self.tokenizer.tokenize(caption)

        if len(tokens_text) > self.max_seq - 2:
            tokens_text = tokens_text[:self.max_seq - 2]

        tokens = ["[CLS]"] + tokens_text + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_seq - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq
        assert len(input_mask) == self.max_seq
        assert len(segment_ids) == self.max_seq

        if len(tokens_caption) > self.max_seq - 2:
            tokens_caption = tokens_caption[:self.max_seq - 2]

        cap_tokens = ["[CLS]"] + tokens_caption + ["[SEP]"]
        cap_input_ids = self.tokenizer.convert_tokens_to_ids(cap_tokens)
        cap_input_mask = [1] * len(cap_tokens)
        cap_segment_ids = [0] * len(cap_tokens)

        padding = [0] * (self.max_seq - len(cap_input_ids))
        cap_input_ids += padding
        cap_input_mask += padding
        cap_segment_ids += padding

        assert len(cap_input_ids) == self.max_seq
        assert len(cap_input_mask) == self.max_seq
        assert len(cap_segment_ids) == self.max_seq

        # 对图片进行处理
        if self.img_path is not None:
            # image process
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()

        img_mask = [1] * 50

        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), \
               torch.tensor(cap_input_ids), torch.tensor(cap_input_mask), torch.tensor(cap_segment_ids), \
               torch.tensor(img_mask), torch.tensor(label), image

