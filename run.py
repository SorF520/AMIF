import os
import argparse
import logging

import torch
import numpy as np
import random
from time import strftime, localtime

from torchvision import transforms

from models.unimo_model import UnimoModelF
from processor.dataset import MSDProcessor, MSDDataset
from modules.train import MSDTrainer
from torch.utils.data import DataLoader

from transformers import BertConfig, CLIPConfig, BertModel, CLIPProcessor, CLIPModel

import fitlog
import warnings


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=2023):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model path")
    parser.add_argument("--vit_name", default="openai/clip-vit-base-patch32", type=str, help="The name of vit")
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=3e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=2023, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default='./output/', type=str, help="save best model at save_path")
    parser.add_argument('--write_path', default=None, type=str,
                        help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")

    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--ignore_idx', default=0, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")

    parser.add_argument('--alpha', default=0.6, type=float, help="CCR")
    parser.add_argument('--margin', default=0.5, type=float, help="CCR")

    parser.add_argument('--beta', default=0, type=float, help="SoftContrastiveLoss")
    parser.add_argument('--mild_margin', default=0.7, type=float, help="SoftContrastiveLoss")
    parser.add_argument('--hetero', default=0.9, type=float, help="SoftContrastiveLoss")
    parser.add_argument('--homo', default=0.9, type=float, help="SoftContrastiveLoss")

    parser.add_argument('--SGR_step', default=3, type=int, help="SGR module steps")
    parser.add_argument('--weight_js', default=0.1, type=float, help="JS divergence")

    args = parser.parse_args()

    data_path = {
        'train': 'twitter/dataset_text/trainknow.json',
        'dev': 'twitter/dataset_text/valknow.json',
        'test': 'twitter/dataset_text/testknow.json'
    }

    img_path = 'twitter/dataset_image'

    data_process, dataset_class = (MSDProcessor, MSDDataset)

    set_seed(args.seed)
    if args.save_path is not None:  # make save_path dir
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)

    # if not os.path.exists('./log'):
    #     os.makedirs('./log', mode=0o777)
    # log_file = '{}-{}-{}.log'.format(args.bert_name, "sarcasm", strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    # logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    logger.info(args)

    writer = None
    if args.do_train:
        clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
        clip_model = CLIPModel.from_pretrained(args.vit_name)
        clip_vit = clip_model.vision_model

        processor = data_process(data_path, args.bert_name, clip_processor=clip_processor)

        train_dataset = MSDDataset(processor, img_path=img_path, max_seq=args.max_seq, mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16,
                                      pin_memory=True)

        dev_dataset = MSDDataset(processor, img_path=img_path, max_seq=args.max_seq, mode="dev")
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                    pin_memory=True)

        test_dataset = MSDDataset(processor, img_path=img_path, max_seq=args.max_seq, mode="test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                     pin_memory=True)

        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        text_config = BertConfig.from_pretrained(args.bert_name)

        model = UnimoModelF(args=args, vision_config=vision_config, text_config=text_config)

        trainer = MSDTrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader,
                              model=model, args=args, logger=logger, writer=writer)

        bert = BertModel.from_pretrained(args.bert_name)
        clip_model_dict = clip_vit.state_dict()
        text_model_dict = bert.state_dict()
        trainer.train(clip_model_dict, text_model_dict)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
