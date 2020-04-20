from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import cv2
import numpy as np

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    # 参数定义
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    # parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    # print(opt)
    print("*%"*100)

    #日志初始化
    logger = Logger("logs")   # ？？？？？？？？？？？

    # 处理器选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出路径创建
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    #参数解析
    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    # print("data_config:")
    # print(data_config)

    # coco2017数据路径定义（训练集、验证集、种类名称）
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # 初始化Yolov3模型
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    # print("Darknet_model:")
    # print(model)

    
    # 初始化模型权重
    model.apply(weights_init_normal)

    # 加载预训练模型
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    # dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)

    #建立coco2017训练集的dataset类，并以此对dataloader进行定义
    # for win10 coco2017
    train_path_2017 = "E:\\dataset_work\\coco\\train_file_list.txt"
    val_path_2017 = "E:\\dataset_work\\coco\\val_file_list.txt"
    dataset_2017 = ListDataset_win_cc2017(train_path_2017, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        # dataset,
        dataset_2017, 
        batch_size=opt.batch_size,
        shuffle=True,
        # num_workers=opt.n_cpu,
        num_workers=0,
        pin_memory=True,
        # collate_fn=dataset.collate_fn,
        collate_fn=dataset_2017.collate_fn,
    )

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters())

    # 设置评价指标
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # 训练部分

    # 每一轮训练
    # for epoch in range(opt.epochs):
    for epoch in range(1):

        # 切换模型到训练状态
        model.train()

        # 记录时间
        start_time = time.time()

        # 每一批
        for batch_i, (_, imgs, targets) in enumerate(dataloader):

            # 总批数
            batches_done = len(dataloader) * epoch + batch_i
            print(epoch, batch_i)

            # 从dataloader中读出图片和标签
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # print("output: img ", imgs.shape)
            # print("output: targets", targets.shape)
            # print("~"*60)
            if batch_i == 2:
                break

            # 前馈，得到损失和输出
            loss, outputs = model(imgs, targets)

            #下方两步肯定是看是求导以及参数更新相关
            # ？？？？？？？？
            loss.backward()

            # ？？？？？
            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            #计算各评价指标
            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        # # 若干个循环训练之后，对模型进行评价
        # if epoch % opt.evaluation_interval == 0:
        #     print("\n---- Evaluating Model ----")
        #     # Evaluate the model on the validation set
        #     precision, recall, AP, f1, ap_class = evaluate(
        #         model,
        #         path=val_path_2017,
        #         iou_thres=0.5,
        #         conf_thres=0.5,
        #         nms_thres=0.5,
        #         img_size=opt.img_size,
        #         batch_size=1,
        #     )
        #     evaluation_metrics = [
        #         ("val_precision", precision.mean()),
        #         ("val_recall", recall.mean()),
        #         ("val_mAP", AP.mean()),
        #         ("val_f1", f1.mean()),
        #     ]
        #     logger.list_of_scalars_summary(evaluation_metrics, epoch)

        #     # Print class APs and mAP
        #     ap_table = [["Index", "Class name", "AP"]]
        #     for i, c in enumerate(ap_class):
        #         ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        #     print(AsciiTable(ap_table).table)
        #     print(f"---- mAP {AP.mean()}")

        # # 保存模型
        # if epoch % opt.checkpoint_interval == 0:
        #     torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
