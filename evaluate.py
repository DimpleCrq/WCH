import os
import numpy as np
import torch
from tqdm import tqdm

# 预定义数据集范围
cifar_d_range = [i for i in range(0, 10000, 500)] + [i for i in range(10000, 50001, 2500)]
coco_d_range = [i for i in range(0, 20000, 1000)] + [i for i in range(20000, 100001, 5000)] + [105000, 110000, 115000, 117218]
nuswide_10_d_range = [i for i in range(0, 30000, 1500)] + [i for i in range(40000, 175001, 7500)] + [180000, 181577]
nuswide_21_d_range = [i for i in range(0, 30000, 1500)] + [i for i in range(40000, 175001, 7500)] + [180000, 187500, 193734]
flickr_d_range = [i for i in range(0, 5000, 250)] + [i for i in range(5000, 22501, 1250)] + [23000]

# 调整初始值
cifar_d_range[0] = 1
coco_d_range[0] = 1
nuswide_10_d_range[0] = 1
nuswide_21_d_range[0] = 1
flickr_d_range[0] = 1


def compute_result(data_loader, net):
    """
    计算给定数据加载器和网络的二进制代码和标签。
    """
    binary_list, class_list = [], []
    net.eval()
    for img, cls, _ in tqdm(data_loader):
        class_list.append(cls)
        binary_list.append((net(img.cuda())).data.cpu())
    return torch.cat(binary_list).sign(), torch.cat(class_list)


def compute_hamming_distance(B1, B2):
    """
    计算两个二进制矩阵 B1 和 B2 之间的汉明距离。
    """
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def compute_topk_map(rB, qB, retrievalL, queryL, topk):
    """
    计算检索集和查询集之间的Top-k平均精度（mAP）。
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        ground_truth = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = compute_hamming_distance(qB[iter, :], rB)
        ind = np.argsort(hamm)
        ground_truth = ground_truth[ind]

        topk_ground_truth = ground_truth[0:topk]
        true_sum = np.sum(topk_ground_truth).astype(int)
        if true_sum == 0:
            continue
        count = np.linspace(1, true_sum, true_sum)

        true_index = np.asarray(np.where(topk_ground_truth == 1)) + 1.0
        topkmap_ = np.mean(count / (true_index))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def pr_curve(retrievalF, queryF, retrievalL, queryL, draw_range):
    """
    计算Precision-Recall (PR) 曲线数据。
    """
    num_query = queryF.shape[0]
    ground_truth = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)
    rank = np.argsort(compute_hamming_distance(queryF, retrievalF))
    precision_list, recall_list = [], []
    for k in tqdm(draw_range):
        precision = np.zeros(num_query)
        recall = np.zeros(num_query)
        for it in range(num_query):
            gnd = ground_truth[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            precision[it] = gnd_r / k
            recall[it] = gnd_r / gnd_all
        precision_list.append(np.mean(precision))
        recall_list.append(np.mean(recall))
    return precision_list, recall_list


def evalModel(test_loader, dataset_loader, net, best_mAP, bit, config, epoch, f):
    """
    计算mAP，并保存结果。
    """
    print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net)

    print("calculating dataset binary code.......")
    trn_binary, trn_label = compute_result(dataset_loader, net)

    mAP = compute_topk_map(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])

    if mAP > best_mAP:
        best_mAP = mAP

        if "logs_path" in config:
            if not os.path.exists(config["logs_path"]):
                os.makedirs(config["logs_path"])

            print("save in ", config["logs_path"])

            dataset_name = config["dataset"]
            bit_str = "%d" % bit
            mAP_str = "%.5f" % round(mAP, 5)

            np.save(os.path.join(config["logs_path"], f"{dataset_name}-{bit_str}-{mAP_str}-tst_label.npy"), tst_label.numpy())
            np.save(os.path.join(config["logs_path"], f"{dataset_name}-{bit_str}-{mAP_str}-tst_binary.npy"), tst_binary.numpy())
            np.save(os.path.join(config["logs_path"], f"{dataset_name}-{bit_str}-{mAP_str}-trn_binary.npy"), trn_binary.numpy())
            np.save(os.path.join(config["logs_path"], f"{dataset_name}-{bit_str}-{mAP_str}-trn_label.npy"), trn_label.numpy())
            torch.save(net.state_dict(), os.path.join(config["logs_path"], f"{dataset_name}-{bit_str}-{mAP_str}-model.pth"))

            for file in os.listdir(config['logs_path']):
                if f"{dataset_name}-{bit_str}-" in file:
                    os.remove(os.path.join(config['logs_path'], file))

            if "cifar" in dataset_name:
                precision, recall = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), cifar_d_range)
            elif "coco" in dataset_name:
                precision, recall = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), coco_d_range)
            elif "nuswide_10" in dataset_name:
                precision, recall = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), nuswide_10_d_range)
            elif "nuswide_21" in dataset_name:
                precision, recall = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), nuswide_21_d_range)
            elif "flickr" in dataset_name:
                precision, recall = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), flickr_d_range)

            print(f'Precision Recall Curve data:')
            print(f'"{config["info"]}":[{precision},{recall}],')
            f.write('PR | Epoch %d | ' % (epoch))
            f.write(f'[{precision}, {recall}]')
            f.write('\n')

    print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.5f, Best MAP: %.5f" % (config["info"], epoch, bit, config["dataset"], mAP, best_mAP))
    f.write('Test | Epoch %d | MAP: %.5f | Best MAP: %.5f\n'% (epoch, mAP, best_mAP))

    return best_mAP
