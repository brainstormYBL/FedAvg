"""
Author   : Bao-lin Yin
Data     : 10.23 2023
Version  : V1.0
Function : Train the model by the FedAvg
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import visdom

from Client.Client import Client
from Server.Server import Server
from Utils.Parameters import parameters
from Utils.ProcessData import ProcessData
import torch

if __name__ == "__main__":
    # print("Training the model")
    # 非联邦学习过程 #
    # path_data_set = r"/Volumes/科研/4.AI/2.联邦学习/2.Project/1.FedAvg/Data/heart.csv"
    # 预处理过程 #
    # 1. 准备/下载数据集, 划分数据集，包括将所有的数据划分给Client，对每一个Client划分测试集与训练集
    # data_holder = ProcessData(path_data_set, 0.8, 4)
    # # 非联邦学习学习过程 #
    # # 1. 定义模型
    # model = LinearRegression(dim_input=data_holder.dim_feature, dim_output=data_holder.dim_label)
    # # 2. 定义损失函数
    # loss_func = torch.nn.MSELoss()
    # # 3. 定义优化器
    # lr = 1e-5
    # optimizer = torch.optim.Adam(model.parameters(), lr)
    # # 3 训练过程
    # echo_nfl = 3000
    # bath_size_nfl = 256
    # loss_dis = np.zeros(echo_nfl)
    # for index_echo in range(echo_nfl):
    #     # 3.1 选择bath_size个数据用于训练
    #     idx_train = np.array(np.random.choice(np.linspace(0, data_holder.num_data - 1, data_holder.num_data),
    #                                           bath_size_nfl), dtype=int)
    #     feature_train_current = data_holder.all_feature_tensor[idx_train]
    #     label_train_current = data_holder.all_feature_tensor[idx_train]
    #     # 3.2 预测结果
    #     y_pre = model(feature_train_current)
    #     # 3.3 计算损失
    #     loss_value = loss_func(y_pre, label_train_current)
    #     loss_dis[index_echo] = loss_value.item()
    #     # 3.4 梯度清零
    #     optimizer.zero_grad()
    #     # 3.5 反向传播
    #     loss_value.backward()
    #     # 3.6 更新参数
    #     optimizer.step()
    #     print("第" + str(index_echo) + "个训练回合的loss值为" + str(loss_value.item()) + "\n")
    # plt.figure()
    # plt.plot(loss_dis)
    # plt.show()

    # 联邦学习过程 #
    # 1. 初始化 参数 Server Client 在线打印窗口
    par = parameters()
    viz = None
    if par.visdom:
        viz = visdom.Visdom()
        viz.close()
    path_data_set = r"/Volumes/科研/4.AI/2.联邦学习/2.Project/2.FedAvgV2.0/Data/heart.csv"
    data_holder = ProcessData(path_data_set, par.ratio_train, par.num_client)
    client = Client(par)
    server = Server(par.model_name, data_holder.dim_feature, data_holder.dim_label, par)
    # 2. 进行多次的FL
    loss_dis = np.zeros(par.epochs)
    for index_fl in range(par.epochs):
        print("-------------开始第 " + str(index_fl + 1) + " 个epoch的训练-------------\n")
        # 2.1 本地模型获取最新的全局模型参数
        w_local_newest = [server.get_parameters_global_model() for index in range(server.num_client_selected)]
        # 2.2 针对每个Client并行执行1-5个Echos，并进行梯度下降，更新参数，并返回各自最新的参数
        loss_value = np.zeros(server.num_client_selected)
        for index_client in range(server.num_client_selected):
            # 使用copy.deepcopy创建了一个同参数的模型副本 copy.deepcopy是深度拷贝 与原始对象相互独立
            w_local_newest[index_client], loss_value[index_client] = client.train_local(
                data_holder.train_feature_tensor["client" + str(server.idx_client_selected[index_client])],
                data_holder.train_label_tensor["client" + str(server.idx_client_selected[index_client])],
                copy.deepcopy(server.global_model))
        loss_dis[index_fl] = sum(loss_value) / len(loss_value)
        if par.visdom:
            viz.line(X=[index_fl + 1], Y=[loss_dis[index_fl]], win='loss', opts={'title': 'Training Loss'},
                     update='append')
        # 2.3 根据本地参数进行全局模型参数的聚合 聚合方式很多 这里采用平均
        w_global_newest = server.calculate_newest_parameters_global_model(w_local_newest)
        # 2.4 更新全局模型参数
        server.load_parameters_to_global_model(w_global_newest)
        print("-------------第 " + str(index_fl + 1) + " 个epoch训练结束，Loss值为" + str(
            loss_dis[index_fl]) + "-------------\n")
    # 3. 画图 LOSS
    plt.figure()
    plt.plot(loss_dis)
    plt.show()
    # 4.保存模型参数
    torch.save(server.global_model.state_dict(), 'model.pth')
