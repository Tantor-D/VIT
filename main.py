import os.path
from pprint import pprint
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
import torch
import torch.optim as optim
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import PIL
import matplotlib.pyplot as plt


def eval(model, dataloader):
    model.eval()  # 设置网络为测试状态，需注意的话其仅对部分层有作用，如：dropout，batchnorm
    total_test_loss = 0  # 累加loss
    total_accu_num = 0
    test_data_size = 0
    with torch.no_grad():
        for data in dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accu_num = (outputs.argmax(1) == targets).sum()
            total_accu_num = total_accu_num + accu_num
            test_data_size = test_data_size + imgs.shape[0]

    print(f"  测试集上的Loss: {total_test_loss}")
    print(f"  测试集上的正确率: {(total_accu_num / test_data_size * 100):.3f}%")
    print("  测试集上的错误个数: {}".format(test_data_size - total_accu_num))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)


def getDataSet(dataset_path):
    # TODO 需要优化裁剪方式，目前的方式会裁处黑边，有些图片大小不一样
    transform = transforms.Compose([
        transforms.CenterCrop(size=522),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return dataset


def get_dict_folder_index():
    for i in range(1000):
        if not os.path.exists(f"./model_dict/dict{i}"):
            return i
    return 1001


if __name__ == '__main__':
    model_name = 'vit_base_patch16_224'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    need_save_model = False

    dict_folder_index = get_dict_folder_index()
    dict_path = f"./model_dict/dict{dict_folder_index}"
    os.mkdir(dict_path)

    writer = SummaryWriter(f"./logs/writer_{dict_folder_index}")

    # prepare dataset
    train_dataset = getDataSet("./warwick_CLS/train")
    test_dataset = getDataSet("./warwick_CLS/test")

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    # out = model(torch.randn(2, 3, 224, 224))
    # print(out.shape)

    # img, t = train_dataset[0]
    # plt.imshow(img.permute(1, 2, 0).numpy())
    # plt.show()
    # pprint(len(train_dataset))

    # prepare to train
    model = timm.create_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # hyper parameter
    lr = 1e-4
    Batch_size = 4
    total_epoch = 50

    optimizer = optim.Adam(model.parameters(), lr)

    # start to train
    now_train_step = 0
    for now_epoch in range(total_epoch):
        epoch_loss_sum = 0
        print("-------第 {} 轮训练开始-------".format(now_epoch + 1))

        model.train()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum = epoch_loss_sum + loss.item()
            writer.add_scalar("train_step_loss", loss.item(), now_train_step)
            now_train_step = now_train_step + 1

            if now_train_step % 10 == 0:  # 每训练10次输出一次
                print("训练次数：{}, Loss: {}".format(now_train_step, loss.item()))

        print(f"第{now_epoch + 1}个epoch，loss之和为{epoch_loss_sum}")
        writer.add_scalar("train_epoch_loss", epoch_loss_sum, now_epoch)

        # 最后一个epoch的模型无论如何都要存，中途存不存在need_save_model
        if (now_epoch > 25 and need_save_model) or now_epoch == total_epoch:
            torch.save(model.state_dict(), dict_path + f"/model_epoch{now_epoch}_loss{epoch_loss_sum:.4f}.pth")

        eval(model, test_dataloader)

    print("-----现在进行value集上的测试-----")
    val_dataset = getDataSet("./warwick_CLS/val")
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    eval(model, val_dataloader)
