import os
import datetime
import time
import torch
from torch.utils.data import DataLoader

from model import MyNet
from data import MyDataset, DataAugmentation
from my_arguments import Args


class TrainInterface(object):
    def __init__(self, opts):
        self.opts = opts

    @staticmethod
    def __train(model, train_loader, optimizer, epoch, num_train, opts):
        model.train()
        device = opts.GPU_id
        avg_metric = 0.
        avg_loss = 0.
        log_file = open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+")
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(localtime)
        log_file.write("\n======================training epoch %d======================\n"%epoch)
        for i,(imgs, labels) in enumerate(train_loader):
            if opts.use_GPU:
                imgs = imgs.to(device)
                labels = labels.to(device)
            preds = model(imgs)
            loss = model.calculate_loss(labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metric = model.calculate_metric(preds, labels)
            avg_metric = (avg_metric*i+metric)/(i+1)
            avg_loss = (avg_loss*i+loss.item())/(i+1)
            if i%opts.print_freq==0:
                print("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f | metric = %.2f, avg_metric = %.2f" %
                      (epoch, opts.epoch, i, num_train//opts.batch_size, loss.item(), avg_loss, metric, avg_metric))
                log_file.write("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f | metric = %.2f, avg_metric = %.2f\n" %
                      (epoch, opts.epoch, i, num_train//opts.batch_size, loss.item(), avg_loss, metric, avg_metric))
                log_file.flush()
        log_file.close()

    @staticmethod
    def __validate(model, val_loader, epoch, num_val, opts):
        model.eval()
        log_file = open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+")
        log_file.write("======================validate epoch %d======================\n"%epoch)
        preds = None
        gts = None
        avg_metric = 0.
        with torch.no_grad():  # 加上这个可以减少在validation过程时的显存占用，提高代码的显存利用率
            for i,(imgs, labels) in enumerate(val_loader):
                if opts.use_GPU:
                    imgs = imgs.to(opts.GPU_id)
                pred = model(imgs).cpu()
                if preds is None:
                    preds = pred
                else:
                    preds = torch.cat((preds, pred), dim=0)
                if gts is None:
                    gts = labels
                else:
                    gts = torch.cat((gts, labels), dim=0)
                metric = model.calculate_metric(pred, labels)
                avg_metric = (avg_metric*i+metric)/(i+1)
                if i%opts.print_freq==0:
                    print("Epoch %d/%d | Iter %d/%d | metric = %.2f, avg_metric = %.2f" %
                          (epoch, opts.epoch, i, num_val//opts.batch_size, metric, avg_metric))
                    log_file.write("Epoch %d/%d | Iter %d/%d | metric = %.2f, avg_metric = %.2f\n" %
                          (epoch, opts.epoch, i, num_val//opts.batch_size, metric, avg_metric))
                    log_file.flush()
            metric = model.calculate_metric(preds, gts)
            print("Evaluation of validation result: average L2 distance = %.5f"%(metric))
            log_file.write("Evaluation of validation result: average L2 distance = %.5f\n"%(metric))
            log_file.flush()
            log_file.close()
        return metric

    @staticmethod
    def __save_model(model, epoch, opts):
        model_name = "epoch%d.pkl" % epoch
        save_dir = os.path.join(opts.checkpoints_dir, model_name)
        torch.save(model, save_dir)


    def main(self):
        opts = self.opts
        if not os.path.exists(opts.checkpoints_dir):
            os.mkdir(opts.checkpoints_dir)

        random_seed = opts.random_seed
        my_augmentation = None # DataAugmentation()
        train_dataset = MyDataset(opts.dataset_dir, seed=random_seed, mode="train", train_val_ratio=0.9, trans=my_augmentation)
        val_dataset = MyDataset(opts.dataset_dir, seed=random_seed, mode="val", train_val_ratio=0.9)
        train_loader = DataLoader(train_dataset, opts.batch_size, shuffle=True, num_workers=opts.num_workers)
        val_loader = DataLoader(val_dataset, opts.batch_size, shuffle=False, num_workers=opts.num_workers)
        num_train = len(train_dataset)
        num_val = len(val_dataset)

        if opts.pretrain is None:
            model = MyNet(n_points=98)
        else:
            model = torch.load(opts.pretrain)
        if opts.use_GPU:
            model.to(opts.GPU_id)

        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=opts.lr_patience, verbose=True)  # 学习率衰减

        best_metric=1000000
        for e in range(opts.start_epoch, opts.epoch+1):
            t = time.time()
            self.__train(model, train_loader, optimizer, e, num_train, opts)
            t2 = time.time()
            print("Training consumes %.2f second\n" % (t2-t))
            with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                log_file.write("Training consumes %.2f second\n" % (t2-t))
            if e % opts.save_freq==0 or e == opts.epoch+1:
                t = time.time()
                metric = self.__validate(model, val_loader, e, num_val, opts)
                t2 = time.time()
                scheduler.step(metric)  # 判断是否需要衰减lr
                print("Validation consumes %.2f second\n" % (t2 - t))
                with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                    log_file.write("Validation consumes %.2f second\n" % (t2 - t))
                if best_metric>metric:
                    best_metric = metric
                    print("Epoch %d is now the best epoch with metric %.4f\n"%(e, best_metric))
                    with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                        log_file.write("Epoch %d is now the best epoch with metric %.4f\n"%(e, best_metric))
                self.__save_model(model, e, opts)


if __name__ == '__main__':
    args = Args()
    args.set_train_args()
    train_interface = TrainInterface(args.get_opts())
    train_interface.main()
