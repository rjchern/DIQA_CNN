from models import CNNDIQAnet
from Performance import DIQAPerformance
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from DIQADataset import DIQADataset
import numpy as np
import yaml
from pathlib import Path
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric
from DataInfoLoader import DataInfoLoader
import math

def ensure_dir(path):
    p=Path(path)
    if not p.exists():
        p.mkdir()

def loss_fn(y_pred, y):
    return F.l1_loss(y_pred, y) 

def get_data_loaders(dataset_name,config,train_batch_size):
    datainfo=DataInfoLoader(dataset_name,config) 
    img_num=datainfo.img_num
    index=np.arange(img_num)
    np.random.shuffle(index)
    
    train_index=index[0:math.floor(img_num*0.6)]
    val_index=index[math.floor(img_num*0.6):math.floor(img_num*0.8)]
    test_index=index[math.floor(img_num*0.8):]
    
    #test only
    #train_index=np.array([1,2,3])
    #val_index=np.array([4,5,6])
    #test_index=np.array([7,8,9])
    
    train_dataset = DIQADataset(dataset_name,config,train_index,status='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4)

    val_dataset = DIQADataset(dataset_name,config,val_index,status='val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = DIQADataset(dataset_name,config,test_index,status='test')
        test_loader = torch.utils.data.DataLoader(test_dataset)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader

class Solver:
    def __init__(self):
        self.model=CNNDIQAnet()

    def run(self,dataset_name,train_batch_size,epochs,lr,weight_decay,model_name,config,trained_model_file,save_result_file,disable_gpu=False):
        if config['test_ratio']:
            train_loader, val_loader, test_loader = get_data_loaders(dataset_name,config,train_batch_size)
        else:
            train_loader, val_loader = get_data_loaders(dataset_name,config,train_batch_size)

        device = torch.device("cuda" if not disable_gpu and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        global best_criterion
        best_criterion = -1 
        trainer = create_supervised_trainer(self.model, optimizer, loss_fn, device=device)
        evaluator = create_supervised_evaluator(self.model,metrics={'DIQA_performance': DIQAPerformance()},device=device)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            SROCC, PLCC= metrics['DIQA_performance']
            print("Validation Results - Epoch: {} SROCC: {:.4f} PLCC: {:.4f} ".format(engine.state.epoch, SROCC, PLCC))
            global best_criterion
            global best_epoch
            if SROCC > best_criterion:
                best_criterion = SROCC
                best_epoch = engine.state.epoch
                torch.save(self.model.state_dict(), trained_model_file)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_testing_results(engine):
            if config["test_ratio"] > 0 and config['test_during_training']:
                evaluator.run(test_loader)
                metrics = evaluator.state.metrics
                SROCC,PLCC= metrics['DIQA_performance']
                print("Testing Results    - Epoch: {} SROCC: {:.4f} PLCC: {:.4f} ".format(engine.state.epoch, SROCC, PLCC))

        @trainer.on(Events.COMPLETED)
        def final_testing_results(engine):
            if config["test_ratio"] > 0:
                self.model.load_state_dict(torch.load(trained_model_file))
                evaluator.run(test_loader)
                metrics = evaluator.state.metrics
                SROCC,PLCC= metrics['DIQA_performance']
                global best_epoch
                print("Final Test Results - Epoch: {} SROCC: {:.4f} PLCC: {:.4f} ".format(best_epoch, SROCC, PLCC))
                np.save(save_result_file, (SROCC, PLCC))
        # kick everything off
        trainer.run(train_loader, max_epochs=epochs)

if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNDIQA')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='1', type=str,
                        help='exp id (default: 1)')
    parser.add_argument('--dataset_name', default='SOC', type=str,
                        help='dataset name (default: SOC)')
    parser.add_argument('--model', default='CNNDIQA', type=str,
                        help='model name (default: CNNIQAplusplus)')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    print('exp id: ' + args.exp_id)
    print('model: ' + args.model)
    #config.update(config[args.dataset_name])
    #config.update(config[args.model])
    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-EXP{}-lr={}.pth'.format(args.model, args.dataset_name, args.exp_id, args.lr)
    ensure_dir('results')
    save_result_file = 'results/{}-{}-EXP{}-lr={}.npy'.format(args.model, args.dataset_name, args.exp_id, args.lr)

    dataset_name='SOC'

    solver=Solver()
    solver.run(dataset_name,args.batch_size, args.epochs, args.lr, args.weight_decay, args.model, config,\
    trained_model_file, save_result_file, args.disable_gpu)
    