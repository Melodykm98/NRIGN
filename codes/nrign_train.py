from graph_constructor import *
from utils import *
from model import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
from sklearn.metrics import matthews_corrcoef,accuracy_score
import warnings
import torch
import pandas as pd
import os
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
import argparse
path_marker = '/'
limit = None
num_process = 12

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device):
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        bg, bg3, bg2, bg4, Ys, keys = batch
        bg, bg3, bg2, bg4, Ys = bg.to(device), bg3.to(device), bg2.to(device), bg4.to(device), Ys.to(device)
        outputs = model(bg, bg3, bg2, bg4)
        loss = loss_fn(outputs, Ys)
        loss.backward()
        optimizer.step()

def run_a_eval_epoch(model, validation_dataloader, device):  
    true = []
    pred = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            bg, bg3, bg2, bg4, Ys, keys = batch
            bg, bg3, bg2, bg4, Ys = bg.to(device), bg3.to(device),bg2.to(device), bg4.to(device), Ys.to(device)
            outputs = model(bg, bg3,bg2, bg4)
            result = torch.softmax(outputs,dim=1)
            true.append(Ys.data.cpu())
            pred.append(result.data.cpu())
            key.append(keys)
    return true, pred, key


if __name__ == '__main__':
    argparser = argparse.ArgumentParser() 
    argparser.add_argument('--gpuid', type=str, default=0, help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -3.0, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=5000, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=200, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=70, help="early stopping patience")
    argparser.add_argument('--l2', type=float, default=10 ** -6, help="L2 regularization")
    argparser.add_argument('--seed', type=int, default=0, help="random seed") 
    argparser.add_argument('--node_feat_size', type=int, default=40)
    argparser.add_argument('--edge_feat_size_2d', type=int, default=12)
    argparser.add_argument('--edge_feat_size_3d', type=int, default=21)
    argparser.add_argument('--graph_feat_size', type=int, default=128)
    argparser.add_argument('--num_layers', type=int, default=2, help='the number of intra-molecular layers')
    argparser.add_argument('--outdim_g3', type=int, default=128, help='the output dim of inter-molecular layers')
    argparser.add_argument('--d_FC_layer', type=int, default=128, help='the hidden layer size of task networks')
    argparser.add_argument('--n_FC_layer', type=int, default=2, help='the number of hidden layers of task networks')
    argparser.add_argument('--dropout', type=float, default=0.15, help='dropout ratio')
    argparser.add_argument('--n_tasks', type=int, default=2)
    argparser.add_argument('--num_workers', type=int, default=0,
                           help='number of workers for loading data in Dataloader')
    argparser.add_argument('--model_save_dir', type=str, default='./model_save', help='path for saving model')
    argparser.add_argument('--mark', type=str, default='3d')
    argparser.add_argument('--dis_threshold',type=int, default=10, help='atoms interaction distance')
    argparser.add_argument('--train_result',type=str,default='./train_result')

    args = argparser.parse_args()  
    gpuid, lr, epochs, batch_size, num_workers, model_save_dir = args.gpuid, args.lr, args.epochs, args.batch_size, args.num_workers, args.model_save_dir
    tolerance, patience, l2, seed = args.tolerance, args.patience, args.l2, args.seed

    # paras for model
    node_feat_size, edge_feat_size_2d, edge_feat_size_3d = args.node_feat_size, args.edge_feat_size_2d, args.edge_feat_size_3d
    graph_feat_size, num_layers = args.graph_feat_size, args.num_layers
    outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks, mark = args.outdim_g3, args.d_FC_layer, args.n_FC_layer, args.dropout, args.n_tasks, args.mark
    dis_threshold,train_result=args.dis_threshold,args.train_result


    HOME_PATH = os.getcwd() 
    all_data = pd.read_csv('./label.csv') 
    os.makedirs(model_save_dir,exist_ok=True)

    # data
    train_dir = './data_graph/training/complex' 
    valid_dir = './data_graph/validation/complex'
    test_dir = './data_graph/test/complex'

    # training data
    train_keys = os.listdir(train_dir) 
    train_labels= []
    train_data_dirs = []
    for key in train_keys:
        train_labels.append(all_data[all_data['ligand'] == key]['label'].values[0]) 
        train_data_dirs.append(train_dir + path_marker + key) 

    # validtion data
    valid_keys = os.listdir(valid_dir)
    valid_labels = []
    valid_data_dirs = []
    for key in valid_keys:
        valid_labels.append(all_data[all_data['ligand'] == key]['label'].values[0])
        valid_data_dirs.append(valid_dir + path_marker + key)

    # testing data
    test_keys = os.listdir(test_dir)
    test_labels = []
    test_data_dirs = []
    for key in test_keys:
        test_labels.append(all_data[all_data['ligand'] == key]['label'].values[0])
        test_data_dirs.append(test_dir + path_marker + key)

    train_dataset = GraphDatasetV2MulPro(keys=train_keys[:limit], labels=train_labels[:limit], data_dirs=train_data_dirs[:limit],
                                        graph_ls_path='./data_graph/training/graph_ls_path',
                                        graph_dic_path='./data_graph/training/graph_dic_path',
                                        num_process=num_process, path_marker=path_marker,dis_threshold=dis_threshold) 
    valid_dataset = GraphDatasetV2MulPro(keys=valid_keys[:limit], labels=valid_labels[:limit], data_dirs=valid_data_dirs[:limit],
                                         graph_ls_path='./data_graph/validation/graph_ls_path',
                                         graph_dic_path='./data_graph/validation/graph_dic_path',
                                         num_process=num_process, path_marker=path_marker,dis_threshold=dis_threshold)
    test_dataset = GraphDatasetV2MulPro(keys=test_keys[:limit], labels=test_labels[:limit],data_dirs=test_data_dirs[:limit],
                                         graph_ls_path='./data_graph/test/graph_ls_path',
                                         graph_dic_path='./data_graph/test/graph_dic_path',
                                         num_process=num_process, path_marker=path_marker,dis_threshold=dis_threshold)
    
    set_random_seed(seed)
    print('the number of train data:', len(train_dataset))
    print('the number of valid data:', len(valid_dataset))
    print('the number of test data:', len(test_dataset))
    train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                    collate_fn=collate_fn_v2_MulPro)
    valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                    collate_fn=collate_fn_v2_MulPro)
    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                    collate_fn=collate_fn_v2_MulPro)

    # model
    DTIModel = DTIPredictorV4_V2(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d, num_layers=num_layers,
                                    graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                                    d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout, n_tasks=n_tasks)
    print('number of parameters : ', sum(p.numel() for p in DTIModel.parameters() if p.requires_grad))
    print(DTIModel)
    device = torch.device("cuda:%s" % gpuid if torch.cuda.is_available() else "cpu")
    DTIModel.to(device)
    optimizer = torch.optim.Adam(DTIModel.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.CrossEntropyLoss().to(device)
    dt = datetime.datetime.now()
    filename = './model_save/{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second,
                                                                          dt.microsecond)
    stopper = EarlyStopping(mode='higher', patience=patience, tolerance=tolerance,
                                filename=filename)

    train_mccs=[]
    valid_mccs=[]
    train_accs=[]
    valid_accs=[]
    for epoch in range(epochs):
        st = time.time()
        # train
        run_a_train_epoch(DTIModel, loss_fn, train_dataloader, optimizer, device)
       
        #validation
        train_true, train_pred, _= run_a_eval_epoch(DTIModel, train_dataloader, device)
        valid_true, valid_pred, _ = run_a_eval_epoch(DTIModel, valid_dataloader, device)
    
        train_true = torch.cat(train_true, dim=0)
        train_pred = torch.max(torch.cat(train_pred, dim=0),dim=1)[1]

        valid_true = torch.cat(valid_true, dim=0)
        valid_pred = torch.max(torch.cat(valid_pred, dim=0),dim=1)[1]

        train_acc = accuracy_score(train_true, train_pred)
        valid_acc = accuracy_score(valid_true, valid_pred)
        train_accs.append(f'{train_acc:.4f}')
        valid_accs.append(f'{valid_acc:.4f}')

        train_mcc = matthews_corrcoef(train_true, train_pred)
        valid_mcc = matthews_corrcoef(valid_true, valid_pred)
        train_mccs.append(f'{train_mcc:.4f}')
        valid_mccs.append(f'{valid_mcc:.4f}')
    
        early_stop = stopper.step(valid_mcc, DTIModel)
        end = time.time()
        if early_stop:
                break
        print(f'epoch:{epoch}')
        print("train_mcc:%.4f \t train_acc:%.4f \nvalid_mcc:%.4f \t valid_acc:%.4f" 
            % (train_mcc, train_acc, valid_mcc, valid_acc))
        print("time:%.3f s" % (end - st))
   