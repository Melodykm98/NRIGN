from graph_constructor import *
from utils import *
from model import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import matthews_corrcoef,accuracy_score
import warnings
import torch
import pandas as pd
import os
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
import argparse

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


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


lr = 0.0003
epochs = 600
batch_size = 64
num_workers = 0
tolerance = 0.0
patience = 70
l2 = 10 ** -6
repetitions = 1
# paras for model
node_feat_size = 40
edge_feat_size_2d = 12
edge_feat_size_3d = 21
graph_feat_size = 128
num_layers = 2
outdim_g3 = 128
d_FC_layer, n_FC_layer = 128, 2
dropout = 0.15
n_tasks = 2
mark = '3d'
path_marker = '/'



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--graph_ls_path', type=str, default='./data_graph/test/graph_ls_path',
                           help="absolute path for storing graph list objects")
    argparser.add_argument('--graph_dic_path', type=str, default='./data_graph/test/graph_dic_path',
                           help="absolute path for storing graph dictionary objects (temporary files)")
    argparser.add_argument('--model_path', type=str, default='./model_save/alignment_six_model.pth',
                           help="absolute path for storing pretrained model")
    argparser.add_argument('--cpu', type=bool, default=True,
                           help="using cpu for the prediction (default:True)")
    argparser.add_argument('--gpuid', type=int, default=0,
                           help="the gpu id for the prediction")
    argparser.add_argument('--num_process', type=int, default=12,
                           help="the number of process for generating graph objects")
    argparser.add_argument('--input_path', type=str, default='./data_graph/test/complex',
                           help="the absoute path for storing input files")
    argparser.add_argument('--dis_threshold',type=int, default=10, help='atoms interaction distance')
    args = argparser.parse_args()
    graph_ls_path, graph_dic_path, model_path, cpu, gpuid, num_process, input_path,dis_threshold = args.graph_ls_path, \
                                                                                                    args.graph_dic_path, \
                                                                                                    args.model_path, \
                                                                                                    args.cpu, \
                                                                                                    args.gpuid, \
                                                                                                    args.num_process, \
                                                                                                    args.input_path, \
                                                                                                    args.dis_threshold

    all_data = pd.read_csv('./label.csv')

    keys = os.listdir(input_path)
    labels = []
    data_dirs = []
    for key in keys:
        data_dirs.append(input_path + path_marker + key)
        labels.append(all_data[all_data['ligand'] == key]['label'].values[0])
    limit = None


    # generating the graph objective using multi process
    test_dataset = GraphDatasetV2MulPro(keys=keys[:limit], labels=labels[:limit], data_dirs=data_dirs[:limit],
                                        graph_ls_path=graph_ls_path,
                                        graph_dic_path=graph_dic_path,
                                        num_process=num_process, dis_threshold=dis_threshold, path_marker=path_marker)
    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_fn_v2_MulPro)

    DTIModel = DTIPredictorV4_V2(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d, num_layers=num_layers,
                                 graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                                 d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout, n_tasks=n_tasks)
    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%s" % gpuid)
    DTIModel.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    DTIModel.to(device)

    test_true, test_pred,_ = run_a_eval_epoch(DTIModel, test_dataloader, device)

    test_true = torch.cat(test_true, dim=0)
    test_pred = torch.max(torch.cat(test_pred, dim=0),dim=1)[1]

    test_acc = accuracy_score(test_true, test_pred)
    test_mcc = matthews_corrcoef(test_true, test_pred)
    
    print('The result of prediction:')
    print(f'mcc: {test_mcc:.4f} acc:{test_acc:.4f}')