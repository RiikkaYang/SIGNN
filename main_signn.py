import sys
import os
import numpy as np
import pandas as pd

import training
from src import signn
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Features, Value, ClassLabel
from distutils.util import strtobool
import dgl
from dgl.nn import GraphConv

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import configargparse
from scipy import interpolate
import json
from scipy.interpolate import interp1d

p = configargparse.ArgumentParser()
p.add_argument('--method', type=str, default="SIGNN")
p.add_argument('--dataset', type=str, default="synthetic_consensus", 
               help='Options are "synthetic_consensus","synthetic_clustering","synthetic_polarization","GMF","USP"')
p.add_argument('--save_dir', type=str, default="output/") 
p.add_argument('--batch_size', type=int, default=256, 
               help='Number of epochs to train for.')
p.add_argument('--hidden_features', type=int, default=1,
               help='Number of units in neural network. $L\in\{8,12,16\}$.')
p.add_argument('--num_hidden_layers', type=int, default=7, 
               help='Number of layers in neural network. $L\in\{3,5,7\}$.')
p.add_argument('--alpha', type=float, default=1.0, 
               help='$\\alpha\in\{0.1,1.0,5.0\}$. ')
p.add_argument('--beta', type=float, default=0.1, 
               help='$\\beta\in\{0.1,1.0,5.0\}$. ')
p.add_argument('--lambda', type=float, default=0,
               help='opinion weight coefficient')
p.add_argument('--num_epochs', type=int, default=1000)
p.add_argument('--lr', type=float, default=0.001, 
               help='learning rate. default=0.001')
p.add_argument('--K', type=int, default=1, 
               help='dimension of latent space $K\in\{1,2,3\}$. ')
p.add_argument('--type_odm', type=str, default="SBCM",
               help='Options are "SBCM", "EPO"')
p.add_argument('--use_graph', type=strtobool, default=True)
p.add_argument('--activation_func', type=str, default='tanh',
               help='Options are "sigmoid", "tanh", "relu", "selu", "softplus", "elu"')
opt = p.parse_args()


def prediction2label(x):
    f_x = np.exp(x) / np.sum(np.exp(x), axis=-1)[:,None]
    label = np.argmax(f_x, axis=-1)
    
    return label


def rolling_matrix(x,window_size=21):
    x = x.flatten()
    n = x.shape[0]
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n-window_size+1, window_size), strides=(stride,stride) ).copy()


class load_data(Dataset):

    def __init__(self, sequence, num_users, initial_u):
        super().__init__()

        uids = sequence[:, 0]
        times = sequence[:, 2]
        opinions = sequence[:, 1]
        interactions = inter_sequence
        
        user1_ids = interactions[:,0]
        print(user1_ids)
        user2_ids = interactions[:,1]

        g = dgl.graph((user1_ids, user2_ids))
        g = dgl.add_self_loop(g)
        
        self.initial_u = np.array(initial_u)

        history = []
        previous = []
        model_out = []
       
        previous = []
        for iu in range(num_users):
            tmpx = sequence[sequence[:,0]==iu,2]
            tmpy = sequence[sequence[:,0]==iu,1]
            if len(tmpx)>0: 
                tmpx = np.append(tmpx, 1.)
                tmpy = np.append(tmpy, tmpy[-1])
            else:
                tmpx = np.array([0,1])
                tmpy = np.array([initial_u[iu],initial_u[iu]])
            tmpf = interp1d(tmpx, tmpy, kind='next', fill_value="extrapolate")
            previous.append( tmpf(times) )
        previous = np.array(previous).T 

        user_history = rolling_matrix(sequence[:,0])
        opinion_history = rolling_matrix(sequence[:,1])
        time_history = rolling_matrix(sequence[:,2])

        dT = np.stack([user_history,opinion_history,time_history], axis=-1) 
        history = dT[:,:-1,:]
        model_out = dT[:,-1,:]

        self.previous = np.array(previous) 
        self.history = history 
        self.model_out = model_out 
        self.interactions = inter_sequence
        self.g = g

        self.datanum = len(self.model_out)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):

        history = self.history[idx]
        previous = self.previous[idx]
        model_out = self.model_out[idx]

        return {'history': torch.from_numpy(history).float(), 'previous': torch.from_numpy(previous).float(), 
                'initial': torch.from_numpy(self.initial_u).float(), 
                'g': self.g,
                'ii': torch.from_numpy(self.interactions).float(),
                'ui': torch.from_numpy(model_out[:1]).long(), 
                'ti': torch.from_numpy(model_out[2:]).float()}, \
               {'opinion': torch.from_numpy(model_out[1:2]).float()}

def evaluate(model, sequence, train_sequence, num_users, initial_u, batch_size, nclasses, val_period):

    test_indices = np.where(sequence[:,2]>=val_period)[0]

    opinions = train_sequence[:, 1]
    uids = train_sequence[:, 0]
    initial_u = np.array(initial_u).reshape(1,-1)

    dfs = []
    for ii in test_indices[20:]: 
        current_time = sequence[ii,2]
        tmphistory = train_sequence[train_sequence[:, 2]<current_time,:]
        if len(tmphistory)<50: continue
        
        prev_u = []
        for iu in range(num_users):
            tmpprev_u = train_sequence[(train_sequence[:, 0]==iu)&(train_sequence[:, 2]<current_time),1] 
            if len(tmpprev_u)>0:  
                prev_u.append(tmpprev_u[-1])
            else:
                prev_u.append(0.)
        print(prev_u)
        if isinstance(prev_u, list):
            prevu_numpy_list = []
            for out in prev_u:
                if isinstance(out, torch.Tensor):
                    prevu_numpy_list.append(out.cpu().detach().numpy())
                elif isinstance(out, (float, int)):
                    prevu_numpy_list.append(np.array(out))
        print(prevu_numpy_list)
        prev_u = np.array(prevu_numpy_list)
        print(prev_u)

        history = tmphistory[np.newaxis,-50:].cpu().detach().numpy()
        print(history)
        previous = prev_u[np.newaxis]
        model_out = sequence[np.newaxis,ii]

        model_input = {'history': torch.from_numpy(history).float(), 'previous': torch.from_numpy(previous).float(), 
                       'initial': torch.from_numpy(initial_u).float(), 
                       'ui': torch.from_numpy(model_out[:,:1]).long(), 'ti': torch.from_numpy(model_out[:,2:]).float()}
        model_output = model(model_input)
        test_pred = model_output['opinion'].detach().numpy().flatten()
        tmpop = sequence[ii,1:2]
        if 'opinion_label' in model_output.keys(): 
            test_pred_label = prediction2label(model_output['opinion_label'].detach().numpy())
            tmpop = tmpop/(nclasses-1)
        else: 
            test_pred_label = test_pred * (nclasses-1)

        new_item = np.c_[sequence[ii,:1],test_pred,sequence[ii,2:]]
        train_sequence = np.r_[train_sequence, new_item]

        tmpdf = pd.DataFrame(data = np.c_[sequence[ii,:1], sequence[ii,2:], tmpop, test_pred, test_pred_label], columns=["user","time","gt","pred","pred_label"]) 
        dfs.append(tmpdf)

    res_df = pd.concat(dfs)

    return res_df

def prediction(use_graph, dataloader, model, batch_size, nclasses):

    model.train = False
    dfs = []
    att_dfs = []
    zu_dfs = []
    for (model_input, gt) in dataloader:
         model_output = model(model_input)
         #print(model_output)
         for key, value in model_output.items():
            if value is not None:
                model_output[key] = value.to('cpu')
         test_ui = model_input['ui'].numpy().flatten()
         test_ti = model_input['ti'].numpy().flatten()
         test_oi = gt["opinion"].detach().numpy().flatten()
         test_pred = model_output['opinion'].detach().numpy().flatten()
         if 'opinion_label' in model_output.keys(): 
             test_pred_label = prediction2label(model_output['opinion_label'].detach().numpy())
             test_oi = test_oi/(nclasses-1) 
         else: 
             test_pred_label = test_pred * (nclasses-1)
         tmpdf = pd.DataFrame(data = np.c_[test_ui, test_ti, test_oi, test_pred, test_pred_label], columns=["user","time","gt","pred","pred_label"]) 
         dfs.append(tmpdf)
         if 'zu' in model_output.keys() and not model_output['zu'] is None: 
             zu_pred = model_output['zu'].detach().numpy()
             print(zu_pred.shape, test_ui.shape)
             if test_ui.shape[0]==zu_pred.shape[0]:
                zu_tmpdf = pd.DataFrame(data = np.c_[test_ui[:,np.newaxis], zu_pred], columns=["user"]+list(range(zu_pred.shape[1]))) 
                zu_dfs.append(zu_tmpdf)
    res_df = pd.concat(dfs)

    att_df = None

    if len(zu_dfs)>0: 
        zu_df = pd.concat(zu_dfs)
    else:
        zu_df = None

    return res_df, att_df, zu_df

def collate_fn(batch):
    graphs = [item[0]['graph'] for item in batch]
    
    batched_graph = dgl.batch(graphs)

    history = torch.stack([item[0]['history'] for item in batch])
    previous = torch.stack([item[0]['previous'] for item in batch])
    initial = torch.stack([item[0]['initial'] for item in batch])
    ui = torch.stack([item[0]['ui'] for item in batch])
    ti = torch.stack([item[0]['ti'] for item in batch])
    opinions = torch.stack([item[1]['opinion'] for item in batch])

    return {
        'graph': batched_graph,
        'history': history,
        'previous': previous,
        'initial': initial,
        'ui': ui,
        'ti': ti
    }, {
        'opinion': opinions
    }

def main_signn(data_type, method, root_path):

    batch_size = opt.batch_size
    use_graph = opt.use_graph 
    num_epochs = opt.num_epochs

    str_params = str(opt.hidden_features)+"_"+str(opt.num_hidden_layers)+"_"+str(opt.alpha)+"_"+str(opt.beta)+"_"+opt.type_odm
    outdir = os.path.join(root_path, str_params)
    if not os.path.exists(outdir): os.makedirs(outdir)

    print("Loading dataset")
    df = pd.read_csv("working/posts_final_"+data_type+".tsv", delimiter="\t") 
    nclasses = len(df["opinion"].unique())
    if not (method=="SINN" or method=="NN"):
        df["opinion"] = df["opinion"]/(nclasses-1)

    sequence = np.array(df[["user_id","opinion","time"]])
    initial_u = np.loadtxt("working/initial_"+data_type+".txt", delimiter=',', dtype='float')

    interactions = pd.read_csv("working/synthetic_interaction_"+data_type+"_g.csv")
    interactions_sequence = np.array(interactions[["user1_id", "user2_id", "time"]])
    
    num_users = int(1 + np.max(sequence[:,0]))
    initial_u = initial_u[:num_users]
    print(len(initial_u))
    print("Finished loading dataset")
    global_graph = dgl.graph(([],[]), num_nodes=num_users)
    global_graph.add_edges(interactions_sequence[:,0], interactions_sequence[:,1])
    global_graph = dgl.add_self_loop(global_graph)
    print(global_graph)
    global_graph.ndata['feat'] = torch.from_numpy(initial_u).float()

    if use_graph:
        profiles = global_graph 
    else:
        profiles = None

    if "synthetic" in data_type:
        train_period = 0.5
        val_period = 0.7
    else:
        train_period = 0.7
        val_period = 0.8

    train_sequence = sequence[sequence[:,2]<train_period,:]
    train_inter_sequence = interactions_sequence[interactions_sequence[:,2]<train_period,:]
    train_dataset = load_data(train_sequence, num_users=num_users, initial_u=initial_u)

    val_sequence = sequence[(sequence[:,2]>=train_period)&(sequence[:,2]<val_period),:]
    val_inter_sequence = interactions_sequence[(interactions_sequence[:,2]>=train_period)&(interactions_sequence[:,2]<val_period),:]
    val_dataset = load_data(val_sequence, num_users=num_users, initial_u=initial_u)

    test_sequence = sequence[(sequence[:,2]>=val_period),:]
    test_inter_sequence = interactions_sequence[(interactions_sequence[:,2]>=val_period),:]
    test_dataset = load_data(test_sequence, num_users=num_users, initial_u=initial_u)

    #train_graphs, train_labels = get_graph_and_labels(train_intervals, opinions_gnn, interactions_gnn)
    #val_graphs, val_labels = get_graph_and_labels(val_intervals, opinions_gnn, interactions_gnn)
    #test_graphs, test_labels = get_graph_and_labels(test_intervals, opinions_gnn, interactions_gnn)

    #train_dataloader = DataLoader(list(zip(train_graphs, train_labels)), batch_size=32, shuffle=True, collate_fn=collate)
    #val_dataloader = DataLoader(list(zip(val_graphs, val_labels)), batch_size=32, shuffle=False, collate_fn=collate)
    #test_dataloader = DataLoader(list(zip(test_graphs, test_labels)), batch_size=32, shuffle=False, collate_fn=collate)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=0)

    if method == 'SIGNN':
        _method = signn
    elif method=="EPOGNN":
        _method = epognn

    torch.manual_seed(100)
    model = _method.model(type=opt.activation_func, num_users=num_users, hidden_features=opt.hidden_features, 
                          num_hidden_layers=opt.num_hidden_layers, type_odm=opt.type_odm, alpha=opt.alpha, beta=opt.beta, K=opt.K, 
                          df_profile=profiles, nclasses=nclasses, dataset=data_type)

    ###############################################################################
    # Training 
    ###############################################################################
    
    print("Training network...")
    training.train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=num_epochs, lr=opt.lr,
                       loss_fn=_method.loss_function, method=method, input_sequence=sequence)
    print("Network trained...")

    model.eval()
    torch.save(model.state_dict(), outdir+"/model_state_dict_"+method)

    ###############################################################################
    # Evaluation
    ###############################################################################

    print("Evaluating network...")
    test_res, att_res, zu_res = prediction(use_graph, test_dataloader, model, batch_size, nclasses)
    if not zu_res is None:
        zu_res.to_csv(outdir+"/interaction_predicted_"+method+".csv", index=False)

    test_res.to_csv(outdir+"/test_predicted_"+method+".csv", index=False)

    train_res, _, _ = prediction(use_graph, train_dataloader, model, batch_size, nclasses)
    train_res.to_csv(outdir+"/train_predicted_"+method+".csv", index=False)

    val_res, _, _ = prediction(use_graph, val_dataloader, model, batch_size, nclasses)
    val_res.to_csv(outdir+"/val_predicted_"+method+".csv", index=False)

    mae = (test_res["pred"]-test_res["gt"]).abs()
    print('#######################################')
    print('## Performance for', method, 'on', data_type, 'dataset')
    print("## MAE:", mae.mean())
    if (method=="SIGNN" or method=="EPOGNN"):
        truth_label = ((nclasses-1)*test_res["gt"]).astype(int)
        acc = accuracy_score(truth_label, test_res["pred_label"])
        f1 = f1_score(truth_label, test_res["pred_label"], average='macro')
        print("## ACC:", acc, "  F1:", f1)
    print('#######################################')
    print()
    print("Network evaluated....")


if __name__ == "__main__":

    logging_root = os.path.join(opt.save_dir, opt.dataset)
    if not os.path.exists(logging_root): os.makedirs(logging_root)

    main_signn(opt.dataset, opt.method, logging_root)