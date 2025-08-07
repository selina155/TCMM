import os
import numpy as np


def get_dataset_path(dataset):
    if 'netsim' in dataset:
        dataset_path = 'netsim'
    elif 'dream3_combined' in dataset:
        dataset_path = 'dream3'
    elif 'yeast' in dataset:
        dataset_path = 'dream3'
    elif 'ecoli' in dataset:
        dataset_path = 'dream3'
    elif dataset in ['traffic','medical','pm25','traffic_medical']:
        dataset_path = 'Traffic'
    elif dataset in ['edges1','edges2','edges3','edges4']:
        dataset_path='simdata'
        
    elif 'snp100' in dataset:
        dataset_path = 'snp100'
    elif "lorenz96" in dataset:
        dataset_path='lorenz96'
    elif dataset == ['finance', 'fluxnet']:
        dataset_path = dataset
    else:
        dataset_path = 'Macaque'

    return dataset_path



def create_save_name(dataset, cfg):
    if dataset == 'lorenz96':
        return f'lorenz96_N={cfg.num_nodes}_T={cfg.timesteps}_num_graphs={cfg.num_graphs}'
    return dataset


def load_synthetic_from_folder(dataset_dir, dataset_name):
    X = np.load(os.path.join(dataset_dir, dataset_name, 'X.npy'))
    adj_matrix = np.load(os.path.join(
        dataset_dir, dataset_name, 'adj_matrix.npy'))

    return X, adj_matrix

def norm(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    normalized_data = (data - mean) / (std)
    return normalized_data

def minmax(X):
    min_val = np.min(X, axis=1, keepdims=True)  # shape (N, 1, M)
    max_val = np.max(X, axis=1, keepdims=True)  # shape (N, 1, M)
    X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
    return X_normalized
    
def zscore(X):
    mean = np.mean(X, axis=1, keepdims=True)  # shape (N, 1, M)
    std = np.std(X, axis=1, keepdims=True)  # shape (N, 1, M)
    X_normalized = (X - mean) / (std + 1e-8)  # ∑¿÷π≥˝¡„
    return X_normalized
           
def prepross_data(data):
    num_samples, T, num_nodes = data.shape
    new_data = np.zeros_like(data, dtype=float)
    for i in range(num_nodes):
        node = data[:,:,i]
        new_data[:,:,i] = node - node.mean()
    return new_data
    
    
def load_netsim(dataset_dir, dataset_file):
    # load the files
    data = np.load(os.path.join(dataset_dir, dataset_file + '.npz'))
    X = data['X']
    X=norm(X)
    adj_matrix = data['adj_matrix']
    return X, adj_matrix
    
def load_traffic(dataset_dir, dataset_file):
    # load the files
    X = np.load(os.path.join(dataset_dir, f'traffic_gen_data.npy'))
    X=X[:,:,:20]
    X=X-X.mean()
    adj_matrix=np.load(os.path.join(dataset_dir, f'traffic_graph.npy'))
    adj_matrix=np.tile(adj_matrix, (X.shape[0], 1, 1))
    return X,adj_matrix  
    
def load_medical(dataset_dir, dataset_file):
    # load the files
    X = np.load(os.path.join(dataset_dir, f'medical_gen_data.npy'))
    X=X[:,:,:20]
    X=X-X.mean()
    adj_matrix=np.load(os.path.join(dataset_dir, f'medical_graph.npy'))
    adj_matrix=np.tile(adj_matrix, (X.shape[0], 1, 1))
    return X,adj_matrix  
    
def load_pm25(dataset_dir, dataset_file):
    # load the files
    X = np.load(os.path.join(dataset_dir, f'pm25_gen_data.npy'))
    X=X[:,:,:36]
    X=X-X.mean()
    #X=prepross_data(X)
    adj_matrix=np.load(os.path.join(dataset_dir, f'pm25_graph.npy'))
    adj_matrix=np.tile(adj_matrix, (X.shape[0], 1, 1))
    return X,adj_matrix 
    
def load_tm(dataset_dir, dataset_file):
    # load the files
    X = np.load(os.path.join(dataset_dir, f'traffic_medical_data.npy'))
    X=X-X.mean()
    adj_matrix=np.load(os.path.join(dataset_dir, f'traffic_medical_adj.npy'))
    return X,adj_matrix
    
def load_simdata(dataset_dir, dataset_file):
    X = np.load(os.path.join(dataset_dir, f'edges4_data2.npy'))
    print(X.shape)
    adj_matrix=np.load(os.path.join(dataset_dir, f'edges4_graph2.npy'))
    return X,adj_matrix

 
def load_dream3_combined(dataset_dir, size):
    data = np.load(os.path.join(dataset_dir, f'combined_{size}.npz'))
    #data = np.load(os.path.join(dataset_dir, f'ecoli_{size}\\ecoli_1_diff.npy'))
    #data=np.load(os.path.join(dataset_dir, f'yeast_{size}/yeast_3.npz'))
    X=data['X']
    X=X-X.mean()
    #X=prepross_data(X)
    adj_matrix = data['adj_matrix']
    return X, adj_matrix

def load_Macaque(dataset_dir, dataset_file):
    # load the files
    data = np.load(os.path.join(dataset_dir, dataset_file + '.npz'))
    X = data['data']
    adj_matrix = data['matrix']
    # adj_matrix = np.transpose(adj_matrix, (0, 2, 1))
    return X,adj_matrix

def load_yeast(dataset_dir, size,index):
    # load the files
    data = np.load(os.path.join(dataset_dir, f'yeast_{size}/yeast_{index}.npz'))
    #data = np.load(os.path.join(dataset_dir, f'ecoli_{size}\\ecoli_1_diff.npy'))
    #data=np.load(os.path.join(dataset_dir, f'yeast_{size}/yeast_3.npz'))
    X=data['X']
    #X=prepross_data(X)
    X=X-X.mean()
    adj_matrix = data['adj_matrix']
    return X, adj_matrix

def load_lorenz96(dataset_dir, dataset_file):
    data = np.load(os.path.join(dataset_dir, dataset_file+'.npz'))
    X = data['X']
    #X = np.transpose(X, (0, 2, 1))
    adj_matrix = data['adj_matrix']
    #adj_matrix = data['adj_matrix']
    return X, adj_matrix

def load_ecoli(dataset_dir, size,index):
    # load the files
    data = np.load(os.path.join(dataset_dir, f'ecoli_{size}/ecoli_{index}.npz'))
    #data = np.load(os.path.join(dataset_dir, f'ecoli_{size}\\ecoli_1_diff.npy'))
    #data=np.load(os.path.join(dataset_dir, f'yeast_{size}/yeast_3.npz'))
    X=data['X']
    #X=X-X.mean()
    X=prepross_data(X)
    adj_matrix = data['adj_matrix']
    return X, adj_matrix
    
def load_data(dataset, dataset_dir, config):
    if dataset in ['edges1','edges2','edges3','edges4']:
        X, adj_matrix = load_simdata(dataset_dir=dataset_dir, dataset_file=dataset)
        N=10
        T=100
        aggregated_graph = True
        # read lag from config file
        lag = int(config['lag'])
        data_dim = 1
        X = X.reshape(N,T,-1)
        X = np.expand_dims(X, axis=-1)
        adj_matrix=np.tile(adj_matrix, (X.shape[0], 1, 1))

    elif 'netsim' in dataset:
        X, adj_matrix = load_netsim(
            dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        # read lag from config file
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'dream3_combined':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_dream3_combined(
            dataset_dir=dataset_dir, size=dream3_size)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'yeast1':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_yeast(
            dataset_dir=dataset_dir, size=dream3_size,index=1)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'yeast2':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_yeast(
            dataset_dir=dataset_dir, size=dream3_size,index=2)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'yeast3':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_yeast(
            dataset_dir=dataset_dir, size=dream3_size,index=3)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'ecoli1':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_ecoli(
            dataset_dir=dataset_dir, size=dream3_size,index=1)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'ecoli2':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_ecoli(
            dataset_dir=dataset_dir, size=dream3_size,index=2)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'Macaque':
        X, adj_matrix=load_Macaque(dataset_dir=dataset_dir,dataset_file=dataset)
        X=norm(X)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        
    elif dataset == "traffic_medical":
        X, adj_matrix = load_tm(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == "traffic":
        X, adj_matrix = load_traffic(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == "medical":
        X, adj_matrix = load_medical(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == "pm25":
        X, adj_matrix = load_pm25(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'MDD':
        X, adj_matrix=load_MDD(dataset_dir=dataset_dir,dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == "RestingLeft":
        X, adj_matrix = load_Resting(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
    elif dataset == "RestingRight":
        X, adj_matrix = load_Resting(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
    elif 'lorenz96' in dataset:
        X, adj_matrix = load_lorenz96(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    else:
        X, adj_matrix = load_synthetic_from_folder(
            dataset_dir=dataset_dir, dataset_name=dataset)
        lag = adj_matrix.shape[1] - 1
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
        aggregated_graph = False
    print("Loaded data of shape:", X.shape)
    return X, adj_matrix, aggregated_graph, lag, data_dim
