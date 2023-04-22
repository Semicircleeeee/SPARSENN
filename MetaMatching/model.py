from .utils import *
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import random
import warnings
import os
import csv

class Net(nn.Module):
    """
    The network structure
    """
    def __init__(self,partition_mtx_dict, num_hidden_layer_neuron_list, keep_prob):
        super(Net,self).__init__()
        layer1 = nn.Sequential()
        for i in range(len(partition_mtx_dict)):
            mtx = partition_mtx_dict["p%d" % i]  # the mask matrix
            layer1.add_module('f'+ str(i), SparseLinear(mtx.shape[0], mtx.shape[1], mtx))
            layer1.add_module("f_relu"+str(i), nn.ReLU(True))
            #layer1.add_module("bn1"+str(i), nn.BatchNorm1d(mtx.shape[1]))
        self.layer1 = layer1
        
        layer2 = nn.Sequential()
        num_hidden_layer_neuron_list = [mtx.shape[1]] + num_hidden_layer_neuron_list + [2]
        for j in range(1, len(num_hidden_layer_neuron_list)-1):
            layer2.add_module('h'+str(j), myLinear(num_hidden_layer_neuron_list[j-1], num_hidden_layer_neuron_list[j]))
            layer2.add_module('h_relu'+str(j), nn.ReLU(True))
            #layer2.add_module("bn2"+str(j), nn.BatchNorm1d(num_hidden_layer_neuron_list[j]))
            layer2.add_module('h_drop'+str(j), nn.Dropout(p=keep_prob))
        layer2.add_module('h'+str(j+1), myLinear(num_hidden_layer_neuron_list[j], num_hidden_layer_neuron_list[j+1]))
        self.layer2 = layer2
        
    def forward(self,input):
        out = self.layer1(input)
        out = self.layer2(out)
        return out
    

# Define the function for train the model 
def sparse_nn(expression, target, partition, feature_meta, sparsify_coefficient=0.3, threshold_layer_size=100, 
              num_hidden_layer_neuron_list=[20], drop_out=0.3, random_seed=10, batch_size=32, lr=0.001, weight_decay=0,
              num_epoch=100):

    # set the partition to be self-connected
    np.fill_diagonal(partition, 1)

    # For sparsity control
    sparsify_hidden_layer_size_dict = getLayerSizeList(partition, threshold_layer_size, sparsify_coefficient)
    degree_dict = getNodeDegreeDict(partition)
    partition_mtx_dict = getPartitionMatricesList(sparsify_hidden_layer_size_dict, degree_dict, feature_meta, partition)

    # split train and test set
    x_train, x_val, y_train, y_val = train_test_split(
            expression, target,test_size = 0.3,random_state = 1)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(random_seed)
    else:
        device = torch.device('cpu')

    # set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    net = Net(partition_mtx_dict, num_hidden_layer_neuron_list, drop_out).to(device)
    warnings.filterwarnings("ignore", category=UserWarning)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)

    loss_func = torch.nn.CrossEntropyLoss()

    x = (torch.from_numpy(x_train)).type(torch.FloatTensor)
    y = (torch.from_numpy(y_train)).type(torch.LongTensor)
    x = Variable(x).to(device)
    y = Variable(y).to(device)

    torch_dataset = Data.TensorDataset(x, y) 

    loader = Data.DataLoader(
        dataset=torch_dataset,      
        batch_size=batch_size,      
        shuffle=True,  
        num_workers=0   
    )

    acc_train = []
    acc_val = []
    acc_val_i = 0
    accuracy = 0
    best_test = 0
        
    # save the result in a new folder
    if not os.path.exists("res"):
        os.mkdir("res")

    for epoch in range(num_epoch): 
        if ((acc_val_i >0.9) &(accuracy>0.9)):
            break
        net.train()
        for step, (x_batch, y_batch) in enumerate(loader):  # 每个训练步骤
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            prediction = net(x_batch)
            loss = loss_func(prediction,y_batch)

            loss.backward()

            optimizer.step()

        with torch.no_grad():
            net.eval()
            prediction = net(x)
            pred_y = prediction.cpu().data.numpy().squeeze()
            target_y = y.cpu().data.numpy()

            accuracy = sum(target_y ==np.argmax(pred_y, axis=1))/len(target_y)
            acc_train.append(accuracy)

            val_input_tensor = (torch.from_numpy(x_val)).type(torch.FloatTensor).to(device)
            out_probs = net(val_input_tensor).cpu().data.numpy().squeeze()
            out_classes = np.argmax(out_probs, axis=1)
            acc_val_i = sum(out_classes == y_val) / len(y_val)
            acc_val.append(sum(out_classes == y_val) / len(y_val))
            if acc_val_i > best_test:
                best_test = acc_val_i
                corr_train = accuracy
                torch.save(net, 'res/model'+str(random_seed)+'.pkl')
    
    # Downstream analysis
    params = []
    for parameters in net.parameters():
        params.append(parameters.cpu().detach().numpy())

    h0 = params[0].T
    h1 = params[2].T
    h2 = params[4].T

    meta_left = abs(h1).sum(axis = 0)
    meta_right = abs(h2).sum(axis = 1)

    meta_imp = meta_right/meta_right.sum() + meta_left/meta_left.sum()

    degree = np.sum(partition,0)+1
    count_feature = np.sum(feature_meta, 0)+1
    f_left = np.sum(abs(h0),0)/count_feature**0.5
    f_right = np.sum(abs(h1),1)/degree**0.5
    f_imp = (f_left/f_left.sum()) + (f_right/f_right.sum())

    n_feature = expression.shape[1]

    # use the h0 multiply by both of meta_imp and f_imp
    max_idx = np.argmax(abs(h0) / f_imp / meta_imp, axis=1)
    Link = abs(h0)/f_imp/meta_imp

    h0_new = np.zeros(h0.shape)
    h0_new[range(n_feature), max_idx] = h0[range(n_feature), max_idx]
    h0_new[h0_new!=0] = 1
    feature_imp = np.dot(h0_new, f_imp)

    file_name = "res/meta_imp.csv"
    if not(os.path.exists(file_name)):
        with open(file_name,"a") as csvfile:
            writer = csv.writer(csvfile)
    with open((file_name),"a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(f_imp))

    file_name = "res/feature_imp.csv"
    if not(os.path.exists(file_name)):
        with open(file_name,"a") as csvfile:
            writer = csv.writer(csvfile)
    with open((file_name),"a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(feature_imp))

    file_name = "res/link.txt"
    if not(os.path.exists(file_name)):
        np.savetxt("res/link.txt",Link)
    else:
        old=np.loadtxt("res/link.txt", delimiter=' ')
        new = old + Link
        np.savetxt("res/link.txt",Link)