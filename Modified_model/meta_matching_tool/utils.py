import pandas as pd
import numpy as np
import igraph as ig
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import os
import zipfile

# Windows
# package_dir = "E:\\SPARSENN\\Modified_model\\meta_matching_tool"
# Macos
package_dir = "/Users/watertank/Desktop/SPARSENN/Modified_model/meta_matching_tool"
# package_dir = os.path.abspath(os.path.dirname(__file__))


zip_path = os.path.join(package_dir, 'data', 'kegg.txt.zip')
output_dir = "data/"

# Check if the zip file exists
if os.path.exists(zip_path):
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the files
        zip_ref.extractall(output_dir)
    os.remove(zip_path)

#################### Function for data pre-processing ####################

# define a function to remove rows with more than 75% of zeros
def remove_rows(dat, data, thres = 0.75):
    """
    dat contains mz and time and other info
    data has shape (n_features, n_samples)
    """
    rowsum = np.sum(data==0,1)
    new_dat = dat.iloc[np.where((rowsum < thres * data.shape[1])==True)[0],:]
    return new_dat

# define a function to find the potential KEGGID for each feature
def find_keggid(dat, kegg_sub, this_adductlist, match_tol_ppm=9):
    """
    dat contains mz and time and other info
    kegg_sub is a subset of kegg that contains only the compounds that are in the graph
    """
    this_kegg = kegg_sub[kegg_sub['Adduct'].isin(this_adductlist)]

    dic = {}
    for i in range(dat.shape[0]):
        # If the mz matches that in the database, we claim that this is a match.
        idx = list(np.where(np.abs(this_kegg['mz']-dat['mz'].iloc[i])/(dat['mz'].iloc[i])<=match_tol_ppm/1e6)[0])
        
        # Get the corresponding KEGGID.
        dic[dat.index[i]] = list(this_kegg['KEGGID'].values[idx])
    return dic

# define a function to get the feature-metabolite matching matrix, adj matrix and feature data
def get_data(dic, new_dat, g):
    # Get all the features and the metabolites.
    features = [key for key, value in dic.items() if value!=[]]
    
    metabolites = np.unique(sum([value for key, value in dic.items() if value!=[]], []))

    
    # get feature data  
    data_anno_new = new_dat.loc[features,:]
    print("The shape of data:", data_anno_new.shape)

    # get feature-metabolite matching matrix
    matching = np.zeros([len(features), len(metabolites)])
    for ix,i in enumerate(features):
        idx = np.where(np.in1d(metabolites, dic[i]))[0]
        matching[ix, idx] = 1

    print("The shape of feature-metabolites matching:", matching.shape)

    # get adjacency matrix of metabolites
    subgraph = ig.Graph()

    # add the vertices from order_list to the subgraph
    for v in metabolites:
        if v in g.vs["name"]:
            subgraph.add_vertex(v)

    # add the edges that connect the vertices in the subgraph
    for e in g.es:
        source = e.source
        target = e.target
        if g.vs['name'][source] in metabolites and g.vs['name'][target] in metabolites:
            subgraph.add_edge(g.vs['name'][source], g.vs['name'][target])

    # Not quite clear why didn't use the original adj matrix but a new one. TODO: check this
    adj_new = np.array(subgraph.get_adjacency().data)
    g_sub = ig.Graph.Adjacency((adj_new > 0).tolist(), mode = "undirected")

    adj_matrix = np.array(g_sub.get_adjacency().data)
    print("The shape of metabolic network:", adj_matrix.shape)
    
    return(data_anno_new, matching, adj_matrix, metabolites)


def data_preprocessing(pos=None, neg=None, 
                       pos_adductlist=["M+H","M+NH4","M+Na","M+ACN+H","M+ACN+Na","M+2ACN+H","2M+H","2M+Na","2M+ACN+H"], 
                       neg_adductlist = ["M-H", "M-2H", "M-2H+Na", "M-2H+K", "M-2H+NH4", "M-H2O-H", "M-H+Cl", "M+Cl", "M+2Cl"], 
                       idx_feature = 4, match_tol_ppm=5, zero_threshold=0.75, log_transform=True, scale=1000):

    # Load data
    g = ig.Graph.Read_GraphML(os.path.join(package_dir, 'data', 'graph.graphhml'))
    all_compound = list(g.vs["name"])
    
    # Filter out the lines in the DB where **(kegg['mz']-kegg['AdductMass']==kegg['MonoisotopicMass'])**.
    kegg = pd.read_csv(os.path.join(package_dir, 'data', 'kegg.txt'), sep='\t')
    kegg['r'] = (kegg['mz']-kegg['AdductMass']==kegg['MonoisotopicMass'])
    kegg = kegg[kegg['r']==True]
    kegg_sub = kegg[kegg['KEGGID'].isin(all_compound)]

    if pos is not None:
        pos.columns.values[0] = 'mz'
        pos.columns.values[1] = 'time'
        pos.columns = pos.columns.str.replace('pos', '')
        pos.index = ['pos.' + str(i) for i in pos.index]

    if neg is not None:
        neg.columns.values[0] = 'mz'
        neg.columns.values[1] = 'time'
        neg.columns = neg.columns.str.replace('neg', '')
        neg.index = ['neg.' + str(i) for i in neg.index]

    # concatenate the two dataframes
    if pos is not None and neg is not None:
        dat = pd.concat([pos, neg], axis=0)
    elif pos is not None:
        dat = pos
    elif neg is not None:
        dat = neg
        
    # leave out those with very low expression rate
    new_dat = remove_rows(dat, dat.iloc[:,idx_feature:], thres = zero_threshold)

    # select only the compounds that are in the graph
     
    if pos is not None and neg is not None:
        dic_pos = find_keggid(new_dat.loc[new_dat.index.str.contains('pos')], kegg_sub, pos_adductlist)
        dic_neg = find_keggid(new_dat.loc[new_dat.index.str.contains('neg')], kegg_sub, neg_adductlist)

        dic = {**dic_pos, **dic_neg}
    elif pos is not None:
        dic = find_keggid(new_dat.loc[new_dat.index.str.contains('pos')], kegg_sub, pos_adductlist)
    elif neg is not None:
        dic = find_keggid(new_dat.loc[new_dat.index.str.contains('neg')], kegg_sub, neg_adductlist)

    data_annos, matchings, adj_matrices, metabolites = get_data(dic, new_dat, g)
    
    if log_transform:
        data_annos.iloc[:,idx_feature:] = np.log(data_annos.iloc[:,idx_feature:]+1)
        
    if scale:
        expression = data_annos.iloc[:,idx_feature:].T
        m_min = np.min(expression, 0)
        m_max = np.max(expression, 0)

        expression = ((expression - m_min)/(m_max - m_min)-0.5) * scale
        
        data_annos.iloc[:,idx_feature:] = expression.T

    return(data_annos, matchings, adj_matrices, metabolites)

###################### Function for main model ######################

def getLayerSizeList(partition, threshold_layer_size, sparsify_coefficient):
    """
    Obtain the size of each sparse layer
    
    INPUT:
    partition: the adjacent matrix of metabolic network
    threshold_layer_size: the threshold of sparese layer
    sparsify_coefficient: the coefficient of each sparse level
    
    OUTPUT:
    sparsify_hidden_layer_size_dict: a dictionary indicating the sparse layer
    """
    n_meta = np.shape(partition)[0]
    n_layer = math.floor(np.log10(1.0 * threshold_layer_size / n_meta) / np.log10(sparsify_coefficient)) + 3
    
    # dict for number of neurons in each layer
    sparsify_hidden_layer_size_dict = {}

    sparsify_hidden_layer_size_dict['n_hidden_0'] = int(n_meta)

    # How is this sparsing rate determined? TODO: check this
    for i in range(1,n_layer):
        sparsify_hidden_layer_size_dict['n_hidden_%d' % (i)] = int(n_meta * (sparsify_coefficient) ** (i-1))
    return sparsify_hidden_layer_size_dict


def getPartitionMatricesList(sparsify_hidden_layer_size_dict, degree_dict, feature_meta, partition):
    """
    Obtain the linkage matrix among two sparse layers
    """
    np.random.seed(1);  # for reproducable result
    g = ig.Graph.Adjacency((partition).tolist(), mode = "undirected")
    dist = np.array(g.shortest_paths()) # use the shortest distance matrix to assign links
    
    sum_remove_node_list = []  # keep note of which nodes are already removed
    
    partition_mtx_dict = {}
    residual_connection_dic = {}

    partition_mtx_dict["p0"] = feature_meta  # first matrix being the connection from features to meta
    partition_mtx_dict["p1"] = partition  # first matrix being the whole adjacency matrix

    # The code below adopted a seemingly very **stupid** way of determining the linkage. TODO: rewrite this
    for i in range(2, len(sparsify_hidden_layer_size_dict)):
        num_nodes_to_remove = sparsify_hidden_layer_size_dict["n_hidden_%d" % (i-1)] - \
                              sparsify_hidden_layer_size_dict["n_hidden_%d" % (i)]
        # sort node degree dict according to number of degrees
        sorted_node_degree_list = sorted(degree_dict.items(), key=lambda item: item[1])

        # Directly take the position of the nodes that are needed to be removed.
        temp_remove_list = []
        max_to_remove_node_degree = sorted_node_degree_list[num_nodes_to_remove - 1][1]
        
        # any node with degree less than `max_to_remove_node_degree` is certain to be removed
        for j in range(num_nodes_to_remove):  
            if sorted_node_degree_list[j][1] < max_to_remove_node_degree:
                id_to_remove_node = sorted_node_degree_list[j][0]
                # print(sorted_node_degree_list[j])
                temp_remove_list.append(id_to_remove_node)
            else:
                break  # node with more degrees is not under consideration
        
        # sample from all nodes that have max_to_remove_node_degree to reach number of nodes to remove
        sample_list = []
        for j in range(len(temp_remove_list), len(sorted_node_degree_list)):
            if sorted_node_degree_list[j][1] == max_to_remove_node_degree:
                sample_list.append(sorted_node_degree_list[j])
            else:
                break  # node with more degrees is not under consideration
            
        # Very interesting way of determining connection...
        sample_idx_list = sorted(
            np.random.choice(len(sample_list), num_nodes_to_remove - len(temp_remove_list), replace=False))
        for idx in sample_idx_list:
            temp_remove_list.append(sample_list[idx][0])

        # sum up add nodes to be removed
        all_list = np.arange(partition.shape[0])
        previous_layer_list = [x for x in all_list if x not in sum_remove_node_list]
        temp_partition = np.delete(partition, sum_remove_node_list, axis=0)
        sum_remove_node_list += temp_remove_list
        temp_partition = np.delete(temp_partition, sum_remove_node_list, axis=1)
        next_layer_list = [x for x in all_list if x not in sum_remove_node_list]

        # Residual connection layer
        residual_location = [previous_layer_list.index(x) for x in next_layer_list]
        
        # assign each neuron at least one linkage
        # I believe this is a mistake...
        # for k in range(len(previous_layer_list)):
        #     if sum(dist[k,next_layer_list]==float("inf"))==len(next_layer_list):
        #         idx = np.random.choice(len(next_layer_list), 1, replace=False)
        #     else:
        #         idx = np.argsort(dist[k,next_layer_list], axis = -1)[0]
        #     temp_partition[k, idx] = 1
            
            
        # Alternative version
        for k in range(len(previous_layer_list)):
            pos = previous_layer_list[k]
            if sum(dist[pos,next_layer_list]==float("inf"))==len(next_layer_list):
                idx = np.random.choice(len(next_layer_list), 1, replace=False)
            else:
                idx = np.argsort(dist[pos,next_layer_list], axis = -1)[0]
            temp_partition[k, idx] = 1
        
        for j in range(len(temp_remove_list)):
            degree_dict.pop(temp_remove_list[j])
            
        # if i == len(sparsify_hidden_layer_size_dict) - 1:
        #     print(next_layer_list)

        partition_mtx_dict["p%d" % i] = temp_partition

        residual_connection_dic["p%d" % i] = residual_location

        print(residual_location)

    return partition_mtx_dict, residual_connection_dic


# This might not be used in my settings.
def getNodeDegreeDict(partition):
    """
    Obtain the node degree using the adjacent matrix of metabolic network
    """
    degree_dict = {}
    row, col = partition.shape
    for i in range(row):
        degree_dict[i] = -1  # decrease its own
        for j in range(0, col):
            if partition[i, j] == 1:
                degree_dict[i] += 1

    return degree_dict


## Functions for backward selection.
def getKeggidByIndex(raw_keggid, idxs, output_dir):
    match_dic = {}

    raise NotImplementedError
