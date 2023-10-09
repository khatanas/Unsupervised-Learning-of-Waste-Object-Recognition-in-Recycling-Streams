
import numpy as np
import networkx as nx
from heapq import heappop, heappush

#**************************************************************************************************************
def collectEdges(index,X,k):
    """
    Builds a graph by searching for the k-NN of each embeddings
    Returns a list of edges [u,v,weight] where the weight is the L2 distance between u and v
    """
    # split embeddings array [belongs_to_array | do_not_belong]
    splitted = [X[:index.ntotal],X[index.ntotal:]]
    acc_I = []
    acc_D = []
    
    E = []
    for idx,part in enumerate(splitted):
        # search for k+1 if part belongs to index (first neighbor is self)
        D,I = index.search(part, k+1 if idx==0 else k)
        # keep k last element of the output
        D = D[:,-k:]
        I = I[:,-k:]
        acc_I.append(I)
        acc_D.append(D)
    
    # create edges: [u,v,weight]
    I = np.vstack(acc_I)
    D = np.vstack(acc_D)
    for u in range(I.shape[0]):
        vs = list(I[u])
        weights = list(D[u])
        edges = [[str(u),str(v),w] for v,w in zip (vs,weights)]
        E+=edges
    return E


def buildGraph(E):
    """
    Builds an undirected weighted graph using the {E} edges
    """
    # build undirected weighted graph
    G = nx.Graph()
    G.add_edges_from([(edge[0],edge[1],{'weight':edge[2]}) for edge in E])
    
    return G


def buildGraphCollection(index,X,k,verbose=False):
    """
    Builds a main {k}-NN graph using X. The latter can be disconnected
    The edges of the principal connected graph, the graph itself and the list of its SCC are always returned
    If verbose == True, the same information is returned for all other connected graphs
    Output: [list of list of edges | list of graphs | list of SCCs]
    """
    # collect edges
    E = collectEdges(index,X,k)
    
    nb_edge = len(E)
    nb_node = X.shape[0]
    
    # find disconnected graphs
    groups_of_nodes = [list(vertex) for vertex in nx.connected_components(buildGraph(E))]
    nb_graph = len(groups_of_nodes)
    
    if verbose: print(f'\nThe main graph is connected') if nb_graph==1 else print(f'\nThe main graph is made of {nb_graph} disconnected graphs:')
    
    # build a dictionary to assign a subgraph to each node (speed up edge filtering)
    belongs_to = {node:-1 for node in range(nb_node)}
    for graph_id in range(nb_graph):
        for node_id in groups_of_nodes[graph_id]:
            belongs_to[node_id] = graph_id
            
    # init list to store filtered vertices, subgraphs, and SCCs 
    subgraphs = [[] for _ in range(3)]
    for graph_id in range(nb_graph if verbose else 1):
        # filter edges
        subgraphs[0].append([E[edge_id] for edge_id in range(nb_edge) if belongs_to[E[edge_id][0]]==graph_id])
        # build unweighted directed subgraph
        subG = nx.DiGraph()
        subG.add_edges_from([(edge[0],edge[1]) for edge in subgraphs[0][graph_id]])
        subgraphs[1].append(subG)
        # find SCC of subgraph with cardinality>1
        subgraphs[2].append([nodes for nodes in list(nx.strongly_connected_components(subgraphs[1][graph_id])) if len(nodes)>1])
        
        flattened = [node for group in subgraphs[2][graph_id] for node in group]
        if verbose: print(f'Subgraph {graph_id}: {len(groups_of_nodes[graph_id])} vertices, with {len(subgraphs[2][graph_id])} SCC, totalizing {len(flattened)} vertices')
    return subgraphs


def shortestPath(G,src,target,forbidden_nodes=[]):
    """
    Returns the shortest path from src to target.
    The nodes listed in forbidden_nodes are never used to build the path
    """
    # graph nodes are of type str
    src = str(src)
    target = str(target)
    forbidden_nodes = [str(item) for item in forbidden_nodes]
    
    # init dict for distances and path retrieval
    parents = {node: -1 for node in G.nodes()}
    distances = {node: float('inf') for node in G.nodes()}
    distances[src] = 0
    
    # assess existence of target in graph
    if target not in distances.keys(): 
        #print(f'{target} not in G')
        return [[] for _ in range(2)]
    
    # init searching 
    visited = set()
    heap = [(0, src)]
    while heap:
        _,current_node = heappop(heap)
        if current_node == target: break
        if current_node not in visited:
            visited.add(current_node)
            # iterate over current node's neighbors
            for neighbor, attributes in G[current_node].items():
                # check if neighbor is not forbidden
                distance_step = float('inf') if neighbor in forbidden_nodes else attributes['weight']
                # update distance and parent if better path is found
                if distances[current_node] + distance_step < distances[neighbor]:
                    distances[neighbor] = distances[current_node] + distance_step
                    parents[neighbor] = current_node
                    heappush(heap,(distances[neighbor],neighbor))
                    
    if distances[target]<float('inf'):
        # retrieve path src ==> dest
        path = [target]
        while path[0]!=src: path = [parents[path[0]]] + path  
        # collect distances 
        dists = [distances[node] for node in path]
        
        return path,dists
    
    else:
        #print(f'{src} and {target} disconnected')
        return [[] for _ in range(2)]