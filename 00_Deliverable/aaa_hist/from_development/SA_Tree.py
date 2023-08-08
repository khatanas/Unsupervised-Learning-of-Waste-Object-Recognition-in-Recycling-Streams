from h_SA import IoU
import numpy as np

class Tree():
    # for quick check of existing nodes
    existing_idx = []
    # node storage
    nodes = {}
    
    def __init__(self,idx):
        self.idx = idx
        self.height = 0
        self.parent = -1
        self.children = []
        Tree.existing_idx.append(idx)
        
    def get(idx):
        '''load node from storage'''
        return Tree.nodes[idx] if Tree.exists(idx) else print(f'Node {idx} does not exist')
    
    def exists(idx):
        return True if idx in Tree.existing_idx else False
    
    def setChild(self,child):
        '''add child_idx node as child to another node'''
        if child not in self.children and Tree.get(child).parent == -1:
            #assert Tree.get(child).parent == -1, 'Cannot have two parents (sad)'
        # manage new height of nodes
            join_height = self.height
            child_height = Tree.get(child).height
            # head_idx of tree to be updated
            head_idx = child if child_height<join_height else self.getHead()
            to_be_updated = [head_idx]+Tree.get(head_idx).getChildren()
            delta = (join_height-1-child_height) if child_height<join_height else (child_height+1-join_height)
            for i in to_be_updated:
                Tree.get(i).height += delta
        # add child
            self.children.append(child)
        # assign parent
            Tree.get(child).parent = self.idx
        
    def getChildren(self):
        '''returns all nodes of lower level'''
        tmp = [i for i in self.children]
        for i in tmp:
            tmp+=Tree.get(i).children
        return tmp
    
    def isHead(self):
        return True if self.parent==-1 else False
    
    def getHead(self):
        '''returns tree head idx'''  
        curr = Tree.get(self.idx)
        while not curr.isHead():
            curr = Tree.get(curr.parent)
        return curr.idx
    
    def getAncestors(self):
        '''returns a list of parent, great-parent,.... up to head'''
        curr = Tree.get(self.idx)
        ancestors = []
        while not curr.isHead():
            curr = Tree.get(curr.parent)
            ancestors.append(curr.idx)
        return ancestors
    
    def isComposed(self,masks):
        '''True if area(children) == area(parent) else False'''
        merged = np.zeros_like(masks[self.idx]['segmentation'])
        for child in self.children:
            merged = np.logical_or(merged,masks[child]['segmentation'])
        
        #little trick to use IoU function
        merged = {'segmentation':merged}
        self.composed=True if IoU(masks[self.idx],merged)>0.95 else False
        
    def make(idx):
        '''init node'''  
        if not Tree.exists(idx): Tree.nodes[idx] = Tree(idx)
    
    def updateHeight(self):
        '''update height when two trees are linked to one another'''
        assert self.idx == self.getHead(), 'Is not head'
        leafs = [i for i in self.getChildren() if len(Tree.get(i).children)==0]
        min_height = 0 if not leafs else min([Tree.get(i).height for i in leafs])
        members = [self.idx] + self.getChildren()
        for i in members:
            Tree.get(i).height -= min_height
    
    def clean(self):
        '''correct height, parents and children before to delete a node'''
        for child in self.children:
            tmp = Tree.get(child)
            tmp.parent = -1
            tmp.updateHeight()
        if self.parent>=0: 
            Tree.get(self.parent).children.remove(self.idx)
            Tree.get(self.getHead()).updateHeight()
                
    def cut(idx):
        '''delete a node'''
        if Tree.exists(idx):
            Tree.get(idx).clean()
            del Tree.nodes[idx]
            Tree.existing_idx.remove(idx)
        else: print(f'Node {idx} does not exist')
        
    def heads():
        '''returns all heads of existing trees'''
        return [Tree.get(i).idx for i in Tree.existing_idx if Tree.get(i).isHead()]
        
    def reset():
        '''delete all nodes'''
        Tree.existing_idx=[]
        Tree.nodes={}
        
    def info(self):
        print('index: {0}\n'
            'height: {1}\n'
            'parent: {2}\n'
            'siblings: {6}\n'
            'direct children {3}\n'
            'head of tree: {4}\n'
            'all children: {5}'.format(
                self.idx,
                self.height,
                self.parent,
                self.children,
                self.getHead(),
                self.getChildren(),
                -1 if self.parent == -1 else [i for i in Tree.get(self.parent).children if i != self.idx]))
        
        
'''def maskOverlay(masks, th_IoU=0.015):
    """
    Returns two lists of lists
    The first one is of len=len(masks). For each mask, a list of overlapping masks ordered by area i decreasing order is returned.
    ex: the i-th list refers to mask_i: the list could be [mask_a, mask_b, mask_i, mask_k] with area_a > area_b > area_i > area_u
    The second list is a subset of the first one. If the i-th list has mask_i as first element, then the i-th list is contained in the second list of lists
    """
    are_overlapping = []
    for idx_ref,mask_ref in enumerate(masks):
        mask_ref = mask_ref['segmentation']
        overlapping_masks = [idx_ref]
        
        for idx_tested,mask_tested in enumerate(masks):
            if idx_ref==idx_tested:continue
            mask_tested = mask_tested['segmentation']
            if np.logical_and(mask_ref,mask_tested).any():
                if IoU(masks[idx_ref],masks[idx_tested])>th_IoU: overlapping_masks.append(idx_tested)
        
        are_overlapping.append(sorted(overlapping_masks, key=lambda i: masks[i]["area"], reverse=True))
    heads_only = [i for idx,i in enumerate(are_overlapping) if i[0]==idx]
    
    return are_overlapping, heads_only'''


def nodesOnly(list_):
    '''return True if {all nodes in list_ exist} else False'''
    return all([Tree.exists(i) for i in list_])

def aintNodes(list_):
    '''return list of item form list_ that are not a node of a tree'''
    return [i for i in list_ if not Tree.exists(i)]

def buildTree(list_):
    '''
    build a tree form a given list
    '''
    # init head
    pre = list_[0]
    Tree.make(pre)
    
    # init child and link to parent
    for i in list_[1:]:
        Tree.make(i)
        Tree.get(pre).setChild(i)
        pre = i
        
def buildForest(masks,are_overlapping):
    '''
    build all trees from are_overlapping
    '''
    Tree.reset()
    for list_ in are_overlapping:
        while not nodesOnly(list_):
            smallest = aintNodes(list_)[-1]
            buildTree(are_overlapping[smallest])
        else: continue
    for i in Tree.existing_idx:
        Tree.get(i).isComposed(masks)