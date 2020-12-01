''' 
该方法的作用为若 num_nodes 参数中有值，则返回该值；若该参数无值，返回index中的最大值+1
 
index存储图中节点连接对，例如在Cora数据集中，index的维度为[2,10556]
max()返回输入tensor中所有元素的最大值；item()获得tensor中的值，加1是因为下标从0开始
'''
def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes
