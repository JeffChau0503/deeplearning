from numpy import *
from functools import reduce
import random
random.random()


# 实现一个输出恒为1的节点,用于计算偏置项w_b
class ConstNode(object):
    def __init__(self, layer_index, node_index):
        """
        :param layer_index:  # 节点所属层的编号
        :param node_index:  # 节点的编号
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
        self.delta = 0  # 在学习资料里这行没有。

    def append_downstream_connection(self, conn):
        """
        添加一个下游节点的连接
        """
        self.downstream.append(conn)

    def clac_hidden_layer_delta(self):
        """
        节点属于隐藏层时，计算delta
        """
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        """
        打印节点的信息
        """
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, ' ')
        return node_str, '\n\tdownstream:' + downstream_str
