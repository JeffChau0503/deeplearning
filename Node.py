from numpy import *
from functools import reduce
import random
random.random()


def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))


#  节点类，负责记录和维护节点自身信息以及这个节点相关的上下连接，实现输出值和误差项计算。
class Node(object):
    def __init__(self, layer_index, node_index):
        """
        构造节点对象
        :param layer_index:节点所属层的编号
        :param node_index: 节点的编号
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        """
        设置节点的输出值。如果节点属于输入层会用到这个函数
        """
        self.output = output

    def append_downstream_connection(self, conn):
        """
        添加一个到下游节点的连接
        """
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        """
        添加一个到下游节点的连接
        """
        self.upstream.append(conn)

    def calc_output(self):
        """
        根据 y = sigmoid（W·X)计算节点的输出
                                其中sigmoid(x) = 1/(1+e^(-x))
        """
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight,
                        self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        """
        节点属于隐藏层时，计算节点误差 delta = a_i(1-a_i) * [w_ki * delta_k]之和
        """
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_ouput_layer_delta(self, label):
        """
        节点属于输出层时，计算节点误差
        """
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        """
        打印节点信息
        """
        node_str = "%u-%u: output: %f delta: %f" % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str
