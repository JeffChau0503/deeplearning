from Connections import Connections
from Connection import Connection
from Layer import Layer


#  Network对象，提供API。API是一个接口，负责一个程序和其他软件的沟通，本质是预先定义的函数。
class Network(object):
    def __init__(self, layers):
        """
        初始化一个全连接神经网络
        :param layers:一个二维数组，描述神经网络每层节点数
        """
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, iteration):
        """
        训练神经网络
        :param labels:数组，训练样本标签。每个元素是样本的标签。
        :param data_set:二维数组，训练样本特征。每个元素是一个样本特征
        :param rate:学习速度
        :param iteration: 迭代
        :return:
        """
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        """
        内部函数，用一个样本训练网络
        """
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        """
        内部函数，计算每个节点的delta
        """
        output_node = self.layers[-1].nodes
        for i in range(len(label)):
            output_node[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        """
        内部函数，更新每个连接权重
        """
        for layer in self.layers[:-1]:
            for node in layer.node:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self, label, sample):
        """
        获得网络下在一个样本下，每个连接的梯度
        :param label:样本标签
        :param sample:样本输入
        """
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        """
        根据输入的样本预测输出值
        :param sample:数组，样本的特征，也就是网络的输入向量
        """
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        """
        打印网络信息
        """
        for layer in self.layers:
            layer.dump()
