class Perceptron(object):  # 感知器
    def __init__(self, input_num, activator):  # 输入个数，激活函数
        """
        初始化感知器
        :param input_num:输入参数的个数
        :param activator:激活函数的类型 double -> double
        """
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]  # 该数组被 0.0 填充
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        """
        打印学习到的权重，偏置项。
        """
        return 'weight\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        :param input_vec:输入向量
        :return:输出感知器的计算结果
        """
        #  把input_vec[x1,x2,x3...]和weight[w1,w2,w3...]打包一起 zip()
        #  变成[(x1,w1),(x2,w2),(x3,w3),...]
        #  然后利用map函数计算[x1*w1,x2*w2,x3*w3]
        #  最后利用reduce求和
        from functools import reduce  # 先把reduce从functools模块中调用出来
        return self.activator(
            reduce(lambda a, b: a+b,
                   [x_w[0] * x_w[1] for x_w in zip(input_vec, self.weights)], 0.0) + self.bias)
        # 0.0这里作为reduce的初始值

    def train(self, input_vecs, labels, iteration, rate):
        """
        输入训练数据：
        :param input_vecs:一组向量
        :param labels:每一个向量对应的标签
        :param iteration:训练轮数，即迭代次数
        :param rate:学习率
        :return:
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
            """
            一次迭代，把所有数据过一遍
            """
            #  把输入和输出打包在一起，成为样本的列表[(input_vec,label), ... ]
            #  而每个训练的样本是(input_vec,label)

            samples = list(zip(input_vecs, labels))

            #  对每个样本，按照感知器规则更新权重
            for (input_vec, label) in samples:
                #  计算感知器在当前权重下的输出
                output = self.predict(input_vec)
                #  更新权重
                self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        """
        按照感知器的规则更新权重
        """
        #  把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        #  变成[(x1,w1),(x2,w2),(x3,w3),...]
        #  然后利用感知器规则更新权重
        delta = label - output
        self.weights = [x_w1[1] + rate * delta * x_w1[0] for x_w1 in zip(input_vec, self.weights)]
        #  更新bias
        self.bias += rate * delta
