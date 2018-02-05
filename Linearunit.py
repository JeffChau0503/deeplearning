from Perceptron import Perceptron


def f(x):  # 定义激活函数,f(x)=x
    return x


class Linearunit(Perceptron):
    def __init__(self, input_num):
        """初始化线性单元，设置输入参数的个数"""
        Perceptron.__init__(self, input_num, f)
