
class A:

    def __init__(self):
        self.apple = 33

    def f(self):
        # does not work fine .... does not display warning
        return self.banana

    def __getattribute__(self, item):
        return super(A, self).__getattribute__()
