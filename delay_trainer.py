""" delay trainer for gluon"""


def set_delay_trainer(trainer, delay):
    '''
    Parameters:
    -----------
    trainer: mxnet.gluon.Trainer
        gluon trainer
    delay: int
        the times to accumulate gradients

    Examples:
    ---------
        set_delay_trainer(trainer, delay=4)
    '''
    assert delay
    old_init_optimizer = trainer._init_optimizer

    def _init_optimizer(self, optimizer, optimizer_params):
        old_init_optimizer(optimizer, optimizer_params)
        self._updaters =\
            [DelayOptimizer(opt, delay) for opt in self._updaters]
    trainer._init_optimizer = _init_optimizer
    trainer._updaters =\
        [DelayOptimizer(opt, delay) for opt in trainer._updaters]


class DelayOptimizer:
    def __init__(self, opt, delay):
        self.opt = opt
        self.delay = delay
        self.grads = dict()
        attrs = ['sync_state_context', 'set_states', 'get_states']
        for attr in attrs:
            self.__dict__[attr] = lambda self, *args, **kwargs:\
                    getattr(self.opt, attr)(*args, **kwargs)

    def __call__(self, index, grad, weight):
        if index not in self.grads:
            grad_info = [grad, 1]
            self.grads[index] = grad_info
        else:
            grad_info = self.grads[index]
            grad_info[0] += grad
            grad_info[1] += 1
        if grad_info[1] >= self.delay:
            self.opt(index, grad_info[0], weight)
            del self.grads[index]
