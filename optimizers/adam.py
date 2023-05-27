class Adam:
    def __init__(self,
                 params,
                 betas = (0.01, 0.02),
                 lr=0.03,
                 optimized_update=True, eps= 1e-16):
        self.params = params
        self.lr = lr
        self.optimized_update = optimized_update
        self.eps = eps
        
    def init_state(self, 
                   state: Dict[str, any],
                   param: nn.Parameter):
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(param)
        state['exp_avg_sq'] = torch.zeros_like(param)
        return state
    
    def get_mv(self,
               state: Dict[str, any],
               group: Dict[str, any],
               grad: torch.Tensor):
        
        beta1, beta2 = group['betas']
        m, v = state['exp_avg'], state['exp_avg_sq']
        # inplace operation
        m.mul_(beta1).add_(grad, alpha = 1-beta1)
        v.mul_(beta2).add_(grad, alpha = 1-beta2)
        return m, v
    
    def get_lr(self, group:Dict[str, any]):
        return group['lr']
    
    def adam_update(self, 
                    state: Dict[str, any],
                    group: Dict[str, any],
                    param: nn.Parameter,
                    m: torch.Tensor,
                    v: torch.Tensor):
        beta1, beta2 = group['betas']
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        lr = self.get_lr(group)
        print(f'param, m, v: {param, m, v}')
        if self.optimized_update:
            denominator = v.sqrt().add_(group['eps'])
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            ## inplace operation
            param.data.addcdiv_(m, denominator, value = -step_size)
#             param.data = param.data - step_size * (m/denominator)
            print( f'm/denominator, step_size: {step_size * (m/denominator)}')
#             print(f'param.data.addcdiv(m, denominator, value = -step_size):')
            
        else:
            denominator = (v.sqrt() / torch.sqrt(bias_correction2)).add_(group['eps'])
            step_size = lr / bias_correction1
            print( f'm/denominator: {m/denominator}')
            param.data.addcdiv_(m, denominator, value=-step_size)
        print(step_size)
        return param
        
    def step_param(self, state, group,grad, param):
#         grad = self.weight_decay(param, grad, group)
        m, v = self.get_mv(state, group, grad)
        state['step'] += 1
        p = self.adam_update(state, group, param, m, v)
#         print(p is None)
        return p

f_in = 2
f_out = 3
params =  torch.nn.Parameter(torch.randn(f_in, f_out)) # weights
y = torch.sum((params * 10 + 0.01))
y.backward()
grad =  params.grad# gradients

opt = Adam(
    params,
    betas = (0.01, 0.02),
    lr=0.03,
    optimized_update=True, eps= 1e-16)   

# state = {'step': 0,
#         'exp_avg': torch.tensor([0], dtype=torch.float, requires_grad=False),
#         'exp_avg_sq': torch.tensor([0], dtype=torch.float, requires_grad=False)}

state = opt.init_state(state={}, param=params)

print(params)
group = {'lr': 0.1,
         'betas': (0.01, 0.02),
         'eps': 1e-16}

opt.step_param(state, group, grad, param = params)
# print(params)