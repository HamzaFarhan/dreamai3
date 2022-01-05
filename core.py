from .data import *

def efficientnet_b0(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b0',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b2(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b2',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b4(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b4',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b5(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b5',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b6(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b6',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b7(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b7',
           num_classes=num_channels, in_channels=in_channels)

models_meta = {resnet34: {'cut': -2, 'conv_channels': 512},
               resnet50: {'cut': -2, 'conv_channels': 2048},
               resnet101: {'cut': -2, 'conv_channels': 2048},
               resnext50_32x4d: {'cut': -2, 'conv_channels': 2048},
               resnext101_32x8d: {'cut': -2, 'conv_channels': 2048},
               densenet121: {'cut': -1, 'conv_channels': 1024},
               densenet169: {'cut': -1, 'conv_channels': 1664},
               efficientnet_b0: {'cut': -5, 'conv_channels': 1280},
               efficientnet_b2: {'cut': -5, 'conv_channels': 1408},
               efficientnet_b4: {'cut': -5, 'conv_channels': 1792},
               efficientnet_b5: {'cut': -5, 'conv_channels': 2048},
               efficientnet_b6: {'cut': -5, 'conv_channels': 2304},
               efficientnet_b7: {'cut': -5, 'conv_channels': 2560}}

DEFAULTS = {'models_meta': models_meta, 'metrics': ['loss', 'accuracy', 'multi_accuracy'],
            'imagenet_stats': imagenet_stats, 'image_extensions': image_extensions}

class BodyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if isinstance(self.model, EfficientNet): # TODO : Implement more
            return self.model.extract_features(x)
        return self.model(x)

def create_body(arch, pretrained=True, cut=None, num_extra=3):
    model = arch(pretrained=pretrained)
    if isinstance(model, EfficientNet):
        body_model = BodyModel(model)
    else:
        if cut is None:
            ll = list(enumerate(model.children()))
            cut = next(i for i,o in reversed(ll) if has_pool_type(o))
        modules = list(model.children())[:cut]
        body_model = BodyModel(nn.Sequential(*modules))
    if num_extra > 0:
        channels = models_meta[arch]['conv_channels']
        extra_convs = [conv_block(channels, channels)]*num_extra
        extra_model = nn.Sequential(*extra_convs)
        body_model = nn.Sequential(body_model, extra_model)
    else:
        body_model = nn.Sequential(body_model)
    return body_model

class HeadModel(nn.Module):
    def __init__(self, pool, linear):
        super().__init__()
        store_attr(self, 'pool,linear')
    def forward(self, x, meta=None):
        if meta is None:
            return self.linear(self.pool(x))  
        return self.linear(torch.cat([self.pool(x), meta], dim=1))

class MultiHeadModel(nn.Module):
    def __init__(self, head_list):
        super().__init__()
        self.head_list = head_list
    def forward(self, x, meta=None):
        return [h(x, meta) for h in self.head_list]

def create_head(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True,
                bn_final=False, lin_first=False, y_range=None, actv=None,
                relu_fn=nn.ReLU(inplace=True), trial=None, num_lin_ftrs=None, n_lin_ftrs=None, trial_num_lin_ftrs=[1,3],
                trial_n_lin_ftrs=[256,512,1024]):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    if trial is not None:
        lin_ftrs = [nf]
        num_lin_ftrs = trial.suggest_int("num_lin_ftrs",
                                         trial_num_lin_ftrs[0],
                                         trial_num_lin_ftrs[1])
        for i in range(num_lin_ftrs):
            n_lin_ftrs = trial.suggest_categorical(f"n_lin_ftrs_{i}", trial_n_lin_ftrs)
            # n_lin_ftrs = trial.suggest_categorical(f"n_lin_ftrs_{i}", [1024,1224,1424,1624,1824,2000])
            lin_ftrs.append(n_lin_ftrs)
        lin_ftrs.append(n_out)
    elif num_lin_ftrs is not None and n_lin_ftrs is not None:
        lin_ftrs = [nf]
        for i in range(num_lin_ftrs):
            lin_ftrs.append(n_lin_ftrs[i])
        lin_ftrs.append(n_out)
    else:
        lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    ps = [ps]
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [relu_fn] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    pool_layers = nn.Sequential(*[pool, Flatten()])
    layers = []
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += LinBnDrop(ni, no, bn=True, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if actv is not None:
        layers.append(actv)
    layers = nn.Sequential(*layers)
    return HeadModel(pool=pool_layers, linear=layers)

def create_model(arch, num_classes, num_extra=3, meta_len=0, body_out_mult=1,
                 relu_fn=nn.ReLU(inplace=True), actv=None, pretrained=True,
                 only_body=False, state_dict=None, strict_load=True, trial=None, num_lin_ftrs=None, n_lin_ftrs=None,
                 trial_num_lin_ftrs=[1,3], trial_n_lin_ftrs=[256,512,1024]):
    meta = models_meta[arch]
    body = create_body(arch, pretrained=pretrained, cut=meta['cut'], num_extra=num_extra)
    if only_body:
        return body
    if is_iterable(num_classes):
        heads = []
        for nc in num_classes:
            heads.append(create_head(nf=((meta['conv_channels']*2)*body_out_mult)+meta_len, n_out=nc, relu_fn=relu_fn, actv=actv, trial=trial, num_lin_ftrs=num_lin_ftrs, n_lin_ftrs=n_lin_ftrs, trial_num_lin_ftrs=trial_num_lin_ftrs, trial_n_lin_ftrs=trial_n_lin_ftrs)) 
        head = MultiHeadModel(nn.ModuleList(heads))
    else:
        head = create_head(nf=((meta['conv_channels']*2)*body_out_mult)+meta_len, n_out=num_classes, relu_fn=relu_fn, actv=actv, trial=trial, num_lin_ftrs=num_lin_ftrs, n_lin_ftrs=n_lin_ftrs, trial_num_lin_ftrs=trial_num_lin_ftrs, trial_n_lin_ftrs=trial_n_lin_ftrs)
    net = nn.Sequential(body, head)
    load_state_dict(net, sd=state_dict, strict=strict_load)
    return net

def model_splitter(model, cut_percentage=0.2, only_body=False):
    
    if not is_sequential(model):
        p = params(model)
        cut = int(len(p)*(1-cut_percentage))
        ret1 = p[:cut]
        ret2 = p[cut:]
        return ret1,ret2

    elif len(model) > 2:
        p = params(model)
        cut = int(len(p)*(1-cut_percentage))
        ret1 = p[:cut]
        ret2 = p[cut:]
        return ret1,ret2
        
    if not only_body:
        ret1, ret2 = params(model[0]), params(model[1])            
        p = params(model[0][0])
        cut = int(len(p)*(1-cut_percentage))
        ret1 = p[:cut]
        ret2 = p[cut:] + params(model[1])
        if len(model[0]) > 1:
            # print('yeesdsssss')
            ret2 += params(model[0][1])
        return ret1, ret2

    if cut_percentage == 0.:
        print("Must pass a cut percentage in the case of 'only_body'. Setting it to 0.2.")
        cut_percentage = 0.2
    p = params(model[0])
    cut = int(len(p)*(1-cut_percentage))
    ret1 = p[:cut]
    ret2 = p[cut:]
    if len(model) > 1:
        ret2 += params(model[1])
    return ret1,ret2

class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, loss):
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1-pt)**self.gamma * loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def focal_loss_(loss, alpha=1, gamma=2, reduce=True):
    pt = torch.exp(-loss)
    F_loss = alpha * (1-pt)**gamma * loss
    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss

class Printer(nn.Module):
    def forward(self,x):
        print(x.size())
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
               relu=True, bn=True, dropout=True, dropout_p=0.2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if relu: layers.append(nn.ReLU(True))
    if bn: layers.append(nn.BatchNorm2d(out_channels))
    if dropout: layers.append(nn.Dropout2d(dropout_p))
    return nn.Sequential(*layers) 

def cnn_input(o,k,s,p):
    return ((o*s)-s)+k-(2*p) 

def cnn_output(w,k,s,p):
    return np.floor(((w-k+(2*p))/s))+1

def cnn_stride(w,o,k,p):
    return np.floor((w-k+(2*p))/(o-1))

def cnn_padding(w,o,k,s):
    return np.floor((((o*s)-s)-w+k)/2 )

class BasicModel(nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head
        self.model = nn.Sequential(body, head)
    def forward(self, x):
        return self.model(x)
    def split_params(self):
        return self.body.parameters(), self.head.parameters()