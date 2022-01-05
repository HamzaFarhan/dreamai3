from .utils import *
from .dai_imports import *

class DaiDataset(Dataset):
    
    def __init__(self, data, data_path='', tfms=None, channels=3, img_idx=[0], label_idx=[1], class_names=None, **kwargs):
        super(DaiDataset, self).__init__()
        
        data_path = Path(data_path)
        if is_list(data):
            data = pd.DataFrame({'Images':data})
            label_idx = []
            
        if class_names is None:
            class_names = [list(np.unique(data.iloc[:, x])) for x in label_idx]
        if is_list(class_names) and not is_list(class_names[0]):
            class_names = [class_names]
        # assert len(class_names) == len(label_idx), f'{len(class_names)} != {len(label_idx)}'
        idx_to_class = [{i:x for i,x in enumerate(cn)} for cn in class_names]
        class_to_idx = [{v:k for k,v in ic.items()} for ic in idx_to_class]
        if channels == 3:
            self.img_reader = rgb_read
        elif channels == 1:
            self.img_reader = c1_read
             
        store_attr(self,','.join(dict_keys(locals_to_params(locals()))))
        
    def __len__(self):
        return len(self.data)
    
    def get_img_names(self, index):
        return [self.data.iloc[index, x] for x in self.img_idx]
    
    def get_img_paths(self, index):
        return [str(self.data_path/x) for x in self.get_img_names(index)]
        
    def get_imgs(self, index):
        return [self.img_reader(x) for x in self.get_img_paths(index)]
    
    def get_tensors(self, index, to_tensor=True):
        imgs = self.get_imgs(index)
        if self.tfms is not None:
            x = [apply_tfms(i.copy(), self.tfms) for i in imgs]
            if self.channels == 1:
                x = [i.unsqueeze(0) for i in x]
            if not to_tensor:
                x = [tensor_to_img(i) for i in x]
        else:
            x = imgs
        return x
    
    def get_labels(self, index, get_idx=True):
        if get_idx:
            return [cid[self.data.iloc[index, x]] for cid,x in zip(self.class_to_idx,self.label_idx)]
        return [self.data.iloc[index, x] for x in self.label_idx]

    def __getitem__(self, index, to_tensor=True, get_idx=True, get_names=False, get_paths=False):
        ret = {'images':self.get_tensors(index, to_tensor=to_tensor), 'labels':self.get_labels(index, get_idx=get_idx)}
        if get_names:
            ret['image_names'] = self.get_img_names(index)
        if get_paths:
            ret['image_paths'] = self.get_img_paths(index)
        return ret

    def denorm_data(self, data, keys=['images']):
        if self.tfms is not None:
            norm_t = get_norm(self.tfms)
            if norm_t:
                mean = norm_t.mean
                std = norm_t.std
                for k in keys:
                    data[k] = denorm_img(data[k], mean, std)

    def get_at_index(self, index, denorm=True, to_tensor=False, get_names=True, get_paths=True):

        data = self.__getitem__(index, to_tensor=to_tensor, get_idx=False, get_names=get_names, get_paths=get_paths)
        if denorm:
            self.denorm_data(data=data)
        return data