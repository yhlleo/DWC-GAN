from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


def get_loader(image_dir, crop_size=178, image_size=128, batch_size=16, attr_path=None, 
    selected_attrs=None, dataset='CelebA', mode='train', num_workers=4):
    """Build and return a data loader."""
    transform = []
    if mode == 'train' and dataset != 'Clevr':
        transform.append(T.RandomHorizontalFlip())
    if dataset == 'CelebA':
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
    elif dataset == 'Clevr':
        transform.append(T.Resize((image_size, image_size)))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
    else:
        transform.append(T.Resize((image_size, image_size)))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)

    if dataset == 'CelebA':
        from data_ios.celeba_data import CelebA
        cur_dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'Clevr':
        from data_ios.clevr_data import CLEVR
        cur_dataset = CLEVR(image_dir, attr_path, mode, transform)

    data_loader = data.DataLoader(dataset=cur_dataset,
                                  batch_size=batch_size,
                                  shuffle=mode=='train',
                                  num_workers=num_workers)
    return data_loader
