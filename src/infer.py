import time
import torch

import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils import *
from options import TestOptions
from models import NAFNet
from datasets import SingleImgDataset

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

image_dir = opt.outputs_dir + '/' + opt.experiment + '/infer'
clean_dir(image_dir, delete=opt.save_image)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
print('inferring data loading...')
infer_dataset = SingleImgDataset(data_source=opt.data_source)
infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
print('successfully loading inferring pairs. =====> qty:{}'.format(len(infer_dataset)))

print('---------------------------------------- step 3/4 : model defining... ----------------------------------------------')
model = NAFNet().cuda()
print_para_num(model)

model = nn.DataParallel(model)
model.load_state_dict(torch.load(opt.model_path))
model = model.module
print('successfully loading pretrained model.')

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')   
def main():
    model.eval()
    
    psnr_meter = AverageMeter()
    time_meter = AverageMeter()
    
    for i, (img, path) in enumerate(infer_dataloader):
        img = img.cuda()

        with torch.no_grad():
            start_time = time.time()
            pred = model(img)
            times = time.time() - start_time

        pred_clip = torch.clamp(pred, 0, 1)

        time_meter.update(times, 1)

        print('Iteration: ' + str(i+1) + '/' + str(len(infer_dataset)) + '  Processing image... ' + str(path) + '  Time ' + str(times))
            
        if opt.save_image:
            save_image(pred_clip, image_dir + '/' + os.path.basename(path[0]))
            
    print('Avg time: ' + str(time_meter.average()))
        
if __name__ == '__main__':
    main()
    