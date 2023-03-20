# NTIRE Challenge 2023 Team LVGroup_HFUT

> This repository is the official [NTIRE Challenge 2023](https://cvlai.net/ntire/2023/) implementation of Team LVGroup_HFUT in [Image Shadow Removal](https://codalab.lisn.upsaclay.fr/competitions/10253).

> The [restoration results](https://pan.baidu.com/s/1klcUzBUyWXxZ3eHXGestUg) (Extraction Code：a9zs) of the tesing images and [pretrained model](https://pan.baidu.com/s/12hQGGC6IwQ-GhKvw7tqNUQ) (Extraction Code：l9zt) can be downloaded from Baidu Netdisk.

## Usage

### Single image inference

`cd your/script/path`

`python infer.py --data_source your/dataset/path --model_path ../pretrained/epoch_0090.pth --save_image --experiment your-experiment-name`

### Train

`cd your/script/path`

`python train.py --data_source your/dataset/path --experiment your-experiment`

### Dataset format

> The format of the dataset should meet the following code in datasets.py:

`img_paths_train = sorted(glob.glob(data_source + '/train' + '/input' + '/*.*'))`

`gt_paths_train = sorted(glob.glob(data_source + '/train' + '/gt' + '/*.*'))`

> or

`self.img_paths = sorted(glob.glob(data_source + '/test' + '/input' + '/*.*'))`

***data_source*** is given by the command line.

### Path to saving results

***when training and validating:***  the default path is `'../results/your-experiment'`

***when testing:***  the default path is `'../outputs/your-experiment/test'`

***when inferring:***  the default path is `'../outputs/your-experiment/infer'`
