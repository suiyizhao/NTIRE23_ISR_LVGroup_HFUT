# NTIRE Challenge 2023 Team LVGroup_HFUT

> This repository is the official [NTIRE Challenge 2023](https://cvlai.net/ntire/2023/) implementation of Team LVGroup_HFUT in [Image Shadow Removal](https://codalab.lisn.upsaclay.fr/competitions/10253).

> The [restoration results](https://drive.google.com/file/d/1euZTV9OXwq9uctXp14an7cKbHBlh7m8p/view?usp=share_link) of the tesing images and [pretrained model](https://drive.google.com/file/d/1T_yrvs7YjMSa5RRNp_gd1fXUJflrOJdK/view?usp=share_link) can be downloaded from Google Drive.

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
