# Neural-Style-Transfer-pytorch

## A minimal pytorch implementation of Neural Style Transfer

### train.py

`python train.py --image_size 500`

| Arguement | Parse | Default |
| ----------- | ----------- | ----------- |
| Image size | --image_size | 256 |
| Steps | --steps | 6000 |
| Learning Rate | --learning_rate | 0.001 |
| Alpha | --alpha | 1 |
| Beta | --beta | 0.01 |
| Content Image | --content_root | images/content.jpg |
| Style Image | --style_root | images/style.jpg |



### Content Image

<img src="https://github.com/rutvij-25/Neural-Style-Transfer-pytorch/blob/main/images/content.jpg" width="500">

### Style Image

<img src="https://github.com/rutvij-25/Neural-Style-Transfer-pytorch/blob/main/images/style.jpg" width="500">

### Output Image

<img src="https://github.com/rutvij-25/Neural-Style-Transfer-pytorch/blob/main/images/generated.jpg" width="500">
