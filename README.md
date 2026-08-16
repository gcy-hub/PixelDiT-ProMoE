# PixelDiT-ProMoE C2I Training

## Dataset

Set data_dir in the selected YAML to a directory with this layout:

~~~text
ImageNet-1k/
|- images.h5
 - images_h5.json
~~~

images_h5.json is a JSON list of image keys stored in images.h5. Each image
key must end in .png and refer to PNG bytes, for example
00000/img00000000.png. The H5 file must also contain dataset.json, whose JSON
payload includes labels as [image_key, class_id] pairs. class_id must be in
[0, 999].

~~~json
{
  "labels": [
    ["00000/img00000000.png", 726],
    ["00000/img00000001.png", 3]
  ]
}
~~~

## Train

On node01, activate the pixel environment and run one selected model:

~~~bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash train_c2i.sh --num-gpus 4 --master-port 29501 --config configs/pix256_promoe_s.yaml
~~~

Replace the configuration for another scale:

~~~text
configs/pix256_promoe_b.yaml
configs/pix256_promoe_l.yaml
configs/pix256_promoe_xl.yaml
~~~

The configurations train from scratch (auto_resume: false), disable WandB,
validate once per epoch, and retain only the rolling last.ckpt checkpoint in
the corresponding c2i/train_logs/exp_*/ directory.

To resume explicitly:

~~~bash
bash train_c2i.sh --num-gpus 4 --master-port 29501 \
  --config configs/pix256_promoe_b.yaml \
  --ckpt-path train_logs/exp_pixdit_promoe_b_imagenet256/last.ckpt
~~~
