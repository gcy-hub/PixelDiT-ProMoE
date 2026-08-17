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

For the unprocessed ImageNet parquet shards, convert only the training shards
to a new directory. The source directory is read only; the script prints
progress, speed, and ETA.

~~~bash
python c2i/convert_imagenet_parquet_to_h5.py \
  --parquet-dir ImageNet-1k/data \
  --output-dir ImageNet-1k/imagenet256_h5 \
  --workers 32
~~~

After an interruption, run the same command with `--resume`.

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

## Inference

Generate requested ImageNet classes from a checkpoint. `--config` must be the
same ProMoE YAML used for that checkpoint. The JSON dictionary maps class IDs
in `[0, 999]` to output counts; the script uses EMA weights by default.

~~~bash
python c2i/infer_c2i.py \
  --checkpoint c2i/train_logs/exp_pixdit_promoe_xl_imagenet256/last.ckpt \
  --config c2i/configs/pix256_promoe_xl.yaml \
  --gpu-ids 0,1,2,3 \
  --class-counts '{"0": 4, "207": 2}' \
  --output-dir c2i/samples/promoe_xl
~~~

The output contains `class_0000/`, `class_0207/`, and `manifest.json`. Use
`--num-steps`, `--guidance`, or `--batch-size` to override the YAML sampler
settings. `--gpu-ids` starts one inference process per GPU and shows total
progress, speed, and ETA. GPU IDs are indexed within `CUDA_VISIBLE_DEVICES`
when it is set.
