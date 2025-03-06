# Code repo for Complexformer

## Usage

1. Install Python 3.10. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. 


3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/Complexformer_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/Complexformer_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/Complexformer_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/Complexformer.sh
# classification
bash ./scripts/classification/Complexformer.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.
