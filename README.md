# DIQA_CNN
PyTorch 0.4.1 implementation of the following paper:
[Le Kang, et al. "A DEEP LEARNING APPROACH TO DOCUMENT IMAGE QUALITY ASSESSMENT." 2014 ICIP.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.648.7747&rep=rep1&type=pdf)

The **SOC** dataset can be downloaded in [DIQA: Document Image Quality Assesment Datasets](https://lampsrv02.umiacs.umd.edu/projdb/project.php?id=73)

## Note
Download the dataset and put all images in a directory and set this directory as `root` in 'config.yaml'

The ground truth for the dataset has been pre-processed and saved as a excel file `SOC_gt.xlsx` stored in `./data/gt_files/SOC_gt.xlsx`

The ground truth file contains:
- img_name: the image name
- img_set: the index of reference image from which the current degraded image generated.
- acc_f: OCR accuracy by ABBYY Finereader
- acc_t: OCR accuracy by Tesseract
- acc_o: OCR accuracy by Omnipage
- acc_avg: average accuracy of the three OCR engines above

The creating details about this dataset:
- [A Dataset for Quality Assessment of Camera Captured Document Images](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.359.6995&rep=rep1&type=pdf)

## Training and validating
```
python main.py --batch_size=128 --epochs=500 --lr=0.001
```
Before training, the `root` in `config.yaml` must be specified.

## demo_DIQA
```
python demo_DIQA.py
```
When a DIQA model has been trained, `demo_DIQA.py` can be used to predict the quality of a document image directly.

Before running `demo_DIQA.py`, the `model_path` and `img_path` must be specified.

## Requirements
- PyTorch 0.4.1
- [pytorch/ignite](https://github.com/pytorch/ignite)
