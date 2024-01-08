# Synthetic Signature Bankcheck Images (SSBI) Dataset for Signature Detection and Verification

This dataset contains 2180 synthetic bank check images with:
- 1680 genuine signatures
- 500 forged signatures

The signatures are from 19 different signers. In addition to the signature bounding boxes, the dataset
also contains the bounding boxes for the date, amount, and payee fields.

![SSBI Dataset](screenshots/sample_details.png)

![SSBI Dataset](screenshots/samples.png)

## Getting Started

```
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
python create_dataset.py
```