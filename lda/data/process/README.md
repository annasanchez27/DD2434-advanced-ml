# Data processing
The files in this folder are used to prepare data. They're not normally imported - instead, they are run every time there's an update in the data format.

This is done for performance reasons.

To prepare the original data, do:
```sh
python -m lda.data.process.reuters
```
