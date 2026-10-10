# PulmoVision AI - Data Directory

This directory stores the Chest X-ray dataset.
The canonical structure is:

```
data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

- Images are in standard formats (.jpeg, .jpg, .png).
- Preprocessing standardizes all inputs to 224x224x3 with DenseNet121 normalization.
