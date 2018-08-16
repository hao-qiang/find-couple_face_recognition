# Find couple

Find couple in the photo and make the girl up.

![](https://github.com/hao-qiang/find_couple/blob/master/result.png)

## Requirements

- python 3.6
- opencv 3.4.1
- dilb 19.15
- face_recognition 1.2.2

## Getting Started

### 1. Preparing your data

Two facial images of couple should be put into folder `./data/couple` , boy's image need to be named as `name` (like 'Tony.jpg') and girl's image should be named as `name_` (like 'Pepper_.jpg'), and other people's facial images named as `name` should be put into folder `./data/single` . The `face_recognition` library will extract facial features from these pictures.

### 2. Run

If you want to use PC camera to do real-time face recognition, run

```
python find_couple_camera.py
```

If you want to recognize faces from one photo (like `photo.jpg` ), you should save the photo at script path and run

```
python find_couple_image.py photo.jpg
```

## Happy Chinese Valentine's Day!
