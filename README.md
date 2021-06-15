# eyeinpainting
implementation of ExGAN in eye-inpainting, update the mask generation of xx works 

# Exemplar_GAN_Eye_Inpainting
The tensorflow implement of [Eye In-Painting with Exemplar Generative Adversarial Networks](https://arxiv.org/abs/1712.03999)  

- Just use refernece image as the exemplar, not code.
- Our model trained using 256x256 pixels, not 128x128 mentioned in the original paper.

## Dependencies
* [Python 3.6]
* [Tensorflow 1.4+](https://github.com/tensorflow/tensorflow)
* dlib 

## Usage
we change the dataset processing way,
- first: download face image dataset,celeba or ffhq or vggface ;download shape_predictor_68_face_landmarks.dat from web,and put under root dir
- second: unzip dataset into data dir,and crop face into 256*256 ,by 
```bash
python preprocess.py --root ./data/raw --output_dir ./data/crop
```
- train:
change path pram to ./data/crop
run:
```bash
python main.py --OPER_FLAG=0
```
-test:
```bash
python main.py --OPER_FLAG=1 --img_path ./data/celeb_id_aligned/emilio-rivera-3.jpg --ex_path ./data/celeb_id_aligned/emilio-rivera-4.jpg
```

