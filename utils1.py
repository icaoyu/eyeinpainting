import os
import errno
import numpy as np
import scipy
import scipy.misc
import json
import dlib
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_image(image_path , image_size, is_crop= True, resize_w= 64, is_grayscale= False, is_test=False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w, is_test=is_test)

def transform(image, npx = 64 , is_crop=False, resize_w=64, is_test=False):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w=resize_w, is_test=is_test)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64, is_test=False):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    if not is_test:
        rate = np.random.uniform(0, 1, size=1)
        if rate < 0.5:
            x = np.fliplr(x)
    # return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
    #                            [resize_w, resize_w])
    return scipy.misc.imresize(x[20:218 - 20, 0: 178], [resize_w, resize_w])

def save_images(images, size, image_path, is_ouput=False):
    return imsave(inverse_transform(images, is_ouput), size, image_path)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    size = [int(x) for x in size]
    if size[0] + size[1] == 2:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image, is_ouput=False):

    if is_ouput == True:
        print(image[0])
    result = ((image + 1) * 127.5).astype(np.uint8)
    if is_ouput == True:
        print(result)
    return result

log_interval = 1000

def read_image_list_for_Eyes(category):

    json_cat = category + "/data.json"
    with open(json_cat, 'r') as f:
        data = json.load(f)

    all_iden_info = []
    all_ref_info = []

    test_all_iden_info = []
    test_all_ref_info = []

    #c: id
    #k: name of identity
    #v: details.
    '''
     "kerri-verna": [
     {"filename": "kerri-verna-3.jpg", 
         "eye_left": {"x": 73, "y": 80}, 
         "box_left": {"w": 61, "h": 48}, 
         "eye_right": {"x": 183, "y": 96}, 
         "box_right": {"w": 76, "h": 60}, 
         "opened": 0.9564253091812134, 
         "closed": 0.05000000074505806},
     {}}
    '''

    for c, (k, v) in enumerate(data.items()):

        identity_info = []

        is_close = False
        is_close_id = 0

        if c % log_interval == 0:
            print('Processed {}/{}'.format(c, len(data)))

        if len(v) < 2:
            continue
        for i in range(len(v)):

            if is_close or v[i]['opened'] is None or v[i]['opened'] < 0.60:
                is_close = True
            if v[i]['opened'] and v[i]['opened'] < 0.60:
                is_close_id = i

            str_info = str(v[i]['filename']) + "_"

            if 'eye_left' in v[i] and v[i]['eye_left'] != None:
                str_info += str(v[i]['eye_left']['y']) + "_"
                str_info += str(v[i]['eye_left']['x']) + "_"
            else:
                str_info += str(0) + "_"
                str_info += str(0) + "_"

            if 'box_left' in v[i] and v[i]['box_left'] != None:
                str_info += str(v[i]['box_left']['h']) + "_"
                str_info += str(v[i]['box_left']['w']) + "_"
            else:
                str_info += str(0) + "_"
                str_info += str(0) + "_"

            if 'eye_right' in v[i] and v[i]['eye_right'] != None:
                str_info += str(v[i]['eye_right']['y']) + "_"
                str_info += str(v[i]['eye_right']['x']) + "_"
            else:
                str_info += str(0) + "_"
                str_info += str(0) + "_"

            if 'box_right' in v[i] and v[i]['box_right'] != None:
                str_info += str(v[i]['box_right']['h']) + "_"
                str_info += str(v[i]['box_right']['w'])
            else:
                str_info += str(0) + "_"
                str_info += str(0)

            identity_info.append(str_info)

        if is_close == False:

            for j in range(len(v)):

                first_n = np.random.randint(0, len(v), size=1)[0]
                all_iden_info.append(identity_info[first_n])
                middle_value = identity_info[first_n]
                identity_info.remove(middle_value)

                second_n = np.random.randint(0, len(v) - 1, size=1)[0]
                all_ref_info.append(identity_info[second_n])

                identity_info.append(middle_value)

        else:

            #append twice with different reference result.

            middle_value = identity_info[is_close_id]
            test_all_iden_info.append(middle_value)
            identity_info.remove(middle_value)

            second_n = np.random.randint(0, len(v) - 1, size=1)[0]
            test_all_ref_info.append(identity_info[second_n])

            test_all_iden_info.append(middle_value)

            second_n = np.random.randint(0, len(v) - 1, size=1)[0]
            test_all_ref_info.append(identity_info[second_n])

    assert len(all_iden_info) == len(all_ref_info)
    assert len(test_all_iden_info) == len(test_all_ref_info)

    print("train_data", len(all_iden_info))
    print("test_data", len(test_all_iden_info))

    return all_iden_info, all_ref_info, test_all_iden_info, test_all_ref_info

class Eyes(object):

    def __init__(self, image_path,detector,predictor):
        self.dataname = "Eyes"
        self.image_size = 256
        self.channel = 3
        self.detector = detector
        self.predictor = predictor
        self.image_path = image_path
        self.dims = self.image_size*self.image_size
        self.shape = [self.image_size, self.image_size, self.channel]
        self.train_images_name, self.train_eye_pos_name, self.train_ref_images_name, self.train_ref_pos_name, \
            self.test_images_name, self.test_eye_pos_name, self.test_ref_images_name, self.test_ref_pos_name = self.load_Eyes(image_path)

    def load_Eyes(self, image_dir):
        train_images_name = []
        train_eye_pos_name = []
        train_ref_images_name = []
        train_ref_pos_name = []

        test_images_name = []
        test_eye_pos_name = []
        test_ref_images_name = []
        test_ref_pos_name = []
        img_paths = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(image_dir)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        assert len(img_paths) > 2, 'number of files is %s,smaller than 2 ' % len(img_paths)
        l=len(img_paths)
        for i in range(l):
            train_images_name.append(img_paths[i])
            s_idx = random.randrange(l)
            train_ref_images_name.append(img_paths[s_idx])


        assert len(train_images_name) == len(train_ref_images_name)
        assert len(test_images_name) == len(test_ref_images_name)

        return train_images_name, train_eye_pos_name, train_ref_images_name, train_ref_pos_name, \
               test_images_name, test_eye_pos_name, test_ref_images_name, test_ref_pos_name

    def getShapeForData(self, filenames, is_test=False):

        array = [get_image(batch_file, 108, is_crop=False, resize_w=256,
                           is_grayscale=False, is_test=is_test) for batch_file in filenames]
        sample_images = np.array(array)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.train_images_name) // batch_size
        train_pose_list = []
        ref_pose_list=[]
        train_name_list = []
        ref_name_list=[]
        if batch_num // ro_num == 0 and is_shuffle:

            length = len(self.train_images_name)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.train_images_name = np.array(self.train_images_name)
            self.train_images_name = self.train_images_name[perm]

            self.train_ref_images_name = np.array(self.train_ref_images_name)
            self.train_ref_images_name = self.train_ref_images_name[perm]

            train_name_list = self.train_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]
            for i in range(len(train_name_list)):
                pose = self.get_eye_pose(train_name_list[i], self.detector, self.predictor)
                if type(pose) == type(None):
                    for j in range(len(train_name_list)):
                        pose = self.get_eye_pose(train_name_list[j], self.detector, self.predictor)
                        if type(pose) != type(None):
                            break
                train_pose_list.append(pose)
            ref_name_list = self.train_ref_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]
            for i in range(len(ref_name_list)):
                pose = self.get_eye_pose(ref_name_list[i], self.detector, self.predictor)
                if type(pose) == type(None):
                    for j in range(len(ref_name_list)):
                        pose = self.get_eye_pose(ref_name_list[j], self.detector, self.predictor)
                        if type(pose) != type(None):
                            break
                ref_pose_list.append(pose)
        # print(len(train_name_list),len(train_pose_list),len(ref_pose_list))
        # assert len(train_name_list)==len(ref_name_list),'batch not equal'
        return np.array(train_name_list),np.array(train_pose_list),np.array(ref_name_list),np.array(ref_pose_list)

    def getTestNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.test_images_name) // batch_size
        if batch_num == 0 and is_shuffle:

            length = len(self.test_images_name)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.test_images_name = np.array(self.test_images_name)
            self.test_images_name = self.test_images_name[perm]

            self.test_eye_pos_name = np.array(self.test_eye_pos_name)
            self.test_eye_pos_name = self.test_eye_pos_name[perm]

            self.test_ref_images_name = np.array(self.test_ref_images_name)
            self.test_ref_images_name = self.test_ref_images_name[perm]

            self.test_ref_pos_name = np.array(self.test_ref_pos_name)
            self.test_ref_pos_name = self.test_ref_pos_name[perm]

        return self.test_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_eye_pos_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_ref_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_ref_pos_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]


    def getTestData(self,img_path,eximg_path,detector, predictor):

        img_eye_pos = self.get_eye_pose(img_path, detector, predictor)
        ex_eye_pos = self.get_eye_pose(eximg_path, detector, predictor)

        return np.array([img_path]),np.array([img_eye_pos]),np.array([eximg_path]),np.array([ex_eye_pos])
    # def getTestNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):


    def get_eye_pose(self,img_file, detector, predictor):
        try:
            img = dlib.load_rgb_image(img_file)
        except:
            print("Error: 没有找到文件或读取文件失败",img_file)
            return
        dets = detector(img, 1)
        if len(dets) < 0:
            print("no face landmark detected")
            return
        else:
            try:
                shape = predictor(img, dets[0])
            except:
                return
            #         points = np.empty([68, 2], dtype=int)
            #         for b in range(68):
            #             points[b, 0] = shape.part(b).x
            #             points[b, 1] = shape.part(b).y

            landmarks = shape.parts()
            image_size = img.shape
            mask = np.full(image_size[0:2] + (1,), 1, dtype=np.float32)

            left_eye_x = int((landmarks[39].x + landmarks[36].x) / 2)
            left_eye_y = int((landmarks[37].y + landmarks[38].y + landmarks[41].y + landmarks[42].y) / 4)
            left_eye_w = int(landmarks[39].x - landmarks[36].x)
            left_eye_h = int(landmarks[40].y - landmarks[38].y)
            right_eye_x = int((landmarks[42].x + landmarks[45].x) / 2)
            right_eye_y = int((landmarks[43].y + landmarks[44].y + landmarks[47].y + landmarks[46].y) / 4)
            right_eye_w = int(landmarks[45].x - landmarks[42].x)
            right_eye_h = int(landmarks[46].y - landmarks[44].y)
            current_eye_pos = [left_eye_y,left_eye_x, left_eye_h, left_eye_w,right_eye_y,right_eye_x,right_eye_h,
                               right_eye_w]
            return current_eye_pos
