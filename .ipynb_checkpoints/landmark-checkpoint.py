import menpo.io as mio
from menpofit.aam import HolisticAAM
from menpo.feature import igo
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
import pickle
from menpodetect import load_opencv_frontal_face_detector
from menpodetect.detect import detect
from menpo.image import Image
import cv2
import numpy as np
import matplotlib as plt

def train_300W():
    path_to_images = './300W/Both'
    def process(image, crop_proportion=0.2, max_diagonal=400):
        if image.n_channels == 3:
            image = image.as_greyscale()
        # image = image.crop_to_landmarks_proportion(crop_proportion)
        # d = image.diagonal()
        # if d > max_diagonal:
        #     image = image.rescale(float(max_diagonal) / d)
        return image
    training_images = mio.import_images(path_to_images, verbose=True)
    training_images = training_images.map(process)

    aam = HolisticAAM(training_images, reference_shape=None,
                    diagonal=150, scales=(0.5, 1.0),
                    holistic_features=igo, verbose=True)
    print(aam)
    with open('aam.pickle', 'wb') as handle:
        pickle.dump(aam, handle, protocol=pickle.HIGHEST_PROTOCOL)
def fit_landmark():
    aam = pickle.load(open('./aam.pickle', 'rb'))
    fitter = LucasKanadeAAMFitter(aam,
                                lk_algorithm_cls=WibergInverseCompositional,
                                n_shape=[3, 20], n_appearance=[30, 150])
    image = np.transpose(cv2.imread("./300W/Both/indoor_003.png"), axes=[2, 0, 1])/255.
    image = Image(image).as_greyscale(mode="luminosity") 
    opencv_detector = load_opencv_frontal_face_detector()
    bounding_box = opencv_detector(image)[0] #Assume only one face
    result = fitter.fit_from_bb(image, bounding_box, max_iters=20, gt_shape=None,
                                return_costs=False)
    result.view(render_initial_shape=True)
fit_landmark()