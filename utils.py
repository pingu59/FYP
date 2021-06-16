from typing import Sequence
import menpo.io as mio
from menpofit.aam import HolisticAAM, PatchAAM
from menpo.feature import igo, fast_dsift
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
import pickle
from menpodetect import load_dlib_frontal_face_detector
from menpodetect.detect import detect
from menpo.image import Image
import cv2
import numpy as np
import matplotlib as plt
from menpofit.sdm import SupervisedDescentFitter, NonParametricNewton
from menpo.landmark import face_ibug_68_to_face_ibug_51_trimesh
from menpofit.io import load_fitter
from PIL import Image as pil_image
import argparse
import math
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from lxml.etree import Element, SubElement, Comment, tostring
import lxml.etree as etree
import menpo3d
import random
from scipy import interpolate
from scipy.sparse.linalg import svds
import pickle
from scipy.signal import convolve
from numpy import linalg
import pyheif
import trimesh
from plyfile import PlyData, PlyElement


# Camera specifications

#pinhole camera insintrict of ipad pro 11-inch 2018 (see the terminal output of the swift program)
fx = 2872.9004 
fy = 2872.9004 
ox = 1538.0475 
oy = 1154.8513 

pixel_per_meter = 10296.3 
ipad_height_pixel = 3088.
ipad_width_pixel = 2316.
depth_width = 480
depth_height = 640

scale = 1000
coeff = ipad_height_pixel/depth_height

facial_midpoints = np.array([27, 28, 29, 30, 33, 51, 62, 66, 57, 8])
#in the ibug68 definition, starting from zero, from top to down
facial_leftpoints = [22, 23, 24, 25, 26, 34, 35, 52, 53, 54, 55, 56, 63, 64, 65] + list(range(9, 17)) #42, 43, 44, 45, 46, 47
facial_rightpoints = [17, 18, 19, 20, 21, 3, 31, 32, 48, 49, 50, 58, 59, 60, 61, 67] + list(range(8)) #36, 37, 38, 39, 40, 41


#Functions for finding landmarks
def process(image, crop_proportion=0.3, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    #connect pointcloud and change ibug 68 to ibug 51 for the model
    image.landmarks['face_ibug_51_trimesh'] = face_ibug_68_to_face_ibug_51_trimesh(image.landmarks['PTS']) 
    return image

    
def fit_landmark(pic):
    fitter = load_fitter('balanced_frontal_face_aam')
    image = np.transpose(get_heic(pic), axes=[2, 0, 1])
    image = Image(image).as_greyscale(mode="luminosity")
    detector = load_dlib_frontal_face_detector()
    bounding_box = detector(image)[0] #Assume only one face
    bounding_box = image.landmarks['dlib_0']
    result = fitter.fit_from_bb(image, bounding_box, max_iters=[20, 30], gt_shape=None,
                                return_costs=False)
    return result

# Functions for creating .obj, modified from https://github.com/CapObvios/Depth-Map-Visualizer
def create_mtl(mtlPath, matName, texturePath):
    if max(mtlPath.find('\\'), mtlPath.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    with open(mtlPath, "w") as f:
        f.write("newmtl " + matName + "\n"      )
        f.write("Ns 10.0000\n"                  )
        f.write("d 1.0000\n"                    )
        f.write("Tr 0.0000\n"                   )
        f.write("illum 2\n"                     )
        f.write("Ka 1.000 1.000 1.000\n"        )
        f.write("Kd 1.000 1.000 1.000\n"        )
        f.write("Ks 0.000 0.000 0.000\n"        )
        f.write("map_Ka " + texturePath + "\n"  )
        f.write("map_Kd " + texturePath + "\n"  )

def vete(v, vt):
    return str(v)+"/"+str(vt)

def calc_surface_normal(v1, v2, v3):
    return np.cross((v2 - v1), (v3 - v1))
    
def create_obj(depth, objPath, mtlPath, matName, mask, useMaterial = True, up_side_down = False, no_vt= False):
    img = depth
    h = depth_height
    w = depth_width 
    
    FOV = math.pi/4
    D = (img.shape[0]/2)/math.tan(FOV/2)

    if max(objPath.find('\\'), objPath.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    
    with open(objPath,"w") as f:    
        if useMaterial and not no_vt:
            f.write("mtllib " + mtlPath.split('/')[-1] + "\n")
            f.write("usemtl " + matName + "\n")

        ids = np.zeros((h, w), int)
        vid = 1
        heat_map = np.zeros((h, w))
        location = np.zeros((h, w, 3), dtype=np.float64)
        location_no_smooth = np.zeros((h, w, 3), dtype=np.float64)
        normal = np.zeros((h, w, 3))
        if up_side_down:
            for u in range(h):
                for v in range(w):
                    d = 1./ img[h - 1 - u, ]
                    heat_map[u, v] = d
        else:
            for u in range(h):
                for v in range(w):
                    d =  1./ img[u, -v]
                    heat_map[u, v] = d
        #Apply gaussian smooth:
        heat_map_no_smooth = heat_map
        heat_map = gaussian_filter(heat_map.copy(), sigma=4)
        mask_small = cv2.resize(mask.copy().astype('float32'), dsize=(depth_width, depth_height))
        for v in range(w):
            for u in range(h):
                d = heat_map[u, v]
                if mask_small[u, v] > 0:
                    ids[u, v] = vid
                vid += 1
                # calibrate using camera intrinsics
                z = D / coeff
                y = (u - (oy/coeff)) * z / (fy/coeff)
                x = (v - (ox/coeff)) * z / (fx/coeff)
                norm = 1 / math.sqrt(x*x + y*y + z*z)
                t = d/(z*norm)
                x = -t*x*norm
                y = -t*y*norm
                z = -t*z*norm    
                location[u, v] = np.array([x, y, z])
                f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n") 
        for v in range(w):
            for u in range(h):
                d = heat_map_no_smooth[u, v]
                # calibrate using camera intrinsics
                z = D / coeff
                y = (u - (oy/coeff)) * z / (fy/coeff)
                x = (v - (ox/coeff)) * z / (fx/coeff)
                norm = 1 / math.sqrt(x*x + y*y + z*z)
                t = d/(z*norm)
                x = -t*x*norm
                y = -t*y*norm
                z = -t*z*norm    
                location_no_smooth[u, v] = np.array([x, y, z])
        if not no_vt:
            for v in range(w):
                for u in range(h):
                    f.write("vt " + str(v/w) + " " + str((h - u - 1)/h) + "\n")
        for v in range(w - 1):
            for u in range(h - 1):
                v1 = ids[u, v]; v2 = ids[u+1, v]; v3 = ids[u, v+1]; v4 = ids[u+1, v+1]
                l1 = location[u, v]; l2 = location[u+1, v]; l3 = location[u, v+1]; l4 = location[u+1, v+1]
                if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                    continue
                if no_vt:
                    f.write("f " + str(v2) + " " + str(v1) + " " + str(v3) + "\n")
                    f.write("f " + str(v2) + " " + str(v3) + " " + str(v4) + "\n")
                else:
                    f.write("f " + vete(v2,v2) + " " + vete(v1,v1) + " " + vete(v3,v3) + "\n")
                    f.write("f " + vete(v2,v2) + " " + vete(v3,v3) + " " + vete(v4,v4) + "\n")
                normal[u, v] = (calc_surface_normal(l2, l1, l3) + calc_surface_normal(l2, l3, l4))/2
                normal[u, v] /= np.linalg.norm(normal[u, v])
                normal[u, v, 0] = - normal[u, v, 0]
        return location, normal, ids, location_no_smooth
    
def create_obj_with_params(texturePath, mtlPath, depth, objPath, mask, matName, up_side_down=False):
    useMat = texturePath != ''
    if useMat:
        create_mtl(mtlPath, matName, texturePath)
    return create_obj(depth, objPath, mtlPath, matName, mask, useMat, up_side_down = up_side_down)

def create_picked_points(pickedPointsPath, ibug, objPath, location):
    picked_points = Element('PickedPoints')
    document_data = SubElement(picked_points, 'DocumentData')
    data_file_name = SubElement(document_data, 'DataFileName',
                                 {'name':objPath.split('/')[-1]})
    template_name = SubElement(document_data, 'templateName',
                                 {'name':''})
    points = np.zeros((ibug.shape[0], 3))
    for i in range(ibug.shape[0]):
        pos_a, pos_b = ibug[i]
        points[i] = location[min(int(pos_a/coeff), depth_height - 1), min(int(pos_b/coeff), depth_width - 1)]
        x, y, z = points[i]
        SubElement(picked_points, 'point', {'name':str(i+1), 'active':str(1), 'x': str(x), 'y': str(y), 'z': str(z)})
    picked_points_xml = etree.tostring(picked_points, encoding="UTF-8",
                         xml_declaration=True,
                         pretty_print=True,
                         doctype='<!DOCTYPE PickedPoints>').decode("utf-8") 
    f = open(pickedPointsPath, "w")
    f.write(picked_points_xml)
    f.close()
    return points

def landmark_one_object(texturePath, mtlPath, depth, objPath, mask, 
                    pickedPointsPath, matName, up_side_down=False):
    location, normal, ids, location_no_smooth = create_obj_with_params(texturePath, mtlPath, depth, objPath, mask,
                                                    matName, up_side_down)
    landmarks = fit_landmark(texturePath)
    ibug = landmarks.final_shape.points
    points = create_picked_points(pickedPointsPath, ibug, objPath, location_no_smooth)
    # mesh = trimesh.load_mesh(objPath)
    # trimesh.smoothing.filter_humphrey(mesh, iterations=20)
    # faces = mesh.faces
    # face_normals = mesh.face_normals
    # trimesh.geometry.mean_vertex_normals(mesh.vertices.shape[0], faces, face_normals)
    # vertex_normals = mesh.vertex_normals
    # vertices = mesh.vertices
    # normals_smoothed = np.zeros((ids.shape[0], ids.shape[1], 3))
    # for i in range(len(vertices)):
    #    x, y, _ = vertices[i]
    #    pixel_x = int(x /pixel_per_meter)
    #    pixel_y = int(y /pixel_per_meter)
    #    normals_smoothed[pixel_x, pixel_y] = vertex_normals[i]
    # plt.figure()
    # plt.imshow(normals_smoothed)
    # plt.show()
    normals_smoothed = cv2.GaussianBlur(normal,(9, 9),0)
    return landmarks, points, ids, location, ibug, normals_smoothed, location_no_smooth

def find_affine(points_from, points_to, use_ibug68 = False, points_toleft=True):
    valid_points_from = []
    valid_points_to = []
    if use_ibug68:
        start = 0
    else:
        start = 17
    if points_toleft:
        allowed_landmark = facial_leftpoints + facial_midpoints.tolist()
    else:
        allowed_landmark = facial_rightpoints + facial_midpoints.tolist()
    
    for i in range(start, points_from.shape[0]): #some points can be outside of the face area hence have location (0, 0, 0)
        if i in allowed_landmark and points_from[i].sum() != 0 and points_to[i].sum() != 0:
            valid_points_from.append(points_from[i])
            valid_points_to.append(points_to[i])
    p_from = np.stack(valid_points_from)
    p_to = np.stack(valid_points_to)
    #Append a column of 1 to the end of p_from
    p_from_ = np.ones((p_from.shape[0], p_from.shape[1] + 1))
    p_from_[:, :-1] = p_from
    p_to_ = np.ones((p_from.shape[0], p_from.shape[1] + 1))
    p_to_[:, :-1] = p_to
    #p_to_ = A * p_from_
    #A = p_to_ * p_from_^(-1)
    A,resid,rank,sing = np.linalg.lstsq(p_from_, p_to_) #solving for least square
    return A.T

def merge_mtl(mtlPath, matNames, texturePaths):
    if max(mtlPath.find('\\'), mtlPath.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    with open(mtlPath, "w") as f:
        for i in range(len(matNames)):
            f.write("newmtl " + matNames[i] + "\n"      )
            f.write("Ns 10.0000\n"                  )
            f.write("d 1.0000\n"                    )
            f.write("Tr 0.0000\n"                   )
            f.write("illum 2\n"                     )
            f.write("Ka 1.000 1.000 1.000\n"        )
            f.write("Kd 1.000 1.000 1.000\n"        )
            f.write("Ks 0.000 0.000 0.000\n"        )
            f.write("map_Ka " + texturePaths[i] + "\n"  )
            f.write("map_Kd " + texturePaths[i] + "\n"  )

def merge_object(out, mtlPath, pointss, locations, idss, matNames, texturePaths, orientation):
    # all objects affine to the first object
    h = depth_height
    w = depth_width
    num_objects = len(idss)
    ls = np.ones((num_objects, h, w, 4))
    len_ids = w * h
    for i in range(num_objects):
        ls[i][:, :, :3] = locations[i][:h, :w]
    ls_affine = np.zeros((num_objects, h, w, 4)) 
    for v in range(w):
        for u in range(h):
            ls_affine[0, u, v] = ls[0, u, v]
    for i in range(1, num_objects):
        A = find_affine(pointss[i], pointss[0], use_ibug68 = False, points_toleft=orientation[i-1])
        for v in range(w):
            for u in range(h):
                ls_affine[i, u, v] = A @ls[i, u, v]
    merge_mtl(mtlPath, matNames, texturePaths)
    if max(out.find('\\'), out.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    with open(out,"w") as f:   
        f.write("mtllib " + mtlPath.split('/')[-1] + "\n")
        for i in range(num_objects):
            for v in range(w):
                for u in range(h):
                    x, y, z, _ = ls_affine[i, u, v]
                    f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")
            f.write("usemtl " + matNames[i] + "\n")
            for v in range(w):
                for u in range(h):
                    f.write("vt " + str(v/w) + " " + str((h - u - 1)/h)+ "\n")
            for v in range(w - 1):
                for u in range(h - 1):
                    v1 = idss[i][u, v]; v2 = idss[i][u+1, v]; v3 = idss[i][u, v+1]; v4 = idss[i][u+1, v+1]
                    if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                        continue
                    v1 += len_ids * i
                    v2 += len_ids * i
                    v3 += len_ids * i
                    v4 += len_ids * i
                    f.write("f " + vete(v2,v2) + " " + vete(v1,v1) + " " + vete(v3,v3) + "\n")
                    f.write("f " + vete(v2,v2) + " " + vete(v3,v3) + " " + vete(v4,v4) + "\n")
        return ls_affine
    
def affine_object(p_from, p_to, location_from, out, ids, orientation):
    A = find_affine(p_from, p_to, use_ibug68 = False, points_toleft=orientation)
    h = depth_height
    w = depth_width #img.shape[0]
    l1 = np.ones((h, w, 4))
    l1[:, :, :3] = location_from[:h, :w]
    l1_affine = np.zeros(l1.shape) 
    for v in range(w):
        for u in range(h):
            l1_affine[u, v] = A@l1[u, v]
    with open(out,"w") as f:   
        for v in range(w):
            for u in range(h):
                x, y, z, _ = l1_affine[u, v]
                f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")
        for v in range(w):
            for u in range(h):
                f.write("vt " + str(v/w) + " " +  str((h - u - 1)/h)+ "\n")    
        for v in range(w - 1):
            for u in range(h - 1):
                v1 = ids[u, v]; v2 = ids[u+1, v]; v3 = ids[u, v+1]; v4 = ids[u+1, v+1]
                if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                    continue
                f.write("f " + vete(v2,v2) + " " + vete(v1,v1) + " " + vete(v3,v3) + "\n")
                f.write("f " + vete(v2,v2) + " " + vete(v3,v3) + " " + vete(v4,v4) + "\n")
    return l1_affine

def mask_half_face(ibug, ids, to_left=False):
    mid_points = np.zeros((facial_midpoints.shape[0] + 2, 2))
    mid_points[1:-1]= ibug[facial_midpoints]
    mid_points[0, 1] = mid_points[1, 1]
    mid_points[-1, 0] = ipad_height_pixel
    mid_points[-1, 1] = mid_points[-2, 1]
    mid_points[:, 0] = mid_points[:, 0] / ipad_height_pixel * depth_height
    mid_points[:, 1] = mid_points[:, 1] / ipad_width_pixel * depth_width
    f = interpolate.interp1d(mid_points[:, 0], mid_points[:, 1])
    if to_left:
        for i in range(depth_height):
            split = int(f(i))
            for j in range(split):
                ids[i, j] = 0
    else:
        for i in range(depth_height):
            split = int(f(i))
            for j in range(split, depth_width):
                ids[i, j] = 0 
    return ids

def get_input(input_dir, color="L"): # L-intensity RGB-different channels
    ims = ()
    sequence = ['right', 'up', 'left', 'down']
    for i in range(len(sequence)):
        img = input_dir[sequence[i]].copy()
        if color == "R":
            imarray = img[:, :, 0]
        elif color == "G":
            imarray = img[:, :, 1]
        elif color == "B":
            imarray = img[:, :, 2]
        elif color == "L":
            #from pil library
            imarray = (img[:, :, 0] * 299./1000.) + (img[:, :, 1] * 587./1000.) + (img[:, :, 2] * 114./1000.)
        else:
            assert False
        imx = imarray.shape[0]
        imy = imarray.shape[1]
        I = np.reshape(imarray, (imx * imy, 1),order="C")
        ims += (I,)
    return np.column_stack(ims), (imx, imy)

# https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def surface_normal_correction(I, size, mask, normal_depth, show_detail = True):
    h, w = size
    print(I.shape)
    u, _, _ = svds(I.astype(float), k=3)    
    surface_normal = u.reshape(h, w, 3, order="C")
    albedo = np.linalg.norm(surface_normal, axis = -1)
    surface_normal *= np.stack([mask, mask, mask], axis = 2).astype(float)
    surface_normal /= (albedo.reshape(h, w, 1) + 1e-10) #for numeric stability
    normal_depth_print = ((normal_depth + 1) / 2).astype(np.float32)

    if show_detail:
        plt.figure()
        plt.imshow(normal_depth_print)
        plt.imsave("./svd/depth.png", normal_depth_print, cmap = "gray")
        plt.figure()
        plt.imshow(normal_depth_print[:, :, 0], cmap = "Greys")
        plt.imsave("./svd/depth_x.png", normal_depth_print[:, :, 0], cmap = "gray")
        plt.title("2-D image for Normal X - depth")
        plt.figure()
        plt.imshow(normal_depth_print[:, :, 1], cmap = "Greys")
        plt.imsave("./svd/depth_y.png", normal_depth_print[:, :, 1], cmap = "gray")
        plt.title("2-D image for Normal Y - depth")
        plt.figure()
        plt.imshow(normal_depth_print[:, :, 2], cmap = "Greys")
        plt.imsave("./svd/depth_z.png", normal_depth_print[:, :, 2], cmap = "gray")
        plt.title("2-D image for Normal Z - depth")
        plt.show()

    surface_normal_print = (surface_normal + 1)/ 2
    surface_normal_print = surface_normal_print.astype(np.float32)
    if show_detail:
        plt.figure()
        plt.imshow(surface_normal_print)
        plt.title("2-D image for Normal")
        plt.imsave("./svd/svd.png", surface_normal_print, cmap = "gray")
        plt.figure()
        plt.imshow(surface_normal_print[:, :, 0], cmap = "Greys")
        plt.title("2-D image for Normal X")
        plt.imsave("./svd/svd_x.png", surface_normal_print[:, :, 0], cmap = "gray")
        plt.figure()
        plt.imshow(surface_normal_print[:, :, 1], cmap = "Greys")
        plt.imsave("./svd/svd_y.png", surface_normal_print[:, :, 1], cmap = "gray")
        plt.title("2-D image for Normal Y")
        plt.figure()
        plt.imshow(surface_normal_print[:, :, 2], cmap = "Greys")
        plt.imsave("./svd/svd_z.png", surface_normal_print[:, :, 2], cmap = "gray")
        plt.title("2-D image for Normal Z")
        plt.show()
    
    #finding u in the standard x, y, z basis:
    nd = cv2.resize(normal_depth.copy(), dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    nd = nd.reshape(h * w, 3)
    sf = surface_normal.reshape(h * w, 3).transpose()
    svd_result_base = np.zeros((3,3))
    for i in range(3):
        result, _, _, _ = np.linalg.lstsq(nd, sf[i], rcond=None)
        svd_result_base[i] = result / np.linalg.norm(result)
    normal_corrected = np.linalg.inv(svd_result_base).dot(sf).T.reshape(h, w, 3)
    # average_depth = nd.mean(axis=0)
    # average_normal = normal_corrected.mean(axis=0)
    # normal_corrected = normal_corrected.reshape(h, w, 3)
    # r = rotation_matrix_from_vectors(average_normal, average_depth)
    # print(r)
    # normal_corrected = np.einsum('ijk, kk->ijk', normal_corrected, r)
    normal_corrected /= np.linalg.norm(normal_corrected, axis=-1).reshape(normal_corrected.shape[0], normal_corrected.shape[1], 1)
    normal_corrected *= mask.reshape(mask.shape[0], mask.shape[1], 1)
    normal_corrected_print = replaceNan(np.clip((normal_corrected + 1) / 2, a_min = 0, a_max=1).astype(np.float32))

    print(normal_corrected_print.shape)
    
    plt.figure()
    plt.imshow(normal_corrected_print, cmap = "Greys")
    plt.imsave("./svd/c_svd.png", normal_corrected_print, cmap = "gray")
    plt.imsave("./svd/c_svd_x.png", normal_corrected_print[:, :, 0], cmap = "gray")
    plt.imsave("./svd/c_svd_y.png", normal_corrected_print[:, :, 1], cmap = "gray")
    plt.imsave("./svd/c_svd_z.png", normal_corrected_print[:, :, 2], cmap = "gray")
    plt.title("2-D image for Normal - corrected")

    if show_detail:
        plt.figure()
        plt.imshow(normal_corrected_print[:, :, 0], cmap = "Greys")
        plt.imsave("./svd/c_svd_x.png", normal_corrected_print[:, :, 0], cmap = "gray")
        plt.title("2-D image for Normal X  - corrected")
        plt.figure()
        plt.imshow(normal_corrected_print[:, :, 1], cmap = "Greys")
        plt.imsave("./svd/c_svd_y.png", normal_corrected_print[:, :, 1], cmap = "gray")
        plt.title("2-D image for Normal Y  - corrected")
        plt.figure()
        plt.imshow(normal_corrected_print[:, :, 2], cmap = "Greys")
        plt.imsave("./svd/c_svd_z.png", normal_corrected_print[:, :, 2], cmap = "gray")
        plt.title("2-D image for Normal Z - corrected ")
    plt.show()
    return normal_corrected

def bump_object(out, mtlPath, location, matName, texturePath, displacement_map, mask, big = False):
    displacement_map /= 5000
    displacement_map /= coeff
    if big:
        h = int(ipad_height_pixel)
        w = int(ipad_width_pixel)
        location = cv2.resize(location.copy(), dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    else:
        h = depth_height
        w = depth_width
        displacement_map = cv2.resize(displacement_map.copy(), dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        m = cv2.resize(mask.copy(), dsize=(w, h))
    
    plt.figure()
    plt.imshow(location[:, :, 2])
    plt.show()
    location[:, :, 2] -= displacement_map
    plt.figure()
    plt.imshow(location[:, :, 2])
    plt.show()
    ids = np.zeros((h, w)).astype(int)
    id_count = 1
    create_mtl(mtlPath, matName, texturePath)
    if max(out.find('\\'), out.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    with open(out,"w") as f:   
        f.write("mtllib " + mtlPath.split('/')[-1] + "\n")
        for v in range(w):
            for u in range(h):
                x, y, z = location[u, v]
                f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")
                if m[u, v] > 0 and not (np.abs(location[u, v].sum()) < 0.01):
                    ids[u, v] = id_count
                id_count += 1
        f.write("usemtl " + matName + "\n")
        for v in range(w):
            for u in range(h):
                f.write("vt " + str(v/w) + " " + str((h - u - 1)/h)+ "\n")
        for v in range(w - 1):
            for u in range(h - 1):
                v1 = ids[u, v]; v2 = ids[u+1, v]; v3 = ids[u, v+1]; v4 = ids[u+1, v+1]
                if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                    continue
                f.write("f " + vete(v2,v2) + " " + vete(v1,v1) + " " + vete(v3,v3) + "\n")
                f.write("f " + vete(v2,v2) + " " + vete(v3,v3) + " " + vete(v4,v4) + "\n")
    return location

#beyond lambert
# modified from https://gist.github.com/asanakoy/c82420afdab4d5eaa78c9f9481e462e1

'''
explainations found reddit :
    You can imagine rotating the R axis of the RGB color space so it aligns with the S direction.  --reddit
    https://www.reddit.com/r/computervision/comments/5t078u/suv_color_space/

'''

def get_rot_mat(effective_source, unit):
    #            from              to
    v = np.cross(effective_source, unit)
    c = effective_source.dot(unit)
    s = np.linalg.norm(v)
    vx = np.zeros((3, 3))
    vx[0, 1] = -v[2]
    vx[0, 2] = v[1]
    vx[1, 0] = v[2]
    vx[1, 2] = -v[0]
    vx[2, 0] = -v[1]
    vx[2, 1] = v[0]
    R = np.identity(3) + vx + (vx.dot(vx)* (1 - c)) / (s * s)
    return R

def RGBToSUV(I_rgb, effective_source, color='R'):
    '''
    your implementation which takes an RGB image and a vector encoding the orientation of S channel wrt to RGB
    '''
    #accordingto page 3 of the paper
    if color == "R":
        unit = np.asarray([1, 0, 0])
    elif color == "G":
        unit = np.asarray([0, 1, 0])
    else:
        unit = np.asarray([0, 0, 1])
    I_suv=np.zeros(I_rgb.shape)
    R=get_rot_mat(effective_source, unit)
    for i in range(I_rgb.shape[0]):
        for j in range(I_rgb.shape[1]):
            I_suv[i,j,:]=np.matmul(R,I_rgb[i,j,:])
    if color == "R":
        S = I_suv[:,:,0]
        U = -I_suv[:,:,1]
        V = -I_suv[:,:,2]
    elif color == "G":
        S = I_suv[:,:,1]
        U = -I_suv[:,:,0]
        V = -I_suv[:,:,2]
    else:
        S = I_suv[:,:,2]
        U = -I_suv[:,:,1]
        V = -I_suv[:,:,0]
    return (S, U, V)

def min_max_normalization(v):
    return (v - v.min()) / (v.max() - v.min())

def get_specular_and_matt(original_img, mask, display=True, save=False):
    h = int(ipad_height_pixel)
    w = int(ipad_width_pixel)
    img = original_img * (np.stack([mask, mask, mask], axis = 2))
    s_all = np.zeros((h, w, 3))
    g_all = np.zeros((h, w, 3))
    source_colour = np.asarray([1, 1, 1])
    if display:
        plt.figure()
        plt.imshow(img)
        plt.imsave("./separate/image.png", img, cmap = "gray")
        plt.figure()
    colors = ["R", "G", "B"]
    for i in range(3):
        s, u, v = RGBToSUV(img, source_colour, colors[i])
        g = np.sqrt(u**2 + v**2)
        if display:
            plt.figure()
            plt.imshow(s, cmap = "gray")
            plt.figure()
            plt.imshow(u, cmap = "gray")
            plt.figure()
            plt.imshow(v, cmap = "gray")
            plt.figure()
            plt.imshow(g, cmap = "gray") 
            plt.imsave("./separate/s_" + colors[i] +".png", s, cmap = "gray")
            plt.imsave("./separate/u_" + colors[i] +".png", u, cmap = "gray")
            plt.imsave("./separate/v_" + colors[i] +".png", v, cmap = "gray")
            plt.imsave("./separate/g_" + colors[i] +".png", g, cmap = "gray")
        s_all[:, :, i] = s
        g_all[:, :, i] = g
    if display:
        plt.figure()
        plt.imshow(s_all)
        plt.figure()
        plt.imshow(g_all) 
        plt.show() 
    return s_all, g_all

#calculate uv normal
def normal_from_uv(imgs, normal_depth, color="R"):
    sequence = ['right', 'up', 'left', 'down']
    num_input = len(sequence)
    h = int(ipad_height_pixel)
    w = int(ipad_width_pixel)
    size = (h, w)
    I_uv = np.zeros((size[0], size[1], num_input))
    for i in range(num_input):
        s , g = get_specular_and_matt(imgs[sequence[i]], imgs['mask'], False) 
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000, from pil library
        if color == "R":
            I_uv[:, :, i] = g[:, :, 0]
        elif color == "G":
            I_uv[:, :, i] = g[:, :, 1]
        elif color == "B":
            I_uv[:, :, i] = g[:, :, 2]
        elif color == "L":
            I_uv[:, :, i] = g[:, :, 0] * 299/1000 + g[:, :, 1] * 587/1000 + g[:, :, 2] * 114/1000
        else:
            assert False
    normal_uv = surface_normal_correction(I_uv.reshape(h * w, num_input), size, imgs['mask'], normal_depth, True)
    return normal_uv

def get_heic(addr):
    heif_file = pyheif.read(addr)
    image = pil_image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
        )
    image = np.array(image)
    return np.rot90(image, 3)

#exposure ascending order
def get_hdr_image(base_name, exposures=[0., 1., 2., 3.]):
    images = []
    times = []
    for e in exposures:
        image = get_heic(base_name + str(int(e)) + '.HEIC')
        images.append(image)
        times.append(2. ** (e - 2))
    times = np.asarray(times, dtype=np.float32)
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, times)
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(images, times, response)
    # tonemap = cv2.createTonemap(2.2)
    # ldr = tonemap.process(hdr)
    # merge_mertens = cv2.createMergeMertens()
    # fusion = merge_mertens.process(images)
    cv2.imwrite(base_name + str(int(e)) + '.hdr', cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR))
    hdr = min_max_normalization(hdr)
    cv2.imwrite(base_name + str(int(e)) + '_normalized.hdr', cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR))
    return  hdr #fusion #min_max_normalization(hdr) #ldr

def replaceNan(m):
    where_are_NaNs = np.isnan(m)
    m[where_are_NaNs] = 0
    where_are_infs = np.isposinf(m)
    m[where_are_infs] = 1
    where_are_neginfs = np.isneginf(m)
    m[where_are_neginfs] = -1
    return m

def gamma_correction(image, gamma=2.2):
    return np.power(image, 1./gamma)
    
def create_ply(normal, texture, location, addr):
    w = depth_width
    h = depth_height
    color = np.clip(cv2.resize(texture.copy(), dsize=(w, h)) * 255., 0, 255).astype(int)
    size = (h - 1) * 2 * (w - 1)
    color = np.repeat(color[:-1, :-1], 2, axis=0).reshape((size, 3)) 
    location_mid = cv2.resize(location.copy().astype('float32'), dsize=(w, h))
    normal_mid = cv2.resize(normal.astype('float32'), dsize=(w, h))
    normal_mid = normal_mid.reshape(h * w, 3)
    vertex = location_mid.reshape(h * w, 3)
    vertex = np.asarray(list(zip(vertex[:,0], vertex[:,1], vertex[:,2], normal_mid[:, 0], normal_mid[:, 1], normal_mid[:, 2]))).flatten()
    vertex.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]

    face_id = np.zeros(((h - 1) * 2, w - 1, 3), dtype=int)
    for i in range(h - 1):
        for j in range(w - 1):
            v1 = int(i * w + j)
            v2 = int((i + 1) * w + j)
            v3 = int(i * w + j + 1)
            v4 = int((i + 1) * w + j + 1)
            face_id[i * 2, j, :3] = np.asarray([(v2, v1, v3)])
            face_id[i * 2 + 1, j, :3] = np.asarray([(v2, v3, v4)])
    face_id = face_id.reshape(size, 3).astype(int)
    face = np.zeros(size, dtype=[('vertex_indices', 'i4', (3,)),
                            ('red', 'u1'), ('green', 'u1'),
                            ('blue', 'u1')])
    for i in range(size):
        face[i] = ([face_id[i, 0], face_id[i, 1], face_id[i, 2]], color[i, 0], color[i, 1], color[i, 2])
    plydata = PlyData(
                [
                    PlyElement.describe(
                        vertex, 'vertex'
                    )
                    ,
                    PlyElement.describe(face, 'face')
                ],
                text=False, byte_order='>'
            )
    plydata.write(addr)