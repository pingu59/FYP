import numpy as np
import open3d as o3d
import pandas
import cv2
import scipy.ndimage
from open3d import JVisualizer
import copy
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from lxml.etree import Element, SubElement, Comment, tostring
import lxml.etree as etree


from PIL import Image
from scipy.sparse.linalg import svds
import pickle
import mat73


# pixel_per_meter = 4801.4
# ipad_height_pixel = 1440.
# ipad_width_pixel = 1080.
up_side_down = False

pixel_per_meter = 10296.3 
ipad_height_pixel = 3088.
ipad_width_pixel = 2316.
depth_height = 640
depth_width = 480
#per meter
scale = 1000
coeff = ipad_height_pixel/depth_height

#pinhole camera insintrict of ipad pro 11-inch 2018 (see the terminal output of the swift program)
fx = 2872.9004 
fy = 2872.9004 
ox = 1538.0475 
oy = 1154.8513 

if ipad_height_pixel < 3088:
    fx = fx /3088. * ipad_height_pixel
    fy = fy /3088. * ipad_height_pixel
    ox = ox /3088. * ipad_height_pixel
    oy = oy /3088. * ipad_height_pixel


color_path_to = '/Users/tianyizuo/Desktop/Captures/standard/mid/full.jpg'
depth_path_to = '/Users/tianyizuo/Desktop/Captures/standard/mid/depth.txt'
mask_path_to = '/Users/tianyizuo/Desktop/Captures/standard/mid/mask.jpg'
landmark_path_to = '/Users/tianyizuo/Desktop/Captures/standard/mid/landmark.txt'

color_path_from = '/Users/tianyizuo/Desktop/Captures/standard/toleft/full.jpg'
depth_path_from = '/Users/tianyizuo/Desktop/Captures/standard/toleft/depth.txt'
mask_path_from = '/Users/tianyizuo/Desktop/Captures/standard/toleft/mask.jpg'
landmark_path_from = '/Users/tianyizuo/Desktop/Captures/standard/toleft/landmark.txt'

#the maximum distance in which the search tried to find a correspondence for each point
threshold = 0.1 # in meters

facial_midpoints = np.array([46, 47, 48, 49, 52, 30, 40, 41, 37, 67])
inner_face_id = 59

def create_picked_points(pickedPointsPath, landmarks, objPath):

    # <!DOCTYPE PickedPoints>
    # <PickedPoints>
    #  <DocumentData>
    #   <DataFileName name="tz2617_masked_open_mouth.obj"/>
    #   <templateName name=""/>
    #  </DocumentData>

    picked_points = Element('PickedPoints')
    document_data = SubElement(picked_points, 'DocumentData')
    data_file_name = SubElement(document_data, 'DataFileName',
                                 {'name':objPath.split('/')[-1]})
    template_name = SubElement(document_data, 'templateName',
                                 {'name':''})

    #  <point name="1" z="0.25358397" active="1" x="0.031310152" y="-0.056594905"/>
    for i in range(landmarks.shape[0]):
        x, y, z = landmarks[i]
        SubElement(picked_points, 'point', {'name':str(i+1), 'active':str(1), 'x': str(x), 'y': str(y), 'z': str(z)})

    picked_points_xml = etree.tostring(picked_points, encoding="UTF-8",
                         xml_declaration=True,
                         pretty_print=True,
                         doctype='<!DOCTYPE PickedPoints>').decode("utf-8") 
#     print(picked_points_xml)
    f = open(pickedPointsPath, "w")
    f.write(picked_points_xml)
    f.close()

def generate_pcd(color_path, depth_path, mask_path, landmark_path):
    color_raw = cv2.imread(color_path)
    color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
    color_raw = o3d.geometry.Image((color_raw).astype(np.uint8))
    img = pandas.read_csv(depth_path, header=None)
    landmarks = pandas.read_csv(landmark_path, header=None).to_numpy()
    depth_raw = np.zeros((depth_height, depth_width))
    h = depth_height
    w = depth_width
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if ipad_height_pixel == 3088.:
        mask = mask.repeat(2, axis=0).repeat(2, axis=1) #upsample
    mask = (mask > 250)
    print(mask.shape)
    instrinct = o3d.camera.PinholeCameraIntrinsic()
    if up_side_down:
        instrinct.set_intrinsics(int(ipad_height_pixel), int(ipad_width_pixel), fx, fy, ipad_width_pixel - ox, ipad_height_pixel - oy)
    else:
        instrinct.set_intrinsics(int(ipad_height_pixel), int(ipad_width_pixel), fx, fy, ox, oy)

    if up_side_down:
        for u in range(h):
            for v in range(w):
                    depth_raw[u, v] = 1./ img.iloc[v, h - 1 - u] # np.sqrt(max(0, np.power(1./img.iloc[v, h - 1 - u], 2) + np.power(u/pixel_per_meter, 2)))
    else:
        for u in range(h):
            for v in range(w):
                    depth_raw[u, v] = 1./ img.iloc[-v, u] #np.sqrt(max(0, np.power(1./img.iloc[-v, u], 2) - np.power(u/pixel_per_meter, 2)))
    depth_raw = scipy.ndimage.zoom(depth_raw, coeff, order=1) * scale 
    depth_raw = np.expand_dims(depth_raw, axis=2).astype(np.uint16)
    
    print(depth_raw.shape)
    index = 0
    index_array = np.zeros((int(ipad_height_pixel), int(ipad_width_pixel)), int)
    for x in range(int(ipad_height_pixel)):
        for y in range(int(ipad_width_pixel)):
            if not mask[x, y]:
                depth_raw[x, y] = 0 #set to invalid value
            else:
                index += 1
                index_array[x, y] = index
    landmarks_3d = np.zeros((landmarks.shape[0], 3))
    # plt.figure()
    # plt.imshow(color_raw)
    # plt.scatter(landmarks[:, 1], landmarks[:, 0])
    # plt.show()
    depth_raw = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_trunc=1., convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, instrinct)
    print(pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.05, max_nn=8))

    point_cloud_array = np.asarray(pcd.points)
    for i in range(landmarks.shape[0]):
        x = int(landmarks[i][0])
        y = int(landmarks[i][1])
        landmarks_3d[i] = point_cloud_array[index_array[x, y] - 1]
    normal_from_depth = np.zeros((int(ipad_height_pixel), int(ipad_width_pixel), 3))
    normals_np = np.asarray(pcd.normals)
    print("normals shape: ", normals_np.shape)
    print("index : ", index)
    for x in range(int(ipad_height_pixel)):
        for y in range(int(ipad_width_pixel)):
            i = index_array[x, y]
            if i > 0 and i < index - 1:
                normal_from_depth[x, y] = normals_np[i - 1] 
    return pcd, landmarks_3d, normal_from_depth


def generate_pcd_lowres(color_path, depth_path, mask_path, landmark_path):
    h = depth_height
    w = depth_width
    color_raw = cv2.imread(color_path)
    #downsample to 640 x 480
    color_raw = cv2.resize(color_raw, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
    color_raw = o3d.geometry.Image((color_raw).astype(np.uint8))
    img = pandas.read_csv(depth_path, header=None)
    landmarks = pandas.read_csv(landmark_path, header=None).to_numpy()
    depth_raw = np.zeros((h, w))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #down sample to 640 x 480
    mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    mask = (mask > 230)
    print(mask.shape)
    instrinct = o3d.camera.PinholeCameraIntrinsic()
    if up_side_down:
        instrinct.set_intrinsics(h, w, fx/coeff, fy/coeff, (ipad_width_pixel - ox)/coeff, (ipad_height_pixel - oy)/coeff)
    else:
        instrinct.set_intrinsics(h, w, fx/coeff, fy/coeff, ox/coeff, oy/coeff)

    if up_side_down:
        for u in range(h):
            for v in range(w):
                    depth_raw[u, v] = 1./ img.iloc[v, h - 1 - u] # np.sqrt(max(0, np.power(1./img.iloc[v, h - 1 - u], 2) + np.power(u/pixel_per_meter, 2)))
    else:
        for u in range(h):
            for v in range(w):
                    depth_raw[u, v] = 1./ img.iloc[-v, u] #np.sqrt(max(0, np.power(1./img.iloc[-v, u], 2) - np.power(u/pixel_per_meter, 2)))
    
    depth_raw *= scale 
    depth_raw = np.expand_dims(depth_raw, axis=2).astype(np.uint16)
    
    print(depth_raw.shape)
    index = 0
    index_array = np.zeros((h, w), int)
    for x in range(h):
        for y in range(w):
            if not mask[x, y]:
                depth_raw[x, y] = 0 #set to invalid value
            else:
                index += 1
                index_array[x, y] = index
    landmarks_3d = np.zeros((landmarks.shape[0], 3))
    # plt.figure()
    # plt.imshow(color_raw)
    # plt.scatter(landmarks[:, 1]/coeff, landmarks[:, 0]/coeff)
    # plt.show()
    depth_raw = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_trunc=1., convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, instrinct)
    print(pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.01, max_nn=4))
    point_cloud_array = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    for i in range(landmarks.shape[0]):
        x = int(landmarks[i][0]/coeff)
        y = int(landmarks[i][1]/coeff)
        landmarks_3d[i] = (point_cloud_array[index_array[x, y] - 1] / coeff)  
    o3d.visualization.draw_geometries([pcd])
    return pcd, landmarks_3d, normals

# visualizer = JVisualizer()
# visualizer.add_geometry(pcd)
# visualizer.show()

pcd_from, landmark_from, normal_from = generate_pcd_lowres(color_path_from, depth_path_from, mask_path_from, landmark_path_from)
pcd_to, landmark_to, normal_to = generate_pcd_lowres(color_path_to, depth_path_to, mask_path_to, landmark_path_to)

o3d.io.write_point_cloud("./point_cloud/usd_mid.ply", pcd_to) # for meshlab visualization
o3d.io.write_point_cloud("./point_cloud/usd_toright.ply", pcd_from)

