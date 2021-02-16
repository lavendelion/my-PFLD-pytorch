"""
数据准备，将数据处理为两个文件，一个是train.csv,另一个是train.txt
train.csv: 每一行是一张图片的标签，具体储存情况根据不同任务的需求自行设定
train.txt: 每一行是图片的路径，该文件每行的图片和train.csv的每一行标注应该是一一对应的
另外，需要将图片稍微离线处理一下，将原图片裁剪出训练使用的图片(resize成训练要求大小)后，保存在自定义文件夹中，train.txt里的路径应与自定义文件夹相同
"""
import numpy as np
import cv2
import os


def crop_according_keypoints(img, kps, paddings=(0,0,0,0)):
    """
    根据人脸关键点坐标,找到最小外接正方形，然后再根据paddings的数值进行扩展，最终裁剪出人脸图像
    :param img: np.array, 待裁剪图像
    :param kps: np.array  2xn, 关键点列表，[[x1,x2,...],[y1,y2,...]]
    :param paddings: tuple, 上下左右的padding数值，(top, bottom, left, right)
    :return: 裁剪后的图像, 以及对齐、归一化后的关键点坐标
    """
    isTranspose = False
    if kps.shape[0]!=2:
        kps = np.transpose(kps, (1,0))
        isTranspose = True
    assert kps.shape[0]==2, "Wrong kps shape (%d, %d)"%(kps.shape[0], kps.shape[1])
    if(kps.dtype!=np.int32):
        kps = kps.astype(np.int32)
    top_left_x, top_left_y = np.min(kps, axis=1).astype(np.int32)
    bottom_right_x, bottom_right_y = np.max(kps, axis=1).astype(np.int32)
    w = bottom_right_x - top_left_x
    h = bottom_right_y - top_left_y
    if w > h:
        gap = (w - h)//2
        top_left_y -= gap
        bottom_right_y += gap
    else:
        gap = (h - w)//2
        top_left_x -= gap
        bottom_right_x += gap
    top_left_x -= paddings[2]
    top_left_y -= paddings[0]
    bottom_right_x += paddings[3]
    bottom_right_y += paddings[1]

    # 处理人脸在图像边界，padding超出原图像大小的情况，此时将正方形平移至bbox不越界的位置
    if top_left_x < 0:
        bottom_right_x -= top_left_x
        top_left_x = 0
    if top_left_y < 0:
        bottom_right_y -= top_left_y
        top_left_y = 0
    if bottom_right_x >= img.shape[1]:
        top_left_x -= (bottom_right_x - img.shape[1] + 1)
        bottom_right_x = img.shape[1] - 1
    if bottom_right_y >= img.shape[0]:
        top_left_y -= (bottom_right_y - img.shape[0] + 1)
        bottom_right_y = img.shape[0] - 1

    # 如果bbox仍然越界，则报错
    if top_left_x<0 or top_left_y<0 or bottom_right_x>=img.shape[1] or bottom_right_y>=img.shape[0]:
        # for i in range(kps.shape[1]):
        #     cv2.circle(img, (int(kps[0,i]),int(kps[1,i])), 5, (255,0,0), 2)
        # cv2.imshow("bad_img", cv2.resize(img, (img.shape[1]//5, img.shape[0]//5)))
        # cv2.waitKey(0)
        return None, None
        # raise ValueError("padding in crop is too much.")

    # draw keypoint in img
    # for i in range(kps.shape[1]):
    #     cv2.circle(img, (int(kps[0,i]),int(kps[1,i])), 1, (255,0,0), 2)

    kps[0, :] -= top_left_x
    kps[1, :] -= top_left_y
    imgsize = bottom_right_y - top_left_y
    kps = kps.astype(np.float32)
    kps /= imgsize
    crop_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
    if isTranspose:
        kps = np.transpose(kps, (1,0))
    return crop_img, kps

def process_img_dir(img_dir, anno_dir, save_root_dir, padding=10, debug=False):
    """
    将img_dir文件夹内的图片按keypoints标注裁剪后，存入save_dir
    padding是关键点最小外接正方形向外padding的尺寸
    标注里，关键点进行归一化 + bbox + attribution + 计算出的图片中人脸的三个欧拉角(角度制)
    最终得到图片文件夹及所有图片对应的标注(train.csv/test.csv)和图片列表文件(train.txt, test.txt)
    """
    with open(os.path.join(anno_dir)) as f:
        text = f.readlines()
    isTrain = None
    if "train.txt" in anno_dir:
        save_dir = os.path.join(save_root_dir, "train")
        isTrain = True
    else:
        save_dir = os.path.join(save_root_dir, "test")
        isTrain = False
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    i, j = 0, 0
    result = None
    listtxt = []
    N = len(text)
    for line in text:
        i += 1
        line = line.split()
        kps = list(map(float,line[0:196]))
        bbox = list(map(int, line[196:200]))
        attr = list(map(int, line[200:-1]))
        img_path = os.path.join(img_dir, line[-1])
        img_name = img_path.split('/')[-1].split('.')[0]
        print("%s is processing..... (%d/%d)" % (img_path, i, N))
        kps = np.asarray(kps).reshape(-1, 2)
        bbox = np.asarray(bbox)
        attr = np.asarray(attr)
        img = cv2.imread(img_path)
        crop_img, kps = crop_according_keypoints(img, kps, (padding, padding, padding, padding))
        if crop_img is None:
            j += 1
            continue
        crop_img = cv2.resize(crop_img, (112, 112))
        save_img_path = os.path.join(save_dir, "%d_%s.png"%(i-j,img_name))  # 因为同一张图可能有多个人脸标注，所以用i给每张人脸进行区分
        listtxt.append(save_img_path+'\n')
        cv2.imwrite(save_img_path, crop_img)  # save image
        if debug:
            for i in range(kps.shape[0]):
                cv2.circle(crop_img, (int(kps[i, 0] * 112), int(kps[i, 1] * 112)), 1, (255, 0, 0), 2)
            cv2.imshow("crop_img", crop_img)
            cv2.waitKey(0)
        angles = get_euler_angle(kps, n_points=98, debug=debug, img_path=save_img_path)
        if result is None:
            result = np.concatenate((kps.reshape(-1,), bbox, attr, angles), axis=0).reshape(1,-1)
        else:
            new_row = np.concatenate((kps.reshape(-1,), bbox, attr, angles), axis=0).reshape(1,-1)
            result = np.concatenate((result, new_row), axis=0)
        cv2.waitKey(0)
    result_name = "train.csv" if isTrain else "test.csv"
    list_name = "train.txt" if isTrain else "test.txt"
    np.savetxt(os.path.join(save_root_dir, result_name), result, delimiter=',')  # save kps
    with open(os.path.join(save_root_dir, list_name), "w") as f:
        f.writelines(listtxt)
    print("%d images have been processed. %d images are bad." % (i, j))
    assert result.shape[0]==len(listtxt), "Error: lines in %s is not equal to that in %s."%(result_name, list_name)


def calculate_pitch_yaw_roll(landmarks_2D,
                             cam_w=256,
                             cam_h=256,
                             n_points=68,
                             radians=False):
    """
    根据自定义标准人脸14个关键点的3D坐标(landmarks_3D)，以及图像对应14个关键点的2D图像坐标(landmarks_2D)，
    计算图像中人脸参照于标准正脸在三维的三个偏转欧拉角(默认是角度制)，并返回
    :param landmarks_2D:
    :param cam_w: 相机图像宽
    :param cam_h: 相机图像高
    :param n_points: 人脸关键点个数
    :param radians: true:返回弧度制；false:返回角度制
    :return: list(pitch, yaw, roll)
    """
    assert landmarks_2D is not None, 'landmarks_2D is None'
    assert landmarks_2D.shape[0] == n_points, "landmarks_2D != n_points"

    # Estimated camera matrix values.
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])  # 相机矩阵
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])  # 相机畸变矩阵

    # dlib数据集 (68 landmark) trached points
    # wflw数据集 (98 landmark) trached points
    if n_points==68:
        TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    elif n_points==98:
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    else:
        raise ValueError("n_points must be 68 or 98 but now is %d"%(n_points))
    landmarks_2D = landmarks_2D[TRACKED_POINTS, :]  # 获取计算欧拉角所需要的关键点

    # X-Y-Z with X pointing forward and Y on the left and Z up.
    # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
    # OpenCV uses the reference usually used in computer vision:
    # X points to the right, Y down, Z to the front
    # 标准正脸的14个关键点的三维坐标
    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT,
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT,
        [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
        [0.000000, -7.415691, 4.070434],  # CHIN
    ])

    # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings points from the world coordinate system to the camera coordinate system.
    # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix,
                                 camera_distortion)
    # Get as input the rotational vector, Return a rotational matrix
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))  # 获得人脸姿态变换矩阵
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)  # 由旋转矩阵计算欧拉角
    return list(map(lambda k: k[0], euler_angles))  # euler_angles contain (pitch, yaw, roll)


def get_euler_angle(kps, n_points=68, debug=False, img_path=None):
    """
    在datadir中创建每张图片的欧拉角标签文件，文件名字为图片名+_euler.csv
    :param kps: 计算欧拉角所用的关键点
    :param n_points: 图片对应的关键点标注是68个点还是98个点，不同点数使用的计算欧拉角的关键的序号不同
        68点用序号：[17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
        98点用序号：[33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    :return: 在原文件夹创建对应的欧拉角标签文件
    """
    assert kps.shape[0]==n_points, "kps.shape[0](%d) != n_points(%d)"%(kps.shape[0], n_points)
    angles = calculate_pitch_yaw_roll(kps, n_points=n_points)

    # debug=True
    if debug:
        assert img_path is not None, "img_path is None."
        img = cv2.imread(img_path)
        img_size = img.shape[0]
        if n_points==68:
            TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
        else:
            TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        for i in range(n_points):
            if i in TRACKED_POINTS:
                cv2.circle(img, (int(img_size*kps[i,0]), int(img_size*kps[i,1])), 2, (255, 0, 0), 2)
        print("pitch=%.2f, yaw=%.2f, roll=%.2f"%(angles[0], angles[1], angles[2]))
        cv2.imshow("img", img)
        cv2.waitKey(0)
    else:
        angles = np.array(angles).reshape(-1,)
    return angles


if __name__ == '__main__':
    # 将原始图像切分为人脸图像，并将原图像关键点转换为归一化后的关键点标签文件
    img_dir = r"I:\Dataset\WFLW\WFLW_images"
    anno_dirs = [r"I:\Dataset\WFLW\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt",
                 r"I:\Dataset\WFLW\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt"]
    save_dir = r"I:\Dataset\WFLW\WFLW_for_PFLD"
    for anno_dir in anno_dirs:
        process_img_dir(img_dir, anno_dir, save_dir, debug=False)

    # 测试用
    # afw: 134212_1.jpg
    # img = cv2.imread(r'I:\Dataset\300W\afw\134212_1.jpg')
    # cv2.imshow("img", img)
    # kps = np.array([721.189915,229.034461,
    #                 720.156130,248.439669,
    #                 719.760845,272.888685,
    #                 726.819985,293.581156,
    #                 737.104205,312.765561,
    #                 751.807847,331.323176,
    #                 767.167918,351.351826,
    #                 780.181675,375.774678,
    #                 802.105440,390.787157,
    #                 832.247913,386.166071,
    #                 856.495348,370.332629,
    #                 878.053846,353.122391,
    #                 905.093283,328.567788,
    #                 922.779550,306.858293,
    #                 934.829891,272.601707,
    #                 940.577820,240.007480,
    #                 943.777839,209.792645,
    #                 709.178421,207.890875,
    #                 718.430775,203.253909,
    #                 730.977646,207.526319,
    #                 743.972548,211.985542,
    #                 758.528339,218.060851,
    #                 796.134867,211.060040,
    #                 813.385082,199.734537,
    #                 836.294532,195.056967,
    #                 860.724625,193.449055,
    #                 884.056569,199.214286,
    #                 775.131302,227.928447,
    #                 772.491006,247.606551,
    #                 769.386004,267.414835,
    #                 766.365850,287.332355,
    #                 753.939252,290.019021,
    #                 759.394371,298.261459,
    #                 772.136304,304.551333,
    #                 788.229081,299.723559,
    #                 804.730001,292.217977,
    #                 730.675730,230.253259,
    #                 739.843797,225.657151,
    #                 751.952415,224.030526,
    #                 764.738475,227.763155,
    #                 751.881290,231.802423,
    #                 740.700458,234.011518,
    #                 813.433270,227.624934,
    #                 829.635692,222.537555,
    #                 842.870523,224.663036,
    #                 855.689111,227.232766,
    #                 843.139696,229.891910,
    #                 829.559300,230.780709,
    #                 760.738462,315.213813,
    #                 769.028597,314.723204,
    #                 776.968630,317.691198,
    #                 783.965680,319.824466,
    #                 793.856309,318.110054,
    #                 815.182725,318.830846,
    #                 840.571636,319.364119,
    #                 816.521027,341.617699,
    #                 798.489009,350.658867,
    #                 787.650662,350.137523,
    #                 778.991394,347.557484,
    #                 768.910742,336.390235,
    #                 769.143753,320.856699,
    #                 777.901471,324.535403,
    #                 785.456938,326.832291,
    #                 794.383504,326.779056,
    #                 827.712352,324.274778,
    #                 794.969446,332.883220,
    #                 785.778326,333.014380,
    #                 778.413074,329.165294])
    # kps = np.transpose(kps.reshape((68, 2)), (1,0))
    # crop_img,kps = crop_according_keypoints(img, kps, (10,10,10,10))
    # print(kps.shape)
    # kps = np.transpose(kps, (1,0))
    # ret = calculate_pitch_yaw_roll(kps)
    # img_size = crop_img.shape[0]
    # for i in range(68):
    #     if i in [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]:
    #         cv2.circle(crop_img, (int(img_size*kps[i,0]), int(img_size*kps[i,1])), 2, (255, 0, 0), 2)
    # print(list(ret))
    # cv2.imshow("crop_img", crop_img)
    # cv2.waitKey(0)