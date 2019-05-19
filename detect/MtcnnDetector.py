#coding:utf-8
import cv2
import time
import numpy as np
from nms import py_nms


class MtcnnDetector(object):


    def __init__(self,
                 detectors,
                 min_face_size=25, # 如果图像size 小于25* 25 就ignore
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.79, # 图像金字塔的放缩因子
                 #scale_factor=0.709,#change
                 slide_window=False):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size # 比这个size 小的图片都会被ignore掉
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor # 生成图像金字塔的用的scale_factor
        self.slide_window = slide_window # False：表示PNET 用FCN,  True:表示 用slide window生成

    def convert_to_square(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
            generate bbox from feature cls_map
        Parameters:
        ----------
            cls_map: numpy array , n x m 
                detect score for each position
                也就是heat map
            reg: numpy array , n x m x 4
                bbox 
                调整bounding box 的位置
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        #stride = 4
        cellsize = 12
        #cellsize = 25
        # 得分 大于threshold的 二维的位置下标拿出来
        t_index = np.where(cls_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        
        # 该如何修正 bounding box 位置
        #offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        # feature map （也就是PNET FCN 输出的热度图） 上的点 映射 到原图 中的位置
        # 除以 scale, 就得到原图中相对大一点的框
        # 根据heat map中二维坐标的bounding box位置得到原图的中的bounding box
        # feature map上每一个小框都会对应 原图中 12 * 12的图
        #  stride = 2 因为在 mtcnn_model.py里P_NET 里max_pooling 是2ao
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 # 左上角加 12 就是右下角
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 # bounding box得分
                                 score,
                                 # how to adjust bounding box
                                 reg])

        return boundingbox.T
    #pre-process images PNET 里做图像金字塔的时候用到
    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128 # 图像里每个像素[0, 255] -> (-127.5) [-127.5, 127.5] -> [-1, -1]归一化
        return img_resized

    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration (calibration 表示修正的意思)
        boxes_c: numpy array
            boxes after calibration
        """
        ''' 
        1. 需要做图像金字塔，图像金字塔是用来帮助PNET得到各种大小不同 在不同尺度的proposal
        2. 对图像金字塔的每一层图像 用FCN去做PNET的predict。还需要对PNET全卷积生成的heatmap上生成proposal
        3. 把FCN的predict结果转换成一个list 的bounding box
        4. 对所有proposal 做 NMS + 然后再根据predict 出来结果调整box（变量名：reg_）去adjust一下
        5. NOTE: PNET, RNET 不返回 landmark, 只有最后的ONET 返回landmark
        '''
        # h, w , c 高 宽 channel个数 (厚度)      
        # h, w will be used to build 图像金字塔
        h, w, c = im.shape
        
        # 图像金字塔  start
        # app.py原始参数： min_face_size 是24 也就是规定最小的人脸的size
        net_size = 12 # 12 * 12 的 bounding box
        current_scale = float(net_size) / self.min_face_size  # find initial scale。 找到 这里12 和 原始参数要求24的最小的要求 的scale
        # print("current_scale", net_size, self.min_face_size, current_scale)
        # 按比例放缩,能满足最小人脸的要求
        im_resized = self.processed_image(im, current_scale)
        # 这里得到了 图像金字塔 最下面那张最大的图片 （不需要通道数 所以最后一个 _）
        current_height, current_width, _ = im_resized.shape
        # fcn
        all_boxes = list()
        # 拿到了图像金字塔最下面的最大的图片后，循环得到整个图像金字塔
        # while loop 结束后，就表示，把所有图像金字塔里的图片都送进去PNET 跑了一遍，并且拿到了bounding box 
        while min(current_height, current_width) > net_size:
            # 图像金字塔每一层的图片 去抠出bounding box - start 
            #return the result predicted by pnet
            #cls_cls_map : H*w*2
            #reg: H*w*4
            # cls_prob, bbox_pred  (heat map) 这是 pnet_detector.predict 函数返回的结果
            # Note: 我们之前train好的model都是被pnet_dectect load 进来的 也就是在FcnDetector, 也就是fcn_detector.py里面
            cls_cls_map, reg = self.pnet_detector.predict(im_resized)
            #boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
            # cls_cls_map 也就是 cls_prob...cls_cls_map[:, :,1] 只拿出是人脸的box,
            # reg bound box pred.
            # thresh[0] pnet 的 threshold
            # 这里生成的bound boxes... 可能会有重合 重叠
            # cls_cls_map[:, :,1]:  意思是 把classficiation 中 是人脸的概率拿出来 用来生成bounding box

            """
                上面 self.pnet_detector.predict(im_resized) 实际上是对图像金字塔的图片做预测，拿到cls_cls_map, reg
                但是我们generate 实际的 bounding box 还需要把图像金字塔里的图片，映射到原图中找出bounding box 
            """
            boxes = self.generate_bbox(cls_cls_map[:, :,1], reg, current_scale, self.thresh[0])
            #图像金字塔每一层的图片 去抠出bounding box - end
            
            # generate 整个图像金字塔的图片， 用最下面那个最大的图片每次scale 缩小一定比例（i.e.scale_factor）- start
            current_scale *= self.scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape
            # generate 整个图像金字塔的图片， 用最小面那个最大的图片每次scale 缩小一定比例（i.e.scale_factor）- end

            if boxes.size == 0:
                continue
            # generate_bbox 返回的boxes一共9个参数（左上，右下坐标， score, reg(4个参数)）这里只要前5个参数：左上，右下 坐标，score 5个参数
            # A B 的交集 除以 AB的并集 ，求AB得nms. 这里也就是 面积的交/面积的并
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)
            # 没while loop, 一次，就处理图像金字塔里的一层

        
        if len(all_boxes) == 0:
            return None, None, None

        all_boxes = np.vstack(all_boxes)
        
        # 下面需要对 while 循环 得到的bounding box 在做nms 不过这次threshold是 0.7，while loop里是0.5 
        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        
        boxes = all_boxes[:, :5]
       
        # 相当于 x2- x1得到宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1 
        # 相当于 y2- y1得到高
        bbh = all_boxes[:, 2] - all_boxes[:, 0] + 1

        # refine the boxes 根据reg 的4个值 (也就是对应的 all_boxes[:, 5]，all_boxes[:, 6] all_boxes[:, 7]all_boxes[:, 8])
        # 去调整 bounding box。 NOTE: all_boxes[:, 4] 是 bounding box的score （根据generat_box函数返回值）
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]]) # all_boxes[:, 4] 表示 score
        boxes_c = boxes_c.T # 因为前面是vstack, 这里要装置一下

        return boxes, boxes_c, None # None表示不需要返回landmark
        

    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration， 未修正过的 24 * 24 的图片
        boxes_c: numpy array
            boxes after calibration 修正过的 24 * 24 的图片
        """
        h, w, c = im.shape
        # 把长方形 convert to 正方形 防止后面 24*24 resize 时候形变
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        # 先抠出来， 再24 * 24 的resize boxes
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
       
         # 抠图 并resize 投 24 * 24 的框
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            #抠图
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            # 再把抠出来的图 resize 成 24 * 24
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24))-127.5) / 128 #归一化. RGB 0 ~ 255 归一化到 -1 ~ 1范围
        #cls_scores : num_data*2
        #reg: num_data*4
        #landmark: num_data*10
        # predict 抠出来的 24 * 24 的图片
        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:,1]
        # 只有大于 RNET threshold才会被认为是人脸 送给ONET
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        # 组织一下  [x1, y1, x2, y2, P, x1*, y1*, x2*, y2*], p表示概率， x1* y1* 表示如何调整
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            #landmark = landmark[keep_inds]
        else:
            return None, None, None
        
        # npm 非极大值压制，IOU的threshold 0.6
        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        # 调整 box
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes, boxes_c,None

    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48))-127.5) / 128
            
        cls_scores, reg,landmark = self.onet_detector.predict(cropped_ims)
        #prob belongs to face
        cls_scores = cls_scores[:,1]        
        keep_inds = np.where(cls_scores > self.thresh[2])[0]        
        if len(keep_inds) > 0:
            #pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None
        
        #width
        w = boxes[:,2] - boxes[:,0] + 1
        #height
        h = boxes[:,3] - boxes[:,1] + 1
        landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
        landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T        
        boxes_c = self.calibrate_box(boxes, reg)
        
        
        boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c,landmark

    #use for video
    def detect(self, img):
        """Detect face over image
        """
        boxes = None
        t = time.time()
    
        # pnet
        t1 = 0
        if self.pnet_detector:
            #这里调用detect_pnet的时候，并没有做图像金字塔。。。但实际上图像金字塔是在detect_pnet里做的
            # boxes_c 表示可能包含人脸的bounding boxes， 这个要传递给下面的 rnet
            boxes, boxes_c,_ = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]),np.array([])
    
            t1 = time.time() - t
            t = time.time()
    
        # rnet
        t2 = 0
        if self.rnet_detector:
            boxes, boxes_c,_ = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])
    
            t2 = time.time() - t
            t = time.time()
    
        # onet
        t3 = 0
        if self.onet_detector:
            boxes, boxes_c,landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])
    
            t3 = time.time() - t
            t = time.time()
            print(
                "time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,
                                                                                                                t3))
    
        return boxes_c,landmark
    
    
    def get_face_from_single_image(self, image):
        # 把一个batch 的 image 数组化 （在我们的例子中，实际只有一张）
        images = np.array(image)
        
        boxes_c,landmarks = self.detect(images)
        
        rets = []
        # ??? 这里实现没有用 landmarks 去做alignment
        for i in range(boxes_c.shape[0]): # boxes_c.shape[0] 就是行数，就相当于是proposal的个数
            bbox = boxes_c[i, :4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            print(corpbbox)
            rets.append(image[corpbbox[0]:corpbbox[2], corpbbox[1]:corpbbox[3]].copy())
        
        return rets

