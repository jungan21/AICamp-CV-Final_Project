#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import re
import cv2

class FaceNet(object):
    # load model and init graph
    def __init__(self, model_path):
        # Read model files and init the tf graph and model
        graph = tf.Graph()
        # all loadded stuff will be stored in the graph
        with graph.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            print('Model directory: %s' % model_path)
            # get model files and will load them into the tf session
            meta_file, ckpt_file = self.get_model_filenames(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            """
                对于facenet, 这里我们用了下面的语句直接import graph , 所以不用像mtcnn那样，再需要写个mtcnn_model.py. 
                否则我们也需要写一个类似的facenet_model.py 来定义graph
                
                但是从别人train好的model直接导入graph有一点不好，就是在predict (做一次前向传播)需要参考别人的源代码里是如何定义graph的，要知道input output是什么                                  
            """

            # 从*.meta 文件里直接load graph
            saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file), input_map=None)
            # 把每一个卷积 每一个全连接 和 ckpt_file中的对应的参数 binding 起来
            saver.restore(self.sess, os.path.join(model_path, ckpt_file))

    # checkpoint 存了 网络的参数        
    # meta file 存了 网络的结构
    def get_model_filenames(self, model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

    # 用load进来的model 搞出embeding
    def predict(self, image):
        """Get the embedding vector of face by facenet
        
        Parameters:
        ----------
        image: numpy array
            input image array
        
        Returns:
        ----------
        embedding: numpy array
            the embedding vector of the face
        """
        #这里的input, embeddings 对应 于 facenet模型 定义时候的名字
        # 因为对于facenet我们直接导入的graph, 所以不知道要feed 什么，所以要参考facenet源代码里如何构建graph的
        
        # 0 表示第0个output
        # https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L128
        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        # https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L136
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")
        # 因为最后要Fully Connected Layer。。所以size  要固定
        image = cv2.resize(image, (160, 160))
        # Run forward pass to calculate embeddings
        feed_dict = { images_placeholder: np.stack([image]), phase_train_placeholder:False }
        emb = self.sess.run(embeddings, feed_dict=feed_dict)
        # 因为input是数组，即使只有一个图片，所以要 
        return emb[0, :]
