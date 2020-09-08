import os
import io
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


class TensorboardTools(object):
    
    def __init__(self):
        self.summary_writer=None


    def summary_img(self, img, tag, new_size=(256, 256), step=None):

        # img=cv2.resize(img, new_size)

        if self.summary_writer == None:
            raise Exception("summary_writer is None")

        im_summaries = []
        # Write the img to a string
        s = io.BytesIO()
        plt.imsave(s,img)

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_img_string=s.getvalue(),
                                    height=new_size[0],
                                    width=new_size[1])
        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag=tag, img=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.summary_writer.add_summary(summary, step)


    def summary_scalar(self, value, tag, step=None):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        if self.summary_writer == None:
            raise Exception("summary_writer is None")
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, step)