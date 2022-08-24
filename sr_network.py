''' @author: Andrew Glaws, Karen Stengel, Ryan King
'''
import tensorflow as tf
from utils import *

class SR_NETWORK(object):
    def __init__(self, x_LR=None, x_HR=None, static_HR=None, r=None, status='pretraining', alpha_advers=0.001, isWGAN=False, usemodgen=0, is_stochastic=False):

        status = status.lower()
        if status not in ['pretraining', 'training', 'testing']:
            print('Error in network status.')
            exit()
        self.x_LR, self.x_HR, self.static_HR = x_LR, x_HR, static_HR
        self.is_stochastic=is_stochastic

        if r is None:
            print('Error in SR scaling. Variable r must be specified.')
            exit()

        if status in ['pretraining', 'training']:
            self.x_SR = self.generator(self.x_LR, self.static_HR, r=r, is_training=True, use_mod_gen=usemodgen)
        else:
            self.x_SR = self.generator(self.x_LR, self.static_HR, r=r, is_training=False, use_mod_gen=usemodgen)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        if status == 'pretraining':
            self.g_loss = self.compute_losses(self.x_HR, self.x_SR, None, None, None, alpha_advers, isGAN=False)

            self.d_loss, self.disc_HR, self.disc_SR, self.d_variables = None, None, None, None
            self.advers_perf, self.content_loss, self.g_advers_loss = None, None, None

        elif status == 'training':
            self.disc_HR = self.discriminator(self.x_HR, reuse=False)
            self.disc_SR = self.discriminator(self.x_SR, reuse=True)
            if isWGAN:
                self.alpha = tf.random_uniform(
                    shape=[tf.shape(self.x_HR)[0],1,1,1], #tf.get_shape(x_HR)[0]?
                    minval=0.,
                    maxval=1.
                )
                self.interpolates = self.alpha*self.x_HR + ((1-self.alpha)*self.x_SR)
                self.gradients = tf.gradients(self.discriminator(self.interpolates,reuse=True), [self.interpolates])[0]
            else:
                self.gradients=None
            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            loss_out = self.compute_losses(self.x_HR, self.x_SR, self.disc_HR, self.disc_SR, self.gradients,alpha_advers, isGAN=True, isWGAN =isWGAN)
            if isWGAN:
                self.g_loss = loss_out[0]
                self.d_loss = loss_out[1]
                self.grad_pen = loss_out[2]
                self.content_loss = loss_out[3]
                self.d_SR = loss_out[4] #mean over batch
                self.d_HR = loss_out[5] #mean over batch
                self.advers_perf, self.g_advers_loss = None, None
            else:
                self.g_loss = loss_out[0]
                self.d_loss = loss_out[1]
                self.advers_perf = loss_out[2]
                self.content_loss = loss_out[3]
                self.g_advers_loss = loss_out[4]
                self.d_HR, self.d_SR, self.grad_pen = None, None, None

        else:
            self.g_loss, self.d_loss = None, None
            self.disc_HR, self.disc_SR, self.d_variables = None, None, None
            self.advers_perf, self.content_loss, self.g_advers_loss = None, None, None
            self.d_HR, self.d_SR, self.grad_pen = None, None, None

    
    def generator(self, x, static_HR, r, is_training=False, reuse=False, use_mod_gen=0):
        if is_training:
            N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]
        else:
            N, h, w, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.get_shape()[3]

        k, stride = 3, 1
        output_shape = [N, h+2*k, w+2*k, -1]    
        
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                if (static_HR is not None) & (not use_mod_gen):
                    C_in, C_out = C, 32
                    if self.is_stochastic:
                         C_out-=4
                else:
                    C_in, C_out = C, 64
                    if self.is_stochastic:
                          C_out-=8
                output_shape[-1] = C_out
                x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                x = tf.nn.relu(x)
            
            if (static_HR is not None) & (not use_mod_gen):
                #HR input branch, assuming r=[2,5]
                C_in, C_out = 2, 32
                if self.is_stochastic:
                    C_out-=4
                x2=static_HR
                ii=0
                for i, r_i in enumerate(r):
                    with tf.variable_scope('conv{}_HR'.format(i+ii)):
                        k1=r_i+1-i
                        k2=r_i+1-i-2*i
                        x2 = conv_layer_2d_special_stride(x2, [k1, k2, C_in, C_out], [r_i,1])
                        x2 = tf.nn.leaky_relu(x2, alpha=0.2)
                        C_in=C_out
                        ii+=1
                    with tf.variable_scope('conv{}_HR'.format(i+ii)):
                        k1=r_i+1-i-2*i
                        k2=r_i+1-i
                        x2 =conv_layer_2d_special_stride(x2, [k1, k2, C_in, C_out], [1,r_i])
                        x2 = tf.nn.leaky_relu(x2, alpha=0.2)

                x=tf.concat([x, x2], axis=3)
            if self.is_stochastic:
                noise=tf.random.normal([N, h, w, 8] ,
                                       mean=0.0,
                                       stddev=1.0)
                x=tf.concat([x,noise],axis=3)


            skip_connection = x

            # B residual blocks
            C_in, C_out = 64, 64
            output_shape[-1] = C_out
            for i in range(16):
                B_skip_connection = x

                with tf.variable_scope('block_{}a'.format(i+1)):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.nn.relu(x)

                with tf.variable_scope('block_{}b'.format(i+1)):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)

                x = tf.add(x, B_skip_connection)

            with tf.variable_scope('deconv2'):
                x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                x = tf.add(x, skip_connection)

            # Super resolution scaling
            r_prod = 1
            for i, r_i in enumerate(r):
                C_out = (r_i**2)*C_in
                with tf.variable_scope('deconv{}'.format(i+3)):
                    output_shape = [N, r_prod*h+2*k, r_prod*w+2*k, C_out]
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.depth_to_space(x, r_i)
                    x = tf.nn.relu(x)

                r_prod *= r_i
            
            output_shape[1] = r_prod*h+2*k
            output_shape[2] = r_prod*w+2*k
            if use_mod_gen:
                                    
                C_out=16
                output_shape[-1] = C_out
                with tf.variable_scope('deconv_SR'):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.nn.relu(x)
                
                if static_HR is not None:
                    
                    C_in=2
                    with tf.variable_scope('deconv_HRstatic1'):
                        x2 = deconv_layer_2d(static_HR, [k, k, C_out, C_in], output_shape, stride, k)
                        x2 = tf.nn.relu(x2)

                    C_in=C_out
                    with tf.variable_scope('deconv_HRstatic2'):
                        x2 = deconv_layer_2d(x2, [k, k, C_out, C_in], output_shape, stride, k)
                        x2 = tf.nn.relu(x2)

                    with tf.variable_scope('deconv_HRstatic3'):
                        x2 = deconv_layer_2d(x2, [k, k, C_out, C_in], output_shape, stride, k)
                        x2 = tf.nn.relu(x2)

                    x=tf.concat([x, x2], axis=3)
                
                    C_in=C_out*2
                    with tf.variable_scope('deconv_concat'):
                        x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                        x = tf.nn.relu(x)
                    
                C_in=C_out
                
                skip_connection = x

                # B residual blocks
                #C_in, C_out = 64, 64
                #output_shape[-1] = C_out
                for i in range(4):
                    B_skip_connection = x

                    with tf.variable_scope('block2_{}a'.format(i+1)):
                        x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                        x = tf.nn.relu(x)

                    with tf.variable_scope('block2_{}b'.format(i+1)):
                        x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)

                    x = tf.add(x, B_skip_connection)

                with tf.variable_scope('deconvX'):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.add(x, skip_connection)
                    
            C_out = 2
            output_shape[-1] = C_out
            with tf.variable_scope('deconv_out'):
                x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)

        return x



    def discriminator(self, x, reuse=False):
        N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer_2d(x, [3, 3, C, 32], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv2'):
                x = conv_layer_2d(x, [3, 3, 32, 32], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv3'):
                x = conv_layer_2d(x, [3, 3, 32, 64], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv4'):
                x = conv_layer_2d(x, [3, 3, 64, 64], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv5'):
                x = conv_layer_2d(x, [3, 3, 64, 128], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv6'):
                x = conv_layer_2d(x, [3, 3, 128, 128], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv7'):
                x = conv_layer_2d(x, [3, 3, 128, 256], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv8'):
                x = conv_layer_2d(x, [3, 3, 256, 256], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = flatten_layer(x)
            with tf.variable_scope('fully_connected1'):
                x = dense_layer(x, 1024)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('fully_connected2'):
                x = dense_layer(x, 1)

        return x

    def compute_losses(self, x_HR, x_SR, d_HR, d_SR, gradients, alpha_advers=0.001, isGAN=False, isWGAN=False):
        
        content_loss = tf.reduce_mean((x_HR - x_SR)**2, axis=[1, 2, 3])

        if isGAN:
            if isWGAN:
                LAMBDA=10
                d_SR=tf.reduce_mean(d_SR)
                d_HR=tf.reduce_mean(d_HR)
                g_loss = -alpha_advers*d_SR +  tf.reduce_mean(content_loss)
                d_loss = d_SR - d_HR
                
#                 alpha = tf.random_uniform(
#                     shape=[tf.shape(x_HR)[0],], #tf.get_shape(x_HR)[0]?
#                     minval=0.,
#                     maxval=1.
#                 )
#                 interpolates = alpha[:,None,None,None]*x_HR + ((1-alpha[:,None,None,None])*x_SR)
#                 gradients = tf.gradients(self.discriminator(interpolates,reuse=True), [interpolates])[0]

                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))


                gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                d_loss += LAMBDA*gradient_penalty
                
                return g_loss, d_loss, gradient_penalty, tf.reduce_mean(content_loss), d_SR, d_HR
            else:
                g_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_SR, labels=tf.ones_like(d_SR))

                d_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR, d_SR], axis=0),
                                                                    labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0))

                advers_perf = [tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) > 0.5, tf.float32)), # % true positive
                               tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) < 0.5, tf.float32)), # % true negative
                               tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) > 0.5, tf.float32)), # % false positive
                               tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) < 0.5, tf.float32))] # % false negative

                g_loss = tf.reduce_mean(content_loss) + alpha_advers*tf.reduce_mean(g_advers_loss)
                d_loss = tf.reduce_mean(d_advers_loss)

                return g_loss, d_loss, advers_perf, content_loss, g_advers_loss
        else:
            return tf.reduce_mean(content_loss)
    
