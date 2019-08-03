#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf 
import numpy as np 
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
rng = np.random.RandomState(23455)


class IR_quantum(object):
	def __init__(
		self, max_input_query,max_input_docu, vocab_size, embedding_size ,batch_size,
		embeddings,filter_sizes,num_filters,l2_reg_lambda = 0.0,trainable = True,
		pooling = 'max',overlap_needed = True,extend_feature_dim = 10):

		# self.dropout_keep_prob = dropout_keep_prob
		self.num_filters = num_filters
		self.embeddings = embeddings
		self.embedding_size = embedding_size
		self.vocab_size = vocab_size
		self.trainable = trainable
		self.filter_sizes = filter_sizes
		self.pooling = pooling
		self.total_embedding_dim = embedding_size
		self.batch_size = batch_size
		self.l2_reg_lambda = l2_reg_lambda
		self.para = []
		self.max_input_query = max_input_query
		self.max_input_docu = max_input_docu
		self.hidden_num = 128
		self.rng = 23455
		self.overlap_need = overlap_needed
		# if self.overlap_need:
		# 	self.total_embedding_dim = embedding_size + extend_feature_dim
		# else:
		# 	self.total_embedding_dim = embedding_size
		self.extend_feature_dim = extend_feature_dim
		self.conv1_kernel_num = 32
		self.conv2_kernel_num = 32

		print (self.max_input_query)
		print (self.max_input_docu)

	def creat_placeholder(self):
		self.query = tf.placeholder(tf.int32,[self.batch_size,self.max_input_query],name = "input_query")
		self.document = tf.placeholder(tf.int32,[self.batch_size,self.max_input_docu],name = "input_document")
		self.input_label = tf.placeholder(tf.float32,[self.batch_size,1],name = "input_label")

		self.q_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_query],name = "q_overlap")
		self.d_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_docu],name = "d_overlap")
		
	def density_weighted(self):
		self.weighted_q = tf.Variable(tf.ones([1,self.max_input_query,1,1]),name = "weight_q")
		self.para.append(self.weighted_q)
		self.weighted_d = tf.Variable(tf.ones([1,self.max_input_docu,1,1]),name = "weight_d")
		self.para.append(self.weighted_d)

	def load_embeddings(self):
		with tf.name_scope("embedding"):
			print ("load embeddings")
			self.words_embeddings = tf.Variable(np.array(self.embeddings),name = "W",dtype = "float32",trainable = self.trainable)
			# self.para.append(self.words_embeddings)
			self.overlap_w = tf.get_variable("overlap_w",shape = [3,self.embedding_size],initializer = tf.random_normal_initializer())
			# self.para.append(self.overlap_w)
		self.embedded_chars_q = tf.expand_dims(self.concat_embedding(self.query,self.q_overlap),-1)
		self.embedded_chars_d = tf.expand_dims(self.concat_embedding(self.document,self.d_overlap),-1)

		print ("embedded_chars_q shape:{}".format(self.embedded_chars_q.get_shape()))
		print ("embedded_chars_d shape:{}".format(self.embedded_chars_d.get_shape()))
		# exit()
	
	def concat_embedding(self, words_indice, overlap_indice):
		embedded_chars = tf.nn.embedding_lookup(self.words_embeddings,words_indice)
		overlap_embedding = tf.nn.embedding_lookup(self.overlap_w,overlap_indice)
		if self.overlap_need:
			return tf.reduce_sum([embedded_chars,overlap_embedding],0)
			# return tf.concat([embedded_chars,overlap_embedding],2)
		else:
			return embedded_chars

	def joint_representation(self):
		self.density_q = self.density_matrix(self.embedded_chars_q,self.weighted_q)
		self.density_d = self.density_matrix(self.embedded_chars_d,self.weighted_d)
		self.M_qd = tf.matmul(self.density_q,self.density_d)

	def density_matrix(self, embedding , weighted):
		self.norm = tf.nn.l2_normalize(embedding,2)
		# reverse_matrix = tf.transpose(self.norm, perm=[0,1,3,2])
		print (self.norm.shape)
		reverse_matrix = tf.transpose(self.norm,perm = [0,1,3,2])
		#out pruduct
		q_d = tf.matmul(self.norm, reverse_matrix)
		return tf.reduce_sum(tf.multiply(q_d, weighted),1)

	def feed_neural_work(self):
		with tf.name_scope('regression'):
			W = tf.Variable(tf.zeros(shape = [(self.total_embedding_dim - self.filter_sizes[0]+1) * self.num_filters * 2, 1]),name = "W")
			b = tf.Variable(tf.zeros([1]),name = "b")
			
			self.para.append(W)
			self.para.append(b)

			self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "score")
			# self.scores = tf.nn.relu(self.logits)
			self.scores = self.logits
	
	def create_loss(self):
		l2_loss = tf.constant(0.0)
		for p in self.para:
			l2_loss += tf.nn.l2_loss(p)
		with tf.name_scope("loss"):
			# p_pre = tf.nn.softmax(self.logits)
			self.p_label = tf.nn.softmax(self.input_label,dim = 0)
			# p_label1 = tf.nn.softmax(self.input_label)
			# self.p_label = self.calculate_probability(self.input_label)
			# print ("p_label")
			# print (p_label)
			# losses = tf.square(self.input_label - self.logits)
			# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.logits,dim = 0))
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.p_label, logits=self.logits,dim = 0))
			# pi_regularization = tf.reduce_sum(self.weighted_q) -1 + tf.reduce_sum(self.weighted_d) - 1
			# self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss + 0.0001 * tf.nn.l2_loss(pi_regularization)
			# print ("cross_entropy")
			# print (cross_entropy)

			# in_label = tf.reshape(self.input_label,[self.batch_size])
			# out_logits = tf.reshape(self.logits,[self.batch_size])
			# zero = tf.constant(0.0)
			# pos_index = tf.where(tf.equal(in_label>zero,True))
			# neg_index = tf.where(tf.equal(in_label>zero,False))
			# pos = tf.reduce_sum(tf.gather_nd(out_logits,pos_index))
			# neg = tf.reduce_sum(tf.gather_nd(out_logits,neg_index))			

			# margin_pos_neg = tf.maximum(0.0,1.0+ neg -pos)

			# print ("margin_pos_neg shape:{}".format(margin_pos_neg.get_shape()))
			self.loss = cross_entropy

			# self.loss = cross_entropy + self.l2_reg_lambda * l2_loss
			# self.loss = cross_entropy + self.l2_reg_lambda * l2_loss
		# with tf.name_scope("accuracy"):


	def convolution(self):
		self.kernels = []
		for i,filter_size in enumerate(self.filter_sizes):
			with tf.name_scope('conv-pool-%s'% filter_size):
				filter_shape = [filter_size,filter_size,1,self.num_filters]
				fan_in = np.prod(filter_shape[:-1])
				fan_out = (filter_shape[-1] * np.prod(filter_shape[:2]))
				W_bound = np.sqrt(6.0/(fan_out+fan_in))

				W = tf.Variable(np.asarray(rng.uniform(low = -W_bound,high = W_bound,size = filter_shape),dtype = 'float32'))
				b = tf.Variable(tf.constant(0.0, shape = [self.num_filters]), name = "b")
				self.kernels.append((W,b))
				self.para.append(W)
				self.para.append(b)

		# self.num_filters_total = self.num_filters * len(self.filter_sizes)
		# self.qd = self.narrow_convolution(tf.expand_dims(self.M_qd,-1))
		# print (self.M_qd.get_shape())
		# shape = (?,300,300)
		self.qd = self.narrow_convolution(tf.expand_dims(self.M_qd,-1))

	def max_pooling(self,conv):
		pooled = tf.nn.max_pool(
				conv,
				ksize = [1,self.max_input_query,self.max_input_docu,1],
				strides = [1,1,1,1],
				padding = "VALID",
				name = "pool"
			)
		return pooled

	def pooling_graph(self):
		with tf.name_scope("pooling"):
			raw_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qd,1))
			col_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qd,2))
			self.represent = tf.concat([raw_pooling,col_pooling],1)

	def narrow_convolution(self,embedding):
		cnn_outputs = []
		for i, filter_size in enumerate(self.filter_sizes):
			conv = tf.nn.conv2d(
				embedding,
				self.kernels[i][0],
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="conv-1"
			)
			self.see = conv
			h = tf.nn.tanh(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
			cnn_outputs.append(h)
		cnn_reshaped = tf.concat(cnn_outputs,3)
		return cnn_reshaped


	def build_graph(self):
		self.creat_placeholder()
		self.load_embeddings()
		self.density_weighted()
		self.joint_representation()
		self.convolution()
		self.pooling_graph()
		self.feed_neural_work()
		self.create_loss()

		# self.creat_placeholder()
		# self.load_embeddings()
		# self.density_weighted()
		# # self.composite_and_partialTrace()
		# # self.convolution()
		# # self.pooling_graph()
		# self.CNN_network()
		# # self.CNN_network11()
		# self.feed_neural_work()
		# # self.feed_neural_work1()
		# self.create_loss()
		print ("end build graph")
		# exit()






























# class IR_quantum(object):
# 	def __init__(
# 		self, max_input_query,max_input_docu, vocab_size, embedding_size ,batch_size,
# 		embeddings,dropout_keep_prob,filter_sizes,num_filters,l2_reg_lambda = 0.0,trainable = True,
# 		pooling = 'max'):

# 		self.dropout_keep_prob = dropout_keep_prob
# 		self.num_filters = num_filters
# 		self.embeddings = embeddings
# 		self.embedding_size = embedding_size
# 		self.vocab_size = vocab_size
# 		self.trainable = trainable
# 		self.filter_sizes = filter_sizes
# 		self.pooling = pooling
# 		self.total_embedding_dim = embedding_size
# 		self.batch_size = batch_size
# 		self.l2_reg_lambda = l2_reg_lambda
# 		self.para = []
# 		self.max_input_query = max_input_query
# 		self.max_input_docu = max_input_docu
# 		self.rng = 23455
# 		print (self.max_input_query)
# 		print (self.max_input_docu)
# 	def creat_placeholder(self):
# 		self.query = tf.placeholder(tf.int32,[None,self.max_input_query],name = "input_query")
# 		self.document = tf.placeholder(tf.int32,[None,self.max_input_docu],name = "input_document")
# 		self.input_label = tf.placeholder(tf.float32,[None,1],name = "input_label")

# 	def density_weighted(self):
# 		self.weighted_q = tf.Variable(tf.ones([1,self.max_input_query,1,1]),name = "weight_q")
# 		self.para.append(self.weighted_q)
# 		self.weighted_d = tf.Variable(tf.ones([1,self.max_input_docu,1,1]),name = "weight_d")
# 		self.para.append(self.weighted_d)

# 	def load_embeddings(self):
# 		with tf.name_scope("embedding"):
# 			print ("load embeddings")
# 			words_emb= tf.Variable(np.array(self.embeddings),name = "W",dtype = "float32",trainable = self.trainable)
# 			self.words_embeddings = words_emb
# 			self.para.append(self.words_embeddings)

# 		self.embedded_chars_q = self.get_embedding(self.query)
# 		self.embedded_chars_d = self.get_embedding(self.document)

# 	def get_embedding(self,words_indice):
# 		embedded_chars = tf.nn.embedding_lookup(self.words_embeddings,words_indice)
# 		# return embedded_chars
# 		return tf.expand_dims(embedded_chars,-1)
# 	def joint_representation(self):
# 		self.density_q = self.density_matrix(self.embedded_chars_q,self.weighted_q)
# 		self.density_d = self.density_matrix(self.embedded_chars_d,self.weighted_d)
# 		self.M_qd = tf.matmul(self.density_q,self.density_d)

# 	def density_matrix(self, embedding , weighted):
# 		self.norm = tf.nn.l2_normalize(embedding,2)
# 		# reverse_matrix = tf.transpose(self.norm, perm=[0,1,3,2])
# 		print (self.norm.shape)
# 		reverse_matrix = tf.transpose(self.norm,perm = [0,1,3,2])
# 		#out pruduct
# 		q_d = tf.matmul(self.norm, reverse_matrix)
# 		return tf.reduce_sum(tf.multiply(q_d, weighted),1)

# 	def feed_neural_work(self):
# 		with tf.name_scope('regression'):
# 			W = tf.Variable(tf.zeros(shape = [(self.total_embedding_dim - self.filter_sizes[0]+1) * self.num_filters * 2, 1]),name = "W")
# 			b = tf.Variable(tf.zeros([1]),name = "b")
			
# 			self.para.append(W)
# 			self.para.append(b)

# 			self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "score")
# 			# self.scores = tf.nn.relu(self.logits)
# 			self.scores = self.logits

# 			# self.prediction = 
# 	# def calculate_probability(self, score):
# 	# 	l = []
# 	# 	for i in score:
# 	# 		l.append(i[0])
# 	# 	exp_s = np.exp(l)
# 	# 	return exp_s/np.sum(exp_s)

# 	def create_loss(self):
# 		l2_loss = tf.constant(0.0)
# 		for p in self.para:
# 			l2_loss += tf.nn.l2_loss(p)
# 		with tf.name_scope("loss"):
# 			# p_pre = tf.nn.softmax(self.logits)
# 			self.p_label = tf.nn.softmax(self.input_label,dim = 0)
# 			# p_label1 = tf.nn.softmax(self.input_label)
# 			# self.p_label = self.calculate_probability(self.input_label)
# 			# print ("p_label")
# 			# print (p_label)
# 			# losses = tf.square(self.input_label - self.logits)
# 			cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.p_label, logits=self.logits,dim = 0))
# 			pi_regularization = tf.reduce_sum(self.weighted_q) -1 + tf.reduce_sum(self.weighted_d) - 1
# 			# self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss + 0.0001 * tf.nn.l2_loss(pi_regularization)
# 			print ("cross_entropy")
# 			print (cross_entropy)
# 			self.loss = tf.reduce_mean(cross_entropy)
# 		# with tf.name_scope("accuracy"):

# 	def convolution(self):
# 		self.kernels = []
# 		for i,filter_size in enumerate(self.filter_sizes):
# 			with tf.name_scope('conv-pool-%s'% filter_size):
# 				filter_shape = [filter_size,filter_size,1,self.num_filters]
# 				fan_in = np.prod(filter_shape[:-1])
# 				fan_out = (filter_shape[-1] * np.prod(filter_shape[:2]))
# 				W_bound = np.sqrt(6.0/(fan_out+fan_in))

# 				W = tf.Variable(np.asarray(rng.uniform(low = -W_bound,high = W_bound,size = filter_shape),dtype = 'float32'))
# 				b = tf.Variable(tf.constant(0.0, shape = [self.num_filters]), name = "b")
# 				self.kernels.append((W,b))
# 				self.para.append(W)
# 				self.para.append(b)

# 		self.num_filters_total = self.num_filters * len(self.filter_sizes)
# 		self.qd = self.narrow_convolution(tf.expand_dims(self.M_qd,-1))

# 	def max_pooling(self,conv):
# 		pooled = tf.nn.max_pool(
# 				conv,
# 				ksize = [1,self.max_input_query,self.max_input_docu,1],
# 				strides = [1,1,1,1],
# 				padding = "VALID",
# 				name = "pool"
# 			)
# 		return pooled

# 	def pooling_graph(self):
# 		with tf.name_scope("pooling"):
# 			raw_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qd,1))
# 			col_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qd,2))
# 			self.represent = tf.concat([raw_pooling,col_pooling],1)

# 	def narrow_convolution(self,embedding):
# 		cnn_outputs = []
# 		for i, filter_size in enumerate(self.filter_sizes):
# 			conv = tf.nn.conv2d(
# 				embedding,
# 				self.kernels[i][0],
# 				strides=[1, 1, 1, 1],
# 				padding='VALID',
# 				name="conv-1"
# 			)
# 			self.see = conv
# 			h = tf.nn.tanh(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
# 			cnn_outputs.append(h)
# 		cnn_reshaped = tf.concat(cnn_outputs,3)
# 		return cnn_reshaped


# 	def build_graph(self):
# 		self.creat_placeholder()
# 		self.load_embeddings()
# 		self.density_weighted()
# 		self.joint_representation()
# 		self.convolution()
# 		self.pooling_graph()
# 		self.feed_neural_work()
# 		self.create_loss()

# # if __name__ == "__main__":
