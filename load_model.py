# check 1 : 128*128, 256*256
# check 2 : validation_data_dir, test_data_dir
# check 3 : model001, model002, ...
# check 4 : model3, Mean_7, model4, Mean_9, ...

###################################################
# Code starts
import tensorflow as tf
import numpy as np
import load_dataset
n_classes = 5 # 01_BA, 02_EO, 03_LY, 04_MO, 05_NE

'''
###################################################
# load the images (128*128)
training_data_dir = '/home/ch/workspace/wbc/db/128_128_training/'
validation_data_dir = '/home/ch/workspace/wbc/db/128_128_validation/'
test_data_dir = '/home/ch/workspace/wbc/db/128_128_test/'
vali_test_data_dir = '/home/ch/workspace/wbc/db/128_128_vali_test/'
'''

'''
###################################################
# load the images (256*256)
training_data_dir = '/home/ch/workspace/wbc/db/256_256_training/'
validation_data_dir = '/home/ch/workspace/wbc/db/256_256_validation/'
test_data_dir = '/home/ch/workspace/wbc/db/256_256_test/'
'''

training_data_dir = '/home/ch/workspace/wbc/db/empty/'
validation_data_dir = '/home/ch/workspace/wbc/db/empty/'
test_data_dir = '/home/ch/workspace/wbc/gan/keras/classification_test/'

# training_data_dir, validation_data_dir, test_data_dir
training_data_x, training_data_y, test_data_x, test_data_y = load_dataset.main(training_data_dir, test_data_dir) #  training set, validation set or test set
print('mizno, training data x (image data) = ' + str(len(training_data_x)))
print('mizno, training data y (label) = ' + str(len(training_data_y)))
print('mizno, test data x (image data) = ' + str(len(test_data_x)))
print('mizno, test data y (label) = ' + str(len(test_data_y)))

###################################################
# load the model
sess = tf.Session()
# saver = tf.train.import_meta_graph('./model_holdout/model006-0/model.meta') # /model001/, /model002/, ...
# saver.restore(sess, tf.train.latest_checkpoint('./model_holdout/model006-0/')) # /model001/, /model002/, ...
saver = tf.train.import_meta_graph('./model_cv_sm/model003/model.meta') # /model001/, /model002/, ...
saver.restore(sess, tf.train.latest_checkpoint('./model_cv_sm/model003/')) # /model001/, /model002/, ...
print(str(saver))
# ./model/model000/model.meta
# ./model/model000/





###################################################
# load the function of the model
# need to check the saved model number
training = sess.graph.get_tensor_by_name("model1/Placeholder:0") # model0, 1, 2, 3, 4
print(training)
X = sess.graph.get_tensor_by_name("model1/Placeholder_1:0") # model0, 1, 2, 3, 4
print(X)
Y = sess.graph.get_tensor_by_name("model1/Placeholder_2:0") # model0, 1, 2, 3, 4
print(Y)
m_dropout_dense = sess.graph.get_tensor_by_name("model1/dropout_3/cond/Merge:0") # model0, 1, 2, 3, 4
print(m_dropout_dense)
logits = sess.graph.get_tensor_by_name("model1/dense_1/BiasAdd:0") # model0, 1, 2, 3, 4
print(logits)
correct_prediction = sess.graph.get_tensor_by_name("Equal_1:0") # Equal:0, Equal_1:0, Equal_2:0, Equal_3:0, Equal_4:0
print(correct_prediction)
accuracy = sess.graph.get_tensor_by_name("Mean_3:0") # Mean_1:0, Mean_3:0, Mean_5:0, Mean_7:0, Mean_9:0
print(accuracy)

####################################################
# Test model and check accuracy with BATCH for TEST SET
acc_batch_size = 10
acc_cur_idx = 0
acc_start = 0
acc_end = 0
acc_total_batch_size = int(len(test_data_x) / acc_batch_size)
acc_avg = 0

arr_correct = [0, 0, 0, 0, 0] # n_classes = 5
arr_count = [0, 0, 0, 0, 0] # n_classes = 5

for acc_step in range(0, acc_total_batch_size):
	acc_start = acc_cur_idx
	acc_end = acc_cur_idx + acc_batch_size
	acc_batch_x = np.array(test_data_x[acc_start:acc_end])
	acc_batch_y = np.array(test_data_y[acc_start:acc_end])
	acc_cur_idx = acc_cur_idx + acc_batch_size
	# print('mizno, batch_x: ' + str(acc_batch_x.size))
	# print('mizno, batch_y: ' + str(acc_batch_y.size))
	# print('mizno, cur_batch_idx: ' + str(acc_cur_idx))

	acc_result = sess.run(accuracy, feed_dict={X: acc_batch_x, Y: acc_batch_y, training: False})
	acc_avg += acc_result / acc_total_batch_size
	# print('Test Set Accuracy:', acc_avg)

	# to check the accuracy of each
	mizno_logits = sess.run(logits, feed_dict={X: acc_batch_x, training: False})
	# print(mizno_logits.size)
	for i in range(0, acc_batch_size):
		# print(str(np.argmax(mizno_logits[i])) + ' / ' + str(np.argmax(acc_batch_y[i])))
		for j in range(0, n_classes):
			if int(str(np.argmax(acc_batch_y[i]))) == j:
				arr_count[j] = arr_count[j]+1
				if int(str(np.argmax(mizno_logits[i]))) == int(str(np.argmax(acc_batch_y[i]))):
					arr_correct[j] = arr_correct[j]+1
				'''
				else: # to check the images that can not be claasified by model
					print("check: " + str(acc_step) + ", " + str(i))
				'''
				'''
                else: # for calculating the top k
                    # print(str(mizno_logits[i]) + ' / ' + str(np.argmax(mizno_logits[i])) + ' / ' + str(np.argmax(acc_batch_y[i])))
                    temp1 = np.argsort(mizno_logits[i])
                    # print(str(temp1) + ' / ' + str(temp1[0]) + ' / ' + str(temp1[1]) + ' / ' + str(temp1[2]) + ' / ' + str(temp1[3]) + ' / ' + str(temp1[4]))
                    for l in range(3, 4): # top 2 = (3, 4), top 3 = (2, 4)
                        if int(str(temp1[l])) == int(str(np.argmax(acc_batch_y[i]))):
                            # print('mizno, top 2')
                            arr_correct[j] = arr_correct[j]+1
                '''

print('Total Accuracy:', acc_avg)

temp1 = 0;
temp2 = 0;

for k in range(0, n_classes):
	print('correct images of ' + str(k) + ' class: ' + str(int(arr_correct[k])))
	print('total images of ' + str(k) + ' class: ' + str(int(arr_count[k])))
	print('accuracy of '+ str(k) + ' class: ' + str(float(arr_correct[k])/float(arr_count[k])))
	temp1 = temp1 + float(arr_correct[k]);
	temp2 = temp2 + float(arr_count[k]);

print('Total Accuracy:', float(temp1)/float(temp2))


	
