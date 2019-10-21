import tensorflow as tf
import numpy as np
import load_dataset
import sklearn.metrics as metrics

n_classes = 5 # 01_BA, 02_EO, 03_LY, 04_MO, 05_NE
f1_y_label = []
f1_y_pred = []

###################################################
# load the dataset

training_data_dir = '/home/ch/workspace/wbc/db/empty/'
test_data_dir = '/home/ch/workspace/wbc/gan/keras/crop_resize/'

training_data_x, training_data_y, test_data_x, test_data_y = load_dataset.main(training_data_dir, test_data_dir)
print('mizno, training data x (image data) = ' + str(len(training_data_x)))
print('mizno, training data y (label) = ' + str(len(training_data_y)))
print('mizno, test data x (image data) = ' + str(len(test_data_x)))
print('mizno, test data y (label) = ' + str(len(test_data_y)))

###################################################
# load the model

sess = tf.Session()
saver = tf.train.import_meta_graph('./model_cv_sm/model003/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model_cv_sm/model003/'))
print(str(saver))

###################################################
# load the function of the model

training = sess.graph.get_tensor_by_name("model1/Placeholder:0")
X = sess.graph.get_tensor_by_name("model1/Placeholder_1:0")
Y = sess.graph.get_tensor_by_name("model1/Placeholder_2:0")
m_dropout_dense = sess.graph.get_tensor_by_name("model1/dropout_3/cond/Merge:0")
logits = sess.graph.get_tensor_by_name("model1/dense_1/BiasAdd:0")
correct_prediction = sess.graph.get_tensor_by_name("Equal_1:0")
accuracy = sess.graph.get_tensor_by_name("Mean_3:0")
print(training)
print(X)
print(Y)
print(m_dropout_dense)
print(logits)
print(correct_prediction)
print(accuracy)

####################################################
# Test model and check accuracy with BATCH for TEST SET

acc_batch_size = 5
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
		f1_y_label.append(np.argmax(acc_batch_y[i]))
		f1_y_pred.append(np.argmax(mizno_logits[i]))
		for j in range(0, n_classes):
			if int(str(np.argmax(acc_batch_y[i]))) == j:
				arr_count[j] = arr_count[j]+1
				if int(str(np.argmax(mizno_logits[i]))) == int(str(np.argmax(acc_batch_y[i]))):
					arr_correct[j] = arr_correct[j]+1

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

# Print the confusion matrix
print(metrics.confusion_matrix(f1_y_label, f1_y_pred))

# Print the precision and recall, among other metrics
print(metrics.classification_report(f1_y_label, f1_y_pred, digits=3))	



