import sys
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

class vocab:
	def __init__(self):
		self.label_to_id = {}
		self.id_to_label = []

	def convert_label(self, s):
		if s not in self.label_to_id:
			self.label_to_id[s] = len(self.id_to_label)
			self.id_to_label.append(s)
		return self.label_to_id[s]

	def convert_id(self, i):
		assert(i < len(self.id_to_label))
		return self.id_to_label[i]

	def __len__(self):
		return len(self.id_to_label)

source_label_vocab = vocab()
target_label_vocab = vocab()
counts = {}
def convert_index(i):
	assert i >= 0 and i < len(source_label_vocab) + len(target_label_vocab)
	if i < len(source_label_vocab):
		return source_label_vocab.convert_id(i)
	else:
		return target_label_vocab.convert_id(i - len(source_label_vocab))

print >>sys.stderr, 'Reading data...'
for line in sys.stdin:
	# Read each line in, and convert the count to an integer
	source_label, target_label, count = line.split('\t')
	count = int(count)

	source_label = source_label_vocab.convert_label(source_label)
	target_label = target_label_vocab.convert_label(target_label)

	counts[source_label, target_label] = count

print >>sys.stderr, 'Creating matrix...'
size = len(source_label_vocab) + len(target_label_vocab)
count_matrix = np.zeros((size, size))
for (source_label, target_label), count in counts.iteritems():
	target_label += len(source_label_vocab)
	count_matrix[source_label][target_label] = count
	count_matrix[target_label][source_label] = count

# Add a small amount to all entries. This helps to smooth the distribution.
epsilon = 1.0e-1
for i in range(len(source_label_vocab)):
	for j in range(len(source_label_vocab), len(source_label_vocab) + len(target_label_vocab)):
		count_matrix[i][j] += epsilon
		count_matrix[j][i] += epsilon

probability_matrix = np.copy(count_matrix)
for i in range(size):
	s = sum(probability_matrix[i])
	probability_matrix[i] /= s
	#probability_matrix[i] *= np.log(s)

print >>sys.stderr, 'Computing covariance...'
random_matrix = count_matrix
mean = np.zeros(size)
for i in range(size):
	mean += random_matrix[i]
mean /= size

covariance_matrix = np.zeros((size, size))
for i in range(size):
	for j in range(i + 1, size):
		d = np.dot(random_matrix[i] - mean, random_matrix[j] - mean)
		covariance_matrix[i][j] = d
		covariance_matrix[j][i] = d

print >>sys.stderr, 'Running SVD...'
matrix_to_decompose = covariance_matrix
u, s, v = np.linalg.svd(matrix_to_decompose, True, True)
non_zero_guys = [v for v in s if abs(v) > 1.0e-7]
print >>sys.stderr, len(non_zero_guys), '/', len(s), 'non-zero SVs'
"""print count_matrix.shape
print u.shape
print s.shape
print v.shape"""

total_s = sum(s)
n = 0
p = 0.99
sum_so_far = 0.0
while n < len(s):
	sum_so_far += s[n]
	if sum_so_far >= p * total_s:
		break
	n += 1
print >>sys.stderr, '%d dimensions explain %f of the variance' % (n, p)

print >>sys.stderr, 'Running k-means...'
vectors = matrix_to_decompose.dot(u.T[:, :n])
clusterer = KMeans(init='k-means++', n_clusters=n)
clusters = clusterer.fit_predict(vectors)

cluster_map = defaultdict(list)
for i in range(len(clusters)):
	side = "Source"
	vocab = source_label_vocab
	j = i
	if i >= len(source_label_vocab):
		j = i - len(source_label_vocab)
		side = "Target"
		vocab = target_label_vocab	

	cluster_map[clusters[i]].append((side, vocab.convert_id(j), sum(count_matrix[i])))
	#print '%s\t%s\t%d' % (side, vocab.convert_id(j), clusters[i])

for cluster_id, labels in sorted(cluster_map.iteritems()):
	print 'C%d' % cluster_id, ' '.join(["%s_%s(%d)" % label for label in sorted(labels, key=lambda (side, label, count): (1 if side == "Source" else 0, count), reverse=True)])
