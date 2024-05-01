################################################################################
#                                                                              #
#  #   #  #             #####  #####  #  #####  #####  #####  #####  #  #####  #
#  ## ##  #             #      #   #  #  #        #    #   #  #      #  #      #
#  # # #  #      #####  #####  #####  #  #####    #    #####  #####  #  #####  #
#  #   #  #             #      #      #      #    #    #   #      #  #      #  #
#  #   #  #####         #####  #      #  #####    #    #   #  #####  #  #####  #
#                                                                              #
#  ###    #####  #####  #####  #####  #####  #  #####  #   #                   #
#  #  #   #        #    #      #        #    #  #   #  ##  #                   #
#  #   #  #####    #    #####  #        #    #  #   #  # # #                   #
#  #  #   #        #    #      #        #    #  #   #  #  ##                   #
#  ###    #####    #    #####  #####    #    #  #####  #   #                   #
#                                                                              #
#                                                                              #
#  Transformer Neural Network for High Order Epistasis Detection               #
#  Contact: miguel.graca@inesc-id.pt                                           #
#                                                                              #
################################################################################

from argparse import ArgumentParser
import tensorflow as tf
from itertools import combinations
from os import listdir
from os.path import isfile, join, isdir
import zipfile
import numpy as np
import time

#Class to implement the Top-KAST constraint function
class Top_KAST(tf.keras.constraints.Constraint):
    def __init__(self, S):
        self.S = S

    def __call__(self, w):
        wshape = w.shape 
        non_zero = wshape[0]*wshape[1] - int(wshape[0]*wshape[1]*self.S)
        w = tf.reshape(w, shape = [wshape[0]*wshape[1]])
        values, indices = tf.math.top_k(tf.abs(w), k=non_zero)
        indices = tf.cast(indices, dtype = tf.int32)
        indices = tf.expand_dims(indices, axis = 1)
        shape = tf.constant([wshape[0]*wshape[1]])
        new_w = tf.scatter_nd(indices, values, shape)
        new_w = tf.reshape(new_w, shape = [wshape[0],wshape[1]])
        return new_w

#Class to implement Single-Query Attention
class Attention(tf.keras.layers.Layer):
    def __init__(self, key_dim, num_heads, qkv_activation, sparsity, n_devices):
        super(Attention, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.qkv_activation = qkv_activation
        self.sparsity = sparsity
        self.n = n_devices

    def build(self, input_shape):
        qkv_shape = input_shape[2]*self.num_heads
        self._v_layer = tf.keras.layers.Dense(qkv_shape, activation = self.qkv_activation, kernel_initializer = tf.keras.initializers.Identity(gain=1.0), kernel_constraint = Top_KAST(S = self.sparsity))
        self._output_layer = tf.keras.layers.Dense(self.key_dim, activation = self.qkv_activation, kernel_constraint = Top_KAST(S = self.sparsity))

    def compute_attention(self, query, key, value):
        kv_shape = (value.shape[0], value.shape[1]-1, value.shape[2], self.num_heads)
        value = tf.reshape(self._v_layer(value[:,0:value.shape[1]-1,:]), kv_shape)
        key = tf.transpose(key, [0, 2, 1, 3])
        attention_scores_2 = tf.einsum('ijkl,inol->iljo', query, key)

        attention_scores = tf.keras.layers.Softmax(axis = -1)(attention_scores_2)        
        attention_output = tf.einsum('ijkl,ilmj->ijlm',attention_scores, value)
        return attention_output, tf.reduce_sum(attention_scores, axis = 1)

    def call(self, query, key, value, return_attention_scores = True):

        attention_output, attention_scores = self.compute_attention(query, key, value)
        attention_output = self._output_layer(tf.reduce_mean(attention_output, axis = 1))

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

#Class to implement Embedding layers
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim, trainable = False)

    def call(self, x):
        
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

#Class to implement a Transformer-Encoder Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, batch, input_size, sparsity, n_devices):
        super(TransformerBlock, self).__init__()
        self.att = Attention(embed_dim, num_heads, 'tanh', sparsity, n_devices)
        self.w = tf.Variable(initial_value = tf.zeros_initializer()(shape=(int(batch/n_devices), 
                            1,input_size-1)), trainable = False, dtype=tf.float32)

        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(embed_dim, activation = 'tanh'),]
        )
       
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.01)
        self.dropout2 = tf.keras.layers.Dropout(0.01)
        
    def call(self, inputs, query, key, training=True):
        attn_output, attn_scores = self.att(query = query, key = key, value = inputs)
        self.w.assign_add(tf.cast(attn_scores, dtype = tf.float32))
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs[:,0:inputs.shape[1]-1,:] + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(ffn_output + out1), attn_scores

#Function to build transformer neural network
def build_model(input_size, batch, splits, comb_order, sparsity, n_devices):

    num_heads = 1
    vocab = 3
    embed_dim = 32

    inputs = tf.keras.layers.Input(shape=(input_size,), batch_size=int(batch))
    outputs = []
    comb = list(combinations([i for i in range(splits)],comb_order))

    maxlen = input_size - 1
    embedding_layer = TokenAndPositionEmbedding(maxlen+1, vocab, int(embed_dim))
    y = embedding_layer(inputs)
    
    kv_shape = (y.shape[0], y.shape[1]-1, y.shape[2], num_heads)
    q_shape = (y.shape[0], 1, y.shape[2], num_heads)
    q_layer = tf.keras.layers.Dense(q_shape[2]*num_heads, activation = 'tanh', kernel_initializer = tf.keras.initializers.Identity(gain=1.0), kernel_constraint = Top_KAST(S = sparsity))
    k_layer = tf.keras.layers.Dense(kv_shape[2]*num_heads, activation = 'tanh', kernel_initializer = tf.keras.initializers.Identity(gain=1.0), kernel_constraint = Top_KAST(S = sparsity))
    query = tf.reshape(q_layer(y[:,y.shape[1]-1:,:]), q_shape)
    key = tf.reshape(k_layer(y[:,0:y.shape[1]-1,:]), kv_shape)

    for i in range(len(comb)):
        indexes = [comb[i][j] for j in range(comb_order)]
        sizes = [int((indexes[j]+1)*maxlen/splits) - int(indexes[j]*maxlen/splits) for j in range(comb_order)]
        key_list = [key[:,int(indexes[j]*maxlen/splits):int((indexes[j]+1)*maxlen/splits),:,:] for j in range(comb_order)]
        y_list = [y[:,int(indexes[j]*maxlen/splits):int((indexes[j]+1)*maxlen/splits),:] for j in range(comb_order)]

        key_list.append(key[:,maxlen:maxlen+1,:,:])
        y_list.append(y[:,maxlen:maxlen+1,:])

        k2 = tf.keras.layers.Concatenate(axis = 1)(key_list)
        y2 = tf.keras.layers.Concatenate(axis = 1)(y_list)

        transformer_block = TransformerBlock(embed_dim, num_heads, batch, sum(sizes) + 1, sparsity, n_devices)
        z, scores = transformer_block(y2,query,k2)
        z = tf.keras.layers.GlobalAveragePooling1D(data_format = 'channels_first')(z)
        z = tf.keras.layers.Dropout(0.01)(z)
        z = tf.keras.layers.Dense(embed_dim, activation = 'tanh')(z) 
        z = tf.keras.layers.Dropout(0.01)(z)
        z = tf.keras.layers.Dense(1, activation = "sigmoid")(z)
        outputs.append(z)

    model = tf.keras.Model(inputs, outputs)
    return model

#Function to read dataset
def get_dataset(path):
    #Read file
    snp_names = path.readline()
    data = np.array(np.loadtxt(path, delimiter='\t'))
    return data, data[:,data.shape[1] - 1]

#Spectral Embedding Function
def se_numpy(X, k_neighbors, low_dims):
    n_samples = X.shape[0]

    # neighbor combination matrix
    A = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        Y = X[i, None, :] - X[None, :, :]
        dist_mat = np.sqrt(np.einsum('ijk,ijk->ij', Y, Y))
        neighbors = np.argsort(dist_mat, axis = 1)[:, 1 : k_neighbors + 1]
        A[i, neighbors[0,:]] += 1

    A = 0.5*(A + A.T)
    np.fill_diagonal(A, 0)
    w = A.sum(axis=0)

    # sparse matrix M
    lap = A*(-1) + np.diag(w)
    eigen_values, eigen_vectors = np.linalg.eigh(lap)   
    index = np.argsort(np.abs(eigen_values))[1 : low_dims + 1]
    selected_eig_vectors = eigen_vectors[:,index]

    low_X = selected_eig_vectors
    return low_X

#Locally Linear Embedding Function
def lle_numpy(X, k_neighbors, low_dims):
        n_samples = X.shape[0]

        # neighbor combination matrix
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            Y = X[i, None, :] - X[None, :, :]
            dist_mat = np.sqrt(np.einsum('ijk,ijk->ij', Y, Y))
            neighbors = np.argsort(dist_mat, axis = 1)[:, 1 : k_neighbors + 1]
            mat_z = X[i] - X[neighbors[0,:]]
            mat_c = np.matmul(mat_z, np.transpose(mat_z))
            w = np.matmul(np.matmul(np.linalg.inv(np.matmul(mat_c.T,mat_c)), mat_c.T), np.ones(mat_c.shape[0]))
            W[i, neighbors[0,:]] = w / np.sum(w) 

        # sparse matrix M
        I_W = np.eye(n_samples) - W
        M = np.matmul(np.transpose(I_W), I_W)
        # solve the d+1 lowest eigen values
        eigen_values, eigen_vectors = np.linalg.eigh(M)   
        index = np.argsort(np.abs(eigen_values))[1 : low_dims + 1]
        selected_eig_values = eigen_values[index]
        selected_eig_vectors = eigen_vectors[:,index]

        eig_values = selected_eig_values
        low_X = selected_eig_vectors
        return low_X

#Principal Components Embedding Function
def pca_numpy(data, k):
    data = data - data.mean(axis = 0)
    cov = np.cov(data.T) / data.shape[0]
    v, w = np.linalg.eig(cov)

    idx = v.argsort()[::-1]
    v = v[idx]
    w = w[:,idx]

    return data.dot(w[:, :k])

def decode_attention_scores(weights):
    som = tf.math.reduce_sum(weights, 0)
    som = tf.math.reduce_sum(som, 0)
    final = som.numpy()

    final = final/(weights.shape[2]*weights.shape[0])
    return final

def choose_snps(scores, k):
    snps = []
    for i in range(k):
        idx = np.argmax(scores)
        snps.append(idx)
        scores[idx] = -100000

    return snps

def training_model(path, splits, comb_order, top, strategy, n_devices, sparsity, epoch_num):

    #Checking path for datasets
    if(isdir(path)): #Check if it is a directory
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        out_filename = files[0][0:len(files[0]) - 5] + '_Results.txt'
    elif(zipfile.is_zipfile(path)): #Check if it is a zip file
        zip_epi = zipfile.ZipFile(path)
        files = zip_epi.namelist()
        out_filename = path[0:len(path[0]) - 5] + '_Results.txt'
    else: 
        files = [path] #Single file
        out_filename = path[0:(len(path[0]) - 5)] + '_Results.txt'

    batch, snpflag = 32, 0
    outfile = open(out_filename, 'w')

    for f in files:
        print(f)
        if(zipfile.is_zipfile(path)):
            f_op = zip_epi.open(f)
        else:
            f_op = open(f, "r")
        
        #Opening Dataset
        dataset, label = get_dataset(f_op)
        snpsize = dataset.shape[1] - 1
        samples = int(dataset.shape[0]) - int(dataset.shape[0]%(int(batch/n_devices)))

        #Calculating Positional Embeddings
        pos = se_numpy(dataset.T, 5, 32)

        if(strategy is None):
            comb = list(combinations([i for i in range(splits)],comb_order))
            if(snpsize != snpflag):
                snpflag = snpsize
                model = build_model(dataset.shape[1], batch, splits, comb_order, sparsity, 1)
                optim = tf.keras.optimizers.Adam(learning_rate = 0.001)
                model.compile(optimizer = optim, loss = ['binary_crossentropy']*len(comb), metrics = [tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])
        else:
            comb = list(combinations([i for i in range(splits)],comb_order))
            with strategy.scope():
                if(snpsize != snpflag):
                    snpflag = snpsize
                    model = build_model(dataset.shape[1], batch, splits, comb_order, sparsity, n_devices)
                    optim = tf.keras.optimizers.Adam(learning_rate = 0.001)
                    model.compile(optimizer = optim, loss = ['binary_crossentropy']*len(comb), metrics = [tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])

        #Training the transformer
        x_train, y_train = dataset[0:samples,:], label[0:samples]
        for layer in model.layers:
            if 'token_and_position_embedding' in layer.name:
                layer.non_trainable_weights[0].assign(pos)
        
        tic = time.time()
        history = model.fit(
            x_train, [y_train]*len(comb), batch_size=batch, epochs = epoch_num, verbose = 2
        )
        total = time.time() - tic
        print("Total Training Time: %.6f\n" % (total))

        #1. Gradient Score Calculation
        loss_object = tf.keras.losses.BinaryCrossentropy()
        output = []

        for layer in model.layers:
            if 'global_average_pooling1d' in layer.name:
                output.append(model.get_layer(layer.name).output)

        if(strategy is None): temp_model = tf.keras.Model(model.input, [model.output, output])
        else:
            with strategy.scope():
                temp_model = tf.keras.Model(model.input, [model.output, output])
        x_test = tf.Variable(x_train[0:int(batch/n_devices),:])

        with tf.GradientTape() as tape:
            tape.reset()
            tape.watch(x_test)
            prediction, layer_output = temp_model(x_test)
            loss = loss_object([label[0:int(batch/n_devices)]]*(len(comb)), tf.squeeze(prediction))
        
        gradient = tape.gradient(loss, layer_output)
        gradients = np.zeros(snpsize)

        for i in range(len(comb)):
            indexes = [comb[i][j] for j in range(comb_order)]
            grad_list = []
            for j in range(comb_order):
                grad_list.append([j for j in range(int(indexes[j]*snpsize/splits),int((indexes[j]+1)*snpsize/splits))])
            grad_list = sum(grad_list, [])
            gradient[i] = tf.reduce_sum(gradient[i],axis=0)
            gradients[np.r_[grad_list]] += np.abs(gradient[i])
        
        #2. Attention Score Calculation
        final_scores = np.zeros(snpsize)
        index = 0
        for layer in model.layers:
            if 'transformer_block' in layer.name:
                scores = layer.non_trainable_weights[0]
                indexes = [comb[index][j] for j in range(comb_order)]
                sizes = [int((indexes[j]+1)*snpsize/splits) - int(indexes[j]*snpsize/splits) for j in range(comb_order)]
                scores_list = []
                for j in range(comb_order):
                    scores_list.append([j for j in range(int(indexes[j]*snpsize/splits),int((indexes[j]+1)*snpsize/splits))])

                scores_list = sum(scores_list, [])
                final_scores[np.r_[scores_list]] += decode_attention_scores(scores)
                index += 1
                layer.non_trainable_weights[0].assign(tf.zeros(shape=(int(batch/n_devices),1,sum(sizes))))                
               
        final_scores_ = (final_scores - np.min(final_scores))/(np.max(final_scores) - np.min(final_scores))
        gradients_ = (gradients - np.min(gradients))/(np.max(gradients) - np.min(gradients))

        final_scores_2 = (final_scores - np.min(final_scores))/(np.max(final_scores) - np.min(final_scores))
        gradients_2 = (gradients - np.min(gradients))/(np.max(gradients) - np.min(gradients))

        #SNP Selection - Gradients
        finalsnps_grad = choose_snps(gradients, int(top*snpsize))

        #SNP Selection - Attention Scores
        finalsnps_scores = choose_snps(final_scores, int(top*snpsize))

        #SNP Selection - Gradients*Scores Scaled
        finalsnps_mixed = choose_snps(gradients_*final_scores_, int(top*snpsize))

        #SNP Selection - Gradients+Scores Scaled
        finalsnps_mixed2 = choose_snps(gradients_2+final_scores_2, int(top*snpsize))

        #Writing output files with results
        outfile.write(str(f_op))
        outfile.write("\n")
        outfile.write("Best SNPs (Gradient Scores)\n")
        outfile.write(str([finalsnps_grad[i] for i in range(len(finalsnps_grad))]))
        outfile.write("\n")

        outfile.write("Best SNPs (Attention Scores)\n")
        outfile.write(str([finalsnps_scores[i] for i in range(len(finalsnps_scores))]))
        outfile.write("\n")

        outfile.write("Best SNPs (Scaled Gradient*Attention Scores)\n")
        outfile.write(str([finalsnps_mixed[i] for i in range(len(finalsnps_mixed))]))  
        outfile.write("\n")

        outfile.write("Best SNPs (Scaled Gradient+Attention Scores)\n") 
        outfile.write(str([finalsnps_mixed2[i] for i in range(len(finalsnps_mixed2))])) 
        outfile.write("\n")
        f_op.close()

def main(argv):
    path, splits, comb, top, device, n, sparsity, epochs = argv.path, argv.partitions, argv.comb, argv.top, argv.device, argv.n, argv.sparsity, argv.epochs
    list_device = tf.config.list_logical_devices(device)

    #Checking for argument correctness
    if(comb > splits or comb < 0):
        raise ValueError("Invalid number of combinations (must be >= 0 and <= partitions)")
    elif(splits <= 0):
        raise ValueError("Number of partitions must be > 0")
    elif len(list_device) == 0 or n <= 0:
        raise ValueError("Device not found")
    elif n > len(list_device):
        raise ValueError("Invalid number of devices (asked for %d %s but found %d)" % (n,device,len(list_device)))
    elif top < 0 or top > 1:
        raise ValueError("Invalid value (SNP top percentage must be between 0 and 1)")
    elif sparsity < 0 or sparsity > 1:
        raise ValueError("Invalid value (sparsity must be between 0 and 1)")
    elif epochs < 1:
        raise ValueError("Invalid value (epochs must be greater than 0)")

    #Checking Devices
    if(device == 'TPU'):
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        training_model(path, splits, comb, top, strategy, n, sparsity, epochs)
        return
    elif((device == 'GPU' or (device == 'CPU' and n != 1))):
        strategy = tf.distribute.MirroredStrategy(tf.config.list_logical_devices(device)[0:n])
        training_model(path, splits, comb, top, strategy, n, sparsity, epochs)
        return
    elif(device == 'IPU'):
        from tensorflow.python import ipu
        ipu_config = ipu.config.IPUConfig()
        ipu_config.auto_select_ipus = n
        ipu_config.configure_ipu_system()
        strategy = ipu.ipu_strategy.IPUStrategy()
        training_model(path, splits, comb, top, strategy, n, sparsity, epochs)
        return
    
    #Training the transformer
    training_model(path, splits, comb, top, None, n, sparsity)

if __name__ == "__main__":
   parser = ArgumentParser(description='Transformer Neural Network For Epistasis Detection')
   parser.add_argument('-path', required=True, type=str, help='a path to epistasis files (can be a .txt file, a folder with files, or a zip with files)')
   parser.add_argument('-partitions', required=True, type=int, help='the number of partitions')
   parser.add_argument('-comb', required=True, type=int, help='the combination order to merge partitions')
   parser.add_argument('-top', required=True, type=float, help='best SNP percentage to report after training (between 0 and 1)')
   parser.add_argument('-device', required=True, type=str, help='the device to use (e.g., CPU, GPU)')
   parser.add_argument('-n', required=True, type=int, help='number of devices')
   parser.add_argument('-sparsity', type=float, help='an optional float (between 0 and 1) for the sparsity percentage on the transformer\'s attention modules. Defaults to 0.9.', default = 0.9)
   parser.add_argument('-epochs', type=int, help='an optional int for the epochs to train the transformer. Defaults to 15.', default = 15)
   args = parser.parse_args()
   main(args)