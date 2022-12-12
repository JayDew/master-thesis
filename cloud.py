#!/usr/bin/env python3

import socket
import sys, struct
import json
from gmpy2 import mpz
import paillier
import numpy as np
import time
from common import get_params

# Attempted implementation of the paper from CDC 2016

DEFAULT_KEYSIZE = 1024
DEFAULT_MSGSIZE = 32
DEFAULT_PRECISION = int(DEFAULT_MSGSIZE / 2)  # of fractional bits
DEFAULT_FACTOR = 80
DEFAULT_STATISTICAL = 40  # The multiplication by random numbers offers DEFAULT_STATISTICAL security
# 2**(DEFAULT_MSGSIZE+DEFAULT_STATISTICAL < N)
NETWORK_DELAY = 0  # 10 ms

try:
    import gmpy2
    HAVE_GMP = True
except ImportError:
    HAVE_GMP = False

seed = 42

def encrypt_vector(pubkey, x, coins=None):
    if (coins == None):
        return [pubkey.encrypt(y) for y in x]
    else:
        return [pubkey.encrypt(y, coins.pop()) for y in x]


def encrypt_matrix(pubkey, x, coins=None):
    if (coins == None):
        return [[pubkey.encrypt(y) for y in z] for z in x]
    else:
        return [[pubkey.encrypt(y, coins.pop()) for y in z] for z in x]


def decrypt_vector(privkey, x):
    return [privkey.decrypt(i) for i in x]

def sum_encrypted_vectors(x, y):
    return [x[i] + y[i] for i in range(np.size(x))]

def diff_encrypted_vectors(x, y):
    return [x[i] - y[i] for i in range(len(x))]

def mul_sc_encrypted_vectors(x, y): # x is encrypted, y is plaintext
    return [y[i]*x[i] for i in range(len(x))]

def dot_sc_encrypted_vectors(x, y): # x is encrypted, y is plaintext
    return sum(mul_sc_encrypted_vectors(x,y))

def dot_m_encrypted_vectors(x, A):
    return [dot_sc_encrypted_vectors(x, vec) for vec in A]

####### We take the convention that a number x < N/3 is positive, and that a number x > 2N/3 is negative.
####### The range N/3 < x < 2N/3 allows for overflow detection.

def fp(scalar, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    if prec < 0:
        return gmpy2.t_div_2exp(mpz(scalar), -prec)
    else:
        return mpz(gmpy2.mul(scalar, 2 ** prec))


def fp_vector(vec, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    if np.size(vec) > 1:
        return [fp(x, prec) for x in vec]
    else:
        return fp(vec, prec)

def fp_matrix(mat,prec=DEFAULT_PRECISION+DEFAULT_FACTOR):
	return [fp_vector(x,prec) for x in mat]

def retrieve_fp(scalar,prec=DEFAULT_PRECISION+DEFAULT_FACTOR):
	return scalar/(2**prec)
	# return gmpy2.div(scalar,2**prec)

def retrieve_fp_vector(vec, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return [retrieve_fp(x, prec) for x in vec]

def retrieve_fp_matrix(mat,prec=DEFAULT_PRECISION+DEFAULT_FACTOR):
	return [retrieve_fp_vector(x,prec) for x in mat]


class Agents:
    def __init__(self, pubkey, fileb, filec):
        self.pubkey = pubkey
        b_A = np.loadtxt(fileb)
        c_A = np.loadtxt(filec)
        self.enc_b_A = encrypt_vector(pubkey, fp_vector(b_A))
        self.enc_c_A = encrypt_vector(pubkey, fp_vector(c_A))
        self.m = np.size(b_A)
        self.n = np.size(c_A)

    def send_data(self):
        return self.enc_b_A, self.enc_c_A, self.m, self.n

class Cloud:
    def __init__(self, pubkey, fileA, fileQ):
        self.pubkey = pubkey
        self.N = pubkey.n
        self.N2 = pubkey.nsquare
        self.N_len = (self.N).bit_length()
        self.l = DEFAULT_MSGSIZE
        A = np.loadtxt(fileA, delimiter=',')
        Q = np.loadtxt(fileQ, delimiter=',')
        self.A = A
        m = np.size(A, 0)
        self.m = m
        At = A.transpose()
        self.Q = Q
        invQ = np.linalg.inv(Q)  ### Q^{-1}
        AinvQ = np.dot(A, invQ)  ###  AQ^{-1}
        AinvQA = np.dot(AinvQ, At)  ### AQ^{-1}A'
        eigs = np.linalg.eigvals(AinvQA)
        eta = 1 / np.real(max(eigs))
        self.delta_A = [0] * m
        # param = np.loadtxt(fileparam, delimiter='\n')
        # self.K = param[0]
        # self.K = int(self.K)
        self.K = 10  #TODO! change this value !!!

        coeff_mu = fp_matrix(np.identity(m) - eta * AinvQA)  ### I-\eta AQ^{-1}A'
        self.coeff_mu = coeff_mu
        coeff_c = fp_matrix(-eta * AinvQ)  ### -\etaAQ^{-1}
        self.coeff_c = coeff_c
        coeff_muK = fp_matrix(np.dot(-invQ, At))  ### -Q^{-1}A'
        self.coeff_muK = coeff_muK
        coeff_cK = fp_matrix(-invQ)
        self.coeff_cK = coeff_cK
        etabar = fp(-eta)
        self.etabar = etabar
        self.gen_rands()

    def gen_rands(self):
        lf = DEFAULT_PRECISION
        gamma = DEFAULT_STATISTICAL
        sigma = DEFAULT_FACTOR
        m = self.m
        l = self.l
        K = self.K
        random_state = gmpy2.random_state(seed)
        mu = np.zeros(m).astype(int)
        # mu = fp_vector([gmpy2.mpz_urandomb(random_state,self.l-DEFAULT_PRECISION-1) for i in range(0,m)])
        self.mu = encrypt_vector(self.pubkey, mu)

        # Noise for truncation
        rn = [gmpy2.mpz_urandomb(random_state, int(l + gamma)) for i in range(0, m * K)]
        # print(rn)
        self.obfuscations = rn
        erobfs = [fp(1 / x, lf + sigma) + 1 for x in rn]
        # print(erobfs)
        self.erobfs = erobfs

    def compute_grad(self, b_A, c_A):
        mu_bar = sum_encrypted_vectors(np.dot(self.coeff_mu, self.mu), np.dot(self.coeff_c, c_A))
        mu_bar = sum_encrypted_vectors(mu_bar, [x * self.etabar for x in b_A])
        self.mu_bar = mu_bar  ### \mu_bar*2^{2*lf}

    def compute_primal_optimum(self, c_A):
        x = np.dot(self.coeff_muK, self.mu)
        x = sum_encrypted_vectors(x, np.dot(self.coeff_cK, c_A))
        return x

    def obfuscate(self):
        m = self.m
        for i in range(0, m):
            r = self.obfuscation[i]
            self.mu_bar[i] = r * self.mu_bar[i]
        return self.mu_bar

    def update(self, temp_mu):
        m = self.m
        for i in range(0, m):
            r = self.erobf[i]
            self.mu[i] = r * temp_mu[i]


def key(serialised):
    received_dict = json.loads(serialised)
    pk = received_dict['public_key']
    n = int(pk['n'])
    public_key = paillier.PaillierPublicKey(n=n)
    return public_key

def send_encr_data(encrypted_number_list):
    time.sleep(NETWORK_DELAY)
    enc_with_one_pub_key = {}
    enc_with_one_pub_key = [str(x.ciphertext()) for x in encrypted_number_list]
    return json.dumps(enc_with_one_pub_key)

def send_plain_data(data):
    time.sleep(NETWORK_DELAY)
    return json.dumps([str(x) for x in data])

def recv_size(the_socket):
    # data length is packed into 4 bytes
    total_len = 0;
    total_data = [];
    size = sys.maxsize
    size_data = sock_data = bytes([]);
    recv_size = 4096
    while total_len < size:
        sock_data = the_socket.recv(recv_size)
        if not total_data:
            if len(sock_data) > 4:
                size = struct.unpack('>i', sock_data[:4])[0]
                recv_size = size
                if recv_size > 262144: recv_size = 262144
                total_data.append(sock_data[4:])
            else:
                size_data += sock_data

        else:
            total_data.append(sock_data)
        total_len = sum([len(i) for i in total_data])
    return b''.join(total_data)

def get_enc_data(received_dict,pubkey):
	return [paillier.EncryptedNumber(pubkey, int(x)) for x in received_dict]

def main():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Cloud: Socket successfully created')
    port = 10007
    # Bind the socket to the port
    localhost = [l for l in (
    [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [
        [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in
         [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]
    server_address = (localhost, port)
    print('Cloud: Starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)
    print('Cloud: Socket is listening')
    # Wait for a connection
    print('Cloud: Waiting for a connection')
    connection, client_address = sock.accept()

    n, m = get_params()
    time.sleep(1)
    print('Cloud: Connection from', client_address)
    data = recv_size(connection)
    if data:
        pubkey = key(data)
        fileA = "Data_2/A"+str(n)+"_"+str(m)+".txt"
        fileQ = "Data_2/Q"+str(n)+"_"+str(m)+".txt"
        fileb = "Data_2/b"+str(n)+"_"+str(m)+".txt"
        filec = "Data_2/c"+str(n)+"_"+str(m)+".txt"
        # fileparam = "Data_2/param"+str(n)+"_"+str(m)+".txt"
        filepriv = "Keys/privkey"+str(DEFAULT_KEYSIZE)+".txt"
        with open(filepriv, 'r') as fin:
            data=[line.split() for line in fin]
        p = mpz(data[0][0])
        q = mpz(data[1][0])
        privkey = paillier.PaillierPrivateKey(pubkey, p, q)
        v = []; t = []
        agents = Agents(pubkey,fileb,filec)
        cloud = Cloud(pubkey,fileA,fileQ)
        b_A, c_A, m, n = agents.send_data()
        # Send m and K
        K = cloud.K
        data = send_plain_data([m,K])
        connection.sendall(struct.pack('>i', len(data))+data.encode('utf-8'))
        # Iterations of the projected gradient descent
        # print(n,m)
        for k in range(0,K):
            print(k)
            cloud.obfuscation = cloud.obfuscations[k*m:(k+1)*m]
            cloud.erobf = cloud.erobfs[k*m:(k+1)*m]
            cloud.compute_grad(b_A,c_A)
            # print(decrypt_vector(privkey,cloud.mu_bar))
            # Begin comparison procedure
            temp_mu = cloud.obfuscate()
            # print('temp_mu before sending',decrypt_vector(privkey,temp_mu))
            # Send temp_mu
            data = send_encr_data(temp_mu)
            connection.sendall(struct.pack('>i', len(data))+data.encode('utf-8'))
            # Receive max(0,temp_mu')
            data = json.loads(recv_size(connection))
            temp_mu2 = get_enc_data(data,pubkey)
            # print('temp_mu after receiving',decrypt_vector(privkey,temp_mu2))
            cloud.update(temp_mu2)
            # print('mu',decrypt_vector(privkey,cloud.mu))

        x = cloud.compute_primal_optimum(c_A)
        # print("x:", retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privkey,x))))
        # Send x
        data = send_encr_data(x)
        connection.sendall(struct.pack('>i', len(data))+data.encode('utf-8'))
        # Wait for the target to finish its tasks -- this is for when consecutive problems are run
        data = json.loads(recv_size(connection))
        # Send 1 if the target should keep the connection open and 0 otherwise
        data = json.dumps(0)
        connection.sendall(struct.pack('>i', len(data))+data.encode('utf-8'))

    else:
        print('Cloud: No data from', client_address)
        # break

    # finally:
    # Clean up the connection
    print('Cloud: Closing connection')
    connection.close()

if __name__ == '__main__':
    main()
