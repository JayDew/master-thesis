#!/usr/bin/env python3

import socket
import sys,struct
import json
from gmpy2 import mpz
import paillier
# from phe import util
import numpy as np
import time
import random
import os

# Attempted implementation of the paper from CDC 2016

DEFAULT_KEYSIZE = 1024
DEFAULT_MSGSIZE = 32
DEFAULT_PRECISION = int(DEFAULT_MSGSIZE/2) # of fractional bits
DEFAULT_FACTOR = 80
DEFAULT_STATISTICAL = 40 # The multiplication by random numbers offers DEFAULT_STATISTICAL security
# 2**(DEFAULT_MSGSIZE+DEFAULT_STATISTICAL < N)
NETWORK_DELAY = 0 #10 ms


try:
    import gmpy2
    HAVE_GMP = True
except ImportError:
    HAVE_GMP = False

seed = 42

def encrypt_vector(pubkey, x, coins=None):
	if (coins==None):
		return [pubkey.encrypt(y) for y in x]
	else: return [pubkey.encrypt(y,coins.pop()) for y in x]

def encrypt_matrix(pubkey, x, coins=None):
	if (coins==None):
		return [[pubkey.encrypt(y) for y in z] for z in x]
	else: return [[pubkey.encrypt(y,coins.pop()) for y in z] for z in x]

def decrypt_vector(privkey, x):
    return [privkey.decrypt(i) for i in x]

def sum_encrypted_vectors(x, y):
	return [x[i] + y[i] for i in range(np.size(x))]

def diff_encrypted_vectors(x, y):
	return [x[i] - y[i] for i in range(len(x))] 


####### We take the convention that a number x < N/3 is positive, and that a number x > 2N/3 is negative. 
####### The range N/3 < x < 2N/3 allows for overflow detection.

def fp(scalar,prec=DEFAULT_PRECISION+DEFAULT_FACTOR):
	if prec < 0:
		return gmpy2.t_div_2exp(mpz(scalar),-prec)
	else: return mpz(gmpy2.mul(scalar,2**prec))

def fp_vector(vec,prec=DEFAULT_PRECISION+DEFAULT_FACTOR):
	if np.size(vec)>1:
		return [fp(x,prec) for x in vec]
	else:
		return fp(vec,prec)

def fp_matrix(mat,prec=DEFAULT_PRECISION+DEFAULT_FACTOR):
	return [fp_vector(x,prec) for x in mat]

def retrieve_fp(scalar,prec=DEFAULT_PRECISION+DEFAULT_FACTOR):
	return scalar/(2**prec)

def retrieve_fp_vector(vec,prec=DEFAULT_PRECISION+DEFAULT_FACTOR):
	return [retrieve_fp(x,prec) for x in vec]

def retrieve_fp_matrix(mat,prec=DEFAULT_PRECISION+DEFAULT_FACTOR):
	return [retrieve_fp_vector(x,prec) for x in mat]


class Target:
	def __init__(self, l=DEFAULT_MSGSIZE):
		# keypair = paillier.generate_paillier_keypair(n_length=DEFAULT_KEYSIZE)
		# self.pubkey, self.privkey = keypair
		self.l = l
		filepub = "Keys/pubkey"+str(DEFAULT_KEYSIZE)+".txt"
		with open(filepub, 'r') as fin:
			data=[line.split() for line in fin]
		Np = int(data[0][0])
		pubkey = paillier.PaillierPublicKey(n=Np)

		filepriv = "Keys/privkey"+str(DEFAULT_KEYSIZE)+".txt"
		with open(filepriv, 'r') as fin:
			data=[line.split() for line in fin]
		p = mpz(data[0][0])
		q = mpz(data[1][0])
		privkey = paillier.PaillierPrivateKey(pubkey, p, q)		
		self.pubkey = pubkey; self.privkey = privkey

	def gen_rands(self,m,K):
		# Noise for encryption
		self.m = m
		Nlen = self.pubkey.n.bit_length()
		random_state = gmpy2.random_state(seed)
		coinsP = [gmpy2.mpz_urandomb(random_state,Nlen-1) for i in range(0,K*m)]
		coinsP = [gmpy2.powmod(x, self.pubkey.n, self.pubkey.nsquare) for x in coinsP]		
		self.coinsP = coinsP

	def comparison(self,msg):
		lf = DEFAULT_PRECISION
		sigma = DEFAULT_FACTOR
		temp_mu = decrypt_vector(self.privkey,msg)
		# print(temp_mu)
		temp_mu = fp_vector(temp_mu,-2*(lf+sigma))
		mu = np.maximum(0,temp_mu)
		# print(mu)
		mu = encrypt_vector(self.pubkey,mu,self.coinsP[-self.m:])
		self.coinsP = self.coinsP[:-self.m]
		return mu


def keys(pubkey):
	pubkeys = {}
	pubkeys['public_key'] = {'n': pubkey.n}
	serialized_pubkeys = json.dumps(pubkeys)
	return serialized_pubkeys

def get_enc_data(received_dict,pubkey):
	return [paillier.EncryptedNumber(pubkey, int(x)) for x in received_dict]

def get_plain_data(data):
	return [int(x) for x in data]

def recv_size(the_socket):
	#data length is packed into 4 bytes
	total_len=0;total_data=[];size=sys.maxsize
	size_data=sock_data=bytes([]);recv_size=4096
	while total_len<size:
		sock_data=the_socket.recv(recv_size)
		if not total_data:
			if len(sock_data)>4:
				size=struct.unpack('>i', sock_data[:4])[0]
				recv_size=size
				if recv_size>262144:recv_size=262144
				total_data.append(sock_data[4:])
			else:
				size_data+=sock_data

		else:
			total_data.append(sock_data)
		total_len=sum([len(i) for i in total_data ])
	return b''.join(total_data)

def send_encr_data(encrypted_number_list):
	time.sleep(NETWORK_DELAY)
	encrypted = {}
	encrypted = [str(x.ciphertext()) for x in encrypted_number_list]
	return json.dumps(encrypted)

def main():
		lf = DEFAULT_PRECISION
		sigma = DEFAULT_FACTOR
		start = time.time()
	# Create a TCP/IP socket
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		port = 10000

		# Connect the socket to the port where the server is listening
		localhost = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]
		server_address = (localhost, port)
		print('Target: Connecting to {} port {}'.format(*server_address))
		sock.connect(server_address)			
		target = Target()
		pubkey = target.pubkey
		privkey = target.privkey
		serialized_pubkeys = keys(pubkey)
		cont = True
		off = time.time() - start
		try:		
			while cont:
				time.sleep(1)
				start = time.time()
				# Send public key
				sock.sendall(struct.pack('>i', len(serialized_pubkeys))+serialized_pubkeys.encode('utf-8'))					
				# Receive m and K
				data = json.loads(recv_size(sock))
				m,K = get_plain_data(data)
				target.gen_rands(m,K)
				offline = off + time.time()-start
				start = time.time()
				l = target.l
				for k in range(0,K):
					# Begin comparison procedure
					# Receive temp_mu
					data = json.loads(recv_size(sock))
					temp_mu = get_enc_data(data,pubkey)
					mu = target.comparison(temp_mu)
					# Send mu*r/2**2f
					serialized_data = send_encr_data(mu)
					sock.sendall(struct.pack('>i', len(serialized_data))+serialized_data.encode('utf-8'))
				# Receive x
				data = json.loads(recv_size(sock))
				x = get_enc_data(data,pubkey)
				x = retrieve_fp_vector(decrypt_vector(privkey,x),2*(lf+sigma))
				print(["%.8f"% i for i in x])
				end = time.time()
				sec = end-start
				print("%.2f" % sec)
				n = len(x)
				sys.stdout.flush()
				# with open(os.path.abspath('Results/delay_'+str(NETWORK_DELAY)+'_plot_'+str(DEFAULT_KEYSIZE)+'_'+str(DEFAULT_MSGSIZE)+'_results_'+str(K)+'_'+str(int(DEFAULT_STATISTICAL))+'.txt'),'a+') as f: f.write("%d, %d, %.2f, %.2f\n" % (n,m,sec,offline))
				with open(os.path.abspath('Results/plot_'+str(DEFAULT_KEYSIZE)+'_'+str(DEFAULT_MSGSIZE)+'_results_'+str(K)+'_'+str(int(DEFAULT_STATISTICAL))+'_'+str(NETWORK_DELAY)+'.txt'),'a+') as f: f.write("%d, %d, %.2f, %.2f\n" % (n,m,sec,offline))
				# Let the cloud know that it is ready
				data = json.dumps(1)
				sock.sendall(struct.pack('>i', len(data))+data.encode('utf-8'))
				# Receive 1 if to continue and 0 if to end
				data = json.loads(recv_size(sock))
				cont = bool(int(data))

		finally:
			print('Target: Closing socket')
			sock.close()				


if __name__ == '__main__':
	main()