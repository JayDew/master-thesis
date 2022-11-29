from functools import reduce

import numpy as np

import paillier

DEFAULT_MSGSIZE = 32  # set here the default number of bits the plaintext can have
DEFAULT_PRECISION = int(DEFAULT_MSGSIZE / 2)  # set here the default number of fractional bits


def generate_keypair():
    """
    Generate Public and Private keys

      Returns:
      tuple: The generated :class:`PaillierPublicKey` and
      :class:`PaillierPrivateKey`
    """
    return paillier.generate_paillier_keypair()


def mult_vect_by_constant(x, const):
    """
    Perform multiplication between an encrypted vector and a constant
    :param x: the encrypted vector
    :param const: constant
    :return: multiplication between the encrypted vector and the constant
    """
    return [x[i] * const for i in range(np.size(x))]


def sum_encrypted_vectors(x, y):
    """
    Perform addition of two encrypted vectors of the same size
    :param x: first encrypted vector
    :param y: second encrypted vector
    :return: sum of the two encrypted vectors
    """
    return [x[i] + y[i] for i in range(np.size(x))]


def diff_encrypted_vectors(x, y):
    """
    Perform difference of two encrypted vectors of the same size
    :param x: first encrypted vector
    :param y: second encrypted vector
    :return: difference of the two encrypted vectors
    """
    return [x[i] - y[i] for i in range(np.size(x))]


def dot(x, consts):
    """
    Perform dot product between a vector of scalars
    and a vector of encrypted values
    :param x: encrypted vector
    :param consts: vector of scalars
    :return: Encrypted vector corresponding to dot product
    """
    temp = []
    for i in range(np.size(x)):
        temp.append(mult_vect_by_constant([x[i]], consts[i])[0])
    return [reduce(lambda a, b: a + b, temp)]


def mat_mul(x, c):
    """
    Perform dot product between a vector of scalars
    and a vector of encrypted values
    :param x: encrypted vector
    :param c: matrix of scalars
    :return: Encrypted vector corresponding to matrix multiplication
    """
    temp = [0] * len(c)
    for i in range(len(c)):
        temp[i] = dot(x, c[i])[0]
    return temp


def encrypt_vector(pubkey, x, coins=None):
    """
    Encrypt a vector
    :param pubkey: public key used for encryption
    :param x: the vector to be encrypted
    :param coins:
    :return: encrypted vector
    """
    if (coins == None):
        return [pubkey.encrypt(y) for y in x]
    else:
        return [pubkey.encrypt(y, coins.pop()) for y in x]


def decrypt_vector(privkey, x):
    """
    Decrypt an encrypted vector
    :param privkey: private key used for decryption
    :param x: encrypted vector to be decrypted
    :return: decrypted vector
    """
    return np.array([privkey.decrypt(i) for i in x])


def fp(scalar, prec=DEFAULT_PRECISION):
    """
    Quantize a scalar by eliminating its fractional part
    :param scalar: the number to be quantized
    :param prec: precision level
    :return: 'mpz' object of quantized scalar
    """
    return int(np.round(scalar * (2 ** prec)))


def fp_vector(vec, prec=DEFAULT_PRECISION):
    """
    Quantize a vector
    :param vec: vector to be quantized
    :param prec: precision level
    :return: vector of quantized elements
    """
    return [fp(x, prec) for x in vec]


def fp_matrix(matrix, prec=DEFAULT_PRECISION):
    """
    Quantize a matrix
    :param vec: vector to be quantized
    :param prec: precision level
    :return: vector of quantized elements
    """
    result = [0] * matrix.shape[0]
    for i in range(matrix.shape[0]):
        result[i] = (fp_vector(matrix[i], prec))
    return result


def retrieve_fp(scalar, prec=DEFAULT_PRECISION):
    """
    Remove quantization from a scalar
    :param scalar: scalar to be un-quantized
    :param prec: precision level
    :return: scalar
    """
    return scalar / (2 ** prec)


def retrieve_fp_vector(vec, prec=DEFAULT_PRECISION):
    """
    Remove quantization from a vector
    :param vec: vector to be un-quantized
    :param prec: precision level
    :return: vector of scalars
    """
    return [retrieve_fp(x, prec) for x in vec]
