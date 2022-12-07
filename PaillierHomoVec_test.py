import unittest

from PaillierHomoVec import *


class TestPaillierHomoVec(unittest.TestCase):

    def test_addition_two_arrays(self):
        x = np.asarray([1.5, 2.33])
        y = np.asarray([3.1, 4.123])
        # generate public and private keys
        publicKey, privateKey = generate_keypair()
        # encrypt the arrays
        x_enc = encrypt_vector(publicKey, fp_vector(x))
        y_enc = encrypt_vector(publicKey, fp_vector(y))
        # add the two encrypted arrays
        result_enc = sum_encrypted_vectors(x_enc, y_enc)
        # decrypt and check that the sum is correct
        result = retrieve_fp_vector(decrypt_vector(privateKey, result_enc))
        assert np.allclose(result, x + y)

    def test_subtract_two_arrays(self):
        x = np.asarray([1.5, 2.33])
        y = np.asarray([-3.1, 4.123])
        # generate public and private keys
        publicKey, privateKey = generate_keypair()
        # encrypt the arrays
        x_enc = encrypt_vector(publicKey, fp_vector(x))
        y_enc = encrypt_vector(publicKey, fp_vector(y))
        #
        result_enc = diff_encrypted_vectors(x_enc, y_enc)
        # decrypt and check that the diff is correct
        result = retrieve_fp_vector(decrypt_vector(privateKey, result_enc))
        assert np.allclose(result, x - y)

    def test_multiplication_by_constant(self):
        x = np.asarray([1.5, 2.33])
        const = 2.5
        # generate public and private keys
        publicKey, privateKey = generate_keypair()
        # encrypt the arrays
        x_enc = encrypt_vector(publicKey, fp_vector(x))
        # add the two encrypted arrays
        result_enc = mult_vect_by_constant(x_enc, fp(const))
        # decrypt and check that the multiplication is correct
        result = retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privateKey, result_enc)))
        assert np.allclose(result, x * const)

    def test_dot_product(self):
        x = np.asarray([3.2, 4])
        consts = np.asarray([1, 2.5])
        # generate public and private keys
        publicKey, privateKey = generate_keypair()
        # encrypt the arrays
        x_enc = encrypt_vector(publicKey, fp_vector(x))
        # add the two encrypted arrays
        result_enc = dot(x_enc, fp_vector(consts))
        # decrypt and check that the multiplication is correct
        result = retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privateKey, result_enc)))
        assert np.allclose(result, np.dot(consts, x))

    def test_dot_product_complex(self):
        x = np.asarray([3.2, 4, -1])
        consts = np.asarray([1, 2.5, 11])
        # generate public and private keys
        publicKey, privateKey = generate_keypair()
        # encrypt the arrays
        x_enc = encrypt_vector(publicKey, fp_vector(x))
        # add the two encrypted arrays
        result_enc = dot(x_enc, fp_vector(consts))
        # decrypt and check that the multiplication is correct
        result = retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privateKey, result_enc)))
        assert np.allclose(result, np.dot(consts, x))

    def test_mat_mul(self):
        x = np.asarray([3.2, 4])
        consts = np.asarray([[1, 2.5], [-1, 3]])
        # generate public and private keys
        publicKey, privateKey = generate_keypair()
        # encrypt the arrays
        x_enc = encrypt_vector(publicKey, fp_vector(x))
        # add the two encrypted arrays
        result_enc = mat_mul(x_enc, fp_matrix(consts))
        # decrypt and check that the multiplication is correct
        result = retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privateKey, result_enc)))
        assert np.allclose(result, np.dot(consts, x))

    def test_mat_mul_complex(self):
        x = np.asarray([3.2, 4, -2, 33])
        consts = np.asarray([[1, 2.5, -11.123, -4], [-1, 3, -0.9876, 1.1112]])
        # generate public and private keys
        publicKey, privateKey = generate_keypair()
        # encrypt the arrays
        x_enc = encrypt_vector(publicKey, fp_vector(x))
        # add the two encrypted arrays
        result_enc = mat_mul(x_enc, fp_matrix(consts))
        # decrypt and check that the multiplication is correct
        result = retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privateKey, result_enc)))
        assert np.allclose(result, np.dot(consts, x))
