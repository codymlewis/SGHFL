import numpy as np
from Pyfhel import Pyfhel


def fhe_dot(ctxt_a, ctxt_b, length):
    c_dot = ctxt_a * ctxt_b
    c_mul = ctxt_a * ctxt_b
    HE.relinKeyGen()
    ~c_mul
    for i in range(1, length):
        c_dot += (c_mul << i)
    return c_dot


if __name__ == "__main__":
    HE = Pyfhel()
    ckks_params = {
        'scheme': 'CKKS',
        'n': 2**14,
        'scale': 2**30,
        'qi_sizes': [60, 30, 30, 30, 60]
    }
    HE.contextGen(**ckks_params)  # Generate context for ckks scheme
    HE.keyGen()             # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()

    arr_x = np.array([0.1, 0.2, -0.3], dtype=np.float64)    # Always use type float64!
    arr_y = np.array([-1.5, 2.3, 4.7], dtype=np.float64)

    ctxt_x = HE.encryptFrac(arr_x)
    ctxt_y = HE.encryptFrac(arr_y)

    c_dot = fhe_dot(ctxt_x, ctxt_y, 3)
    p_dot = HE.decryptFrac(c_dot)
    print(f"true dot: {np.dot(arr_x, arr_y)}, fhe dot: {p_dot[0]}")
