import numpy as np


def mat_update(prev_mat, mat):
    if np.linalg.det(mat) == 0:
        return prev_mat
    else:
        return mat


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


import torch


def quaternion_conjugate(q):
    # q: [B, 4] = [x, y, z, w]
    # Conjugate: q* = [-x, -y, -z, w]
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack([-x, -y, -z, w], dim=-1)


def quaternion_mul(q1, q2):
    # q1, q2: [B,4] = [x, y, z, w]
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack([x, y, z, w], dim=-1)


def rotate_quaternion(A, B):
    # Apply rotation represented by B to A: Q = B * A * B*
    B_conj = quaternion_conjugate(B)
    return quaternion_mul(quaternion_mul(B, A), B_conj)


# Example
if __name__ == "__main__":
    # Assume A and B are [B,4] and are unit quaternions
    A = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],  # Unit quaternion (no rotation)
            [0.0, 0.7071, 0.0, 0.7071],
        ]
    )  # Approximately 90 degrees rotation around y-axis
    B = torch.tensor(
        [
            [
                0.7071,
                0.0,
                0.0,
                0.7071,
            ],  # Approximately 90 degrees rotation around x-axis
            [0.0, 0.0, 0.7071, 0.7071],
        ]
    )  # Approximately 90 degrees rotation around z-axis

    Q = rotate_quaternion(A, B)
    print(Q)
