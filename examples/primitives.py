import numpy as np

from simdkalman.primitives import predict
from simdkalman.primitives import update


def to_3d(v):
    if len(v.shape) == 1:
        return v[np.newaxis, :, np.newaxis]
    elif len(v.shape) == 2:
        return v[np.newaxis, ...]
    else:
        return v


# define model
state_transition = np.array([[1, 1], [0, 1]])
process_noise = np.eye(2)*0.01
observation_model = np.array([[1, 0]])
observation_noise = np.array([[1.0]])


state_transition = to_3d(state_transition)
process_noise = to_3d(process_noise)
observation_model = to_3d(observation_model)
observation_noise = to_3d(observation_noise)
# initial state
m = to_3d(np.array([[0, 1]]))
P = to_3d(np.eye(2))

# predict next state
m, P = predict(m, P, state_transition, process_noise)

# first observation
y = to_3d(np.array([4]))
y = to_3d(y)
# m, P = update(m, P, observation_model, observation_noise, y)

# predict second state
# m, P = predict(m, P, state_transition, process_noise)

print('mean')
print(m)

print('cov')
print(P)
