from soil_field import generate_soil_field
import numpy as np
import matplotlib.pyplot as plt

f1 = generate_soil_field(seed=1)
f2 = generate_soil_field(seed=2)

print("Identical:", np.array_equal(f1, f2))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(f1, aspect="auto")
ax[0].set_title("Seed 1")

ax[1].imshow(f2, aspect="auto")
ax[1].set_title("Seed 2")

plt.tight_layout()
plt.show()
