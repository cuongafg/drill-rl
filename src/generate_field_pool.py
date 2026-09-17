import argparse
import numpy as np

from soil_field import generate_soil_field


def main(n_fields, seed_start, out_path):

    fields = []

    for i in range(n_fields):

        seed = seed_start + i

        print(
            f"Generating field {i+1}/{n_fields} "
            f"(seed={seed})"
        )

        field = generate_soil_field(seed=seed)

        fields.append(field)

    pool = np.stack(fields)

    np.save(out_path, pool)

    print(
        f"Saved {pool.shape} to {out_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=20
    )

    parser.add_argument(
        "--seed_start",
        type=int,
        default=0
    )

    parser.add_argument(
        "--out",
        type=str,
        default="../data/field_pool_train.npy"
    )

    args = parser.parse_args()

    main(
        args.n,
        args.seed_start,
        args.out
    )
