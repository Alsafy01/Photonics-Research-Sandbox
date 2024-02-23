
"""
    Reference: @ARTICLE{9716852,
  author={Rahman, Lubaba Tazrian and Akhter, Mahmud Elahi, and Sayem, Faizul Rakib and Hossain, Mainul and Ahmed, Rajib and Elahi, M. M. Lutfe and Ali, Khaleda and Islam, Sharnali},
  journal={IEEE Access},
  title={A 1.55 μm Wideband 1 × 2 Photonic Power Splitter With Arbitrary Ratio: Characterization and Forward Modeling},
  year={2022},
  volume={10},
  number={},
  pages={20149-20158},
  doi={10.1109/ACCESS.2022.3151722}}

  RunSample from Terminal:  python main.py Al2O3_TE.csv 2
"""

import argparse
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Visualize Hole Vectors')
    parser.add_argument('hv_csv', default=None, metavar='HV_csv', type=argparse.FileType('r'),
                        help="Path to the hole vector csv file")
    parser.add_argument('index', default=None, metavar='INDEX', type=int, help="sample choice")
    args = parser.parse_args()

    df = pd.read_csv(args.hv_csv)

    print(df.describe())
    print('')
    print(df.head(5))
    print('')

    HV = df.iloc[:, 0:400]

    image = np.array(HV.loc[[args.index]]).astype(np.float32).reshape(20, 20)
    title = "Hole Vector sample " + f'{args.index}'

    k = plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.show()
    k.savefig(title + '.jpg', dpi=300)
    print("generated")


if __name__ == '__main__':
    main()
