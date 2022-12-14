# %%

from project.Benchmarks_base.benchmark import Benchmark
import numpy as np
import numpy.typing as npt
from typing import Annotated, Tuple

number = int | float | complex

# %%

base_params = (1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
               0.01, 1, 1, 0)


class Mishra4(Benchmark):

    def __init__(self, params: Annotated[Tuple, 14] = base_params,
                 lower: npt.NDArray = np.array([-10, -10]),
                 upper: npt.NDArray = np.array([10, 10]), D=2):
        super(Mishra4, self).__init__(params, lower, upper)

    def _get_value(self, inp: npt.NDArray) -> npt.NDArray:
        """Get the value of a list of points

        Args:
            inp (npt.ArrayLike[npt.NDArray]): point matrix of shape (N, 2)

        Returns:
            npt.NDArray: list of values of shape (N)
        """
        x_1 = inp[:, 0]
        x_2 = inp[:, 1]

        p = self.params

        s_0 = p[2] * np.sqrt(np.abs(p[3]*(x_1+p[4])**2 +
                             p[5]*(x_2+p[6])**2 + p[7]))
        s_1 = p[0] * np.sqrt(np.abs(p[1] * np.sin(s_0 + p[8]) + p[9]))
        s_2 = p[10] * (p[11]*x_1 + p[12]*x_2)

        return s_1 + s_2 + p[13]

    def _to_latex(self, r=5, line_breaks=True, parameterized=False, double=False):
        p_f = np.round(self.params, r)

        # all parameters that multiply
        p = ["" for _ in p_f]
        for i in [0, 1, 2, 3, 5, 10, 11, 12]:
            p[i] = "" if p_f[i] == 1 else str(p_f[i])

        # all parameters that add
        for i in [4, 6, 7, 8, 9, 13]:
            p[i] = "" if p_f[i] == 0 else str(p_f[i])

        if parameterized:
            p = [f"p_{'{'}{i}{'}'}" for i in range(len(p))]

        res = f"${p[0]}\sqrt{'{'}|{p[1]}\sin({p[2]}\sqrt{'{'}|{p[3]}(x_1+{p[4]})^2 + {p[5]}(x_2+{p[6]})^2 + {p[7]}|{'}'} + {p[8]}) + {p[9]}|{'}'} + {p[10]}({p[11]}x_1 + {p[12]}x_2) + {p[13]}$"

        if line_breaks:
            res = f"${p[0]}\sqrt{'{'}|{p[1]}\sin({p[2]}\sqrt{'{'}|{p[3]}(x_1+{p[4]})^2 + {p[5]}(x_2+{p[6]})^2 + {p[7]}|{'}'} + {p[8]}) + {p[9]}|{'}'} +$ \\\\ ${p[10]}({p[11]}x_1 + {p[12]}x_2) + {p[13]}$"

        return res
