# Standard Library dependencies
import time

# Internal dependencies
from src.autograd.engine.symbolic.derivation import SumGroup, ProductGroup
from src.utils.types import Partial

t0: float = time.time()

X: Partial = Partial(name="X", input=None, order=0)
B: Partial = Partial(name="B", input=X, order=0)
A: Partial = Partial(name="A", input=B, order=0)

d = SumGroup(product=ProductGroup(partial=A))

for i in range(40):
    d: SumGroup = d.derivate(X)
    # how much time does it take to calculate the i-th order differential expression?
    print(f"order-{str(i+1).zfill(2)}: {time.time() - t0:.6f}")
