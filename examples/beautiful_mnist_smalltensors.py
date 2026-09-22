# MNIST CNN, with the test evaluation split into sub-batches so no single forward
# pass exceeds the device's per-buffer limit (e.g. the Adreno 256 MiB cap, see
# extra/vulkan/ADRENO_256MB.md). The reported test accuracy is the sum of the
# per-batch correct counts over the FULL test set, so it is identical to a single
# (full-batch) evaluation -- just safe on devices with a small max buffer size.
#
#   EVAL_BS=1000   # test-set sub-batch size (lower it if you still hit the limit)
from typing import Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, function, Context
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist

EVAL_BS = getenv("EVAL_BS", 1000)   # test-set sub-batch size for get_test_acc

class Model:
  def __init__(self):
    self.layers: list[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10)]

  @function
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

  @TinyJit
  @Context(TRAINING=1)
  def train_step(self, X_train:Tensor, Y_train:Tensor) -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
    loss = self(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
    return loss.realize(*opt.schedule_step())

  def get_test_acc(self, X_test:Tensor, Y_test:Tensor) -> Tensor:
    # Evaluate the full test set in EVAL_BS-sized sub-batches so no single forward
    # pass exceeds the device's per-buffer limit. `correct` is the total number of
    # hits over all `n` images, so (correct/n*100) == the single-batch test accuracy.
    n = X_test.shape[0]
    correct = Tensor(0)
    for i in range(0, n, EVAL_BS):
      bc = (self(X_test[i:i+EVAL_BS]).argmax(axis=1) == Y_test[i:i+EVAL_BS]).sum()
      Tensor.realize(bc)   # compute this sub-batch now, so its buffers are freed
      correct = correct + bc
    return correct * 100 / n

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

  model = Model()
  opt = (nn.optim.Muon if getenv("MUON") else nn.optim.SGD if getenv("SGD") else nn.optim.Adam)(nn.state.get_parameters(model))

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    GlobalCounters.reset()
    loss = model.train_step(X_train, Y_train)
    if i%10 == 9: test_acc = model.get_test_acc(X_test, Y_test).item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))
