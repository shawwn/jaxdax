import unittest
import subprocess
import sys
from jaxdax import core

from absl import logging
from absl.testing import absltest, parameterized

from jax._src import test_util as jtu
from jax._src.util import partial

import jax.numpy as jnp
import numpy as np
import jax

import builtins

def f(x, lib=core):
    y = lib.sin(x) * 2.0
    z = - y + x
    return z

class BasicTest(jtu.JaxTestCase):
    def check(self, f1, f2, *args, **kws):
        jval = f1(*args, **kws)
        nval = f2(*args, **kws)
        logging.info('jval: %s nval: %s', jval, nval)
        self.assertAllClose(jval, nval)

    def test_basic(self):
        logging.info('info')
        self.assertEqual(1, 1)
        fnp = partial(f, lib=jnp)
        self.check(f, fnp, 3.0)
        self.check(core.vmap(f, (0,)), jax.vmap(fnp, (0,)), np.arange(3))



class BackendsTest(jtu.JaxTestCase):

  @unittest.skipIf(not sys.executable, "test requires sys.executable")
  @jtu.skip_on_devices("gpu", "tpu")
  def test_cpu_warning_suppression(self):
    warning_expected = (
      "import jax; "
      "jax.numpy.arange(10)")
    warning_not_expected = (
      "import jax; "
      "jax.config.update('jax_platform_name', 'cpu'); "
      "jax.numpy.arange(10)")

    result = subprocess.run([sys.executable, '-c', warning_expected],
                            check=True, capture_output=True)
    assert "No GPU/TPU found" in result.stderr.decode()

    result = subprocess.run([sys.executable, '-c', warning_not_expected],
                            check=True, capture_output=True)
    assert "No GPU/TPU found" not in result.stderr.decode()


if __name__ == '__main__':
    builtins.__stdout__ = sys.__stdout__
    builtins.__stderr__ = sys.__stderr__
    logging.set_verbosity('DEBUG')
    #logging.get_absl_handler().python_handler.stream = sys.__stdout__
    logging.use_absl_handler()
    absltest.main(testLoader=jtu.JaxTestLoader())

