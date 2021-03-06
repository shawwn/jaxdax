from jax._src.lib.xla_bridge import *
import numpy as np
import jax.numpy as jnp

from dataclasses import dataclass

#@dataclass
class DaxBackend:
    def __init__(self):
        self._local_devices = [DaxDevice(0)]
    #device_count: int = 1
    def device_count(self):
        return len(self._local_devices)
    @property
    def platform(self):
        return 'dax'
    @property
    def platform_version(self):
        return '<unknown>'
    def process_index(self):
        return 0
    def devices(self):
        return list(self._local_devices)
    def local_devices(self):
        return list(self._local_devices)
    def buffer_from_pyval(self, val):
        return jnp.array(val)
    def get_default_device_assignment(self, arg0: int, arg1: int = None):
        assert arg0 == 1
        if arg1 is None:
            return self.local_devices()
        else:
            return [self.local_devices()]
    def compile(self, built_c, compile_options=None):
        #print('compile', built_c, compile_options)
        result = _cpu.compile(built_c, compile_options)
        #print(result)
        return result


class DaxDevice:
    def __init__(self, index):
        self.id = index
        self.host_id = index
        self.device_kind = 'dax'
        self.platform = 'dax'
        self.process_index = 0

register_backend_factory('dax', lambda *args, **kws: (print(args, kws) or DaxBackend(*args, **kws)), priority=500)

_cpu = get_backend('cpu')

