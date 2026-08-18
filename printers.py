import gdb
import fnmatch

class ArrayPrinter:
    def __init__(self, value):
        self.pointer = value['ptr_']
        self.num = int(value['num_'])

    def to_string(self):
        return 'Array[%d]' % self.num

    def children(self):
        yield ('size', self.num)
        yield from (('[%d]' % i, self.pointer[i]) for i in range(self.num))


def ppmatcher(value):
    for wildcard in ['Span<*>', 'Array<*>']:
        if fnmatch.fnmatch(str(value.type), wildcard):
            return ArrayPrinter(value)
    return None

if ppmatcher in gdb.pretty_printers:
    del gdb.pretty_printers[gdb.pretty_printers.index(ppmatcher)]
gdb.pretty_printers.append(ppmatcher)
