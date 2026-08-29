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

class RawPrinter:
    def __init__(self, value):
        ctype = value.type.template_argument(0)
        self.bytes = value['bytes_']
        self.element = self.bytes.cast(ctype.pointer())[0]

    def to_string(self):
        return 'Raw: ' + str(self.element)

    def children(self):
        yield ('*', self.element)
        yield ('[raw]', self.bytes)


def ppmatcher(value):
    for wildcard in ['tbbss::Span<*>', 'tbbss::Array<*,*>']:
        if fnmatch.fnmatch(str(value.type), wildcard):
            return ArrayPrinter(value)
    for wildcard in ['tbbss::Raw<*>']:
        if fnmatch.fnmatch(str(value.type), wildcard):
            return RawPrinter(value)
    return None

if ppmatcher in gdb.pretty_printers:
    del gdb.pretty_printers[gdb.pretty_printers.index(ppmatcher)]
gdb.pretty_printers.append(ppmatcher)
