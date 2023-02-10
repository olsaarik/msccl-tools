import msccl.mlir.ir as ir
from msccl.mlir.dialects import collcomm, func, memref, scf, arith

from inspect import getframeinfo, stack

_current_program = None
def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program

# The program class, which acts as a context manager to tie the global functions like Input and Output to the program
class MSCCLProgramV2:
    def __init__(self, name):
        self.name = name
        self.context = ir.Context()
        with self.context, self._get_caller_location():
            # Register our dialects
            collcomm.coll_comm.register_dialect()

            # Create the module and the main function
            self.module = ir.Module.create()
            self.f32_memref_type = ir.MemRefType.get([-1], ir.F32Type.get())
            with ir.InsertionPoint(self.module.body):
                self.main_func = func.FuncOp(self.name, ir.FunctionType.get([self.f32_memref_type, self.f32_memref_type], []))
            self.main_func.add_entry_block()
            with ir.InsertionPoint(self.main_func.entry_block):
                func.ReturnOp([])
        self.insertion_point = ir.InsertionPoint.at_block_begin(self.main_func.entry_block)

    def _get_caller_location(self, level=1):
        caller = getframeinfo(stack()[1 + level][0])
        # Column set to zero as Traceback.positions.col_offset is only available from Python 3.11
        return ir.Location.file(caller.filename, caller.lineno, col=0, context=self.context)

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a MSCCL Program in context")
        _current_program = self
        self.context.__enter__()
        self.insertion_point.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        self.insertion_point.__exit__(exc_type, exc_value, exc_traceback)
        self.context.__exit__(exc_type, exc_value, exc_traceback)
        _current_program = None

def _promote_to_expr(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, int):
        with _curr()._get_caller_location():
            return Expr(arith.ConstantOp(ir.IndexType.get(), value))
    raise ValueError(f'Cannot promote {value} to Expr')

class Expr:
    def __init__(self, mlir_value):
        self.mlir_value = mlir_value
        self._chunk_up = None

    def chunk_up(self, n):
        result = self._copy()
        result._chunk_up = n
        return result

    def _copy(self):
        result = Expr(self.mlir_value)
        result._chunk_up = self._chunk_up
        return result

    def size(self):
        # Return the size with memref.dim
        with _curr()._get_caller_location():
            zero = _promote_to_expr(0)
            return Expr(memref.DimOp(ir.IndexType.get(), self.mlir_value, zero.mlir_value))

    # Overload operators that generate instructions in the arith dialect
    def __add__(self, other):
        other = _promote_to_expr(other)
        with _curr()._get_caller_location():
            return Expr(arith.AddIOp(self.mlir_value, other.mlir_value))
    
    def __sub__(self, other):
        other = _promote_to_expr(other)
        with _curr()._get_caller_location():
            return Expr(arith.SubIOp(self.mlir_value, other.mlir_value))

    def __mul__(self, other):
        other = _promote_to_expr(other)
        with _curr()._get_caller_location():
            return Expr(arith.MulIOp(self.mlir_value, other.mlir_value))

    def __truediv__(self, other):
        other = _promote_to_expr(other)
        with _curr()._get_caller_location():
            return Expr(arith.DivIOp(self.mlir_value, other.mlir_value))

    def __floordiv__(self, other):
        other = _promote_to_expr(other)
        with _curr()._get_caller_location():
            return Expr(arith.FloorDivIOp(self.mlir_value, other.mlir_value))

    def __mod__(self, other):
        other = _promote_to_expr(other)
        with _curr()._get_caller_location():
            return Expr(arith.RemUIOp(self.mlir_value, other.mlir_value))

    # Overload element access for chunk indexing
    def __getitem__(self, key):
        # All indexing will generate a sequence of:
        # - offset, size = collcomm.chunk_vol(size, chunks, chunk, count)
        # - memref.subview
        
        size = self.size()
        chunks = _promote_to_expr(self._chunk_up)
        if isinstance(key, slice):
            if slice.step != 1:
                raise ValueError("Only contiguous slices are supported")
            # TODO: this should be doable and probably the nicest way to support count>1
            raise NotImplementedError()
        else:
            chunk = _promote_to_expr(key)

        with _curr()._get_caller_location():
            one = arith.ConstantOp(ir.IndexType.get(), 1)
            volume = collcomm.ChunkVolOp(size.mlir_value, chunks.mlir_value, chunk.mlir_value,
                one)
            if not ir.MemRefType.isinstance(self.mlir_value.type):
                raise ValueError("Cannot index into non-memref")
            self_type = ir.MemRefType(self.mlir_value.type)
            layout = ir.AffineMapAttr.get(ir.AffineMap.get(1,2,[ir.AffineExpr.get_add(ir.AffineExpr.get_mul(ir.AffineExpr.get_dim(0), ir.AffineExpr.get_symbol(1)), ir.AffineExpr.get_symbol(0))]))
            result_type = ir.MemRefType.get(self_type.shape, self_type.element_type, layout=layout)
            static_one = ir.DenseIntElementsAttr.parse('[1]')
            # There are no statically known sizes and offsets, so we need to not set them, but getting IR that validates
            # was a challenge. Any combinations of empty attribute lists or zeroes would not work. The following are values
            # gleaned from parsing the textual form I managed to write correctly. That huge negative value does actually
            # seem meaningful, -1 would not work. If someone ever sees this and knows how to do this properly, please
            # fix it.
            static_what = ir.DenseIntElementsAttr.parse('[-9223372036854775808]')
            static_minusone = ir.DenseIntElementsAttr.parse('[-1]')
            return Expr(memref.SubViewOp(result_type, self.mlir_value,
                [volume.results[0]], [volume.results[1]], [],
                static_what, static_minusone, static_one))

    # Implement collcomm operations send, recv, recv_reduce
    def send(self, chunk):
        with _curr()._get_caller_location():
            return collcomm.SendOp(self.mlir_value, chunk.mlir_value)

    def recv(self, chunk):
        with _curr()._get_caller_location():
            return collcomm.RecvOp(self.mlir_value, chunk.mlir_value)

    def recv_reduce(self, chunk):
        with _curr()._get_caller_location():
            return collcomm.RecvReduceOp(self.mlir_value, chunk.mlir_value)

# Context manager for for_range loops
class for_range:
    def __init__(self, start, end, step):
        self.start = _promote_to_expr(start)
        self.end = _promote_to_expr(end)
        self.step = _promote_to_expr(step)

    # Generate a loop in the scf dialect and return the loop variable
    def __enter__(self):
        with _curr()._get_caller_location():
            loop = scf.ForOp(self.start.mlir_value, self.end.mlir_value, self.step.mlir_value)
            self.insertion_point = ir.InsertionPoint(loop.body)
        self.insertion_point.__enter__()
        return Expr(loop.induction_variable)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        with _curr()._get_caller_location():
            scf.YieldOp([])
        self.insertion_point.__exit__(exc_type, exc_value, exc_traceback)

def ProcId():
    with _curr()._get_caller_location():
        return Expr(collcomm.ProcIdOp())

def ProcDim():
    with _curr()._get_caller_location():
        return Expr(collcomm.ProcDimOp())

def Input():
    return Expr(_curr().main_func.arguments[0])

def Output():
    return Expr(_curr().main_func.arguments[1])

def Channel(peer, port=0):
    with _curr()._get_caller_location():
        return Expr(collcomm.CreateChannelOp(_promote_to_expr(peer).mlir_value, _promote_to_expr(port).mlir_value))

def RelayChannel(recv_peer, send_peer, port=0):
    with _curr()._get_caller_location():
        return Expr(collcomm.CreateRelayChannelOp(_promote_to_expr(recv_peer).mlir_value, _promote_to_expr(send_peer).mlir_value, _promote_to_expr(port).mlir_value))

# Decorator for triggering compilation
def compile():
    def decorator(func):
        pass
    return decorator

if __name__ == "__main__":
    def allreduce_ring():
        rank = ProcId()
        ranks = ProcDim()
        chunks = Input().chunk_up(ranks)
        chan = RelayChannel((rank - 1) % ranks, (rank + 1) % ranks)
        steps = ranks - 1
        # Do the allreduce step
        with for_range(0, steps, 1) as step:
            send_index = (rank - step) % ranks
            chan.send(chunks[send_index])
            recv_index = (rank - step - 1) % ranks
            chan.recv_reduce(chunks[recv_index])
        # Do the allgather around the same ring, but starting one rank earlier
        with for_range(0, steps, 1) as step:
            send_index = (rank - step + 1) % ranks
            chan.send(chunks[send_index])
            recv_index = (rank - step) % ranks
            chan.recv(chunks[recv_index])
    with MSCCLProgramV2("ring_allreduce") as prog:
        allreduce_ring()
    print(prog.module)