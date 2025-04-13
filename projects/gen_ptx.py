from typing import List
from tinygrad.helpers import tqdm


from tinygrad import Tensor, dtypes

a = Tensor([2], dtype=dtypes.int32)
b = Tensor([3], dtype=dtypes.int32)
out = a + b

# *****
# 3. Create a schedule.

# The weight Tensors have been assigned to, but not yet realized. Everything is still lazy at this point
# l1.lazydata and l2.lazydata define a computation graph

from tinygrad.engine.schedule import ScheduleItem
schedule: List[ScheduleItem] = Tensor.schedule(out)

print(f"The schedule contains {len(schedule)} items.")
for si in schedule: print(str(si)[:80])

s = schedule[0].ast

# convert the computation to a "linearized" format (print the format)
from tinygrad.engine.realize import get_kernel, CompiledRunner
from tinygrad.renderer.ptx import PTXRenderer
kernel = get_kernel(PTXRenderer(arch="999999"), s).linearize()

# compile a program (and print the source)
fxn = CompiledRunner(kernel.to_program())
print(fxn.p.src)
# NOTE: fxn.clprg is the CPUProgram
# *****
# 4. Lower a schedule.
import sys; sys.exit(0)

from tinygrad.engine.realize import lower_schedule_item, ExecItem
lowered: List[ExecItem] = [lower_schedule_item(si) for si in tqdm(schedule)]

# *****
# 5. Run the schedule

for ei in tqdm(lowered): ei.run()

# *****
# 6. Print the weight change

print("first weight change\n", l1.numpy()-l1n)
print("second weight change\n", l2.numpy()-l2n)
