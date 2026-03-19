
"""
num_epoch split into N_SEG (eg. 4) segs, 
last seg interval=eval_model_interval, 2nd last eval_model_interval*2, ...
"""
from conf import *
eval_model_interval=2


N_SEG = 3
segment_length = num_epoch // N_SEG
i2interval = {}

for epoch in range(num_epoch):
    current_segment = epoch // segment_length
    if current_segment < N_SEG - 1:
        if 0:
            interval = eval_model_interval * (2 ** (N_SEG - 1 - current_segment))
        else:
            interval = eval_model_interval * ( 1+ (N_SEG - 1 - current_segment))
    else:
        interval = eval_model_interval
        
    i2interval[epoch] = interval

print(f'{i2interval=}')



def eval_gate( 
        i,
        num_epoch,
        #
        eval_model_interval,
    ):
    if 0:
        if eval_model_interval > 0 \
            and (i + 1) % eval_model_interval == 0:
                return 1
        else:
            return 0
    if 1:
        if i < len(i2interval):
            interval = i2interval[i]
        else:
            interval = eval_model_interval
        if interval > 0 and (i + 1) % interval == 0:
            return 1
        else:
            return 0
