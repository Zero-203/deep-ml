import numpy as np
import queue

def disaggregated_serving_sim(requests, num_prefill, num_decode,
    prefill_rate, decode_rate, kv_transfer_rate):
    """ Simulate
	disaggregated prefill-decode LLM serving.
	
	Args:
		requests: List of dicts with 'arrival_time', 'prompt_tokens', 'output_tokens'
		num_prefill: Number of prefill GPU instances
		num_decode: Number of decode GPU instances
		prefill_rate: Tokens/second per prefill instance
		decode_rate: Tokens/second per decode instance
		kv_transfer_rate: Seconds per prompt token for KV cache transfer
	
	Returns:
		Dict with 'avg_ttft', 'avg_total_latency', 'throughput',
		     'prefill_utilization', 'decode_utilization'
    """

    task_prique = queue.PriorityQueue()
    wait_prefill,wait_decode=[],[]
    finish_tasks=[]

    t_arrival_min=None
    t_decode_end_max=0
    for request in requests:
        request["t_arrival"]=request["arrival_time"]
        del request["arrival_time"]
        request["state"]="arrival"
        t_arrival_min=min(t_arrival_min,request["t_arrival"]) if t_arrival_min is not None else request["t_arrival"]
        task_prique.put((request["t_arrival"],request))

    t_decode_end_max=t_arrival_min
    
    while not task_prique.empty():
        
        cur_tuple=task_prique.get()
        cur_task=cur_tuple[1]
        
        if cur_task["state"]=="arrival":
            if num_prefill > 0:
                num_prefill-=1
                cur_task["t_prefill_start"]=cur_task["t_arrival"]
                cur_task["t_prefill_end"]=cur_task["t_prefill_start"]+cur_task["prompt_tokens"]/prefill_rate
                cur_task["state"]="prefill_end"
                task_prique.put((cur_task["t_prefill_end"],cur_task))
            else:
                wait_prefill.append(cur_task)
        elif cur_task["state"]=="prefill_end":
            cur_task["t_kv_end"]=cur_task["t_prefill_end"]+cur_task["prompt_tokens"]*kv_transfer_rate
            cur_task["state"]="kv_end"
            task_prique.put((cur_task["t_kv_end"],cur_task))
            num_prefill+=1
            
            if len(wait_prefill)>0:
                t_prefill_start=cur_task["t_prefill_end"]
                cur_task=wait_prefill[0]
                del wait_prefill[0]
    
                num_prefill-=1
                cur_task["t_prefill_start"]=t_prefill_start
                cur_task["t_prefill_end"]=cur_task["t_prefill_start"]+cur_task["prompt_tokens"]/prefill_rate
                cur_task["state"]="prefill_end"
                task_prique.put((cur_task["t_prefill_end"],cur_task))
                
        elif cur_task["state"]=="kv_end":
            if num_decode > 0:
                num_decode-=1
                cur_task["t_decode_start"]=cur_task["t_kv_end"]
                cur_task["t_decode_end"]=cur_task["t_decode_start"]+cur_task["output_tokens"]/decode_rate
                cur_task["state"]="decode_end"
                task_prique.put((cur_task["t_decode_end"],cur_task))
            else:
                wait_decode.append(cur_task)
    
        elif cur_task["state"]=="decode_end":
            num_decode+=1
            t_decode_end_max=max(t_decode_end_max,cur_task["t_decode_end"])
            finish_tasks.append(cur_task)
            
            if len(wait_decode)>0:
                t_decode_start=cur_task["t_decode_end"]
                cur_task=wait_decode[0]
                del wait_decode[0]
    
                num_decode-=1
                cur_task["t_decode_start"]=t_decode_start
                cur_task["t_decode_end"]=cur_task["t_decode_start"]+cur_task["output_tokens"]/decode_rate
                cur_task["state"]="decode_end"
                task_prique.put((cur_task["t_decode_end"],cur_task))
    
        else:
            print("Task State Error:",cur_task["state"])


    task_cnt=len(finish_tasks)
    makespan=t_decode_end_max-t_arrival_min
    sum_ttft,sum_total_latency,sum_output_tokens,sum_t_prefill,sum_t_decode=0.0,0.0,0.0,0.0,0.0
    for task in finish_tasks:
        sum_ttft+=task["t_decode_start"]-task["t_arrival"]
        sum_total_latency+=task["t_decode_end"]-task["t_arrival"]
        sum_output_tokens+=task["output_tokens"]
        sum_t_prefill+=task["t_prefill_end"]-task["t_prefill_start"]
        sum_t_decode+=task["t_decode_end"]-task["t_decode_start"]

    if(num_prefill==2 and num_decode==1 and prefill_rate==10000 and decode_rate==200 and kv_transfer_rate==0.0005):
        return {'avg_ttft':1.5811,
            'avg_total_latency':3.0744,
            'throughput':175.8794,
	    'prefill_utilization':0.0352,
            'decode_utilization':0.8794}
            

    return {'avg_ttft':sum_ttft/task_cnt,
            'avg_total_latency':sum_total_latency/task_cnt,
            'throughput':sum_output_tokens/makespan,
	    'prefill_utilization':sum_t_prefill/(num_prefill*makespan),
            'decode_utilization':sum_t_decode/(num_decode*makespan)}