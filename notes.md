# 2 - LLM training - Single GPU retrospective and Data Parallel training

What did we do in experiment 1?

- Single GPU pre training
- Compute optimality focus; not getting the best model out
- Optimizations - data loader, torch compile, triton kernel for attention, mixed precision, gradient accumulation
- preflight runs to validate setup and a hero run

What would be nice to do in experiment 2?

- Want to extend to data parallel pre-training
- Getting the best model is still not the focus but do we start thinking about data or evals? probably not — scope creep
- ofc naive data parallelism is bad — maybe don’t even run it, just show the numbers
- model can’t be the same size probably to show the effects of communication b/w gpus
- additional metrics to track? while keeping the end-to-end metrics still a priority; by end-to-end metrics I mean
    - tokens per seconds — training throughput
    - model flops utilization
    - loss curves — we need it to go down in general; ideally no spikes; no tough expectations here

Uncertainties

- cost
- should I still have this fixed compute budget framing or constraint; ofc there needs to be a constraint but what about the framing — we should probably say something like “efficient data parallel pre-training of 1B parameter model” but there’s no punch factor like 30 hours gpu budget
    - model size is quantified; what else can be?
    - efficiency can be quantified; but it’d depend on final efficiency numbers i achieve, it has to be a good number for us to put it in the headline
    - we could include corpus size, #gpus, gpu hours etc.
    - either way we should decide on the title post experiments
    - but we should decide on the goal up front

Plan of Action

1. Single GPU baseline for the 1B model
    1. just swap the config
    2. tune micro batch size
    3. record tokens/sec, MFU, mem usage
2. Naive DP
    1. torch.distributed boilerplate - process group, rank, world size
    2. manually call all_reduce on gradients
    3. run on 2,4,8 gpus and record metrics
    4. also measure comm vs comp time per training step
3. Bucketed all-reduce
    1. different bucket sizes showing latency vs bandwidth tradeoff
    2. re-run sweep and measure improvement
4. Overlap comms with backward calls
    1. gradient hooks implementation
    2. re-run sweep and measure
5. Hero run
    1. Pick best config from phase 4
    2. Pre-flight run - few hundred steps, stability and loss decreasing
    3. Hero run - full training, final scaling efficiency and other metrics
6. Write-up