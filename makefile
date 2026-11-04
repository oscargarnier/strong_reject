alloc:
	salloc --account=c2016159 \
       --constraint=MI250 \
       --job-name=dynamic_allocation_qwen \
       --ntasks=1 \
       --nodes=1 \
       --gres=gpu:1 \
       --mem=32G \
       --cpus-per-task=4 \
       --threads-per-core=1 \
       --time=01:00:00

small_alloc:
	salloc --account=c2016159 \
       --constraint=MI250 \
       --job-name=dynamic_allocation_small \
       --ntasks=1 \
       --nodes=1 \
       --gres=gpu:1 \
       --mem=16G \
       --cpus-per-task=4 \
       --threads-per-core=1 \
       --time=01:00:00
